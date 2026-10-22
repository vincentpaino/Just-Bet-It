"""
The Odds API ingestion module.
Implements the "Fetch Odds" -> "Validate & Format" -> "Insert" path
from Doc II, Figure 3.1.

Docs: https://the-odds-api.com/liveapi/guides/v4/
"""
import os
from datetime import datetime
from typing import Any, Optional

import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_ncaaf"


class OddsAPIError(Exception):
    pass


def _api_get(path: str, **params) -> requests.Response:
    """GET an Odds API endpoint under /sports/<SPORT_KEY>, raising on failure."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise OddsAPIError("ODDS_API_KEY is not set in the environment")

    url = f"{ODDS_API_BASE_URL}/sports/{SPORT_KEY}/{path}"
    response = requests.get(url, params={"apiKey": api_key, **params}, timeout=15)
    if response.status_code != 200:
        raise OddsAPIError(
            f"Odds API request failed ({response.status_code}): {response.text}"
        )
    return response


def fetch_odds(
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
) -> list[dict[str, Any]]:
    """Pull current NCAA football odds across all covered sportsbooks."""
    response = _api_get(
        "odds", regions=regions, markets=markets, oddsFormat=odds_format
    )

    # The Odds API returns remaining/used quota in response headers —
    # worth logging so the team can watch the 500 req/mo free-tier limit
    # called out in Doc III, Section 5.1.
    remaining = response.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"[odds_api] requests remaining this period: {remaining}")

    return response.json()


def _implied_prob_from_american(price: int) -> Optional[float]:
    """Convert an American moneyline price to an implied probability."""
    if price is None:
        return None
    if price > 0:
        return 100 / (price + 100)
    return -price / (-price + 100)


def swap_home_away(row: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of an odds row seen from the other side: home and away
    values trade places and the (home) spread flips sign. Used when a game is
    stored with home/away the other way round (common at neutral sites).
    """
    swapped = dict(row)

    for home_key, away_key in [
        ("home_team", "away_team"),
        ("home_moneyline", "away_moneyline"),
        ("implied_prob_home", "implied_prob_away"),
    ]:
        swapped[home_key], swapped[away_key] = row.get(away_key), row.get(home_key)

    if row.get("spread") is not None:
        swapped["spread"] = -row["spread"]

    return swapped


def validate_and_format(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten the nested Odds API response into one row per
    (game, sportsbook) pair matching the `odds` table shape.
    Rows that are missing required fields are dropped rather than
    inserted with nulls in required columns.
    """
    rows = []
    for event in raw_events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        if not (home_team and away_team and commence_time):
            continue

        for bookmaker in event.get("bookmakers", []):
            row = {
                "sportsbook": bookmaker.get("key"),
                "home_team": home_team,
                "away_team": away_team,
                "game_date": commence_time,
                "home_moneyline": None,
                "away_moneyline": None,
                "spread": None,
                "over_under": None,
            }

            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = market.get("outcomes", [])

                if key == "h2h":
                    for outcome in outcomes:
                        if outcome.get("name") == home_team:
                            row["home_moneyline"] = outcome.get("price")
                        elif outcome.get("name") == away_team:
                            row["away_moneyline"] = outcome.get("price")

                elif key == "spreads":
                    for outcome in outcomes:
                        if outcome.get("name") == home_team:
                            row["spread"] = outcome.get("point")

                elif key == "totals":
                    if outcomes:
                        row["over_under"] = outcomes[0].get("point")

            row["implied_prob_home"] = _implied_prob_from_american(row["home_moneyline"])
            row["implied_prob_away"] = _implied_prob_from_american(row["away_moneyline"])
            row["fetched_at"] = datetime.utcnow()
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# QB passing-yards props
# ---------------------------------------------------------------------------

def _remaining_credits(response: requests.Response) -> Optional[int]:
    """Credits left this period, from the quota header (None if absent)."""
    remaining = response.headers.get("x-requests-remaining")
    try:
        return int(float(remaining))
    except (TypeError, ValueError):
        return None


def fetch_events() -> tuple[list[dict[str, Any]], Optional[int]]:
    """
    List upcoming NCAA football events (id, teams, kickoff), plus the credits
    remaining this period. This endpoint costs no credits.
    """
    response = _api_get("events")
    return response.json(), _remaining_credits(response)


def fetch_event_props(
    event_id: str,
    regions: str = "us",
    markets: str = "player_pass_yds",
    odds_format: str = "american",
) -> tuple[dict[str, Any], Optional[int]]:
    """
    Pull player prop odds for one event, plus the credits remaining after.
    Costs 1 credit per market returned per region; a response with no odds
    (props not posted yet) costs nothing.
    """
    response = _api_get(
        f"events/{event_id}/odds",
        regions=regions, markets=markets, oddsFormat=odds_format,
    )
    return response.json(), _remaining_credits(response)


def validate_and_format_props(event: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten one event's player_pass_yds odds into one row per
    (sportsbook, player), matching the `qb_prop_lines` table shape.
    A book can list several lines for a player (alternates), but the table
    holds one line per book per snapshot, so only the main line is kept:
    the one priced on both sides with over and under closest to even.
    A book may list only one side, so over/under odds can be None.
    """
    home_team = event.get("home_team")
    away_team = event.get("away_team")
    commence_time = event.get("commence_time")
    if not (home_team and away_team and commence_time):
        return []

    fetched_at = datetime.utcnow()
    rows = {}

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "player_pass_yds":
                continue

            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                line = outcome.get("point")
                side = (outcome.get("name") or "").lower()
                if not player or line is None or side not in ("over", "under"):
                    continue

                key = (bookmaker.get("key"), player, line)
                row = rows.setdefault(key, {
                    "sportsbook": bookmaker.get("key"),
                    "player_name": player,
                    "line": line,
                    "over_odds": None,
                    "under_odds": None,
                    "home_team": home_team,
                    "away_team": away_team,
                    "game_date": commence_time,
                    "fetched_at": fetched_at,
                })
                row[f"{side}_odds"] = outcome.get("price")

    def main_line_rank(row):
        if row["over_odds"] is None or row["under_odds"] is None:
            return (1, 0.0)
        gap = abs(_implied_prob_from_american(row["over_odds"])
                  - _implied_prob_from_american(row["under_odds"]))
        return (0, gap)

    main_lines = {}
    for row in sorted(rows.values(), key=main_line_rank):
        main_lines.setdefault((row["sportsbook"], row["player_name"]), row)

    return list(main_lines.values())
