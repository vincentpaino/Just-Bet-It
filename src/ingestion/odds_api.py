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


def fetch_odds(
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
) -> list[dict[str, Any]]:
    """Pull current NCAA football odds across all covered sportsbooks."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise OddsAPIError("ODDS_API_KEY is not set in the environment")

    url = f"{ODDS_API_BASE_URL}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }

    response = requests.get(url, params=params, timeout=15)
    if response.status_code != 200:
        raise OddsAPIError(
            f"Odds API request failed ({response.status_code}): {response.text}"
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
