"""
CollegeFootballData.com (CFBD) ingestion module.
Replaces sportsipy as the historical stats source — sportsipy scrapes
Sports-Reference.com, which now returns 403 Forbidden on scraping
attempts (confirmed during Sprint 1 testing). CFBD is a real,
actively-maintained REST API instead of a scraper, so it isn't
subject to the same blocking risk.

Docs: https://collegefootballdata.com
Get a free API key there (1,000 requests/month free tier).
"""
import os
from typing import Any

import pandas as pd

try:
    import cfbd
except ImportError:
    cfbd = None


class CFBDUnavailableError(Exception):
    pass


def _get_api_client() -> "cfbd.ApiClient":
    if cfbd is None:
        raise CFBDUnavailableError(
            "cfbd package is not installed. Run: pip install cfbd"
        )

    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise CFBDUnavailableError("CFBD_API_KEY is not set in the environment")

    configuration = cfbd.Configuration(
        host="https://api.collegefootballdata.com",
        access_token=api_key,
    )
    return cfbd.ApiClient(configuration)


def fetch_team_season_stats(year: int) -> pd.DataFrame:
    """
    Pull one season's worth of game-level results via CFBD.
    Returns a DataFrame — one row per game for that season.
    """
    with _get_api_client() as api_client:
        games_api = cfbd.GamesApi(api_client)
        games = games_api.get_games(year=year, classification="fbs")

    if not games:
        return pd.DataFrame()

    records = [game.to_dict() for game in games]
    return pd.DataFrame(records)


def validate_and_format(raw_games: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Clean CFBD's raw game-result output into the shape our `games`
    table expects. Rows missing a home/away team are dropped.
    """
    if raw_games.empty:
        return []

    df = raw_games.copy()

    rename_map = {
        "homeTeam": "home_team",
        "awayTeam": "away_team",
        "startDate": "game_date",
        "homePoints": "home_score",
        "awayPoints": "away_score",
    }
    df = df.rename(columns=rename_map)

    required = {"home_team", "away_team", "game_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {required} in CFBD response, missing: {missing} — "
            "check whether the CFBD API schema has changed."
        )

    df = df.dropna(subset=["home_team", "away_team"])
    return df.to_dict(orient="records")