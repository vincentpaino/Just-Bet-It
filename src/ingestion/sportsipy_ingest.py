"""
Sportsipy ingestion module.
Implements the "Fetch Stats" -> "Validate & Format" path from Doc II,
Figure 3.1, using sportsipy.ncaafb.

Note (Doc III, Section 5.2 — Road Blocks): sportsipy scrapes
Sports-Reference.com and is not officially maintained. If a season's
data comes back incomplete or the library breaks, fall back to a
manual CSV pull from Sports-Reference and load it with
load_stats_from_csv() below.
"""
from typing import Any

import pandas as pd

try:
    from sportsipy.ncaaf.teams import Teams
except ImportError:
    Teams = None


class SportsipyUnavailableError(Exception):
    pass


def fetch_team_season_stats(year: int) -> pd.DataFrame:
    """
    Pull one season's worth of team-level stats via sportsipy.
    Returns a DataFrame — one row per team for that season.
    """
    if Teams is None:
        raise SportsipyUnavailableError(
            "sportsipy is not installed or failed to import. "
            "Fall back to load_stats_from_csv() with a Sports-Reference export."
        )

    teams = Teams(year=str(year))
    frames = []
    for team in teams:
        df = team.dataframe.copy()
        df["season"] = year
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_stats_from_csv(path: str) -> pd.DataFrame:
    """
    Fallback path per Doc III Section 5.2: load a manually downloaded
    Sports-Reference CSV export when sportsipy is unavailable.
    """
    return pd.read_csv(path)


def validate_and_format(raw_stats: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Clean the raw sportsipy/CSV output into the shape the `teams` and
    `games` tables expect.
    """
    if raw_stats.empty:
        return []

    df = raw_stats.copy()
    df = df.rename(columns={
        "abbreviation": "team_abbreviation",
        "name": "team_name",
    })

    if "team_name" not in df.columns:
        raise ValueError(
            "Expected a 'team_name' (or 'name') column in the raw stats — "
            "check whether the sportsipy schema has changed."
        )

    df = df.dropna(subset=["team_name"])
    return df.to_dict(orient="records")