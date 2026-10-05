"""
CFBD (CollegeFootballData.com) fetching + cleaning.

Nothing in this file touches the database. It returns plain dicts / DataFrames
that run_ingestion.py writes to PostgreSQL.
"""

import os
from datetime import datetime

import pandas as pd

from .dates import as_utc_datetime, to_game_date

try:
    import cfbd
except ImportError:
    cfbd = None


class CFBDUnavailableError(Exception):
    pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _get_api_client():
    """Create and return a CFBD API client."""
    if cfbd is None:
        raise CFBDUnavailableError(
            "cfbd package is not installed. Run: pip install cfbd"
        )

    api_key = os.environ.get("CFBD_API_KEY")

    if not api_key:
        raise CFBDUnavailableError(
            "CFBD_API_KEY is not set in the environment"
        )

    configuration = cfbd.Configuration(
        host="https://api.collegefootballdata.com",
        access_token=api_key,
    )

    return cfbd.ApiClient(configuration)


def _clean(value):
    """NaN / NaT -> None (PostgreSQL needs NULL, not NaN)."""
    if value is None:
        return None

    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _int_or_none(value):
    value = _clean(value)
    return None if value is None else int(value)


def _enum_value(value):
    """CFBD returns enums for some fields; return a plain string."""
    value = _clean(value)

    if value is None:
        return None

    return str(getattr(value, "value", value))


def _to_int(stat):
    """Parse an int from a CFBD stat dict like {'name': ..., 'stat': '312'}."""
    if not stat:
        return None

    try:
        return int(stat["stat"])
    except (KeyError, ValueError, TypeError):
        return None


def _parse_completions_attempts(stat):
    """Parse a 'C/ATT' stat dict such as {'stat': '18/27'} -> (18, 27)."""
    if not stat:
        return None, None

    value = stat.get("stat")

    if value and "/" in value:
        try:
            completions, attempts = value.split("/")
            return int(completions), int(attempts)
        except ValueError:
            pass

    return None, None


def _fetch_games(games_api, year):
    """Single place that defines which games we pull for a season."""
    return games_api.get_games(
        year=year,
        classification="fbs",
        season_type="both",
    )


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

def fetch_team_season_stats(year):
    """
    Fetch FBS game information for a season (one row per game).
    """

    with _get_api_client() as api_client:
        games = _fetch_games(cfbd.GamesApi(api_client), year)

    if not games:
        return pd.DataFrame()

    return pd.DataFrame([game.to_dict() for game in games])


def validate_and_format(raw_games):
    """
    Validate and format CFBD game records.

    Each returned dict contains:
        cfbd_game_id, home_team, away_team, start_time (UTC, tz-aware),
        start_time_tbd, game_date (U.S. calendar day), season, week,
        season_type, neutral_site, home_score, away_score
    Missing numbers are None, never NaN.
    """

    if raw_games.empty:
        return []

    # The CFBD Python package returns camelCase field names from to_dict().
    df = raw_games.rename(
        columns={
            "id": "cfbd_game_id",
            "homeTeam": "home_team",
            "awayTeam": "away_team",
            "startDate": "start_time",
            "startTimeTBD": "start_time_tbd",
            "seasonType": "season_type",
            "neutralSite": "neutral_site",
            "homePoints": "home_score",
            "awayPoints": "away_score",
        }
    )

    required = {
        "cfbd_game_id",
        "home_team",
        "away_team",
        "start_time",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"CFBD game data missing required columns: {missing}"
        )

    df = df.dropna(subset=["home_team", "away_team"])

    records = []

    for row in df.to_dict(orient="records"):

        start_time = as_utc_datetime(_clean(row.get("start_time")))
        cfbd_game_id = _clean(row.get("cfbd_game_id"))

        if start_time is None or cfbd_game_id is None:
            continue

        time_tbd = bool(_clean(row.get("start_time_tbd")) or False)

        neutral = _clean(row.get("neutral_site"))

        records.append({
            "cfbd_game_id": int(cfbd_game_id),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "start_time": start_time,
            "start_time_tbd": time_tbd,
            "game_date": to_game_date(start_time, time_tbd),
            "season": _int_or_none(row.get("season")),
            "week": _int_or_none(row.get("week")),
            "season_type": _enum_value(row.get("season_type")),
            "neutral_site": None if neutral is None else bool(neutral),
            "home_score": _int_or_none(row.get("home_score")),
            "away_score": _int_or_none(row.get("away_score")),
        })

    return records


# ---------------------------------------------------------------------------
# Quarterbacks
# ---------------------------------------------------------------------------

def fetch_qb_season_stats(year):
    """
    Fetch game-level statistics for quarterbacks.

    CFBD's player game statistics endpoint requires a week, team, or
    conference, so the data is requested one week at a time.

    Steps:
        1. Get all FBS games for the season.
        2. Get all players whose position is QB.
        3. Get player game statistics week-by-week.
        4. Match each stat record to its game information.
        5. Extract passing and rushing statistics.
        6. Return flat records ready for database insertion.
    """

    season_is_over = year < datetime.now().year

    with _get_api_client() as api_client:
        games_api = cfbd.GamesApi(api_client)
        players_api = cfbd.PlayersApi(api_client)

        # 1. Games ---------------------------------------------------------

        games = _fetch_games(games_api, year)

        games_by_id = {}

        for game in games:
            game_data = game.to_dict()

            if game_data.get("id") is not None:
                games_by_id[game_data["id"]] = game_data

        # 2. QB player IDs -------------------------------------------------

        qb_ids = set()

        for player in players_api.get_player_usage(year=year):
            player_id = getattr(player, "id", None)

            if getattr(player, "position", None) == "QB" and player_id is not None:
                qb_ids.add(str(player_id))

        print(f"[cfbd_qb] Found {len(qb_ids)} QB players")

        # 3. Player game stats, week by week -------------------------------
        # Weeks come from the games actually returned, so seasons with a
        # different number of weeks (and postseason) are handled.

        weeks = sorted(
            {game.week for game in games if game.week is not None}
        )

        print(f"[cfbd_qb] Fetching QB stats for weeks: {weeks}")

        player_game_stats = []
        failed_weeks = []

        for week in weeks:
            try:
                weekly_stats = games_api.get_game_player_stats(
                    year=year,
                    week=week,
                    season_type="both",
                )
                player_game_stats.extend(weekly_stats)
                print(
                    f"[cfbd_qb] Week {week}: fetched {len(weekly_stats)} "
                    f"player-game stat records"
                )
            except Exception as exc:
                failed_weeks.append(week)
                print(f"[cfbd_qb] Week {week}: FAILED to fetch: {exc}")

        if failed_weeks:
            message = f"QB fetch incomplete for {year}, failed weeks: {failed_weeks}"

            # A finished season must be complete. For the current season,
            # future weeks may legitimately have no data yet, so only warn.
            if season_is_over:
                raise RuntimeError(message)

            print(f"[cfbd_qb] WARNING: {message}")

    # 4-5. Flatten the nested CFBD response --------------------------------
    # (game -> team -> category -> stat type -> athlete)

    records = []

    for game_stats in player_game_stats:

        game_data = game_stats.to_dict()
        game_id = game_data.get("id")

        game = games_by_id.get(game_id)

        # Only use games found in the FBS game list above.
        if game is None:
            continue

        start_date = game.get("startDate")

        if start_date is None:
            continue

        game_date = to_game_date(
            start_date,
            bool(game.get("startTimeTBD")),
        )

        home_team = game.get("homeTeam")
        away_team = game.get("awayTeam")
        season = game.get("season", year)
        game_week = game.get("week")

        for team_data in game_data.get("teams", []):

            team_name = team_data.get("team")
            home_away = team_data.get("homeAway")

            if not team_name:
                continue

            passing = {}   # player_id -> {stat_name: {"name", "stat"}}
            rushing = {}

            for category in team_data.get("categories", []):

                target = {
                    "passing": passing,
                    "rushing": rushing,
                }.get(category.get("name"))

                if target is None:
                    continue

                for stat_type in category.get("types", []):

                    stat_name = stat_type.get("name")

                    for athlete in stat_type.get("athletes", []):

                        player_id = athlete.get("id")

                        if player_id is None:
                            continue

                        player_id = str(player_id)

                        # Only keep actual QBs.
                        if player_id not in qb_ids:
                            continue

                        target.setdefault(player_id, {})[stat_name] = {
                            "name": athlete.get("name"),
                            "stat": athlete.get("stat"),
                        }

            # Combine passing + rushing for each QB on this team.
            for player_id in set(passing) | set(rushing):

                p = passing.get(player_id, {})
                r = rushing.get(player_id, {})

                named = next(iter(p.values()), None) or next(iter(r.values()), None)
                player_name = named.get("name") if named else None

                completions, attempts = _parse_completions_attempts(
                    p.get("C/ATT")
                )

                passing_yards = _to_int(p.get("YDS"))
                passing_tds = _to_int(p.get("TD"))
                passing_ints = _to_int(p.get("INT"))

                rushing_attempts = _to_int(r.get("CAR"))
                rushing_yards = _to_int(r.get("YDS"))
                rushing_tds = _to_int(r.get("TD"))

                completion_pct = None
                if completions is not None and attempts:
                    completion_pct = (completions / attempts) * 100

                yards_per_attempt = None
                if passing_yards is not None and attempts:
                    yards_per_attempt = passing_yards / attempts

                records.append({
                    "cfbd_game_id": int(game_id),
                    "player_id": int(player_id),
                    "player_name": player_name,
                    "team": team_name,
                    "home_team": home_team,
                    "away_team": away_team,
                    "game_date": game_date,
                    "season": season,
                    "week": game_week,
                    "home_away": home_away,
                    "passing_yards": passing_yards,
                    "passing_tds": passing_tds,
                    "passing_ints": passing_ints,
                    "completions": completions,
                    "attempts": attempts,
                    "completion_pct": completion_pct,
                    "yards_per_attempt": yards_per_attempt,
                    "rushing_yards": rushing_yards,
                    "rushing_tds": rushing_tds,
                    "rushing_attempts": rushing_attempts,
                })

    print(f"[cfbd_qb] Found {len(records)} QB game-stat records")

    return pd.DataFrame(records)


def validate_and_format_qb_stats(raw_qb_games):
    """
    Validate and clean QB game statistics before inserting them into
    PostgreSQL.
    """

    if raw_qb_games.empty:
        return []

    required = {
        "cfbd_game_id",
        "player_id",
        "player_name",
        "team",
        "home_team",
        "away_team",
        "game_date",
        "season",
    }

    missing = required - set(raw_qb_games.columns)

    if missing:
        raise ValueError(
            f"QB stats missing required columns: {missing}"
        )

    # Remove rows missing required identifying data.
    raw_qb_games = raw_qb_games.dropna(
        subset=[
            "cfbd_game_id",
            "player_id",
            "player_name",
            "team",
            "home_team",
            "away_team",
        ]
    )

    # Convert pandas NaN / NaT values into Python None so PostgreSQL
    # receives SQL NULL.
    raw_qb_games = raw_qb_games.astype(object).where(
        pd.notna(raw_qb_games),
        None,
    )

    records = raw_qb_games.to_dict(orient="records")

    integer_fields = [
        "cfbd_game_id",
        "player_id",
        "passing_yards",
        "passing_tds",
        "passing_ints",
        "completions",
        "attempts",
        "rushing_yards",
        "rushing_tds",
        "rushing_attempts",
        "season",
        "week",
    ]

    for record in records:
        for field in integer_fields:
            value = record.get(field)

            if value is not None:
                record[field] = int(value)

    # Warn about values that would overflow a 32-bit PostgreSQL INTEGER.
    # cfbd_game_id and player_id are BIGINT columns, so they are skipped.
    postgres_integer_min = -2147483648
    postgres_integer_max = 2147483647
    bigint_fields = {"cfbd_game_id", "player_id"}

    for record in records:
        for field in integer_fields:

            if field in bigint_fields:
                continue

            value = record.get(field)

            if (
                value is not None
                and not (postgres_integer_min <= value <= postgres_integer_max)
            ):
                print(
                    f"[cfbd_qb] OUT-OF-RANGE VALUE: "
                    f"{field}={value}, "
                    f"player={record.get('player_name')}, "
                    f"game_date={record.get('game_date')}"
                )

    return records
