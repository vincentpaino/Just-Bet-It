import os
import pandas as pd

try:
    import cfbd
except ImportError:
    cfbd = None


class CFBDUnavailableError(Exception):
    pass


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


def fetch_team_season_stats(year):
    """
    Fetch FBS game information for a season.
    """

    with _get_api_client() as api_client:
        games_api = cfbd.GamesApi(api_client)

        games = games_api.get_games(
            year=year,
            classification="fbs"
        )

    if not games:
        return pd.DataFrame()

    records = [
        game.to_dict()
        for game in games
    ]

    return pd.DataFrame(records)


def fetch_qb_season_stats(year):
    """
    Fetch game-level statistics for quarterbacks.

    CFBD's player game statistics endpoint requires either
    a week, team, or conference, so we request the data one
    week at a time.

    The process is:

        1. Get all FBS games for the season.
        2. Get all players whose position is QB.
        3. Get player game statistics week-by-week.
        4. Match each player-stat game to its game information.
        5. Extract passing and rushing statistics.
        6. Return flat records ready for database insertion.
    """

    with _get_api_client() as api_client:
        games_api = cfbd.GamesApi(api_client)
        players_api = cfbd.PlayersApi(api_client)

        # ---------------------------------------------------------
        # 1. Get all FBS games for the season
        # ---------------------------------------------------------

        games = games_api.get_games(
            year=year,
            classification="fbs"
        )

        games_by_id = {}

        for game in games:
            game_data = game.to_dict()

            game_id = game_data.get("id")

            if game_id is not None:
                games_by_id[game_id] = game_data

        # ---------------------------------------------------------
        # 2. Get all QB player IDs
        # ---------------------------------------------------------

        player_usage = players_api.get_player_usage(
            year=year
        )

        qb_ids = set()

        for player in player_usage:
            position = getattr(
                player,
                "position",
                None
            )

            player_id = getattr(
                player,
                "id",
                None
            )

            if position == "QB" and player_id is not None:
                qb_ids.add(str(player_id))

        print(
            f"[cfbd_qb] Found {len(qb_ids)} QB players"
        )

        # ---------------------------------------------------------
        # 3. Get game-level player statistics week-by-week
        # ---------------------------------------------------------

        player_game_stats = []

        for week in range(1, 16):

            try:
                weekly_stats = (
                    games_api.get_game_player_stats(
                        year=year,
                        week=week
                    )
                )

                player_game_stats.extend(
                    weekly_stats
                )

                print(
                    f"[cfbd_qb] Week {week}: "
                    f"{len(weekly_stats)} games"
                )

            except Exception as exc:
                print(
                    f"[cfbd_qb] Week {week}: "
                    f"unavailable ({exc})"
                )

        # ---------------------------------------------------------
        # 4. Flatten the nested CFBD response
        # ---------------------------------------------------------

        records = []

        for game_stats in player_game_stats:

            game_data = game_stats.to_dict()

            game_id = game_data.get("id")

            # Only use games that were found in the FBS
            # game list above.
            if game_id not in games_by_id:
                continue

            game = games_by_id[game_id]

            home_team = game.get("homeTeam")
            away_team = game.get("awayTeam")

            start_date = game.get("startDate")

            if start_date is None:
                continue

            # CFBD gives us a datetime object.
            game_date = start_date.date()

            season = game.get(
                "season",
                year
            )

            week = game.get("week")

            # -----------------------------------------------------
            # 5. Process both teams
            # -----------------------------------------------------

            for team_data in game_data.get(
                "teams",
                []
            ):

                team_name = team_data.get(
                    "team"
                )

                home_away = team_data.get(
                    "homeAway"
                )

                if not team_name:
                    continue

                passing_stats = {}
                rushing_stats = {}

                categories = team_data.get(
                    "categories",
                    []
                )

                # -------------------------------------------------
                # Extract passing and rushing categories
                # -------------------------------------------------

                for category in categories:

                    category_name = category.get(
                        "name"
                    )

                    if category_name not in (
                        "passing",
                        "rushing",
                    ):
                        continue

                    for stat_type in category.get(
                        "types",
                        []
                    ):

                        stat_name = stat_type.get(
                            "name"
                        )

                        athletes = stat_type.get(
                            "athletes",
                            []
                        )

                        for athlete in athletes:

                            player_id = athlete.get(
                                "id"
                            )

                            if player_id is None:
                                continue

                            player_id = str(
                                player_id
                            )

                            # Only keep actual QBs.
                            if player_id not in qb_ids:
                                continue

                            player_name = athlete.get(
                                "name"
                            )

                            stat_value = athlete.get(
                                "stat"
                            )

                            if category_name == "passing":

                                passing_stats.setdefault(
                                    player_id,
                                    {}
                                )[stat_name] = {
                                    "name": player_name,
                                    "stat": stat_value,
                                }

                            elif category_name == "rushing":

                                rushing_stats.setdefault(
                                    player_id,
                                    {}
                                )[stat_name] = {
                                    "name": player_name,
                                    "stat": stat_value,
                                }

                # -------------------------------------------------
                # Combine passing + rushing players
                # -------------------------------------------------

                qb_player_ids = (
                    set(passing_stats.keys())
                    | set(rushing_stats.keys())
                )

                for player_id in qb_player_ids:

                    passing = passing_stats.get(
                        player_id,
                        {}
                    )

                    rushing = rushing_stats.get(
                        player_id,
                        {}
                    )

                    # -------------------------------------------------
                    # Player name
                    # -------------------------------------------------

                    player_name = None

                    if passing:
                        first_stat = next(
                            iter(passing.values())
                        )

                        player_name = first_stat.get(
                            "name"
                        )

                    if (
                        player_name is None
                        and rushing
                    ):
                        first_stat = next(
                            iter(rushing.values())
                        )

                        player_name = first_stat.get(
                            "name"
                        )

                    # -------------------------------------------------
                    # Passing: completions / attempts
                    # -------------------------------------------------

                    completions = None
                    attempts = None

                    c_att = passing.get(
                        "C/ATT"
                    )

                    if c_att:
                        value = c_att.get(
                            "stat"
                        )

                        if value and "/" in value:

                            try:
                                completions, attempts = map(
                                    int,
                                    value.split("/")
                                )

                            except ValueError:
                                pass

                    # -------------------------------------------------
                    # Passing yards
                    # -------------------------------------------------

                    passing_yards = None

                    if passing.get("YDS"):

                        try:
                            passing_yards = int(
                                passing["YDS"]["stat"]
                            )

                        except (
                            ValueError,
                            TypeError
                        ):
                            pass

                    # -------------------------------------------------
                    # Passing touchdowns
                    # -------------------------------------------------

                    passing_tds = None

                    if passing.get("TD"):

                        try:
                            passing_tds = int(
                                passing["TD"]["stat"]
                            )

                        except (
                            ValueError,
                            TypeError
                        ):
                            pass

                    # -------------------------------------------------
                    # Passing interceptions
                    # -------------------------------------------------

                    passing_ints = None

                    if passing.get("INT"):

                        try:
                            passing_ints = int(
                                passing["INT"]["stat"]
                            )

                        except (
                            ValueError,
                            TypeError
                        ):
                            pass

                    # -------------------------------------------------
                    # Rushing attempts
                    # -------------------------------------------------

                    rushing_attempts = None

                    if rushing.get("CAR"):

                        try:
                            rushing_attempts = int(
                                rushing["CAR"]["stat"]
                            )

                        except (
                            ValueError,
                            TypeError
                        ):
                            pass

                    # -------------------------------------------------
                    # Rushing yards
                    # -------------------------------------------------

                    rushing_yards = None

                    if rushing.get("YDS"):

                        try:
                            rushing_yards = int(
                                rushing["YDS"]["stat"]
                            )

                        except (
                            ValueError,
                            TypeError
                        ):
                            pass

                    # -------------------------------------------------
                    # Rushing touchdowns
                    # -------------------------------------------------

                    rushing_tds = None

                    if rushing.get("TD"):

                        try:
                            rushing_tds = int(
                                rushing["TD"]["stat"]
                            )

                        except (
                            ValueError,
                            TypeError
                        ):
                            pass

                    # -------------------------------------------------
                    # Completion percentage
                    # -------------------------------------------------

                    completion_pct = None

                    if (
                        completions is not None
                        and attempts is not None
                        and attempts > 0
                    ):
                        completion_pct = (
                            completions / attempts
                        ) * 100

                    # -------------------------------------------------
                    # Yards per attempt
                    # -------------------------------------------------

                    yards_per_attempt = None

                    if (
                        passing_yards is not None
                        and attempts is not None
                        and attempts > 0
                    ):
                        yards_per_attempt = (
                            passing_yards / attempts
                        )

                    # -------------------------------------------------
                    # Create database-ready record
                    # -------------------------------------------------

                    records.append({
                        "player_id": int(
                            player_id
                        ),

                        "player_name": player_name,

                        "team": team_name,

                        "home_team": home_team,
                        "away_team": away_team,

                        "game_date": game_date,

                        "season": season,
                        "week": week,

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

    print(
        f"[cfbd_qb] Found "
        f"{len(records)} QB game-stat records"
    )

    return pd.DataFrame(records)


def validate_and_format(raw_games):
    """
    Validate and format normal CFBD game records.
    """

    if raw_games.empty:
        return []

    df = raw_games.copy()

    # The current CFBD Python package returns camelCase
    # field names from game.to_dict().
    rename_map = {
        "homeTeam": "home_team",
        "awayTeam": "away_team",
        "startDate": "game_date",
        "homePoints": "home_score",
        "awayPoints": "away_score",
    }

    existing_rename = {
        key: value
        for key, value in rename_map.items()
        if key in df.columns
    }

    df = df.rename(
        columns=existing_rename
    )

    required = {
        "home_team",
        "away_team",
        "game_date",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"CFBD game data missing required "
            f"columns: {missing}"
        )

    df = df.dropna(
        subset=[
            "home_team",
            "away_team",
        ]
    )

    # Convert datetime values to Python dates.
    if "game_date" in df.columns:

        df["game_date"] = pd.to_datetime(
            df["game_date"]
        ).dt.date

    return df.to_dict(
        orient="records"
    )


def validate_and_format_qb_stats(raw_qb_games):
    """
    Validate and clean QB game statistics before
    inserting them into PostgreSQL.
    """

    if raw_qb_games.empty:
        return []

    required = {
        "player_id",
        "player_name",
        "team",
        "home_team",
        "away_team",
        "game_date",
        "season",
    }

    missing = required - set(
        raw_qb_games.columns
    )

    if missing:
        raise ValueError(
            f"QB stats missing required columns: "
            f"{missing}"
        )

    # Remove rows missing required identifying data.
    raw_qb_games = raw_qb_games.dropna(
        subset=[
            "player_id",
            "player_name",
            "team",
            "home_team",
            "away_team",
        ]
    )

    # Convert pandas NaN / NaT values into
    # Python None so PostgreSQL receives SQL NULL.
    raw_qb_games = raw_qb_games.astype(
        object
    ).where(
        pd.notna(raw_qb_games),
        None
    )

    records = raw_qb_games.to_dict(
        orient="records"
    )

    # -------------------------------------------------------------
    # Convert database INTEGER fields to actual Python integers.
    # -------------------------------------------------------------

    integer_fields = [
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

                record[field] = int(
                    value
                )

    # -------------------------------------------------------------
    # Check for values that exceed PostgreSQL INTEGER range.
    # -------------------------------------------------------------

    postgres_integer_min = -2147483648
    postgres_integer_max = 2147483647

    for record in records:

        for field in integer_fields:

            value = record.get(field)

            if (
                value is not None
                and not (
                    postgres_integer_min
                    <= value
                    <= postgres_integer_max
                )
            ):

                print(
                    f"[cfbd_qb] OUT-OF-RANGE VALUE: "
                    f"{field}={value}, "
                    f"player={record.get('player_name')}, "
                    f"game_date={record.get('game_date')}"
                )

    return records