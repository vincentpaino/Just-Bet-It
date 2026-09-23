
"""
Top-level ingestion entry point.
Mirrors the ingest_data() pseudocode from Doc II Section 3.3 / Doc III
Section 4.3 — pulls odds, pulls historical stats, logs the run.

Usage:
    python -m src.ingestion.run_ingestion --year 2026
"""

import argparse
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import and_, or_

load_dotenv()

from src.db.models import (
    IngestionLog,
    Team,
    Odds,
    QBGameStat,
    Game,
)

from src.db.session import SessionLocal, init_db

from .odds_api import (
    fetch_odds,
    validate_and_format as validate_odds,
)

from .cfbd_ingest import (
    fetch_team_season_stats,
    fetch_qb_season_stats,
    validate_and_format as validate_stats,
    validate_and_format_qb_stats,
    CFBDUnavailableError,
)


def _get_or_create_team(session, team_name: str) -> Team:
    """
    Find a team by name or create it if it doesn't exist.
    """

    team = (
        session.query(Team)
        .filter_by(team_name=team_name)
        .first()
    )

    if team is None:
        team = Team(
            team_name=team_name
        )

        session.add(team)
        session.flush()

    return team


def ingest_odds(session, year: int) -> int:
    """
    Fetch and store sportsbook odds.
    """

    raw_data = fetch_odds()

    records = validate_odds(raw_data)

    written = 0

    for record in records:

        home_team = _get_or_create_team(
            session,
            record["home_team"]
        )

        away_team = _get_or_create_team(
            session,
            record["away_team"]
        )

        game_date = record.get("game_date")

        if isinstance(game_date, str):
            game_date = datetime.fromisoformat(
                game_date.replace("Z", "+00:00")
            )

        game = (
            session.query(Game)
            .filter_by(
                game_date=game_date.date()
                if hasattr(game_date, "date")
                else game_date
            )
            .filter(
                or_(
                    and_(
                        Game.home_team_id
                        == home_team.team_id,
                        Game.away_team_id
                        == away_team.team_id,
                    ),
                    and_(
                        Game.home_team_id
                        == away_team.team_id,
                        Game.away_team_id
                        == home_team.team_id,
                    ),
                )
            )
            .first()
        )

        if game is None:

            game = Game(
                home_team_id=home_team.team_id,
                away_team_id=away_team.team_id,
                game_date=(
                    game_date.date()
                    if hasattr(game_date, "date")
                    else game_date
                ),
            )

            session.add(game)
            session.flush()

        odds = Odds(
            game_id=game.game_id,
            sportsbook=record.get("sportsbook"),
            home_moneyline=record.get(
                "home_moneyline"
            ),
            away_moneyline=record.get(
                "away_moneyline"
            ),
            spread=record.get("spread"),
            over_under=record.get(
                "over_under"
            ),
            implied_prob_home=record.get(
                "implied_prob_home"
            ),
            implied_prob_away=record.get(
                "implied_prob_away"
            ),
            fetched_at=record.get(
                "fetched_at"
            ),
        )

        session.add(odds)

        written += 1

    session.commit()

    return written


def ingest_team_stats(session, year: int) -> int:
    """
    Fetch and store team season statistics.
    """

    raw_df = fetch_team_season_stats(year)

    records = validate_stats(raw_df)

    written = 0

    for record in records:

        home_team_name = record.get(
            "home_team",
            "Unknown"
        )

        away_team_name = record.get(
            "away_team",
            "Unknown"
        )

        home_team = _get_or_create_team(
            session,
            home_team_name
        )

        away_team = _get_or_create_team(
            session,
            away_team_name
        )

        game_date = record.get(
            "game_date"
        )

        if isinstance(game_date, str):
            game_date = datetime.strptime(
                game_date,
                "%Y-%m-%d"
            ).date()

        game = (
            session.query(Game)
            .filter_by(
                game_date=game_date
            )
            .filter(
                or_(
                    and_(
                        Game.home_team_id
                        == home_team.team_id,
                        Game.away_team_id
                        == away_team.team_id,
                    ),
                    and_(
                        Game.home_team_id
                        == away_team.team_id,
                        Game.away_team_id
                        == home_team.team_id,
                    ),
                )
            )
            .first()
        )

        if game is None:

            game = Game(
                home_team_id=home_team.team_id,
                away_team_id=away_team.team_id,
                game_date=game_date,
            )

            session.add(game)
            session.flush()

        written += 1

    session.commit()

    return written

def ingest_qb_stats(session, year: int) -> int:
    """
    Fetch and store QB game statistics.

    A QB can only have one stat record per game, so
    (game_id, player_id) is treated as the unique identifier.

    Existing records are skipped so that rerunning ingestion
    does not create duplicates.
    """

    raw_df = fetch_qb_season_stats(year)

    records = validate_and_format_qb_stats(
        raw_df
    )

    print(
        f"[cfbd_qb] Processing "
        f"{len(records)} records..."
    )

    written = 0
    skipped = 0

    for index, record in enumerate(
        records,
        start=1
    ):

        try:

            # -----------------------------------------------------
            # Get/create teams
            # -----------------------------------------------------

            home_team = _get_or_create_team(
                session,
                record.get(
                    "home_team",
                    "Unknown"
                )
            )

            away_team = _get_or_create_team(
                session,
                record.get(
                    "away_team",
                    "Unknown"
                )
            )

            # -----------------------------------------------------
            # Game date
            # -----------------------------------------------------

            game_date = record.get(
                "game_date"
            )

            if isinstance(game_date, str):

                game_date = datetime.strptime(
                    game_date,
                    "%Y-%m-%d"
                ).date()

            # -----------------------------------------------------
            # Find existing game
            # -----------------------------------------------------

            game = (
                session.query(Game)
                .filter_by(
                    game_date=game_date
                )
                .filter(
                    or_(
                        and_(
                            Game.home_team_id
                            == home_team.team_id,
                            Game.away_team_id
                            == away_team.team_id,
                        ),
                        and_(
                            Game.home_team_id
                            == away_team.team_id,
                            Game.away_team_id
                            == home_team.team_id,
                        ),
                    )
                )
                .first()
            )

            # -----------------------------------------------------
            # Create game if necessary
            # -----------------------------------------------------

            if game is None:

                game = Game(
                    home_team_id=home_team.team_id,
                    away_team_id=away_team.team_id,
                    game_date=game_date,
                )

                session.add(game)
                session.flush()

            # -----------------------------------------------------
            # Determine QB's team
            # -----------------------------------------------------

            team_name = record.get(
                "team",
                ""
            )

            if team_name == home_team.team_name:

                team_id = home_team.team_id

                opponent_team_id = (
                    away_team.team_id
                )

                home_away = "home"

            elif team_name == away_team.team_name:

                team_id = away_team.team_id

                opponent_team_id = (
                    home_team.team_id
                )

                home_away = "away"

            else:

                team_id = home_team.team_id

                opponent_team_id = (
                    away_team.team_id
                )

                home_away = "home"

            # -----------------------------------------------------
            # Check whether this QB/game already exists
            # -----------------------------------------------------

            existing_stat = (
                session.query(QBGameStat)
                .filter_by(
                    game_id=game.game_id,
                    player_id=record["player_id"],
                )
                .first()
            )

            if existing_stat is not None:

                skipped += 1

                if skipped <= 10:
                    print(
                        f"[cfbd_qb] Skipping existing record: "
                        f"{record.get('player_name')} "
                        f"(game_id={game.game_id})"
                    )

                continue

            # -----------------------------------------------------
            # Create QB stat
            # -----------------------------------------------------

            qb_stat = QBGameStat(
                game_id=game.game_id,

                player_id=record[
                    "player_id"
                ],

                player_name=record[
                    "player_name"
                ],

                team_id=team_id,

                opponent_team_id=(
                    opponent_team_id
                ),

                passing_yards=record.get(
                    "passing_yards"
                ),

                passing_tds=record.get(
                    "passing_tds"
                ),

                passing_ints=record.get(
                    "passing_ints"
                ),

                completions=record.get(
                    "completions"
                ),

                attempts=record.get(
                    "attempts"
                ),

                completion_pct=record.get(
                    "completion_pct"
                ),

                yards_per_attempt=record.get(
                    "yards_per_attempt"
                ),

                rushing_yards=record.get(
                    "rushing_yards"
                ),

                rushing_tds=record.get(
                    "rushing_tds"
                ),

                rushing_attempts=record.get(
                    "rushing_attempts"
                ),

                game_date=game_date,

                season=record[
                    "season"
                ],

                week=record.get(
                    "week"
                ),

                home_away=home_away,
            )

            session.add(qb_stat)

            # -----------------------------------------------------
            # Force PostgreSQL to insert this record.
            # -----------------------------------------------------

            session.flush()

            written += 1

            if written % 25 == 0:

                print(
                    f"[cfbd_qb] Successfully inserted "
                    f"{written}/{len(records)} new records"
                )

        except Exception as exc:

            print()
            print("=" * 70)
            print("[cfbd_qb] ERROR")
            print("=" * 70)

            print(
                f"Record number: {index}"
            )

            print(
                f"Player: "
                f"{record.get('player_name')}"
            )

            print(
                f"Player ID: "
                f"{record.get('player_id')}"
            )

            print(
                f"Team: "
                f"{record.get('team')}"
            )

            print(
                f"Home team: "
                f"{record.get('home_team')}"
            )

            print(
                f"Away team: "
                f"{record.get('away_team')}"
            )

            print(
                f"Game date: "
                f"{record.get('game_date')}"
            )

            print(
                f"Season: "
                f"{record.get('season')}"
            )

            print(
                f"Week: "
                f"{record.get('week')}"
            )

            print()
            print("Database error:")
            print(exc)

            print("=" * 70)

            session.rollback()

            raise

    session.commit()

    print(
        f"[cfbd_qb] Finished processing."
    )

    print(
        f"[cfbd_qb] New records inserted: "
        f"{written}"
    )

    print(
        f"[cfbd_qb] Existing records skipped: "
        f"{skipped}"
    )

    return written


def log_run(
    session,
    source: str,
    record_count: int = 0,
    success: bool = True,
    error_message: str | None = None,
):
    """
    Record the result of an ingestion run.

    Matches the IngestionLog model:
        source
        run_at
        record_count
        success
        error_message
    """

    log = IngestionLog(
        source=source,
        run_at=datetime.utcnow(),
        record_count=record_count,
        success=success,
        error_message=error_message,
    )

    session.add(log)
    session.commit()


def main():

    print("SCRIPT STARTED")

    parser = argparse.ArgumentParser(
        description="Run Just-Bet-It data ingestion."
    )

    parser.add_argument(
        "--year",
        type=int,
        default=datetime.utcnow().year,
        help="Season year to ingest.",
    )

    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip odds ingestion.",
    )

    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip team stats ingestion.",
    )

    parser.add_argument(
        "--skip-qb",
        action="store_true",
        help="Skip QB stats ingestion.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Initialize database tables
    # -------------------------------------------------------------

    init_db()

    session = SessionLocal()

    try:

        # ---------------------------------------------------------
        # Odds
        # ---------------------------------------------------------

        if not args.skip_odds:

            try:

                print(
                    f"[odds] Starting odds ingestion "
                    f"for {args.year}..."
                )

                written = ingest_odds(
                    session,
                    args.year
                )

                print(
                    f"[odds] Successfully wrote "
                    f"{written} records"
                )

                log_run(
                    session,
                    "odds",
                    written,
                    True,
                )

            except Exception as exc:

                session.rollback()

                print(
                    f"[odds] FAILED: {exc}"
                )

                try:

                    log_run(
                        session,
                        "odds",
                        0,
                        False,
                        str(exc),
                    )

                except Exception:
                    session.rollback()

        # ---------------------------------------------------------
        # Team statistics
        # ---------------------------------------------------------

        if not args.skip_stats:

            try:

                print(
                    f"[cfbd_stats] Starting team stats "
                    f"ingestion for {args.year}..."
                )

                written = ingest_team_stats(
                    session,
                    args.year
                )

                print(
                    f"[cfbd_stats] Successfully wrote "
                    f"{written} records"
                )

                log_run(
                    session,
                    "cfbd_stats",
                    written,
                    True,
                )

            except CFBDUnavailableError as exc:

                session.rollback()

                print(
                    f"[cfbd_stats] UNAVAILABLE: {exc}"
                )

                try:

                    log_run(
                        session,
                        "cfbd_stats",
                        0,
                        False,
                        str(exc),
                    )

                except Exception:
                    session.rollback()

            except Exception as exc:

                session.rollback()

                print(
                    f"[cfbd_stats] FAILED: {exc}"
                )

                try:

                    log_run(
                        session,
                        "cfbd_stats",
                        0,
                        False,
                        str(exc),
                    )

                except Exception:
                    session.rollback()

        # ---------------------------------------------------------
        # QB statistics
        # ---------------------------------------------------------

        if not args.skip_qb:

            try:

                print(
                    f"[cfbd_qb] Starting QB stats "
                    f"ingestion for {args.year}..."
                )

                written = ingest_qb_stats(
                    session,
                    args.year
                )

                print(
                    f"[cfbd_qb] Successfully wrote "
                    f"{written} records"
                )

                log_run(
                    session,
                    "cfbd_qb",
                    written,
                    True,
                )

            except CFBDUnavailableError as exc:

                session.rollback()

                print(
                    f"[cfbd_qb] UNAVAILABLE: {exc}"
                )

                try:

                    log_run(
                        session,
                        "cfbd_qb",
                        0,
                        False,
                        str(exc),
                    )

                except Exception:
                    session.rollback()

            except Exception as exc:

                session.rollback()

                print(
                    f"[cfbd_qb] FAILED: {exc}"
                )

                try:

                    log_run(
                        session,
                        "cfbd_qb",
                        0,
                        False,
                        str(exc),
                    )

                except Exception:
                    session.rollback()

    finally:

        print(
            f"Ingestion run complete for {args.year}."
        )

        session.close()


if __name__ == "__main__":
    main()
