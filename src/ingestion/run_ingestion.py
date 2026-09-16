"""
Top-level ingestion entry point.
Mirrors the ingest_data() pseudocode from Doc II Section 3.3 / Doc III
Section 4.3 — pulls odds, pulls historical stats, logs the run.

Usage:
    python -m ingestion.run_ingestion --year 2026
"""
import argparse
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from db.models import IngestionLog, Team, Odds
from db.session import SessionLocal, init_db
from ingestion.odds_api import fetch_odds, validate_and_format as validate_odds
from ingestion.cfbd_ingest import (
    fetch_team_season_stats,
    validate_and_format as validate_stats,
    CFBDUnavailableError,
)


def _get_or_create_team(session, team_name: str) -> Team:
    team = session.query(Team).filter_by(team_name=team_name).first()
    if team is None:
        team = Team(team_name=team_name)
        session.add(team)
        session.flush()
    return team


def ingest_odds(session) -> int:
    raw = fetch_odds()
    rows = validate_odds(raw)

    written = 0
    for row in rows:
        home_team = _get_or_create_team(session, row["home_team"])
        away_team = _get_or_create_team(session, row["away_team"])

        from db.models import Game
        game = (
            session.query(Game)
            .filter_by(home_team_id=home_team.team_id, away_team_id=away_team.team_id)
            .first()
        )
        if game is None:
            game = Game(
                home_team_id=home_team.team_id,
                away_team_id=away_team.team_id,
                game_date=row["game_date"],
            )
            session.add(game)
            session.flush()

        session.add(Odds(
            game_id=game.game_id,
            sportsbook=row["sportsbook"],
            home_moneyline=row["home_moneyline"],
            away_moneyline=row["away_moneyline"],
            spread=row["spread"],
            over_under=row["over_under"],
            implied_prob_home=row["implied_prob_home"],
            implied_prob_away=row["implied_prob_away"],
            fetched_at=row["fetched_at"],
        ))
        written += 1

    session.commit()
    return written


def ingest_historical_stats(session, year: int) -> int:
    raw_df = fetch_team_season_stats(year)
    records = validate_stats(raw_df)

    for record in records:
        _get_or_create_team(session, record["home_team"])
        _get_or_create_team(session, record["away_team"])

    session.commit()
    return len(records)


def log_run(session, source: str, record_count: int, success: bool, error: str = None) -> None:
    session.add(IngestionLog(
        source=source,
        run_at=datetime.utcnow(),
        record_count=record_count,
        success=success,
        error_message=error,
    ))
    session.commit()


def main() -> int:
    print("SCRIPT STARTED")
    parser = argparse.ArgumentParser(description="Run Just Bet It data ingestion")
    parser.add_argument("--year", type=int, default=datetime.utcnow().year,
                         help="Season year for historical stats ingestion")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    exit_code = 0

    if not args.skip_odds:
        try:
            count = ingest_odds(session)
            log_run(session, "odds_api", count, success=True)
            print(f"[odds_api] ingested {count} odds rows")
        except Exception as exc:
            session.rollback()
            log_run(session, "odds_api", 0, success=False, error=str(exc))
            print(f"[odds_api] FAILED: {exc}", file=sys.stderr)
            exit_code = 1

    if not args.skip_stats:
        try:
            count = ingest_historical_stats(session, args.year)
            log_run(session, "cfbd", count, success=True)
            print(f"[cfbd] ingested {count} game records for {args.year}")
        except CFBDUnavailableError as exc:
            session.rollback()
            log_run(session, "cfbd", 0, success=False, error=str(exc))
            print(f"[cfbd] UNAVAILABLE: {exc}", file=sys.stderr)
            exit_code = 1
        except Exception as exc:
            session.rollback()
            log_run(session, "cfbd", 0, success=False, error=str(exc))
            print(f"[cfbd] FAILED: {exc}", file=sys.stderr)
            exit_code = 1

    session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())