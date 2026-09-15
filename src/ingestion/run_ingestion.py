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

from db import IngestionLog, Team, Odds
from db import SessionLocal, init_db
frm ingestion.odds_api import fetch_odds, validate_and_format as validate_odds
from ingestion.sportsipy_ingest import (
    fetch_team_season_stats,
    validate_and_format as validate_stats,
    SportsipyUnavailableError,
)


def _get_or_create_team(session, team_name: str) -> Team:
    team = session.query(Team).filter_by(team_name=team_name).first()
    if team is None:
        team = Team(team_name=team_name)
        session.add(team)
        session.flush()  # populate team.team_id without committing yet
    return team


def ingest_odds(session) -> int:
    """Fetch + validate + persist current odds. Returns rows written."""
    raw = fetch_odds()
    rows = validate_odds(raw)

    written = 0
    for row in rows:
        home_team = _get_or_create_team(session, row["home_team"])
        away_team = _get_or_create_team(session, row["away_team"])

        # NOTE: this creates a bare game shell keyed on team names + date.
        # Once feature engineering (Sprint 3) exists, this should instead
        # look up the matching game_id from the games table by
        # (home_team_id, away_team_id, game_date) and only create one if
        # it truly doesn't exist yet.
        from db import Game
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
    """Fetch + validate + persist one season of team stats. Returns rows written."""
    raw_df = fetch_team_season_stats(year)
    records = validate_stats(raw_df)

    for record in records:
        _get_or_create_team(session, record["team_name"])

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
        except Exception as exc:  # noqa: BLE001 — log and continue to stats
            session.rollback()
            log_run(session, "odds_api", 0, success=False, error=str(exc))
            print(f"[odds_api] FAILED: {exc}", file=sys.stderr)
            exit_code = 1

    if not args.skip_stats:
        try:
            count = ingest_historical_stats(session, args.year)
            log_run(session, "sportsipy", count, success=True)
            print(f"[sportsipy] ingested {count} team-season records for {args.year}")
        except SportsipyUnavailableError as exc:
            session.rollback()
            log_run(session, "sportsipy", 0, success=False, error=str(exc))
            print(f"[sportsipy] UNAVAILABLE: {exc}", file=sys.stderr)
            exit_code = 1
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            log_run(session, "sportsipy", 0, success=False, error=str(exc))
            print(f"[sportsipy] FAILED: {exc}", file=sys.stderr)
            exit_code = 1

    session.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
