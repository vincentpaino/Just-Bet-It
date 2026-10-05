"""
Top-level ingestion entry point.
Mirrors the ingest_data() pseudocode from Doc II Section 3.3 / Doc III
Section 4.3 — pulls odds, pulls historical stats, logs the run.

Usage:
    python -m src.ingestion.run_ingestion --year 2026

Games are matched by their CFBD game ID. Older rows that predate the ID
column are found by matchup + date, then stamped with the ID and given the
corrected game date (this is what migrates existing data on a re-run).

Run order matters: team/game stats must run before QB stats, because QB rows
attach to games by CFBD game ID.
"""

import argparse
from datetime import datetime, timezone

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

from .dates import to_game_date, utc_date

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Naive UTC timestamp (what the DateTime columns store)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _get_or_create_team(session, team_name: str) -> Team:
    """Find a team by name or create it if it doesn't exist."""

    team = (
        session.query(Team)
        .filter_by(team_name=team_name)
        .first()
    )

    if team is None:
        team = Team(team_name=team_name)
        session.add(team)
        session.flush()

    return team


def _same_matchup(team_a_id: int, team_b_id: int):
    """SQL condition: these two teams play each other (either home/away)."""
    return or_(
        and_(
            Game.home_team_id == team_a_id,
            Game.away_team_id == team_b_id,
        ),
        and_(
            Game.home_team_id == team_b_id,
            Game.away_team_id == team_a_id,
        ),
    )


def _find_game_by_matchup(
    session,
    team_a_id: int,
    team_b_id: int,
    dates,
    unstamped_only: bool = False,
):
    """
    Find a game between two teams on any of the given dates.
    With unstamped_only=True, only games that don't have a CFBD ID yet
    are considered (used to migrate old rows).
    """

    dates = [d for d in set(dates) if d is not None]

    if not dates:
        return None

    query = (
        session.query(Game)
        .filter(Game.game_date.in_(dates))
        .filter(_same_matchup(team_a_id, team_b_id))
    )

    if unstamped_only:
        query = query.filter(Game.cfbd_game_id.is_(None))

    return query.first()


def _apply_result(game, cfbd_home_team_id, home_score, away_score) -> bool:
    """
    Write scores onto a game row, in the orientation that row uses.
    (A game created from odds data can have home/away reversed relative to
    CFBD, so scores are swapped to match.) Returns True if swapped.
    Only non-None scores are written, so a missing score never erases one.
    """

    swapped = game.home_team_id != cfbd_home_team_id

    if swapped:
        home_score, away_score = away_score, home_score

    if home_score is not None:
        game.home_score = home_score

    if away_score is not None:
        game.away_score = away_score

    if home_score is not None and away_score is not None:
        game.home_win = home_score > away_score

    return swapped


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

def ingest_odds(session, year: int) -> int:
    """Fetch and store sportsbook odds."""

    raw_data = fetch_odds()

    records = validate_odds(raw_data)

    written = 0

    for record in records:

        raw_date = record.get("game_date")

        local_date = to_game_date(raw_date)

        if local_date is None:
            continue

        home_team = _get_or_create_team(session, record["home_team"])
        away_team = _get_or_create_team(session, record["away_team"])

        game = _find_game_by_matchup(
            session,
            home_team.team_id,
            away_team.team_id,
            [local_date, utc_date(raw_date)],
        )

        if game is None:
            home_score = record.get("home_score")
            away_score = record.get("away_score")

            home_win = None
            if home_score is not None and away_score is not None:
                home_win = home_score > away_score

            game = Game(
                home_team_id=home_team.team_id,
                away_team_id=away_team.team_id,
                game_date=local_date,
                home_score=home_score,
                away_score=away_score,
                home_win=home_win,
            )

            session.add(game)
            session.flush()

        session.add(
            Odds(
                game_id=game.game_id,
                sportsbook=record.get("sportsbook"),
                home_moneyline=record.get("home_moneyline"),
                away_moneyline=record.get("away_moneyline"),
                spread=record.get("spread"),
                over_under=record.get("over_under"),
                implied_prob_home=record.get("implied_prob_home"),
                implied_prob_away=record.get("implied_prob_away"),
                fetched_at=record.get("fetched_at"),
            )
        )

        written += 1

    session.commit()

    return written


# ---------------------------------------------------------------------------
# Games (named "team stats" for historical reasons)
# ---------------------------------------------------------------------------

def ingest_team_stats(session, year: int) -> int:
    """
    Fetch CFBD games for a season and create / update them.

    Matching order for each CFBD game:
        1. by cfbd_game_id
        2. by matchup + date, among games that have no ID yet (old rows);
           these get stamped with the ID and the corrected game date
        3. otherwise create a new game
    """

    raw_df = fetch_team_season_stats(year)

    records = validate_stats(raw_df)

    games_by_cfbd_id = {
        game.cfbd_game_id: game
        for game in session.query(Game).filter(Game.cfbd_game_id.is_not(None))
    }

    created = matched = migrated = swapped = 0

    for record in records:

        home_team = _get_or_create_team(session, record["home_team"])
        away_team = _get_or_create_team(session, record["away_team"])

        cfbd_id = record["cfbd_game_id"]

        game = games_by_cfbd_id.get(cfbd_id)

        if game is not None:
            matched += 1

        else:
            game = _find_game_by_matchup(
                session,
                home_team.team_id,
                away_team.team_id,
                [record["game_date"], utc_date(record["start_time"])],
                unstamped_only=True,
            )

            if game is not None:
                migrated += 1
            else:
                game = Game(
                    home_team_id=home_team.team_id,
                    away_team_id=away_team.team_id,
                    game_date=record["game_date"],
                )
                session.add(game)
                created += 1

        game.cfbd_game_id = cfbd_id
        games_by_cfbd_id[cfbd_id] = game

        game.game_date = record["game_date"]
        game.start_time = record["start_time"]
        game.start_time_tbd = record["start_time_tbd"]
        game.season = record["season"]
        game.week = record["week"]
        game.season_type = record["season_type"]
        game.neutral_site = record["neutral_site"]

        if _apply_result(
            game,
            home_team.team_id,
            record["home_score"],
            record["away_score"],
        ):
            swapped += 1

        session.flush()

    session.commit()

    print(
        f"[cfbd_stats] {year}: {created} created, {matched} matched by ID, "
        f"{migrated} migrated from old rows, "
        f"{swapped} with home/away reversed vs CFBD"
    )

    return len(records)


# ---------------------------------------------------------------------------
# Quarterbacks
# ---------------------------------------------------------------------------

def ingest_qb_stats(session, year: int) -> int:
    """
    Fetch and store QB game statistics.

    Each QB row attaches to a game by CFBD game ID, so team/game stats must
    be ingested first. (game_id, player_id) is the unique identifier; existing
    rows are skipped so re-running never creates duplicates.
    """

    raw_df = fetch_qb_season_stats(year)

    records = validate_and_format_qb_stats(raw_df)

    print(f"[cfbd_qb] Processing {len(records)} records...")

    # Load lookups once instead of querying per record.
    games_lookup = {
        cfbd_id: (game_id, game_date)
        for cfbd_id, game_id, game_date in session.query(
            Game.cfbd_game_id, Game.game_id, Game.game_date
        ).filter(Game.cfbd_game_id.is_not(None))
    }

    existing = {
        (game_id, player_id)
        for game_id, player_id in session.query(
            QBGameStat.game_id, QBGameStat.player_id
        )
    }

    team_ids = {
        name: team_id
        for name, team_id in session.query(Team.team_name, Team.team_id)
    }

    def team_id_for(name: str) -> int:
        if name not in team_ids:
            team_ids[name] = _get_or_create_team(session, name).team_id
        return team_ids[name]

    written = skipped = missing_game = unmatched_team = 0

    for index, record in enumerate(records, start=1):

        try:
            game_info = games_lookup.get(record["cfbd_game_id"])

            if game_info is None:
                missing_game += 1
                continue

            game_id, game_date = game_info

            key = (game_id, record["player_id"])

            if key in existing:
                skipped += 1
                continue

            team_name = record["team"]

            if team_name == record["home_team"]:
                opponent_name = record["away_team"]
                home_away = "home"

            elif team_name == record["away_team"]:
                opponent_name = record["home_team"]
                home_away = "away"

            else:
                # Don't guess which side the QB played for.
                unmatched_team += 1
                print(
                    f"[cfbd_qb] Skipping: team '{team_name}' is neither "
                    f"'{record['home_team']}' nor '{record['away_team']}' "
                    f"({record.get('player_name')})"
                )
                continue

            session.add(
                QBGameStat(
                    game_id=game_id,
                    player_id=record["player_id"],
                    player_name=record["player_name"],
                    team_id=team_id_for(team_name),
                    opponent_team_id=team_id_for(opponent_name),
                    passing_yards=record.get("passing_yards"),
                    passing_tds=record.get("passing_tds"),
                    passing_ints=record.get("passing_ints"),
                    completions=record.get("completions"),
                    attempts=record.get("attempts"),
                    completion_pct=record.get("completion_pct"),
                    yards_per_attempt=record.get("yards_per_attempt"),
                    rushing_yards=record.get("rushing_yards"),
                    rushing_tds=record.get("rushing_tds"),
                    rushing_attempts=record.get("rushing_attempts"),
                    game_date=game_date,
                    season=record["season"],
                    week=record.get("week"),
                    home_away=home_away,
                )
            )

            # Force PostgreSQL to insert now so errors point at this record.
            session.flush()

            existing.add(key)
            written += 1

            if written % 250 == 0:
                print(
                    f"[cfbd_qb] Inserted {written} new records so far "
                    f"(record {index}/{len(records)})"
                )

        except Exception as exc:
            session.rollback()
            print()
            print("=" * 70)
            print(f"[cfbd_qb] ERROR on record {index}:")
            print(record)
            print(f"Database error: {exc}")
            print("=" * 70)
            raise

    session.commit()

    print("[cfbd_qb] Finished processing.")
    print(f"[cfbd_qb] New records inserted: {written}")
    print(f"[cfbd_qb] Existing records skipped: {skipped}")

    if missing_game:
        print(
            f"[cfbd_qb] WARNING: {missing_game} records had no matching game. "
            f"Run team/game stats ingestion for {year} first."
        )

    if unmatched_team:
        print(
            f"[cfbd_qb] WARNING: {unmatched_team} records skipped because "
            f"the team name matched neither side of the game."
        )

    return written


# ---------------------------------------------------------------------------
# Logging + entry point
# ---------------------------------------------------------------------------

def log_run(
    session,
    source: str,
    record_count: int = 0,
    success: bool = True,
    error_message: str | None = None,
):
    """Record the result of an ingestion run in ingestion_log."""

    session.add(
        IngestionLog(
            source=source,
            run_at=_utcnow(),
            record_count=record_count,
            success=success,
            error_message=error_message,
        )
    )
    session.commit()


def _run_step(session, label: str, source: str, func, year: int):
    """
    Run one ingestion step. A failure is printed and logged but does not stop
    the other steps.
    """

    try:
        print(f"[{label}] Starting for {year}...")

        written = func(session, year)

        print(f"[{label}] Successfully wrote {written} records")

        log_run(session, source, written, True)

    except Exception as exc:
        session.rollback()

        kind = "UNAVAILABLE" if isinstance(exc, CFBDUnavailableError) else "FAILED"

        print(f"[{label}] {kind}: {exc}")

        try:
            log_run(session, source, 0, False, str(exc))
        except Exception:
            session.rollback()


def main():

    print("SCRIPT STARTED")

    parser = argparse.ArgumentParser(
        description="Run Just-Bet-It data ingestion."
    )

    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now(timezone.utc).year,
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
        help="Skip team/game ingestion.",
    )

    parser.add_argument(
        "--skip-qb",
        action="store_true",
        help="Skip QB stats ingestion.",
    )

    args = parser.parse_args()

    init_db()

    session = SessionLocal()

    try:

        if not args.skip_odds:
            _run_step(session, "odds", "odds", ingest_odds, args.year)

        if not args.skip_stats:
            _run_step(
                session, "cfbd_stats", "cfbd_stats",
                ingest_team_stats, args.year,
            )

        if not args.skip_qb:
            _run_step(
                session, "cfbd_qb", "cfbd_qb",
                ingest_qb_stats, args.year,
            )

    finally:

        print(f"Ingestion run complete for {args.year}.")

        session.close()


if __name__ == "__main__":
    main()
