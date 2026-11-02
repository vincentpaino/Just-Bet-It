"""
Backfill historical QB passing-yards prop lines from The Odds API (paid plans).

For each game day in a date range, newest first:
  1. list that day's events as of shortly before the first kickoff (1 credit)
  2. for each of our games, fetch the player_pass_yds props as they were
     MINUTES_BEFORE minutes before kickoff (10 credits; free if none posted)

Lines are stored in qb_prop_lines exactly like the weekly snapshots, with
fetched_at = the snapshot's real historical time, so everything that uses
"each book's last pregame line" (backtest, line model, stale-line test)
works on them unchanged.

Two snapshot times (--snapshot):
  closing   MINUTES_BEFORE minutes before each kickoff (default): the closing line
  friday9   the Friday 9:00am CT before each game, what the weekly job captures
            live; with the closing snapshot it measures closing-line value (CLV).
            Like the live job, only games kicking off within 48h of that Friday
            9am (Friday night and Saturday) are included.

Safety:
  --estimate      costs nothing: counts our games and prints the cost range
  --budget N      hard cap on credits this run spends
  resume          progress is saved after every day (data/raw/backfill_progress.json)
                  and games that already have a snapshot shortly before kickoff
                  are skipped, so re-running never pays twice
  newest first    if the budget runs out, it's the oldest games that are skipped

Run from the project root:
    python -m scripts.backfill_props --estimate --start 2024-08-01 --end 2026-11-01
    python -m scripts.backfill_props --start 2024-09-07 --end 2024-09-07 --budget 400   # pilot
    python -m scripts.backfill_props --start 2024-08-01 --end 2026-11-01 --budget 19000
    python -m scripts.backfill_props --snapshot friday9 --start 2025-08-01 --end 2026-11-01 --budget 5000
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text

from src.db.models import QBPropLine
from src.db.session import SessionLocal, engine
from src.ingestion.dates import as_utc_datetime
from src.ingestion.odds_api import (
    OddsAPIError,
    fetch_historical_event_props,
    fetch_historical_events,
    validate_and_format_props,
)
from src.ingestion.run_ingestion import _QBMatcher, _load_odds_name_map
from src.ingestion.team_names import resolve_odds_team_name
from scripts.xgboost_reg import PROJECT_ROOT

PROGRESS_PATH = PROJECT_ROOT / "data" / "raw" / "backfill_progress.json"            # closing
FRIDAY_PROGRESS_PATH = PROJECT_ROOT / "data" / "raw" / "backfill_progress_friday9.json"
CENTRAL = ZoneInfo("America/Chicago")
MINUTES_BEFORE = 60          # snapshot this long before kickoff (~the closing line)
EVENTS_COST, PROPS_COST = 1, 10
PROPS_POSTED_SHARE = 0.65    # rough share of games with QB props (from this season's snapshots)


def plan_refused(exc: Exception) -> bool:
    """The Odds API's 'historical data needs a paid plan' (or other auth) refusal."""
    message = str(exc)
    return "HISTORICAL_UNAVAILABLE" in message or "(401)" in message or "(403)" in message


def iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def load_games(start: str, end: str) -> pd.DataFrame:
    """Our games with a known kickoff in [start, end], already played."""
    games = pd.read_sql(text("""
        SELECT g.game_id, g.game_date, g.start_time, g.season, g.home_team_id, g.away_team_id
        FROM games g
        WHERE g.start_time IS NOT NULL AND g.game_date BETWEEN :start AND :end
          AND g.start_time < now()
    """), engine, params={"start": start, "end": end})
    games["start_time"] = pd.to_datetime(games["start_time"], utc=True)
    return games


def estimate(games: pd.DataFrame):
    days = games.groupby("season")["game_date"].nunique()
    count = games.groupby("season").size()
    print(f"{'season':>8} {'game days':>10} {'games':>7} {'max credits':>12} {'likely credits':>15}")
    for season in sorted(count.index, reverse=True):
        hi = days[season] * EVENTS_COST + count[season] * PROPS_COST
        likely = days[season] * EVENTS_COST + count[season] * PROPS_COST * PROPS_POSTED_SHARE
        print(f"{season:>8} {days[season]:10} {count[season]:7} {hi:12,.0f} {likely:15,.0f}")
    hi = games["game_date"].nunique() * EVENTS_COST + len(games) * PROPS_COST
    likely = games["game_date"].nunique() * EVENTS_COST + len(games) * PROPS_COST * PROPS_POSTED_SHARE
    print(f"{'total':>8} {games['game_date'].nunique():10} {len(games):7} {hi:12,.0f} {likely:15,.0f}")
    print("\n'max' assumes every game had props; 'likely' assumes ~65% did. Games without "
          "props cost nothing, and the --budget cap stops the run before going over.")


def friday_9am_before(kickoff) -> pd.Timestamp:
    """The latest Friday 9:00am Central at least an hour before kickoff (as UTC)."""
    t = (pd.Timestamp(kickoff) - timedelta(hours=1)).tz_convert(CENTRAL)
    friday = t.normalize() - timedelta(days=(t.weekday() - 4) % 7)
    friday = friday.replace(hour=9)
    if friday > t:                       # e.g. a Friday kickoff before 10am
        friday -= timedelta(days=7)
    return friday.tz_convert("UTC")


def already_captured(session, game_id: int, start, end) -> bool:
    """A snapshot for this game already exists between start and end (live or backfilled)."""
    naive = lambda t: pd.Timestamp(t).tz_convert("UTC").tz_localize(None).to_pydatetime()
    return session.query(QBPropLine.id).filter(
        QBPropLine.game_id == game_id,
        QBPropLine.fetched_at >= naive(start),
        QBPropLine.fetched_at < naive(end),
    ).first() is not None


def load_progress(path=None) -> dict:
    path = path or PROGRESS_PATH
    if path.exists():
        return json.loads(path.read_text())
    return {"done_days": []}


def save_progress(progress: dict, path=None):
    path = path or PROGRESS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=1))


def main():
    parser = argparse.ArgumentParser(description="Backfill historical QB prop lines (paid Odds API plans)")
    parser.add_argument("--start", required=True, help="First game date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Last game date (YYYY-MM-DD)")
    parser.add_argument("--estimate", action="store_true", help="Only print the cost estimate (free)")
    parser.add_argument("--budget", type=int, help="Max credits this run may spend (required unless --estimate)")
    parser.add_argument("--minutes-before", type=int, default=MINUTES_BEFORE,
                        help="Closing snapshots: this many minutes before kickoff")
    parser.add_argument("--snapshot", choices=["closing", "friday9"], default="closing",
                        help="closing = before kickoff (default); friday9 = the Friday 9am CT before")
    args = parser.parse_args()

    games = load_games(args.start, args.end)
    print(f"{len(games):,} played games with a kickoff time, {args.start} to {args.end}\n")
    if args.estimate or games.empty:
        if not games.empty:
            estimate(games)
        return 0
    if args.budget is None:
        parser.error("--budget is required (or use --estimate)")

    teams = dict(pd.read_sql(text("SELECT team_name, team_id FROM teams"), engine).values)
    name_map = _load_odds_name_map()
    friday = args.snapshot == "friday9"
    progress_path = FRIDAY_PROGRESS_PATH if friday else PROGRESS_PATH
    progress = load_progress(progress_path)
    # Batches: one per game day (closing) or one per Friday 9am snapshot (friday9)
    if friday:
        games["batch"] = games["start_time"].map(friday_9am_before)
        # Same window as the live weekly job: games kicking off within 48h of it
        live_window = (games["start_time"] - games["batch"]) <= timedelta(hours=48)
        print(f"friday9: {live_window.sum():,} of {len(games):,} games kick off within 48h of a Friday "
              f"9am snapshot (the rest the live job wouldn't capture either)")
        games = games[live_window]
    else:
        games["batch"] = games["game_date"]
    session = SessionLocal()

    spent = requested = with_props = lines_stored = skipped = 0
    remaining = None
    stopped_for_budget = False

    try:
        for batch in sorted(games["batch"].unique(), reverse=True):
            day_key = str(batch)
            if day_key in progress["done_days"]:
                continue
            day_games = games[games["batch"] == batch]
            if spent + EVENTS_COST > args.budget:
                stopped_for_budget = True
                break

            if friday:
                list_at = pd.Timestamp(batch)          # events as listed that Friday morning
            else:
                list_at = day_games["start_time"].min() - timedelta(minutes=args.minutes_before + 60)
            try:
                events, remaining, cost = fetch_historical_events(iso(list_at))
            except OddsAPIError as exc:
                if plan_refused(exc):
                    print(f"STOPPED: the Odds API refused the historical request; historical odds need "
                          f"a paid plan ({spent} credit(s) spent).\n  {str(exc)[:200]}")
                    return 1
                raise
            spent += cost if cost is not None else (EVENTS_COST if events else 0)

            # Match events to our games by teams (either orientation)
            by_matchup = {frozenset((g.home_team_id, g.away_team_id)): g for g in day_games.itertuples()}
            day_complete = True
            for event in events:
                ids = [teams.get(resolve_odds_team_name(event.get(side, ""), name_map) or event.get(side))
                       for side in ("home_team", "away_team")]
                game = by_matchup.get(frozenset(ids)) if None not in ids else None
                if game is None:
                    continue
                if friday:
                    target = pd.Timestamp(batch)
                    window = (target - timedelta(hours=2), target + timedelta(hours=2))
                else:
                    window = (game.start_time - timedelta(hours=6), game.start_time)
                if already_captured(session, game.game_id, *window):
                    skipped += 1
                    continue
                if spent + PROPS_COST > args.budget:
                    stopped_for_budget = True
                    day_complete = False        # come back to this day next run
                    break

                if friday:
                    snapshot_at = pd.Timestamp(batch)
                else:
                    snapshot_at = as_utc_datetime(event["commence_time"]) - timedelta(minutes=args.minutes_before)
                try:
                    data, snapshot_ts, remaining, cost = fetch_historical_event_props(event["id"], iso(snapshot_at))
                except OddsAPIError as exc:
                    if plan_refused(exc):
                        # The free plan can list past events but not their odds
                        print(f"STOPPED: historical odds need a paid Odds API plan "
                              f"({spent} credit(s) spent listing events).\n  {str(exc)[:200]}")
                        return 1
                    raise
                requested += 1
                records = validate_and_format_props(data) if data else []
                spent += cost if cost is not None else (PROPS_COST if records else 0)
                if not records:
                    continue

                fetched_at = pd.Timestamp(snapshot_ts or iso(snapshot_at)).tz_convert("UTC").tz_localize(None).to_pydatetime()
                match_qb = _QBMatcher(session, [game.home_team_id, game.away_team_id])
                for r in records:
                    session.add(QBPropLine(
                        game_id=game.game_id, player_id=match_qb(r["player_name"]),
                        player_name=r["player_name"], sportsbook=r["sportsbook"], line=r["line"],
                        over_odds=r["over_odds"], under_odds=r["under_odds"], fetched_at=fetched_at,
                    ))
                session.commit()
                with_props += 1
                lines_stored += len(records)

            if not day_complete:
                break
            progress["done_days"].append(day_key)
            save_progress(progress, progress_path)
            print(f"[backfill] {day_key}: {len(events)} events listed | so far {requested} games requested, "
                  f"{with_props} with props, {lines_stored} lines, {spent} credits spent, "
                  f"{remaining} remaining")
    finally:
        session.close()

    print(f"\nDone: {requested} games requested, {with_props} had props, {lines_stored} lines stored, "
          f"{skipped} skipped (already captured), {spent} credits spent, {remaining} remaining.")
    if stopped_for_budget:
        print(f"Stopped at the {args.budget}-credit budget. Re-run with a new budget to continue; "
              f"finished days are skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
