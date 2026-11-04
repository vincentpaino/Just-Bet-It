"""
Snapshot confident picks' prop lines shortly before kickoff, for live
closing-line value (did the line move toward our pick after Friday?).

Scheduled Friday 4pm and Saturday 10am, 1pm, 4pm and 7pm (Central). Each run
covers confident picks (from live_picks) kicking off before the next
scheduled run, skips games that already started or already have a snapshot,
and stores the lines in qb_prop_closing_lines (kept apart from the Friday
lines in qb_prop_lines, which picks are graded at).

Credits: 1 per game (live odds, one market, one region); both QBs in a game
come back together. Never goes below RESERVE remaining, at most MAX_GAMES per
run. About 15-25 credits a week.

Run from the project root:
    python -m scripts.snapshot_closing
"""
import sys
from datetime import timedelta

import pandas as pd
from sqlalchemy import text

from src.db.models import QBPropClosingLine
from src.db.session import SessionLocal, engine
from src.ingestion.odds_api import fetch_event_props, fetch_events, validate_and_format_props
from src.ingestion.run_ingestion import _QBMatcher, _load_odds_name_map
from src.ingestion.team_names import resolve_odds_team_name

CENTRAL = "America/Chicago"
RUNS = [(4, 16), (5, 10), (5, 13), (5, 16), (5, 19)]   # (weekday Mon=0, hour) Central
RESERVE = 50
MAX_GAMES = 30


def window_end(now: pd.Timestamp) -> pd.Timestamp:
    """Cover games kicking off before the next scheduled run (+30 min overlap);
    after the week's last run, the next 8 hours (late Saturday games)."""
    now_ct = now.tz_convert(CENTRAL)
    week_start = now_ct.normalize() - timedelta(days=now_ct.weekday())
    runs = [week_start + timedelta(days=d, hours=h) for d, h in RUNS]
    later = [r for r in runs if r > now_ct + timedelta(minutes=5)]
    end = (later[0] + timedelta(minutes=30)) if later else now_ct + timedelta(hours=8)
    return end.tz_convert("UTC")


def targets(now: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Confident picks kicking off in (now, end] with no closing snapshot yet."""
    return pd.read_sql(text("""
        SELECT lp.game_id, lp.player_id, lp.player_name, lp.kickoff, g.home_team_id, g.away_team_id
        FROM live_picks lp JOIN games g USING (game_id)
        WHERE lp.confident AND lp.kickoff > :now AND lp.kickoff <= :end
          AND NOT EXISTS (SELECT 1 FROM qb_prop_closing_lines c
                          WHERE c.game_id = lp.game_id AND c.player_id = lp.player_id)
    """), engine, params={"now": now.to_pydatetime(), "end": end.to_pydatetime()})


def main():
    now = pd.Timestamp.now(tz="UTC")
    end = window_end(now)
    picks = targets(now, end)
    print(f"[closing] {len(picks)} confident picks kick off before {end.tz_convert(CENTRAL):%a %I:%M%p} CT "
          f"without a snapshot yet")
    if picks.empty:
        return 0

    events, remaining = fetch_events()          # free
    teams = dict(pd.read_sql(text("SELECT team_name, team_id FROM teams"), engine).values)
    name_map = _load_odds_name_map()
    by_teams = {}
    for e in events:
        ids = [teams.get(resolve_odds_team_name(e.get(side, ""), name_map) or e.get(side))
               for side in ("home_team", "away_team")]
        if None not in ids:
            by_teams[frozenset(ids)] = e

    session = SessionLocal()
    start_remaining, requested, stored = remaining, 0, 0
    try:
        for game_id, game in picks.groupby("game_id"):
            g = game.iloc[0]
            event = by_teams.get(frozenset((int(g.home_team_id), int(g.away_team_id))))
            if event is None:
                print(f"[closing] no Odds API event found for game {game_id}; skipped")
                continue
            if requested >= MAX_GAMES or (remaining is not None and remaining - 1 < RESERVE):
                print(f"[closing] stopping: credit reserve ({RESERVE}) or cap ({MAX_GAMES} games) reached")
                break
            data, remaining = fetch_event_props(event["id"])
            requested += 1
            match_qb = _QBMatcher(session, [int(g.home_team_id), int(g.away_team_id)])
            for r in validate_and_format_props(data):
                session.add(QBPropClosingLine(
                    game_id=int(game_id), player_id=match_qb(r["player_name"]), player_name=r["player_name"],
                    sportsbook=r["sportsbook"], line=r["line"], over_odds=r["over_odds"],
                    under_odds=r["under_odds"], fetched_at=r["fetched_at"],
                ))
                stored += 1
            session.commit()
    finally:
        session.close()

    used = (start_remaining - remaining) if start_remaining is not None and remaining is not None else "unknown"
    print(f"[closing] requested {requested} games, stored {stored} lines; credits used {used}, "
          f"remaining {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
