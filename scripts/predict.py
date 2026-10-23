"""
Predict one QB's passing yards for a scheduled game, with every trained model.

The game is looked up in the schedule (games table), so the opponent,
home/away and week are real, and features come from the same code the
models were trained with (scripts/xgboost_reg.py). Works for upcoming games
and for past ones (then the actual result is shown too).

Run from the project root (after python -m scripts.train):
    python -m scripts.predict --qb-id 5078810 --game-date 2026-10-24
    python -m scripts.predict --qb-id 5078810 --list-recent
"""
import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from scripts.train import MODELS_PATH, load_models, predict_all
from scripts.xgboost_reg import PROJECT_ROOT, build_upcoming_features, load_qb_games
from src.db.session import engine


def recent_games(qb_id, limit=10) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT s.game_date, t.team_name AS team, o.team_name AS opponent, s.home_away,
               s.passing_yards, s.completions, s.attempts, s.passing_tds, s.passing_ints
        FROM qb_game_stats s
        JOIN teams t ON t.team_id = s.team_id
        LEFT JOIN teams o ON o.team_id = s.opponent_team_id
        WHERE s.player_id = :qb ORDER BY s.game_date DESC LIMIT :n
    """), engine, params={"qb": qb_id, "n": limit})


def find_game(qb_id, game_date, opponent_team_id=None):
    """The QB's team (most recent before the game) and its scheduled game that day."""
    with engine.connect() as c:
        qb = c.execute(text("""
            SELECT player_name, team_id FROM qb_game_stats
            WHERE player_id = :qb AND game_date < :d ORDER BY game_date DESC LIMIT 1
        """), {"qb": qb_id, "d": game_date}).first()
        if qb is None:
            return None, None
        game = c.execute(text("""
            SELECT g.game_id, g.game_date, g.season, g.week, g.home_team_id, g.away_team_id,
                   ht.team_name AS home, at.team_name AS away
            FROM games g
            JOIN teams ht ON ht.team_id = g.home_team_id
            JOIN teams at ON at.team_id = g.away_team_id
            WHERE g.game_date = :d AND :team IN (g.home_team_id, g.away_team_id)
              AND (CAST(:opp AS INTEGER) IS NULL OR :opp IN (g.home_team_id, g.away_team_id))
        """), {"d": game_date, "team": qb.team_id, "opp": opponent_team_id}).first()
    return qb, game


def main():
    parser = argparse.ArgumentParser(description="Predict a QB's passing yards for a scheduled game")
    parser.add_argument("--qb-id", type=int, required=True, help="CFBD player ID of the QB")
    parser.add_argument("--game-date", type=str, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--opponent-team-id", type=int,
                        help="Only needed if the QB's team plays twice that day (it never should)")
    parser.add_argument("--list-recent", action="store_true", help="List the QB's recent games instead")
    args = parser.parse_args()

    if args.list_recent:
        print(recent_games(args.qb_id).to_string(index=False))
        return 0

    if not args.game_date:
        parser.error("--game-date is required (or use --list-recent)")
    try:
        game_date = datetime.strptime(args.game_date, "%Y-%m-%d").date()
    except ValueError:
        parser.error("--game-date must be YYYY-MM-DD")

    bundle = load_models()
    if bundle is None:
        print(f"ERROR: {MODELS_PATH.relative_to(PROJECT_ROOT)} not found. Train first: python -m scripts.train")
        return 1

    qb, game = find_game(args.qb_id, game_date, args.opponent_team_id)
    if qb is None:
        print(f"ERROR: no games on record for QB {args.qb_id} before {game_date}")
        return 1
    if game is None:
        print(f"ERROR: no scheduled game for {qb.player_name}'s team on {game_date}")
        return 1

    is_home = qb.team_id == game.home_team_id
    upcoming = pd.DataFrame([{
        "game_id": game.game_id, "player_id": args.qb_id, "player_name": qb.player_name,
        "team_id": qb.team_id,
        "opponent_team_id": game.away_team_id if is_home else game.home_team_id,
        "game_date": game.game_date, "season": game.season, "week": game.week,
        "home_away": "home" if is_home else "away",
    }])

    features = build_upcoming_features(load_qb_games(), upcoming)
    if features.empty:
        print(f"ERROR: {qb.player_name} doesn't have enough history yet (needs 3+ prior games "
              f"with 10+ attempts)")
        return 1

    preds = predict_all(bundle["models"], features[bundle["features"]])
    row = features.iloc[0]

    print(f"{qb.player_name}: {game.away} @ {game.home}, {game.game_date} "
          f"({'home' if is_home else 'away'}, week {game.week})")
    print(f"Last 5 games avg: {row['avg_pass_yds_last5']:.0f} yds | "
          f"opponent allows {row['opp_pass_yds_allowed_avg']:.0f} pass yds/game\n")
    for name, pred in preds.items():
        print(f"  {name:18} {pred[0]:6.1f} passing yards")
    if pd.notna(row["passing_yards"]):
        print(f"\n  actual             {row['passing_yards']:6.0f} passing yards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
