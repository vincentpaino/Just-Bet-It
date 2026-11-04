"""
Record this week's picks in live_picks: the official record of what the model
said when the Friday job ran (Friday consensus line, all four predictions,
the side, and whether it's a confident pick). Each (game, QB) is recorded
once; re-running never overwrites it, so the record can't be changed later.

Confident pick (the rule registered 2026-10-26): XGBoost at least 15 yards
from the line and all four models on the same side.

The Friday job runs this right after retraining. scripts/snapshot_closing.py
then snapshots the confident picks shortly before kickoff, for live
closing-line value in the weekly report. Uses no API credits.

Run from the project root:
    python -m scripts.record_picks
"""
import sys

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from scripts.predict_upcoming import load_upcoming_lines, upcoming_qbs
from scripts.train import MODELS_PATH, load_models, predict_all
from scripts.xgboost_reg import PROJECT_ROOT, build_upcoming_features, load_qb_games
from src.db.models import LivePick
from src.db.session import SessionLocal, engine

MODELS = ["linear_regression", "random_forest", "xgboost", "neural_network"]
MIN_EDGE = 15.0
HOURS = 48          # same window as the Friday props snapshot


def picks_from(qbs: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """One pick per QB-game: consensus line, side, confident flag. `qbs` already
    carries pred_<model> columns for each QB-game."""
    consensus = lines.groupby(["game_id", "player_id"]).agg(
        line=("line", "median"), books=("sportsbook", "nunique")).reset_index()
    p = qbs.merge(consensus, on=["game_id", "player_id"])
    p["edge"] = p["pred_xgboost"] - p["line"]
    p = p[p["edge"] != 0].copy()
    p["side"] = np.where(p["edge"] > 0, "OVER", "UNDER")
    sides = np.sign(p[[f"pred_{m}" for m in MODELS]].to_numpy() - p[["line"]].to_numpy())
    p["confident"] = (p["edge"].abs() >= MIN_EDGE) & (np.abs(sides.sum(axis=1)) == len(MODELS))
    return p


def main():
    bundle = load_models()
    if bundle is None:
        print(f"[picks] ERROR: {MODELS_PATH.relative_to(PROJECT_ROOT)} not found; train first")
        return 1

    lines = load_upcoming_lines(HOURS)
    if lines.empty:
        print(f"[picks] no stored prop lines for games in the next {HOURS}h; nothing to record")
        return 0

    qbs = upcoming_qbs(lines)
    features = build_upcoming_features(
        load_qb_games(),
        qbs[["game_id", "player_id", "player_name", "team_id", "opponent_team_id",
             "game_date", "season", "week", "home_away"]],
    )
    preds = predict_all(bundle["models"], features[bundle["features"]])
    keyed = features[["game_id", "player_id", "player_name"]].reset_index(drop=True)
    keyed = keyed.assign(**{f"pred_{name}": preds[name] for name in MODELS + ["ensemble"]})
    picks = picks_from(keyed, lines)

    kickoffs = pd.read_sql("SELECT game_id, start_time FROM games WHERE start_time > now()", engine)
    picks = picks.merge(kickoffs, on="game_id")
    trained = pd.Timestamp(bundle["train_end"]).date()

    rows = [{
        "game_id": int(r.game_id), "player_id": int(r.player_id), "player_name": r.player_name,
        "kickoff": r.start_time, "line": float(r.line), "books": int(r.books),
        **{f"pred_{m}": round(float(getattr(r, f"pred_{m}")), 1) for m in MODELS + ["ensemble"]},
        "side": r.side, "edge": round(float(r.edge), 1), "confident": bool(r.confident),
        "model_trained_through": trained,
    } for r in picks.itertuples()]

    session = SessionLocal()
    try:
        added = 0
        if rows:
            result = session.execute(
                pg_insert(LivePick.__table__).values(rows)
                .on_conflict_do_nothing(constraint="uq_live_pick")
                .returning(LivePick.__table__.c.id)
            )
            added = len(result.fetchall())
        session.commit()
    finally:
        session.close()

    print(f"[picks] {len(rows)} picks for games in the next {HOURS}h ({int(picks['confident'].sum())} "
          f"confident); {added} newly recorded, {len(rows) - added} already on record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
