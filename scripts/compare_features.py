"""
Do the market features help? A fair head-to-head of the same models trained
with stats only vs. stats + market features (CFBD game total, spread and
implied team points).

Both versions use identical games, the same date-based split, and the same
model settings (scripts/train.py), so the only difference is the inputs.
They're scored two ways:
  1. the held-out test set (newest 20% of games, never trained on)
  2. the real QB prop lines in the backtest (games after the training cutoff)

Run from the project root:
    python -m scripts.compare_features
"""
import sys

import numpy as np
import pandas as pd

from scripts.backtest import load_graded_games, simulate_betting_strategy
from scripts.train import build_models, predict_all
from scripts.xgboost_reg import (
    FEATURES,
    MARKET_FEATURES,
    STATS_FEATURES,
    build_features,
    filter_trainable,
    load_params,
    load_qb_games,
    split_by_date,
)

VARIANTS = {"stats only": STATS_FEATURES, "stats + market": FEATURES}


def train_variants(train: pd.DataFrame) -> dict:
    params, _ = load_params()
    trained = {}
    for label, features in VARIANTS.items():
        models = build_models(params)
        for model in models.values():
            model.fit(train[features], train["passing_yards"])
        trained[label] = models
    return trained


def accuracy_table(trained: dict, games: pd.DataFrame, extra_rows: dict | None = None):
    y = games["passing_yards"].to_numpy(dtype=float)
    print(f"{'model':18} " + "  ".join(f"{label:>22}" for label in VARIANTS))
    print(f"{'':18} " + "  ".join(f"{'MAE':>10} {'RMSE':>11}" for _ in VARIANTS))
    preds = {label: predict_all(trained[label], games[VARIANTS[label]]) for label in VARIANTS}
    for name in preds["stats only"]:
        cells = []
        for label in VARIANTS:
            p = preds[label][name]
            cells.append(f"{np.mean(np.abs(y - p)):10.2f} {np.sqrt(np.mean((y - p) ** 2)):11.2f}")
        print(f"{name:18} " + "  ".join(cells))
    for name, p in (extra_rows or {}).items():
        cell = f"{np.mean(np.abs(y - p)):10.2f} {np.sqrt(np.mean((y - p) ** 2)):11.2f}"
        print(f"{name:18} {cell}  {'(same)':>22}")
    return preds


def main():
    data = filter_trainable(build_features(load_qb_games()))
    train, test, cutoff = split_by_date(data)
    print(f"Training {len(VARIANTS)} feature sets on {len(train):,} games through {cutoff.date()}; "
          f"market features: {', '.join(MARKET_FEATURES)}\n")
    trained = train_variants(train)

    print(f"1. Held-out test set: {len(test):,} newer games (lower is better)")
    accuracy_table(trained, test, {
        "baseline: last-5": test["avg_pass_yds_last5"].to_numpy(dtype=float),
    })

    _, graded = load_graded_games({"train_end": cutoff})
    if graded.empty:
        print("\n2. No graded prop-line games yet.")
        return 0

    print(f"\n2. Real QB prop lines: {len(graded)} graded games after the cutoff")
    preds = accuracy_table(trained, graded, {"sportsbook line": graded["line"].to_numpy(dtype=float)})

    print(f"\n{'picks vs the line':18} " + "  ".join(f"{label:>22}" for label in VARIANTS))
    for name in preds["stats only"]:
        cells = []
        for label in VARIANTS:
            r = simulate_betting_strategy(preds[label][name], graded, 0.0)
            losses = r["total_bets"] - r["winning_bets"]
            cells.append(f"{r['winning_bets']:>6}-{losses:<3} {r['total_profit']:+7.2f} u   ")
        print(f"{name:18} " + "  ".join(f"{c:>22}" for c in cells))

    print("\nNOTE: the prop-line section has few games so far; the test set is the main evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
