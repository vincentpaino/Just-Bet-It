"""
The real test: our models vs. sportsbook CLOSING lines on every graded game
with one (2024 to now, from the historical backfill + weekly snapshots).

Every game is predicted out of sample by walking forward: before each week,
all four models are retrained on every game played before it. Predictions
are cached in data/raw/walkforward_preds.csv (gitignored) so re-runs are fast.

Pre-registered rules (2026-11-02, before running):
  Test A  The model "beats the books" if betting its side on every graded
          game wins more than 52.4% with a one-sided binomial p < 0.05 and
          positive units, at each book's real closing price. The confident-
          picks rule registered 2026-10-26 (|XGBoost - line| >= 15 and all
          four models on the same side) is judged the same way.
  Test B  A line-aware model (predicts actual - line from the line, the
          models' disagreement with it and a few market/QB inputs), trained
          on 2024-25 lined games and tested on 2026 ones, passes by the same
          rule on the 2026 games.

Run from the project root:
    python -m scripts.vs_closing_lines
    python -m scripts.vs_closing_lines --params configs/xgboost_reg_params_pre2024.json \
        --cache data/raw/walkforward_preds_pre2024.csv    # XGBoost settings tuned before 2024
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import scripts.xgboost_reg as X
from scripts.backtest import load_closing_prop_lines, simulate_betting_strategy
from scripts.train import build_models, predict_all
from src.db.session import SessionLocal

CACHE = X.PROJECT_ROOT / "data" / "raw" / "walkforward_preds.csv"
MODELS = ["linear_regression", "random_forest", "xgboost", "neural_network"]
BREAK_EVEN = 0.524
TEST_SEASON = 2026


def p_value(wins: int, n: int, p0: float = BREAK_EVEN) -> float:
    """One-sided binomial P(at least `wins` of `n` | true win rate p0)."""
    return float(binom.sf(wins - 1, n, p0))   # exact, and safe for thousands of bets


def walk_forward(data: pd.DataFrame, lined: pd.DataFrame, params: dict | None = None,
                 cache: Path = CACHE) -> pd.DataFrame:
    """Out-of-sample predictions for every lined game, retraining before each week."""
    if cache.exists():
        cached = pd.read_csv(cache)
        if set(zip(lined.game_id, lined.player_id)) <= set(zip(cached.game_id, cached.player_id)):
            print(f"(using cached walk-forward predictions: {cache})")
            return cached
    params = params or X.load_params()[0]
    rounds = lined.groupby(["season", "week"])["game_date"].min().sort_values()
    out = []
    for i, ((season, week), start) in enumerate(rounds.items(), start=1):
        rows = lined[(lined.season == season) & (lined.week == week)]
        history = data[data.game_date < start]
        models = build_models(params)
        for m in models.values():
            m.fit(history[X.FEATURES], history["passing_yards"])
        preds = predict_all(models, rows[X.FEATURES])
        out.append(rows[["game_id", "player_id"]].assign(**{k: v for k, v in preds.items()},
                                                          trained_through=history.game_date.max().date()))
        if i % 10 == 0 or i == len(rounds):
            print(f"  walk-forward: {i}/{len(rounds)} weeks")
    result = pd.concat(out, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache, index=False)
    return result


def bet_report(label: str, pred, games: pd.DataFrame, mask=None):
    """Flat 1-unit bets on the side `pred` favors, at the book's real closing price."""
    games = games if mask is None else games[mask]
    pred = np.asarray(pred) if mask is None else np.asarray(pred)[np.asarray(mask)]
    r = simulate_betting_strategy(pred, games, 0.0)
    n, w = r["total_bets"], r["winning_bets"]
    decided = n - int((games["passing_yards"] == games["line"]).sum())
    pv = p_value(w, decided) if decided else float("nan")
    verdict = "PASS" if (r["win_rate"] > BREAK_EVEN and pv < 0.05 and r["total_profit"] > 0) else "fail"
    print(f"  {label:44} {n:5} bets  {w:4}-{n - w:<4} {r['win_rate']:6.1%}  {r['total_profit']:+8.2f} u  "
          f"ROI {r['roi']:+6.1%}  p = {pv:.3f}  {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward models vs. closing lines")
    parser.add_argument("--params", type=Path, help="XGBoost settings file (default: configs/xgboost_reg_params.json)")
    parser.add_argument("--cache", type=Path, default=CACHE, help="Walk-forward prediction cache")
    args = parser.parse_args()
    params = json.loads(args.params.read_text()) if args.params else None

    data = X.filter_trainable(X.build_features(X.load_qb_games()))
    session = SessionLocal()
    try:
        lines = load_closing_prop_lines(session)
    finally:
        session.close()
    lined = data.merge(lines, on=["game_id", "player_id"], how="inner")
    lined = lined[lined.game_date >= "2024-08-01"].reset_index(drop=True)
    print(f"{len(lined):,} graded QB-games with a pregame closing line ({lined.game_date.min().date()} "
          f"to {lined.game_date.max().date()}); by season: {lined.groupby('season').size().to_dict()}")

    preds = walk_forward(data, lined, params, args.cache)
    g = lined.merge(preds, on=["game_id", "player_id"], how="inner")
    y = g["passing_yards"].to_numpy(dtype=float)

    # ---------------- Test A ----------------
    print("\nTEST A: walk-forward models vs. closing lines (rule: > 52.4%, p < 0.05, units > 0)")
    print(f"  {'':44} {'avg miss (yds)':>14}")
    for name in ["sportsbook line", "xgboost", "ensemble"]:
        p = g["line"] if name == "sportsbook line" else g[name]
        print(f"  {name:44} {np.mean(np.abs(y - p)):14.1f}")
    print()
    bet_report("XGBoost, every game", g["xgboost"], g)
    bet_report("ensemble, every game", g["ensemble"], g)
    sides = np.sign(g[MODELS].to_numpy() - g[["line"]].to_numpy())
    confident = ((g["xgboost"] - g["line"]).abs() >= 15) & (np.abs(sides.sum(axis=1)) == len(MODELS))
    bet_report("confident picks (registered 2026-10-26)", g["xgboost"], g, confident)
    print("\n  by season (XGBoost, every game):")
    for season in sorted(g.season.unique()):
        bet_report(f"  {season}", g["xgboost"], g, g.season == season)
    print("\n  descriptive only, by how far XGBoost is from the line:")
    edge = (g["xgboost"] - g["line"]).abs()
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 999)]:
        bet_report(f"  {lo}-{hi if hi < 999 else '+'} yds", g["xgboost"], g, (edge >= lo) & (edge < hi))

    # ---------------- Test B ----------------
    train, test = g[g.season < TEST_SEASON], g[g.season == TEST_SEASON]
    feats = lambda d: pd.DataFrame({
        "ens_minus_line": d["ensemble"] - d["line"], "xgb_minus_line": d["xgboost"] - d["line"],
        "line": d["line"], "last5_minus_line": d["avg_pass_yds_last5"] - d["line"],
        "game_total": d["game_total"], "team_spread": d["team_spread"],
        "implied_team_pts": d["implied_team_pts"], "pass_yds_std_last5": d["pass_yds_std_last5"],
    })
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))
    ridge.fit(feats(train), train["passing_yards"] - train["line"])
    line_aware = test["line"].to_numpy() + ridge.predict(feats(test))

    d_tr = (train["ensemble"] - train["line"]).to_numpy()
    w = float(np.clip(d_tr @ (train["passing_yards"] - train["line"]).to_numpy() / (d_tr @ d_tr), 0, 1))
    blend = test["line"].to_numpy() + w * (test["ensemble"] - test["line"]).to_numpy()

    yt = test["passing_yards"].to_numpy(dtype=float)
    print(f"\nTEST B: line-aware models, trained on {len(train):,} games (2024-25), tested on "
          f"{len(test):,} {TEST_SEASON} games (rule: > 52.4%, p < 0.05, units > 0)")
    for name, p in [("sportsbook line", test["line"]), ("XGBoost (walk-forward)", test["xgboost"]),
                    ("blend: line + w*(ensemble - line)", blend), ("line-aware ridge", line_aware)]:
        print(f"  {name:44} avg miss {np.mean(np.abs(yt - np.asarray(p))):6.1f} yds")
    print(f"  (blend learned w = {w:.2f}: trusts the model {w:.0%}, the line {1 - w:.0%})\n")
    bet_report("XGBoost, every 2026 game", test["xgboost"], test)
    bet_report("line-aware ridge, every 2026 game", line_aware, test)
    if w > 0:
        bet_report("blend, every 2026 game (same sides as ensemble)", blend, test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
