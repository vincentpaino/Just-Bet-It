"""
Line-aware models: use the sportsbook line as an input, compared fairly
against the line itself and the regular models.

The regular models (scripts/train.py) predict yards from stats alone. Books
also know about injuries, game plans and offense style, so these models
start from the line and learn how far the actual result tends to land from
it:

  blend          line + w * (ensemble - line)
                 one learned number: how much to trust the model over the book
  line-aware     (actual - line) predicted by a small ridge regression on the
                 ensemble's disagreement with the line, the line itself, and
                 the QB's last-5 average vs the line

They can only learn from games with a stored pregame line (a few dozen so
far, ~60 more each week), so they're kept tiny, and every game is predicted
by a version fit WITHOUT it, two ways:

  leave-one-out   each game predicted from all the other games
  forward         each date predicted only from earlier dates (strictest;
                  what you'd actually have known at the time)

Run from the project root (after python -m scripts.train), e.g. each week:
    python -m scripts.line_model
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest import load_graded_games, simulate_betting_strategy
from scripts.train import MODELS_PATH, load_models, predict_all
from scripts.xgboost_reg import PROJECT_ROOT

MIN_TRAIN_GAMES = 10          # forward evaluation needs at least this many earlier games
RIDGE_ALPHAS = np.logspace(-1, 3, 20)


def line_aware_inputs(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "model_minus_line": df["ensemble"] - df["line"],
        "line": df["line"],
        "last5_minus_line": df["avg_pass_yds_last5"] - df["line"],
    })


def fit_blend_weight(train: pd.DataFrame) -> float:
    """Least-squares w for actual ≈ line + w * (ensemble - line), kept between 0 and 1."""
    d = (train["ensemble"] - train["line"]).to_numpy()
    e = (train["passing_yards"] - train["line"]).to_numpy()
    return float(np.clip((d @ e) / (d @ d), 0.0, 1.0)) if (d @ d) > 0 else 0.0


def fit_line_aware(train: pd.DataFrame):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(line_aware_inputs(train), train["passing_yards"] - train["line"])
    return model


def predict_fold(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Line-aware predictions for `test`, using only what was learned from `train`."""
    w = fit_blend_weight(train)
    adjust = fit_line_aware(train).predict(line_aware_inputs(test))
    return {
        "blend": test["line"].to_numpy() + w * (test["ensemble"] - test["line"]).to_numpy(),
        "line-aware": test["line"].to_numpy() + adjust,
    }


def leave_one_out(games: pd.DataFrame) -> dict:
    preds = {"blend": np.empty(len(games)), "line-aware": np.empty(len(games))}
    for i in range(len(games)):
        fold = predict_fold(games.drop(index=i), games.iloc[[i]])
        for name in preds:
            preds[name][i] = fold[name][0]
    return preds


def forward_by_date(games: pd.DataFrame):
    """Predict each date from earlier dates only. Returns (mask of predicted games, predictions)."""
    preds = {"blend": np.full(len(games), np.nan), "line-aware": np.full(len(games), np.nan)}
    for date in sorted(games["game_date"].unique()):
        train, test = games[games["game_date"] < date], games[games["game_date"] == date]
        if len(train) < MIN_TRAIN_GAMES:
            continue
        fold = predict_fold(train, test)
        for name in preds:
            preds[name][test.index] = fold[name]
    return ~np.isnan(preds["blend"]), preds


def report(title: str, games: pd.DataFrame, preds: dict):
    y = games["passing_yards"].to_numpy(dtype=float)
    print(f"\n{title} ({len(games)} games)")
    print(f"{'model':16} {'MAE':>6} {'RMSE':>6}   {'picks':>7}  {'win %':>6}  {'units':>6}")
    rows = {"sportsbook line": games["line"].to_numpy(dtype=float), **preds}
    for name, pred in rows.items():
        mae, rmse = np.mean(np.abs(y - pred)), np.sqrt(np.mean((y - pred) ** 2))
        bet = simulate_betting_strategy(pred, games, 0.0)
        if bet["total_bets"]:
            losses = bet["total_bets"] - bet["winning_bets"]
            picks = f"{bet['winning_bets']}-{losses}"
            print(f"{name:16} {mae:6.1f} {rmse:6.1f}   {picks:>7}  {bet['win_rate']:6.1%}  {bet['total_profit']:+6.2f}")
        else:
            print(f"{name:16} {mae:6.1f} {rmse:6.1f}   {'(no bets: it IS the line)':>30}")


def main():
    bundle = load_models()
    if bundle is None:
        print(f"ERROR: {MODELS_PATH.relative_to(PROJECT_ROOT)} not found. Train first: python -m scripts.train")
        return 1

    _, games = load_graded_games(bundle)
    if len(games) < MIN_TRAIN_GAMES + 5:
        print(f"Only {len(games)} graded games with lines; need at least {MIN_TRAIN_GAMES + 5}. "
              f"Run again after more weekly snapshots.")
        return 0

    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    base = predict_all(bundle["models"], games[bundle["features"]])
    games["ensemble"] = base["ensemble"]
    regular = {"xgboost": base["xgboost"], "ensemble": base["ensemble"]}

    counts = games["game_date"].dt.date.value_counts().sort_index()
    print("Line-aware models vs. the sportsbook line and the regular models")
    print("Graded games by date: " + ", ".join(f"{d}: {n}" for d, n in counts.items()))

    report("Leave-one-out: every game predicted from all the others",
           games, {**regular, **leave_one_out(games)})

    mask, fwd = forward_by_date(games)
    if mask.any():
        sub = games[mask].reset_index(drop=True)
        report("Forward: each date predicted only from earlier dates (strictest)", sub,
               {**{k: v[mask] for k, v in regular.items()}, **{k: v[mask] for k, v in fwd.items()}})
    else:
        print("\nForward evaluation needs more dates of data.")

    # What the line-aware models learn from all graded games
    w = fit_blend_weight(games)
    ridge = fit_line_aware(games)
    coefs = ridge[-1].coef_ / ridge[0].scale_
    print(f"\nFit on all {len(games)} games:")
    print(f"  blend trusts the model {w:.0%} and the line {1 - w:.0%}")
    print("  line-aware: actual - line ≈ "
          f"{ridge[-1].intercept_ - coefs @ ridge[0].mean_:+.1f}"
          + "".join(f" {c:+.3f}×{name}" for c, name in zip(coefs, line_aware_inputs(games).columns))
          + f"   (ridge alpha {ridge[-1].alpha_:.0f})")
    print("\nNOTE: with this few games, differences of a few yards or a few picks are noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
