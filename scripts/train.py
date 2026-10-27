"""
Train every QB passing-yards model and save them together.

All models learn from the same real, pregame-only features built by
scripts/xgboost_reg.py (QB rolling averages, opponent defense, team passing
volume, rest, home/away...), with the same starter filter and the same
date-based split: train on older games, test on the newest 20%.

Models (Doc II, Section 5):
  - linear_regression  baseline
  - random_forest
  - xgboost            primary model; tuned settings from configs/xgboost_reg_params.json
  - neural_network     MLP
  - ensemble           average of the four

Linear regression, random forest and the neural net can't take blanks, so the
few missing values (a QB's first game of a season, a defense with < 2 prior
games) are filled with the training median, plus a column flagging that the
value was missing. XGBoost handles blanks itself.

Two steps:
  1. Evaluate: train on older games, score on the newest 20% (the numbers
     printed), and keep those held-out misses for the betting probabilities.
  2. Refit on ALL games and save that version for live predictions, so the
     models know this season's games (walk-forward testing showed a small,
     consistent gain; see reports/model_experiments.md).

Because the saved models have seen every game up to today, backtest.py and
the weekly report only score games played after this run. The Friday job
grades last weekend's picks BEFORE retraining, so each week is graded by the
model that made the picks.

Saves models/qb_models.pkl, used by backtest.py, predict.py,
predict_upcoming.py and the dashboard. Run from the project root:

    python -m scripts.train                  # evaluate, then refit on all games
    python -m scripts.train --holdout-only   # save the evaluation models instead
                                             # (to backtest past graded games)
"""

import argparse
import sys
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.xgboost_reg import (
    FEATURES,
    PROJECT_ROOT,
    build_features,
    filter_trainable,
    load_params,
    load_qb_games,
    make_xgboost,
    seed,
    split_by_date,
)

MODELS_PATH = PROJECT_ROOT / "models" / "qb_models.pkl"


def build_models(xgb_params: dict) -> dict:
    def imputed(*steps):
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), *steps)

    return {
        "linear_regression": imputed(StandardScaler(), LinearRegression()),
        "random_forest": imputed(RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=-1,
        )),
        "xgboost": make_xgboost(xgb_params),
        "neural_network": imputed(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=0.001, learning_rate="adaptive",
            max_iter=500, early_stopping=True, random_state=seed,
        )),
    }


def predict_all(models: dict, X) -> dict:
    """Predictions from every model plus their average (ensemble), floored at 0 yards."""
    preds = {name: np.maximum(0, model.predict(X)) for name, model in models.items()}
    preds["ensemble"] = np.mean(list(preds.values()), axis=0)
    return preds


def load_models(path=MODELS_PATH):
    """The bundle saved by this script (models, features, train_end), or None."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train and save the QB passing-yards models")
    parser.add_argument("--holdout-only", action="store_true",
                        help="Save the models trained before the test split instead of refitting on all games")
    args = parser.parse_args()

    print("Training QB passing-yards models")
    print("-" * 60)

    data = filter_trainable(build_features(load_qb_games()))
    train, test, cutoff = split_by_date(data)
    X_train, y_train = train[FEATURES], train["passing_yards"]
    X_test, y_test = test[FEATURES], test["passing_yards"]
    print(f"{len(data):,} QB games ({data['game_date'].min().date()} to {data['game_date'].max().date()}), "
          f"{len(FEATURES)} features")
    print(f"Train: {len(train):,} games through {cutoff.date()} | Test: {len(test):,} newer games")

    xgb_params, source = load_params()
    print(f"XGBoost settings: {source}\n")

    models = build_models(xgb_params)
    for name, model in models.items():
        print(f"  training {name}...")
        model.fit(X_train, y_train)

    print(f"\n{'model':20} {'MAE':>7} {'RMSE':>7} {'R²':>7}   (test set, passing yards)")
    rows = predict_all(models, X_test)
    # Held-out misses (actual - predicted), saved so src/betting/probability.py
    # can turn a prediction into P(over the line) from the model's real errors
    residuals = {name: (y_test.to_numpy(dtype=float) - pred) for name, pred in rows.items()}
    rows["baseline: last-5 avg"] = X_test["avg_pass_yds_last5"].to_numpy()
    rows["baseline: train mean"] = np.full(len(y_test), y_train.mean())
    for name, pred in rows.items():
        print(f"{name:20} {mean_absolute_error(y_test, pred):7.2f} "
              f"{root_mean_squared_error(y_test, pred):7.2f} {r2_score(y_test, pred):7.3f}")

    # Step 2: refit on every game for live predictions (unless --holdout-only)
    train_end = cutoff
    if not args.holdout_only:
        print(f"\nRefitting on all {len(data):,} games for live predictions...")
        models = build_models(xgb_params)
        for model in models.values():
            model.fit(data[FEATURES], data["passing_yards"])
        train_end = data["game_date"].max()

    MODELS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "models": models,
        "features": FEATURES,
        "train_end": train_end,   # backtest/weekly report only score games after this
        "eval_cutoff": cutoff,    # where the evaluation split was
        "refit_on_all": not args.holdout_only,
        "xgb_params": xgb_params,
        "residuals": residuals,  # per model, held-out misses from the evaluation step
        "trained_at": datetime.now(),
    }, MODELS_PATH)
    print(f"\nSaved {len(models)} models to {MODELS_PATH.relative_to(PROJECT_ROOT)} "
          f"(trained through {train_end.date()}"
          f"{'; evaluation models only' if args.holdout_only else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
