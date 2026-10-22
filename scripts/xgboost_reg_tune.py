# Hyperparameter tuning using Optuna for the QB passing-yards XGBRegressor.
#
# Lives next to xgboost_reg.py (in scripts/). Run from the project root:
#     python3 scripts/xgboost_reg_tune.py
#
# How it avoids cheating:
#   * The test set (most recent TEST_FRACTION of games) is NEVER used while tuning.
#   * Tuning uses time-series cross-validation inside the training data:
#     each fold trains on earlier games and validates on later ones.
#   * Only after the search finishes do we score the best params once on the test set.

import json
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Reuse the data loading / feature engineering from the main script and other globals.
from xgboost_reg import (
    BEST_PARAMS_PATH,
    DEFAULT_PARAMS,
    FEATURES,
    TEST_FRACTION,
    build_features,
    filter_trainable,
    load_qb_games,
    seed,
)

N_TRIALS = 100   # more trials = better search but slower
N_SPLITS = 4    # number of time-series CV folds


def make_model(params: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        **params,
    )


def objective(trial: optuna.Trial, X, y) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    fold_rmses = []
    for step, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(X)):
        model = make_model(params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[va_idx])
        fold_rmses.append(root_mean_squared_error(y.iloc[va_idx], pred))

        # Stop hopeless trials early
        trial.report(float(np.mean(fold_rmses)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_rmses))


if __name__ == "__main__":
    # Same data prep and date split as xgboost_reg.py
    data = filter_trainable(build_features(load_qb_games()))
    data = data.sort_values("game_date").reset_index(drop=True)
    cutoff = data["game_date"].quantile(1 - TEST_FRACTION)
    train = data[data["game_date"] <= cutoff]
    test = data[data["game_date"] > cutoff]

    X_train, y_train = train[FEATURES], train["passing_yards"]
    X_test, y_test = test[FEATURES], test["passing_yards"]
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows (held out until the end)")

    # Tune
    study = optuna.create_study(
        direction="minimize",  # we are minimizing RMSE
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(lambda t: objective(t, X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest CV RMSE: {study.best_value:.2f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    BEST_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))
    print(f"\nSaved to {BEST_PARAMS_PATH} (scripts/xgboost_reg.py uses it from now on)")

    # One-time test evaluation: tuned vs. your original hand-picked settings
    for label, params in [("Default (hand-picked)", DEFAULT_PARAMS), ("Optuna-tuned", study.best_params)]:
        model = make_model(params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(
            f"{label:22} test RMSE: {root_mean_squared_error(y_test, pred):7.2f}   "
            f"MAE: {mean_absolute_error(y_test, pred):7.2f}"
        )