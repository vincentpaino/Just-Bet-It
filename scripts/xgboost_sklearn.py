# Predicting QB passing yards (a continuous value), so we use a regression
# model: XGBRegressor.
#
# Pipeline:
#   1. Load raw per-QB-per-game stats from Postgres (qb_game_stats)
#   2. Engineer "pregame-only" features in pandas (nothing is deleted from the DB)
#   3. Filter out rows we can't train on (in memory only)
#   4. Split by DATE (train on the past, test on the future)
#   5. Train XGBRegressor and compare it against simple baselines
#
# Run from the project root:  python xgboost_sklearn.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

try:
    from dotenv import load_dotenv

    ENV_PATH = Path(__file__).resolve().parent.parent / "src" / ".env"
    load_dotenv(ENV_PATH)
except ImportError:
    pass

# Global variables
seed = 42
MIN_ATTEMPTS = 10      # ignore garbage-time backups / trick-play "QBs"
MIN_PRIOR_GAMES = 3    # need some history before we trust rolling averages
TEST_FRACTION = 0.2    # most recent 20% of games (by date) become the test set

engine = create_engine(
    URL.create(
        drivername="postgresql+psycopg2",
        username=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "") or None,
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        database=os.environ.get("DB_NAME", "just_bet_it"),
    )
)


# ----------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------
def load_qb_games() -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT game_id, player_id, player_name, team_id, opponent_team_id,
               game_date, season, week, home_away,
               passing_yards, passing_tds, passing_ints, completions, attempts,
               rushing_yards, rushing_attempts
        FROM qb_game_stats
        """,
        engine,
    )
    num_cols = [
        "passing_yards", "passing_tds", "passing_ints", "completions",
        "attempts", "rushing_yards", "rushing_attempts",
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    return df


# ----------------------------------------------------------------------------
# 2. Feature engineering
#    Every rolling feature uses shift(1) so a game's own stats are never
#    included in its own features (that would be target leakage).
# ----------------------------------------------------------------------------
def prior_mean(df, keys, col, window, min_periods=1):
    """Mean of the previous `window` games (excluding the current one)."""
    return df.groupby(keys)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
    )


FEATURES = [
    "games_played_prior",
    "avg_pass_yds_last3", "avg_pass_yds_last5", "avg_pass_yds_season",
    "pass_yds_std_last5",
    "avg_attempts_last5", "avg_ypa_last5", "avg_comp_pct_last5",
    "avg_rush_yds_last5",
    "opp_pass_yds_allowed_avg", "team_pass_att_avg",
    "is_home", "rest_days", "week",
    # engineered combinations
    "yds_trend", "exp_yards", "att_share",
]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(subset=["game_date", "passing_yards", "attempts"]).copy()

    # ---- team offense + opponent defense (uses ALL QBs in a game) ----------
    team = (
        df.groupby(["game_id", "team_id"], as_index=False)
        .agg(
            team_pass_yds=("passing_yards", "sum"),
            team_att=("attempts", "sum"),
            opp=("opponent_team_id", "first"),
            game_date=("game_date", "first"),
        )
        .sort_values(["team_id", "game_date", "game_id"])
    )
    team["team_pass_att_avg"] = prior_mean(team, "team_id", "team_att", 5, 2)

    # Yards a defense ALLOWED = passing yards of the team that faced it
    defense = (
        team[["game_id", "opp", "team_pass_yds", "game_date"]]
        .rename(columns={"opp": "defense_id", "team_pass_yds": "yds_allowed"})
        .dropna(subset=["defense_id"])
    )
    defense["defense_id"] = defense["defense_id"].astype(int)
    defense = defense.sort_values(["defense_id", "game_date", "game_id"])
    defense["opp_pass_yds_allowed_avg"] = prior_mean(
        defense, "defense_id", "yds_allowed", 5, 2
    )

    # ---- QB-level rows: real starters only ---------------------------------
    qb = df[df["attempts"] >= MIN_ATTEMPTS].copy()
    qb = qb.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    qb["ypa"] = qb["passing_yards"] / qb["attempts"]
    qb["comp_pct"] = qb["completions"] / qb["attempts"]

    qb["games_played_prior"] = qb.groupby("player_id").cumcount()
    qb["avg_pass_yds_last3"] = prior_mean(qb, "player_id", "passing_yards", 3)
    qb["avg_pass_yds_last5"] = prior_mean(qb, "player_id", "passing_yards", 5)
    qb["avg_attempts_last5"] = prior_mean(qb, "player_id", "attempts", 5)
    qb["avg_ypa_last5"] = prior_mean(qb, "player_id", "ypa", 5)
    qb["avg_comp_pct_last5"] = prior_mean(qb, "player_id", "comp_pct", 5)
    qb["avg_rush_yds_last5"] = prior_mean(qb, "player_id", "rushing_yards", 5)
    qb["pass_yds_std_last5"] = qb.groupby("player_id")["passing_yards"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).std()
    )
    qb["avg_pass_yds_season"] = qb.groupby(["player_id", "season"])[
        "passing_yards"
    ].transform(lambda s: s.shift(1).expanding().mean())

    # Context
    qb["is_home"] = qb["home_away"].str.lower().map({"home": 1, "away": 0})
    qb["rest_days"] = (
        qb.groupby("player_id")["game_date"].diff().dt.days.clip(upper=21)
    )

    # Attach team/defense features (left merges keep QB row order)
    qb = qb.merge(
        team[["game_id", "team_id", "team_pass_att_avg"]],
        on=["game_id", "team_id"], how="left",
    )
    qb = qb.merge(
        defense[["game_id", "defense_id", "opp_pass_yds_allowed_avg"]],
        left_on=["game_id", "opponent_team_id"],
        right_on=["game_id", "defense_id"], how="left",
    ).drop(columns=["defense_id"])

    # Engineered combinations
    qb["yds_trend"] = qb["avg_pass_yds_last3"] - qb["avg_pass_yds_season"]
    qb["exp_yards"] = qb["avg_ypa_last5"] * qb["avg_attempts_last5"]
    qb["att_share"] = qb["avg_attempts_last5"] / qb["team_pass_att_avg"]

    return qb


# ----------------------------------------------------------------------------
# 3. Filter (in memory only -- the DB is untouched)
# ----------------------------------------------------------------------------
def filter_trainable(qb: pd.DataFrame) -> pd.DataFrame:
    # We must have a target and enough history for the core rolling feature.
    # Other feature NaNs are left alone: XGBoost handles missing values natively.
    ok = (
        qb["passing_yards"].notna()
        & qb["avg_pass_yds_last5"].notna()
        & (qb["games_played_prior"] >= MIN_PRIOR_GAMES)
    )
    return qb[ok].copy()


# ----------------------------------------------------------------------------
# Optional: sportsbook prop line baseline (the real bar to beat)
# ----------------------------------------------------------------------------
def load_prop_lines() -> pd.DataFrame:
    try:
        pl = pd.read_sql(
            "SELECT game_id, player_id, line FROM qb_prop_lines WHERE player_id IS NOT NULL",
            engine,
        )
    except Exception:
        return pd.DataFrame(columns=["game_id", "player_id", "line"])
    pl["line"] = pd.to_numeric(pl["line"], errors="coerce")
    return pl.groupby(["game_id", "player_id"], as_index=False)["line"].median()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Clean and load in QB data (filtering happens here, not in the DB)
    raw = load_qb_games()
    print(f"Raw qb_game_stats rows: {len(raw):,}")
    print("home_away values:", raw["home_away"].value_counts(dropna=False).to_dict())

    qb = build_features(raw)
    data = filter_trainable(qb)
    print(f"QB rows with attempts >= {MIN_ATTEMPTS}: {len(qb):,}")
    print(f"Trainable rows (>= {MIN_PRIOR_GAMES} prior games, target present): {len(data):,}")
    print("\nNull rate per feature among trainable rows:")
    print((data[FEATURES].isna().mean() * 100).round(1).to_string())

    # 2. Split the data by date: train on the past, test on the future
    data = data.sort_values("game_date").reset_index(drop=True)
    cutoff = data["game_date"].quantile(1 - TEST_FRACTION)
    train = data[data["game_date"] <= cutoff]
    test = data[data["game_date"] > cutoff]
    print(f"\nTrain: {len(train):,} rows (through {cutoff.date()})")
    print(f"Test:  {len(test):,} rows (after {cutoff.date()})")

    X_train, y_train = train[FEATURES], train["passing_yards"]
    X_test, y_test = test[FEATURES], test["passing_yards"]

    # 3. Initialize the Regressor model
    reg_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",  # Default loss for regression
        random_state=seed,
    )

    # 4. Train and predict
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    # 5. Evaluate performance against baselines
    print("\n--- Results on test set (passing yards) ---")
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"XGBoost                 RMSE: {rmse:7.2f}   MAE: {mae:7.2f}")

    mean_pred = np.full(len(y_test), y_train.mean())
    print(f"Baseline: train mean    RMSE: {root_mean_squared_error(y_test, mean_pred):7.2f}")

    last5 = X_test["avg_pass_yds_last5"]
    print(f"Baseline: last-5 avg    RMSE: {root_mean_squared_error(y_test, last5):7.2f}")

    lines = load_prop_lines()
    cmp = test.assign(pred=y_pred).merge(lines, on=["game_id", "player_id"], how="inner")
    if len(cmp) > 0:
        print(
            f"Sportsbook line ({len(cmp)} test rows with a line): "
            f"line RMSE {root_mean_squared_error(cmp['passing_yards'], cmp['line']):.2f} "
            f"vs model RMSE {root_mean_squared_error(cmp['passing_yards'], cmp['pred']):.2f}"
        )
    else:
        print("Sportsbook line baseline: no matching qb_prop_lines rows yet.")

    print("\n--- Feature importance ---")
    imp = pd.Series(reg_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(imp.round(3).to_string())

    print("\n--- Sample predictions ---")
    sample = test.assign(predicted=y_pred.round(1))[
        ["game_date", "player_name", "passing_yards", "predicted", "avg_pass_yds_last5"]
    ]
    print(sample.head(15).to_string(index=False))