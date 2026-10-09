"""
Quick Exploratory Data Analysis (EDA) for Just Bet It. Run from the project root:

    python explore_data.py

Prints a report to the terminal and saves plots to ./eda_output/.
Uses the same DB_* environment variables as your app (reads .env if python-dotenv is installed).
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Same logic as your get_database_url(); swap for an import from your module if you prefer.
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

os.makedirs("eda_output", exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------- 1. row counts
section("1. ROW COUNTS")
tables = [
    "teams", "games", "features", "odds", "users", "user_predictions",
    "ingestion_log", "qb_game_stats", "qb_game_features", "qb_prop_lines",
]
with engine.connect() as conn:
    for t in tables:
        try:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            print(f"{t:20} {n:>10,}")
        except Exception as e:
            conn.rollback()
            print(f"{t:20} (missing: {type(e).__name__})")

# ---------------------------------------------------------------- 2. master frame
section("2. GAMES + FEATURES (one row per game)")
games = pd.read_sql(
    """
    SELECT g.game_id, g.game_date, g.season, g.week, g.season_type, g.neutral_site,
           g.home_score, g.away_score, g.home_win,
           ht.team_name AS home_team, at.team_name AS away_team,
           ht.conference AS home_conf, at.conference AS away_conf,
           f.rolling_win_pct_home, f.rolling_win_pct_away,
           f.elo_home, f.elo_away,
           f.rest_days_home, f.rest_days_away,
           f.home_away_split, f.sos_home, f.sos_away
    FROM games g
    JOIN teams ht ON ht.team_id = g.home_team_id
    JOIN teams at ON at.team_id = g.away_team_id
    LEFT JOIN features f ON f.game_id = g.game_id
    ORDER BY g.game_date
    """,
    engine,
)

feat_cols = [
    "rolling_win_pct_home", "rolling_win_pct_away", "elo_home", "elo_away",
    "rest_days_home", "rest_days_away", "home_away_split", "sos_home", "sos_away",
]
# Numeric columns arrive as Decimal (object dtype); convert them.
games[feat_cols] = games[feat_cols].apply(pd.to_numeric, errors="coerce")
games["game_date"] = pd.to_datetime(games["game_date"])
print(games.head(10).to_string())

# ---------------------------------------------------------------- 3. coverage
section("3. COVERAGE BY SEASON (do we have features and results for every game?)")
cov = games.groupby("season").agg(
    games=("game_id", "count"),
    has_features=("elo_home", lambda s: s.notna().sum()),
    has_result=("home_win", lambda s: s.notna().sum()),
    first_game=("game_date", "min"),
    last_game=("game_date", "max"),
)
print(cov.to_string())

# ---------------------------------------------------------------- 4. nulls
section("4. NULL RATES")
print((games[feat_cols + ["home_win", "week", "neutral_site"]].isna().mean() * 100).round(1).to_string())
print("\nNull rate by week (early-season NULLs are expected for rolling stats):")
print((games.groupby("week")[["rolling_win_pct_home", "elo_home", "rest_days_home"]]
       .apply(lambda d: d.isna().mean() * 100).round(1)).head(16).to_string())

# ---------------------------------------------------------------- 5. distributions
section("5. FEATURE SUMMARY STATS")
print(games[feat_cols].describe().T.round(3).to_string())

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, c in zip(axes.ravel(), feat_cols):
    games[c].dropna().hist(ax=ax, bins=40)
    ax.set_title(c)
plt.tight_layout()
plt.savefig("eda_output/feature_distributions.png", dpi=120)
plt.close()

# ---------------------------------------------------------------- 6. target
section("6. TARGET (home_win) BALANCE")
done = games.dropna(subset=["home_win"])
print(f"Overall home win rate: {done['home_win'].astype(float).mean():.3f}  (n={len(done):,})")
print(done.groupby("season")["home_win"].apply(lambda s: s.astype(float).mean()).round(3).to_string())
print("\nNeutral-site home win rate (should be near 0.5):")
print(done.groupby("neutral_site")["home_win"].apply(lambda s: s.astype(float).mean()).round(3).to_string())

# ---------------------------------------------------------------- 7. signal + leakage sanity
section("7. SIGNAL / LEAKAGE SANITY CHECK")
d = done.dropna(subset=["elo_home", "elo_away"]).copy()
d["elo_diff"] = d["elo_home"] - d["elo_away"]
d["win_pct_diff"] = d["rolling_win_pct_home"] - d["rolling_win_pct_away"]
d["sos_diff"] = d["sos_home"] - d["sos_away"]
d["rest_diff"] = d["rest_days_home"] - d["rest_days_away"]
d["y"] = d["home_win"].astype(float)
try:
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(d["y"], d["elo_diff"])
    print(f"AUC of elo_diff alone: {auc:.3f}")
    print("  ~0.70-0.78 is plausible for college football.")
    print("  >0.90 means Elo probably already includes the game's result (LEAKAGE).")
except ImportError:
    print("scikit-learn not installed; skipping AUC.")

diff_cols = ["elo_diff", "win_pct_diff", "sos_diff", "rest_diff", "home_away_split"]
corr = d[diff_cols + ["y"]].corr()["y"].drop("y").round(3)
print("\nCorrelation with home_win:")
print(corr.to_string())

# Continuity check: a team's pre-game Elo should move smoothly game to game.
long = pd.concat([
    d[["game_date", "home_team", "elo_home"]].rename(columns={"home_team": "team", "elo_home": "elo"}),
    d[["game_date", "away_team", "elo_away"]].rename(columns={"away_team": "team", "elo_away": "elo"}),
]).sort_values(["team", "game_date"])
long["elo_change"] = long.groupby("team")["elo"].diff()
print("\nGame-to-game change in a team's pre-game Elo (huge jumps = bug or season reset):")
print(long["elo_change"].describe().round(2).to_string())

# ---------------------------------------------------------------- 8. odds
section("8. ODDS COVERAGE")
odds = pd.read_sql(
    "SELECT game_id, sportsbook, home_moneyline, away_moneyline, spread, over_under, "
    "implied_prob_home, implied_prob_away, fetched_at FROM odds",
    engine,
)
num = ["home_moneyline", "away_moneyline", "spread", "over_under", "implied_prob_home", "implied_prob_away"]
odds[num] = odds[num].apply(pd.to_numeric, errors="coerce")
print(f"Odds rows: {len(odds):,}   games with odds: {odds['game_id'].nunique():,} "
      f"of {len(games):,} ({odds['game_id'].nunique() / max(len(games), 1) * 100:.1f}%)")
print("\nRows per sportsbook:")
print(odds["sportsbook"].value_counts().to_string())
per_game = odds.groupby("game_id").agg(snapshots=("fetched_at", "nunique"), books=("sportsbook", "nunique"))
print("\nSnapshots / books per game:")
print(per_game.describe().round(2).to_string())

odds["prob_sum"] = odds["implied_prob_home"] + odds["implied_prob_away"]
print("\nimplied_prob_home + implied_prob_away (>1 means vig is still in):")
print(odds["prob_sum"].describe().round(4).to_string())

# Timing check: were odds fetched before kickoff?
t = odds.merge(games[["game_id", "game_date"]], on="game_id")
t["days_before_game"] = (t["game_date"] - pd.to_datetime(t["fetched_at"]).dt.normalize()).dt.days
print("\nDays between odds snapshot and game date (negative = fetched AFTER the game):")
print(t["days_before_game"].describe().round(2).to_string())

print("\nDone. Plots saved to ./eda_output/")