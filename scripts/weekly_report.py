"""
Weekly results tracker: how the models' picks did against real sportsbook lines.

Writes two files in reports/:
  graded_picks.csv     one row per graded QB-game (line, book, predictions,
                       picks, actual yards, win/loss, units). APPEND-ONLY: a
                       game is added once, when its result first comes in,
                       and never rewritten, so retraining the models later
                       can't change past weeks.
  weekly_results.md    week-by-week summary built from graded_picks.csv,
                       with running season totals.

A game with a pick recorded on Friday (live_picks, scripts/record_picks.py)
is graded with those recorded predictions: the picks actually made. Older
games are graded with whatever models are trained at the time, and only if
they're after the training cutoff (same rule as backtest.py). Picks are flat
1-unit bets on the side the model favors, paid at the book's real price.

Two strategies are tracked: betting every game, and "confident picks", a
rule fixed in advance (CONFIDENT_RULE) that only counts games played after
the date it was registered, so it can't be tuned to results already seen.

Live closing-line value: for confident picks recorded on Friday
(scripts/record_picks.py, table live_picks) and snapshotted shortly before
kickoff (scripts/snapshot_closing.py, table qb_prop_closing_lines), how far
the line moved toward the pick. Positive CLV is the early sign of a real edge.

The weekly job (scripts/snapshot_odds.sh) runs this right after it pulls the
previous weekend's results. Run by hand from the project root:
    python -m scripts.weekly_report
"""
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from scripts.backtest import american_payout, load_graded_games
from scripts.train import MODELS_PATH, load_models, predict_all
from scripts.xgboost_reg import PROJECT_ROOT
from src.db.session import engine

REPORTS_DIR = PROJECT_ROOT / "reports"
PICKS_PATH = REPORTS_DIR / "graded_picks.csv"
SUMMARY_PATH = REPORTS_DIR / "weekly_results.md"

# Models whose picks are tracked: the primary model and the ensemble
TRACKED = ["xgboost", "ensemble"]
BREAK_EVEN = 0.524   # win rate needed at -110

# Pre-registered "confident picks" strategy: decided on the date below, BEFORE
# any of the games it's scored on were played. Don't change it after the fact;
# to try a different rule, register a new one with a new date.
CONFIDENT_RULE = {
    "registered": "2026-10-26",   # only games played after this date count
    "min_edge": 15.0,             # |XGBoost prediction - line| in yards
    # ...and all of these models must pick the same side of the line
    "models": ["linear_regression", "random_forest", "xgboost", "neural_network"],
}


def grade(pred: float, actual: float, line: float, over_odds: float, under_odds: float):
    """(pick, result, units) for a flat 1-unit bet on the side the prediction favors."""
    if pred == line:
        return "NO BET", "no bet", 0.0
    pick = "OVER" if pred > line else "UNDER"
    if actual == line:
        return pick, "push", 0.0
    won = (actual > line) == (pick == "OVER")
    price = over_odds if pick == "OVER" else under_odds
    return pick, ("win" if won else "loss"), round(float(american_payout(price)) if won else -1.0, 3)


def is_confident(preds: dict, line: float, game_date: str, rule: dict = CONFIDENT_RULE):
    """
    Whether a game is a confident pick under the pre-registered rule: XGBoost
    is at least min_edge yards from the line and every listed model is on the
    same side. Returns None for games played on or before the registration
    date (not eligible: the rule was chosen after seeing them).
    """
    if game_date <= rule["registered"]:
        return None
    sides = {np.sign(preds[name] - line) for name in rule["models"]}
    return bool(abs(preds["xgboost"] - line) >= rule["min_edge"] and len(sides) == 1 and 0 not in sides)


ALL_MODELS = CONFIDENT_RULE["models"] + ["ensemble"]


def recorded_predictions() -> pd.DataFrame:
    """Predictions recorded when the picks were made (live_picks), one row per QB-game."""
    try:
        return pd.read_sql(text(
            "SELECT game_id, player_id, model_trained_through, "
            + ", ".join(f"pred_{m}::float AS {m}" for m in ALL_MODELS) + " FROM live_picks"), engine)
    except Exception as exc:   # table missing (older database): grade the old way
        print(f"[report] recorded picks unavailable ({type(exc).__name__}); re-predicting")
        return pd.DataFrame(columns=["game_id", "player_id", "model_trained_through", *ALL_MODELS])


def with_predictions(games: pd.DataFrame, bundle, recorded: pd.DataFrame) -> pd.DataFrame:
    """Each game's predictions: the recorded ones if the pick was recorded on Friday,
    else the current models' (only fair for games after their training cutoff)."""
    current = pd.DataFrame(predict_all(bundle["models"], games[bundle["features"]]))
    current["model_trained_through"] = pd.Timestamp(bundle["train_end"]).date()
    games = games.reset_index(drop=True)
    rec = games[["game_id", "player_id"]].merge(recorded, on=["game_id", "player_id"], how="left")
    is_recorded = rec["xgboost"].notna().to_numpy()
    preds = current.copy()
    preds.loc[is_recorded] = rec.loc[is_recorded, current.columns].to_numpy()
    keep = is_recorded | (games["game_date"] > pd.Timestamp(bundle["train_end"])).to_numpy()
    return games.join(preds.add_prefix("p_"))[keep].reset_index(drop=True)


def new_graded_rows(bundle, already: set) -> pd.DataFrame:
    """Graded games not in the file yet, with each tracked model's pick and result."""
    games, _ = load_graded_games(bundle)
    keys = list(zip(games["game_id"], games["player_id"]))
    games = games[[k not in already for k in keys]].reset_index(drop=True)
    if games.empty:
        return games

    games = with_predictions(games, bundle, recorded_predictions())
    if games.empty:
        return games
    preds = {m: games[f"p_{m}"].to_numpy(dtype=float) for m in ALL_MODELS}
    teams = dict(pd.read_sql(text("SELECT team_id, team_name FROM teams"), engine).values)

    rows = []
    for i, g in games.iterrows():
        row = {
            "season": int(g["season"]), "week": int(g["week"]),
            "game_date": g["game_date"].date().isoformat(),
            "game_id": int(g["game_id"]), "player_id": int(g["player_id"]),
            "qb": g["player_name"], "team": teams.get(g["team_id"]),
            "opponent": teams.get(g["opponent_team_id"]),
            "sportsbook": g["sportsbook"], "line": float(g["line"]),
            "over_odds": float(g["over_odds"]), "under_odds": float(g["under_odds"]),
            "books": int(g["books"]), "actual": float(g["passing_yards"]),
        }
        for name in TRACKED:
            pred = float(preds[name][i])
            pick, result, units = grade(pred, row["actual"], row["line"], row["over_odds"], row["under_odds"])
            row.update({f"{name}_pred": round(pred, 1), f"{name}_pick": pick,
                        f"{name}_result": result, f"{name}_units": units})
        # Confident pick = XGBoost's pick, but only when the rule says so
        row["confident_pick"] = is_confident(
            {name: float(preds[name][i]) for name in CONFIDENT_RULE["models"]},
            row["line"], row["game_date"],
        )
        row["model_trained_through"] = pd.Timestamp(g["p_model_trained_through"]).date().isoformat()
        row["model_features"] = len(bundle["features"])   # 17 = stats only, 20 = stats + market
        row["graded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        rows.append(row)
    return pd.DataFrame(rows)


def record(results: pd.Series, units: pd.Series) -> dict:
    wins, losses = (results == "win").sum(), (results == "loss").sum()
    decided = wins + losses
    return {"wl": f"{wins}-{losses}", "pct": wins / decided if decided else np.nan, "units": units.sum()}


def summary_markdown(picks: pd.DataFrame) -> str:
    out = ["# Weekly results", "",
           f"_Updated {datetime.now():%Y-%m-%d %H:%M}. Generated by `python -m scripts.weekly_report` "
           f"from `reports/graded_picks.csv`; don't edit by hand._", "",
           "Flat 1-unit bets on the side each model favors vs. each book's last pregame line, "
           f"at the book's real price. About {BREAK_EVEN:.1%} wins are needed to break even at -110. "
           "**Avg miss** is how far predictions landed from the actual yards; the sportsbook "
           "line's own miss is the bar to beat.", ""]

    header = ("| Season | Week | Dates | Games | XGBoost W-L | Win % | Units | Ensemble W-L | Win % | Units "
              "| Avg miss: XGBoost | Avg miss: line |")
    out += [header, "|" + "---|" * (header.count("|") - 1)]

    def line_for(label_season, label_week, df, dates):
        x = record(df["xgboost_result"], df["xgboost_units"])
        e = record(df["ensemble_result"], df["ensemble_units"])
        miss_model = (df["xgboost_pred"] - df["actual"]).abs().mean()
        miss_line = (df["line"] - df["actual"]).abs().mean()
        pct = lambda v: "-" if np.isnan(v) else f"{v:.1%}"
        return (f"| {label_season} | {label_week} | {dates} | {len(df)} | {x['wl']} | {pct(x['pct'])} | "
                f"{x['units']:+.2f} | {e['wl']} | {pct(e['pct'])} | {e['units']:+.2f} | "
                f"{miss_model:.1f} | {miss_line:.1f} |")

    for (season, week), df in picks.groupby(["season", "week"], sort=True):
        dates = f"{df['game_date'].min()} to {df['game_date'].max()}"
        out.append(line_for(season, week, df, dates))
    for season, df in picks.groupby("season", sort=True):
        out.append(line_for(f"**{season} total**", "", df, ""))

    out += confident_section(picks)
    out += live_clv_section(load_live_clv())

    out += ["", "Every graded pick, with the line, prices and model version, is in `reports/graded_picks.csv`.",
            "Small samples are noisy: a few hundred bets are needed before a win rate means much.", ""]
    return "\n".join(out)


def confident_section(picks: pd.DataFrame) -> list:
    """Week-by-week record of the pre-registered confident-picks strategy."""
    rule = CONFIDENT_RULE
    out = ["", "## Confident picks (pre-registered)", "",
           f"Rule registered {rule['registered']}, before any of these games were played: bet XGBoost's "
           f"side only when it's at least {rule['min_edge']:g} yards from the line **and** all four models "
           "(linear regression, random forest, XGBoost, neural net) pick the same side. Only games "
           f"played after {rule['registered']} count. Compare against \"bet every game\" above.", ""]

    flag = picks.get("confident_pick")
    eligible = picks[flag.notna()] if flag is not None else picks.iloc[0:0]
    if eligible.empty:
        return out + [f"_No eligible games graded yet; tracking starts with games after {rule['registered']}._"]

    chosen = eligible[eligible["confident_pick"].astype(str) == "True"]
    header = "| Season | Week | Eligible games | Confident picks | W-L | Win % | Units |"
    out += [header, "|" + "---|" * (header.count("|") - 1)]
    pct = lambda v: "-" if np.isnan(v) else f"{v:.1%}"

    def row(label_season, label_week, elig, ch):
        r = record(ch["xgboost_result"], ch["xgboost_units"])
        return (f"| {label_season} | {label_week} | {len(elig)} | {len(ch)} | {r['wl']} | "
                f"{pct(r['pct'])} | {r['units']:+.2f} |")

    for (season, week), elig in eligible.groupby(["season", "week"], sort=True):
        out.append(row(season, week, elig, chosen[(chosen["season"] == season) & (chosen["week"] == week)]))
    for season, elig in eligible.groupby("season", sort=True):
        out.append(row(f"**{season} total**", "", elig, chosen[chosen["season"] == season]))
    return out


def live_clv(picks: pd.DataFrame, closing: pd.DataFrame) -> pd.DataFrame:
    """Confident picks with a closing snapshot: closing consensus (median across
    books, each book's last snapshot) and CLV in yards toward the pick."""
    if picks.empty or closing.empty:
        return picks.iloc[0:0].assign(close_line=[], clv=[])
    last = closing[closing["fetched_at"] == closing.groupby(
        ["game_id", "player_id", "sportsbook"])["fetched_at"].transform("max")]
    close = last.groupby(["game_id", "player_id"])["line"].median().rename("close_line").reset_index()
    d = picks.merge(close, on=["game_id", "player_id"])
    sign = np.where(d["side"] == "OVER", 1.0, -1.0)
    return d.assign(clv=sign * (d["close_line"] - d["line"]))


def load_live_clv() -> dict:
    """Recorded confident picks (whose games have kicked off) and their CLV."""
    try:
        picks = pd.read_sql(text("""
            SELECT lp.game_id, lp.player_id, lp.line::float AS line, lp.side, g.season, g.week
            FROM live_picks lp JOIN games g USING (game_id)
            WHERE lp.confident AND lp.kickoff < now()
        """), engine)
        closing = pd.read_sql(text("""
            SELECT game_id, player_id, sportsbook, line::float AS line, fetched_at
            FROM qb_prop_closing_lines WHERE player_id IS NOT NULL
        """), engine)
    except Exception as exc:   # tables missing (older database): skip the section
        print(f"[report] live CLV skipped ({type(exc).__name__})")
        return {"picks": pd.DataFrame(), "clv": pd.DataFrame()}
    return {"picks": picks, "clv": live_clv(picks, closing)}


def live_clv_section(data: dict) -> list:
    """Week-by-week live CLV of the confident picks."""
    picks, clv = data["picks"], data["clv"]
    out = ["", "## Live closing-line value (confident picks)", "",
           "How far each confident pick's line (Friday consensus, when the pick was recorded) moved "
           "toward our side by a snapshot shortly before kickoff, in yards: positive means the market "
           "agreed with us after we'd have bet. In the 2024-26 backtest confident picks averaged "
           "+2.5 yds. Wins/losses are still graded at the Friday line above.", ""]
    if picks.empty:
        return out + ["_No recorded confident picks have kicked off yet._"]
    header = "| Season | Week | Confident picks | With closing snapshot | Mean CLV (yds) | Moved our way | Moved against |"
    out += [header, "|" + "---|" * (header.count("|") - 1)]

    def row(label_season, label_week, p, c):
        if c.empty:
            return f"| {label_season} | {label_week} | {len(p)} | 0 | - | - | - |"
        return (f"| {label_season} | {label_week} | {len(p)} | {len(c)} | {c['clv'].mean():+.1f} | "
                f"{np.mean(c['clv'] > 0):.0%} | {np.mean(c['clv'] < 0):.0%} |")

    for (season, week), p in picks.groupby(["season", "week"], sort=True):
        out.append(row(season, week, p, clv[(clv["season"] == season) & (clv["week"] == week)]))
    for season, p in picks.groupby("season", sort=True):
        out.append(row(f"**{season} total**", "", p, clv[clv["season"] == season]))
    return out


def main():
    bundle = load_models()
    if bundle is None:
        print(f"ERROR: {MODELS_PATH.relative_to(PROJECT_ROOT)} not found. Train first: python -m scripts.train")
        return 1

    REPORTS_DIR.mkdir(exist_ok=True)
    existing = pd.read_csv(PICKS_PATH) if PICKS_PATH.exists() else pd.DataFrame()
    already = set(zip(existing["game_id"], existing["player_id"])) if not existing.empty else set()

    new = new_graded_rows(bundle, already)
    picks = pd.concat([existing, new], ignore_index=True) if not new.empty else existing

    if picks.empty:
        print("[report] no graded games yet (need a stored pregame line and a final result)")
        return 0

    picks = picks.sort_values(["season", "week", "game_date", "qb"]).reset_index(drop=True)
    picks.to_csv(PICKS_PATH, index=False)
    SUMMARY_PATH.write_text(summary_markdown(picks))

    print(f"[report] added {len(new)} newly graded games ({len(picks)} total) -> "
          f"{PICKS_PATH.relative_to(PROJECT_ROOT)}, {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
