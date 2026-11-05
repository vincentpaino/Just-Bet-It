"""
Closing-line value (CLV): do our picks beat the line they'd actually be bet
at (the Friday 9am snapshot, like the live weekly job) by kickoff?

For every graded QB-game with both a Friday-9am snapshot and a closing
snapshot (about an hour before kickoff):
  - the pick is made against the Friday consensus line (median across
    books) with the walk-forward predictions (models trained only on games
    before that week; data/raw/walkforward_preds.csv from
    scripts/vs_closing_lines.py)
  - CLV = how far the consensus line moved toward the pick by close, in
    yards (over: close - Friday; under: Friday - close). Positive CLV
    means the market moved our way after we "bet", the standard sign of
    a real edge, and it shows up with far fewer bets than win/loss.

Pre-registered rule (2026-11-02, before fetching the Friday snapshots):
the confident picks (|XGBoost - Friday line| >= 15 yds and all four models
on the same side) pass if their mean CLV is above 0 with a one-sided
p < 0.05. Profit/loss betting at Friday prices is reported too.

Run from the project root (after scripts.vs_closing_lines has cached predictions):
    python -m scripts.clv
    python -m scripts.clv --preds data/raw/walkforward_preds_pre2024.csv
"""
import argparse
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import binom, ttest_1samp
from sqlalchemy import text

from scripts.backfill_props import friday_9am_before
from scripts.backtest import american_payout
from scripts.xgboost_reg import PROJECT_ROOT
from src.db.session import engine

PREDS = PROJECT_ROOT / "data" / "raw" / "walkforward_preds.csv"
MODELS = ["linear_regression", "random_forest", "xgboost", "neural_network"]
BREAK_EVEN = 0.524


def book_lines() -> pd.DataFrame:
    """Every stored book line for graded QB-games, with kickoff and result."""
    d = pd.read_sql(text("""
        SELECT p.game_id, p.player_id, p.sportsbook, p.line::float AS line,
               p.over_odds::float AS over_odds, p.under_odds::float AS under_odds,
               p.fetched_at, g.start_time, g.season, s.passing_yards::float AS actual
        FROM qb_prop_lines p
        JOIN games g USING (game_id)
        JOIN qb_game_stats s ON s.game_id = p.game_id AND s.player_id = p.player_id
        WHERE p.player_id IS NOT NULL AND s.passing_yards IS NOT NULL AND g.start_time IS NOT NULL
    """), engine)
    d["start_time"] = pd.to_datetime(d["start_time"], utc=True)
    d["fetched_at"] = pd.to_datetime(d["fetched_at"]).dt.tz_localize("UTC")
    return d


def snapshot(d: pd.DataFrame, start, end) -> pd.DataFrame:
    """Each book's last line within [start, end) per QB-game (start/end are Series)."""
    s = d[(d["fetched_at"] >= start) & (d["fetched_at"] < end)]
    last = s.groupby(["game_id", "player_id", "sportsbook"])["fetched_at"].transform("max")
    return s[s["fetched_at"] == last]


def main():
    parser = argparse.ArgumentParser(description="Closing-line value of the walk-forward picks")
    parser.add_argument("--preds", type=Path, default=PREDS, help="Walk-forward prediction cache")
    args = parser.parse_args()

    d = book_lines()
    friday = d["start_time"].map(friday_9am_before)
    fri = snapshot(d, friday - timedelta(hours=2), friday + timedelta(hours=2))
    close = snapshot(d, d["start_time"] - timedelta(hours=2), d["start_time"])

    keys = ["game_id", "player_id"]
    games = (fri.groupby(keys).agg(fri_line=("line", "median"), fri_books=("sportsbook", "nunique"),
                                   season=("season", "first"), actual=("actual", "first"))
             .join(close.groupby(keys).agg(close_line=("line", "median")), how="inner")
             .reset_index())
    preds = pd.read_csv(args.preds)
    games = games.merge(preds, on=keys, how="inner")
    print(f"{len(games):,} graded QB-games with a Friday-9am snapshot, a closing snapshot and an "
          f"out-of-sample prediction; by season: {games.groupby('season').size().to_dict()}")

    side = np.sign(games["xgboost"] - games["fri_line"])
    games = games[side != 0].copy()
    games["side"] = np.sign(games["xgboost"] - games["fri_line"])
    games["clv"] = games["side"] * (games["close_line"] - games["fri_line"])
    sides = np.sign(games[MODELS].to_numpy() - games[["fri_line"]].to_numpy())
    games["confident"] = ((games["xgboost"] - games["fri_line"]).abs() >= 15) & \
                         (np.abs(sides.sum(axis=1)) == len(MODELS))

    # P&L at Friday prices: the book whose Friday line is nearest the Friday consensus
    f = fri.merge(games[keys + ["fri_line", "side"]], on=keys)
    f = f[f["over_odds"].notna() & f["under_odds"].notna()]
    f = f.assign(dist=(f["line"] - f["fri_line"]).abs()).sort_values(["dist", "sportsbook"]).drop_duplicates(keys)
    games = games.merge(f[keys + ["line", "over_odds", "under_odds"]].rename(columns={"line": "bet_line"}), on=keys, how="left")

    def report(label, g):
        if g.empty:
            print(f"  {label:34} no bets"); return
        t = ttest_1samp(g["clv"], 0, alternative="greater")
        b = g.dropna(subset=["bet_line"])
        over = b["side"] > 0
        win = np.where(over, b["actual"] > b["bet_line"], b["actual"] < b["bet_line"])
        push = b["actual"] == b["bet_line"]
        price = np.where(over, b["over_odds"], b["under_odds"])
        units = np.where(push, 0, np.where(win, american_payout(price), -1.0)).sum()
        w, n = int(win[~push].sum()), int((~push).sum())
        print(f"  {label:34} {len(g):5} picks  CLV {g['clv'].mean():+5.2f} yds "
              f"(moved our way {np.mean(g['clv'] > 0):4.0%}, against {np.mean(g['clv'] < 0):4.0%})  "
              f"p = {t.pvalue:.3f} | at Friday prices {w}-{n - w} {w / n if n else 0:5.1%} {units:+7.2f} u "
              f"p = {binom.sf(w - 1, n, BREAK_EVEN):.3f}")

    print("\nCLV (rule: confident picks' mean CLV > 0, one-sided p < 0.05):")
    report("confident picks (pre-registered)", games[games["confident"]])
    report("all picks", games)
    print("\n  by season (confident picks):")
    for season in sorted(games["season"].unique()):
        report(f"  {season}", games[games["confident"] & (games["season"] == season)])
    print("\n  descriptive: market movement Friday -> close, all games: "
          f"mean |move| {(games['close_line'] - games['fri_line']).abs().mean():.1f} yds, "
          f"lines that moved at all {np.mean(games['close_line'] != games['fri_line']):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
