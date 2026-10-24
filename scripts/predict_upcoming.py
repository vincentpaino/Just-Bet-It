"""
Model predictions vs. sportsbook prop lines for upcoming games.

For every QB with a stored passing-yards line (qb_prop_lines) in a game that
hasn't kicked off, this prints:
  - the consensus line, the XGBoost prediction (the primary model), the
    ensemble (average of all four models) and the QB's last-5-game average,
  - which side the model leans and the best available price for that side
    across books (line shopping, Doc II Section 6.3),
  - the model's win probability (from its real held-out misses, see
    src/betting/probability.py), the market's no-vig probability for the same
    side, expected value per 1-unit bet, and a quarter-Kelly stake for bets
    that clear the minimum EV (src/betting).

These are NOT validated betting recommendations: the model hasn't yet been
shown to beat real lines (see scripts/backtest.py).

Run from the project root (after training: python -m scripts.train):
    python -m scripts.predict_upcoming
    python -m scripts.predict_upcoming --hours 48 --min-edge 15 --min-ev 0.05
    python -m scripts.predict_upcoming --bankroll 500 --kelly 0.5
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

from scripts.train import MODELS_PATH, load_models, predict_all
from scripts.xgboost_reg import PROJECT_ROOT, build_upcoming_features, load_qb_games
from src.betting import BetConfig, best_bet, cap_exposure, from_bundle
from src.db.session import engine


def format_kickoff(kickoff) -> str:
    """e.g. "Sat 2:30PM". Builds the hour by hand: "%-I" (no leading zero)
    works on macOS/Linux but raises ValueError on Windows."""
    t = pd.Timestamp(kickoff)
    return f"{t:%a} {t.hour % 12 or 12}:{t:%M%p}"


def load_upcoming_lines(hours: int) -> pd.DataFrame:
    """Each book's latest line (both sides priced) for QBs in games kicking off within `hours`."""
    return pd.read_sql(text("""
        WITH latest AS (
            SELECT game_id, player_id, sportsbook, MAX(fetched_at) AS fetched_at
            FROM qb_prop_lines WHERE player_id IS NOT NULL
            GROUP BY game_id, player_id, sportsbook
        )
        SELECT p.game_id, p.player_id, p.player_name, p.sportsbook,
               p.line::float AS line, p.over_odds::float AS over_odds, p.under_odds::float AS under_odds,
               g.game_date, g.season, g.week, g.home_team_id, g.away_team_id,
               ht.team_name AS home, at.team_name AS away,
               g.start_time AT TIME ZONE 'America/Chicago' AS kickoff
        FROM qb_prop_lines p
        JOIN latest l USING (game_id, player_id, sportsbook, fetched_at)
        JOIN games g ON g.game_id = p.game_id
        JOIN teams ht ON ht.team_id = g.home_team_id
        JOIN teams at ON at.team_id = g.away_team_id
        WHERE g.start_time > now()
          AND g.start_time <= now() + make_interval(hours => :hours)
          AND p.over_odds IS NOT NULL AND p.under_odds IS NOT NULL
    """), engine, params={"hours": hours})


def upcoming_qbs(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, QB), with the QB's team and opponent for this game."""
    qbs = lines.drop_duplicates(["game_id", "player_id"]).copy()

    # The QB's team is the one he most recently played for.
    latest_team = pd.read_sql(text("""
        SELECT DISTINCT ON (player_id) player_id, team_id
        FROM qb_game_stats ORDER BY player_id, game_date DESC
    """), engine)
    qbs = qbs.merge(latest_team, on="player_id", how="left")

    is_home = qbs["team_id"] == qbs["home_team_id"]
    qbs["is_home"] = is_home
    qbs["home_away"] = np.where(is_home, "home", "away")
    qbs["opponent_team_id"] = np.where(is_home, qbs["away_team_id"], qbs["home_team_id"])
    return qbs


def main():
    parser = argparse.ArgumentParser(description="Model predictions vs. stored prop lines for upcoming games")
    parser.add_argument("--hours", type=int, default=72, help="Only games kicking off within this many hours")
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="Only show QBs where |model - line| is at least this many yards")
    parser.add_argument("--min-ev", type=float, default=BetConfig.min_ev,
                        help="Only size a bet if its expected profit per unit is at least this")
    parser.add_argument("--kelly", type=float, default=BetConfig.kelly_multiplier,
                        help="Fraction of full Kelly to bet (0.25 = quarter Kelly)")
    parser.add_argument("--bankroll", type=float, default=100.0,
                        help="Bankroll in units, to show stakes in units")
    args = parser.parse_args()
    config = BetConfig(min_ev=args.min_ev, min_edge_yards=args.min_edge, kelly_multiplier=args.kelly)

    bundle = load_models()
    if bundle is None:
        print(f"ERROR: {MODELS_PATH.relative_to(PROJECT_ROOT)} not found. Train first: python -m scripts.train")
        return 1

    lines = load_upcoming_lines(args.hours)
    if lines.empty:
        print(f"No stored prop lines for games kicking off in the next {args.hours}h. "
              f"Lines are captured by the weekly job (scripts/snapshot_odds.sh).")
        return 0

    qbs = upcoming_qbs(lines)
    prob_model = from_bundle(bundle, "xgboost")

    # Predictions from every model, using the same feature code they were trained with
    features = build_upcoming_features(
        load_qb_games(),
        qbs[["game_id", "player_id", "player_name", "team_id", "opponent_team_id",
             "game_date", "season", "week", "home_away"]],
    )
    preds = predict_all(bundle["models"], features[bundle["features"]])
    features["xgboost"] = preds["xgboost"]
    features["ensemble"] = preds["ensemble"]
    qbs = qbs.merge(features[["game_id", "player_id", "xgboost", "ensemble", "avg_pass_yds_last5"]],
                    on=["game_id", "player_id"], how="left")

    rows = []
    for q in qbs.itertuples():
        books = lines[(lines["game_id"] == q.game_id) & (lines["player_id"] == q.player_id)]
        line = books["line"].median()
        bet = best_bet(books, q.xgboost, prob_model, config) if pd.notna(q.xgboost) else None
        rows.append({
            "Kickoff (CT)": format_kickoff(q.kickoff),
            "Game": f"{q.away} @ {q.home}",
            "QB": q.player_name,
            "Books": len(books),
            "Line": line,
            "Model": q.xgboost,
            "Ensemble": q.ensemble,
            "Last 5 avg": q.avg_pass_yds_last5,
            "Edge": q.xgboost - line,
            "Lean": bet["side"] if bet else "-",
            "Best price": f"{bet['book']} {bet['line']:g} ({bet['price']:+.0f})" if bet else "-",
            "Model win %": bet["p_win"] * 100 if bet else np.nan,
            "Market %": bet["market_p_win"] * 100 if bet else np.nan,
            "EV/unit": bet["ev"] if bet else np.nan,
            "Stake": bet["stake"] * args.bankroll if bet else 0.0,
        })

    table = pd.DataFrame(rows)
    # These bets are placed together, so cap their combined stake like one day's slate
    table["Stake"] = cap_exposure(table["Stake"] / args.bankroll, config.max_daily_exposure) * args.bankroll
    no_model = table["Model"].isna()
    shown = table[~no_model & (table["Edge"].abs() >= args.min_edge)].sort_values("EV/unit", ascending=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"Model (XGBoost) vs. sportsbook lines, games in the next {args.hours}h")
    print(f"Win %: {prob_model.method}. Stakes: {args.kelly:g}x Kelly on a {args.bankroll:g}-unit "
          f"bankroll, bets with EV >= {args.min_ev:+.2f} only (max {config.max_stake:.0%} each, "
          f"{config.max_daily_exposure:.0%} in total)\n")
    print(shown.to_string(index=False, formatters={
        "Line": "{:.1f}".format, "Model": "{:.0f}".format, "Last 5 avg": "{:.0f}".format,
        "Ensemble": lambda v: "-" if pd.isna(v) else f"{v:.0f}",
        "Edge": "{:+.0f}".format, "Model win %": "{:.0f}%".format, "Market %": "{:.0f}%".format,
        "EV/unit": "{:+.2f}".format, "Stake": lambda v: f"{v:.2f}" if v > 0 else "-",
    }))
    sized = shown[shown["Stake"] > 0]
    print(f"\n{len(sized)} bets clear the filters; total stake {sized['Stake'].sum():.2f} units "
          f"({sized['Stake'].sum() / args.bankroll:.1%} of bankroll)")

    if no_model.any():
        print(f"\nNot enough history to predict: {', '.join(table.loc[no_model, 'QB'])}")

    print("\nNOT betting advice: these predictions haven't been validated against real results yet")
    print("(see scripts/backtest.py). Market % is the books' no-vig probability for the same side;")
    print("a model far more confident than the market is more often wrong than right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
