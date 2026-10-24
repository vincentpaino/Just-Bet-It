"""
QB Passing Yards Model Backtesting Script.
Scores the trained models against real sportsbook lines stored in
qb_prop_lines (captured by `run_ingestion --props`) and simulates flat
1-unit over/under bets at the books' actual prices.

Only games with a stored pregame line, a final CFBD stat line, and a date
after the models' training cutoff are scored, so the sample grows as the
weekly snapshots accumulate. Features come from the same code the models
were trained with (scripts/xgboost_reg.py).

It runs two simulations:
  1. flat 1-unit bets on every game where the model and line differ by more
     than --edge-threshold yards,
  2. a Kelly bankroll (src/betting): only bets with expected value of at least
     --min-ev, sized at --kelly times full Kelly from the model's win
     probability, replayed in date order.

Run from the project root (after python -m scripts.train):
    python -m scripts.backtest
    python -m scripts.backtest --edge-threshold 15
    python -m scripts.backtest --min-ev 0.05 --kelly 0.25 --bankroll 100
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

from scripts.train import MODELS_PATH, load_models, predict_all
from scripts.xgboost_reg import PROJECT_ROOT, build_features, has_history, load_qb_games
from src.betting import BetConfig, from_bundle, simulate_bankroll
from src.db.session import SessionLocal


def load_closing_prop_lines(session):
    """
    One closing passing-yards line per (game, QB), with real prices:
      1. each book's last snapshot taken before kickoff,
      2. that book's main line (books also post alternates): the one whose
         over and under are priced closest to even,
      3. the book whose main line is nearest the consensus (median) line.
    Only lines with both sides priced and a known CFBD player ID are used.
    """
    df = pd.read_sql(
        text("""
            SELECT p.game_id, p.player_id, p.sportsbook, p.line,
                   p.over_odds, p.under_odds, p.fetched_at
            FROM qb_prop_lines p
            JOIN games g USING (game_id)
            WHERE p.player_id IS NOT NULL
              AND p.over_odds IS NOT NULL AND p.under_odds IS NOT NULL
              AND g.start_time IS NOT NULL
              AND p.fetched_at < (g.start_time AT TIME ZONE 'UTC')
        """),
        session.connection(),
    )

    if df.empty:
        return df

    for col in ['line', 'over_odds', 'under_odds']:
        df[col] = pd.to_numeric(df[col])

    keys = ['game_id', 'player_id']

    # 1. Last pregame snapshot per book
    last_fetch = df.groupby(keys + ['sportsbook'])['fetched_at'].transform('max')
    df = df[df['fetched_at'] == last_fetch].copy()

    # 2. Main line per book: smallest gap between over and under implied odds
    def implied(price):
        return np.where(price > 0, 100 / (price + 100), -price / (-price + 100))

    df['price_gap'] = np.abs(implied(df['over_odds']) - implied(df['under_odds']))
    df = df.sort_values('price_gap').drop_duplicates(keys + ['sportsbook'])

    # 3. Book closest to the consensus line
    df['consensus'] = df.groupby(keys)['line'].transform('median')
    df['books'] = df.groupby(keys)['sportsbook'].transform('count')
    df['distance'] = (df['line'] - df['consensus']).abs()
    df = df.sort_values(['distance', 'sportsbook']).drop_duplicates(keys)

    return df[keys + ['sportsbook', 'line', 'over_odds', 'under_odds', 'consensus', 'books']]


def american_payout(price):
    """Profit on a winning 1-unit stake at these American odds."""
    price = np.asarray(price, dtype=float)
    return np.where(price > 0, price / 100, 100 / np.abs(price))


def simulate_betting_strategy(predictions, bets, edge_threshold=0.0):
    """
    Flat 1-unit bets against real closing lines: OVER when the prediction is
    more than `edge_threshold` yards above the line, UNDER when it's more than
    that below. Winners are paid at the book's actual price; a push returns
    the stake.

    Args:
        predictions: Array of predicted passing yards
        bets: DataFrame with passing_yards (actual), line, over_odds, under_odds
        edge_threshold: Minimum |prediction - line| in yards to place a bet

    Returns:
        Dictionary with betting performance metrics
    """
    predictions = np.asarray(predictions, dtype=float)
    actual = bets['passing_yards'].to_numpy(dtype=float)
    lines = bets['line'].to_numpy(dtype=float)

    edges = predictions - lines
    bet_over = edges > edge_threshold
    bet_under = edges < -edge_threshold
    placed = bet_over | bet_under

    if not placed.any():
        return {
            'total_bets': 0, 'overs': 0, 'unders': 0, 'winning_bets': 0,
            'win_rate': 0.0, 'total_profit': 0.0, 'roi': 0.0, 'avg_edge': 0.0,
        }

    over_win = bet_over & (actual > lines)
    under_win = bet_under & (actual < lines)
    lost = (bet_over & (actual < lines)) | (bet_under & (actual > lines))

    profits = np.zeros(len(predictions))
    profits[over_win] = american_payout(bets['over_odds'].to_numpy()[over_win])
    profits[under_win] = american_payout(bets['under_odds'].to_numpy()[under_win])
    profits[lost] = -1.0

    total_bets = int(placed.sum())
    winning_bets = int((over_win | under_win).sum())
    decided = winning_bets + int(lost.sum())  # pushes excluded
    total_profit = float(profits.sum())

    return {
        'total_bets': total_bets,
        'overs': int(bet_over.sum()),
        'unders': int(bet_under.sum()),
        'winning_bets': winning_bets,
        'win_rate': winning_bets / decided if decided else 0.0,
        'total_profit': total_profit,
        'roi': total_profit / total_bets,
        'avg_edge': float(np.abs(edges[placed]).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest QB passing-yards models against stored pregame lines")
    parser.add_argument("--edge-threshold", type=float, default=0.0,
                        help="Minimum |prediction - line| in yards to place a bet")
    parser.add_argument("--min-ev", type=float, default=BetConfig.min_ev,
                        help="Kelly simulation: minimum expected profit per unit to bet")
    parser.add_argument("--kelly", type=float, default=BetConfig.kelly_multiplier,
                        help="Kelly simulation: fraction of full Kelly to bet (0.25 = quarter Kelly)")
    parser.add_argument("--bankroll", type=float, default=100.0,
                        help="Kelly simulation: starting bankroll in units")
    args = parser.parse_args()

    print("QB Passing Yards Model Backtesting")
    print("=" * 50)

    bundle = load_models()
    if bundle is None:
        print(f"ERROR: {MODELS_PATH.relative_to(PROJECT_ROOT)} not found. Train first: python -m scripts.train")
        return 1
    train_end = pd.Timestamp(bundle["train_end"])
    print(f"Loaded {len(bundle['models'])} models (trained through {train_end.date()})")

    session = SessionLocal()
    try:
        lines = load_closing_prop_lines(session)
    finally:
        session.close()

    # Pregame features for every played QB game (same code as training)
    qb = build_features(load_qb_games())
    qb = qb[has_history(qb)]

    bets = (
        qb.merge(lines, on=["game_id", "player_id"], how="inner")
        if not lines.empty else qb.iloc[0:0]
    )

    if bets.empty:
        print("\nNo QB games have both a stored pregame prop line and a final stat line yet.")
        print("Lines are captured by the weekly job (scripts/snapshot_odds.sh); results")
        print("appear once CFBD QB stats for those games are ingested.")
        return 0

    # Never score games the models trained on
    after = bets["game_date"] > train_end
    print(f"\n{len(bets)} QB games with a pregame line and a result; "
          f"{after.sum()} are after the training cutoff")
    bets = bets[after].reset_index(drop=True)

    if bets.empty:
        print("None are after the training cutoff yet; more weeks of lines will fix this.")
        return 0

    print(f"Scoring {len(bets)} games ({bets['game_date'].min().date()} to "
          f"{bets['game_date'].max().date()}, avg {bets['books'].mean():.1f} books per line)")
    if len(bets) < 200:
        print("NOTE: small sample; win rate and ROI are noisy until more weeks accumulate.")

    y = bets["passing_yards"].to_numpy(dtype=float)
    all_predictions = predict_all(bundle["models"], bets[bundle["features"]])

    # Accuracy, with the sportsbook line itself as the bar to beat
    print("\nAccuracy vs actual passing yards:")
    print("-" * 50)
    rows = list(all_predictions.items()) + [("sportsbook line", bets["line"].to_numpy(dtype=float))]
    for name, pred in rows:
        mae = np.mean(np.abs(y - pred))
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        print(f"{name:20} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f}")

    # Betting simulation
    print("\nBetting simulation (flat 1 unit, real pregame prices):")
    print("-" * 50)
    print(f"Minimum edge threshold: {args.edge_threshold} yards")

    for name, pred in all_predictions.items():
        results = simulate_betting_strategy(pred, bets, args.edge_threshold)

        print(f"\n{name.upper()}:")
        print(f"  Bets placed: {results['total_bets']} "
              f"({results['overs']} over / {results['unders']} under)")
        print(f"  Winning bets: {results['winning_bets']}")
        print(f"  Win rate: {results['win_rate']:.1%}  (about 52.4% breaks even at -110)")
        print(f"  Total profit: {results['total_profit']:+.2f} units")
        print(f"  ROI: {results['roi']:.1%}")
        print(f"  Average edge: {results['avg_edge']:.2f} yards")

    # Kelly bankroll simulation
    config = BetConfig(min_ev=args.min_ev, min_edge_yards=args.edge_threshold,
                       kelly_multiplier=args.kelly)
    print("\nKelly bankroll simulation (src/betting):")
    print("-" * 50)
    print(f"{args.kelly:g}x Kelly, bets with EV >= {args.min_ev:+.2f}, max {config.max_stake:.0%} "
          f"per bet and {config.max_daily_exposure:.0%} per day, starting at {args.bankroll:g} units")
    if "residuals" not in bundle:
        print("NOTE: this models file has no saved residuals, so win chances use a normal curve.")
        print("      Retrain (python -m scripts.train) to use the models' real misses.")
    for name, pred in all_predictions.items():
        bets[f"pred_{name}"] = pred
        prob_model = from_bundle(bundle, name)
        _, k = simulate_bankroll(bets, f"pred_{name}", prob_model, config, start=args.bankroll)
        print(f"\n{name.upper()}:  ({prob_model.method})")
        if k["bets"] == 0:
            print("  No bets cleared the EV filter.")
            continue
        print(f"  Bets placed: {k['bets']}  ({k['wins']} won / {k['losses']} lost / {k['pushes']} pushed, "
              f"win rate {k['win_rate']:.1%})")
        print(f"  Staked: {k['staked']:.2f} units | profit {k['profit']:+.2f} | ROI on stakes {k['roi']:.1%}")
        print(f"  Bankroll: {k['start']:g} -> {k['final']:.2f} ({k['growth']:+.1%}), "
              f"worst drawdown {k['max_drawdown']:.1%}")

    print("\n" + "=" * 50)
    print("BACKTESTING COMPLETE")
    print("=" * 50)
    print("Notes:")
    print("- Lines are each book's last snapshot before kickoff, so how close")
    print("  they are to the true closing line depends on snapshot timing")
    print("- Only games after the models' training cutoff are scored")
    print("- The models only cover starters, so games where the QB threw fewer")
    print("  than 10 passes (e.g. injured early) aren't scored")
    print("- Win chances come from the models' misses on the training test set,")
    print("  which can overlap these games; treat Kelly results as optimistic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
