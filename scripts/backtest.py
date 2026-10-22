"""
QB Passing Yards Model Backtesting Script.
Scores the trained models against real sportsbook closing lines stored in
qb_prop_lines (captured by `run_ingestion --props`) and simulates flat
1-unit over/under bets at the books' actual prices.

Only games with both a stored pregame line and a final CFBD stat line are
scored, so the sample grows as the daily snapshots accumulate.
"""
import argparse
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.db.models import QBGameStat, Game, Team, Odds
from src.db.session import SessionLocal, init_db
from sqlalchemy import and_, or_, text


def load_models(model_dir='models'):
    """
    Load trained models and feature names from disk.
    """
    try:
        models = {}
        model_files = {
            'linear_regression': 'linear_regression_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'neural_network': 'neural_network_model.pkl'
        }

        for name, filename in model_files.items():
            model_path = f"{model_dir}/{filename}"
            try:
                models[name] = joblib.load(model_path)
                print(f"Loaded {name} model from {model_path}")
            except FileNotFoundError:
                print(f"Warning: {model_path} not found")
                continue

        # Load feature names
        feature_path = f"{model_dir}/feature_names.pkl"
        try:
            feature_names = joblib.load(feature_path)
            print(f"Loaded feature names from {feature_path}")
        except FileNotFoundError:
            print(f"Warning: {feature_path} not found")
            feature_names = []

        return models, feature_names

    except Exception as e:
        print(f"Error loading models: {e}")
        return [], []


def engineer_qb_features(df):
    """
    Engineer features for QB passing yards prediction (same as in training).
    """
    # Make a copy to avoid warnings
    df = df.copy()

    # The DB returns dates as datetime.date and NUMERIC columns as Decimal
    # (or all-None objects when the features table is empty); convert them
    # so .dt and numeric operations work.
    df['game_date'] = pd.to_datetime(df['game_date'])
    decimal_columns = [
        'completion_pct', 'yards_per_attempt',
        'rolling_win_pct_home', 'rolling_win_pct_away', 'elo_home', 'elo_away',
        'rest_days_home', 'rest_days_away', 'home_away_split', 'sos_home', 'sos_away',
    ]
    for col in decimal_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by QB and date for rolling calculations
    df = df.sort_values(['qb_id', 'game_date']).reset_index(drop=True)

    # QB-specific rolling features (previous 3 games). shift(1) drops the
    # current game, so a row's features never include its own stats (that
    # would leak the target). A QB's first game has no history and stays NaN.
    def prior_3game_avg(col):
        return df.groupby('qb_id')[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

    df['td_rate'] = df['passing_tds'] / df['attempts'].replace(0, np.nan)
    df['int_rate'] = df['passing_ints'] / df['attempts'].replace(0, np.nan)

    df['qb_passing_yards_3game_avg'] = prior_3game_avg('passing_yards')
    df['qb_completion_pct_3game_avg'] = prior_3game_avg('completion_pct')
    df['qb_yards_per_attempt_3game_avg'] = prior_3game_avg('yards_per_attempt')
    df['qb_td_rate_3game_avg'] = prior_3game_avg('td_rate')
    df['qb_int_rate_3game_avg'] = prior_3game_avg('int_rate')

    # Opposing defensive features (would need defensive stats table)
    # For now, we'll use placeholder - in production, join with defensive stats
    df['opp_pass_yards_allowed_avg'] = 240.0  # League average placeholder
    df['opp_sack_rate_avg'] = 6.5             # League average placeholder
    df['opp_pass_yards_per_attempt_allowed'] = 6.8  # League average placeholder

    # Matchup features
    df['pass_yards_vs_opp_allowed'] = df['qb_passing_yards_3game_avg'] - df['opp_pass_yards_allowed_avg']
    df['yards_per_attempt_vs_opp_allowed'] = df['qb_yards_per_attempt_3game_avg'] - df['opp_pass_yards_per_attempt_allowed']

    # Game context features
    df['days_since_last_game'] = df.groupby('qb_id')['game_date'].diff().dt.days.fillna(7)
    df['is_home_game'] = df['is_home']

    # Season progress feature (early/mid/late season)
    df['week_normalized'] = df['week'] / 17.0  # Normalize week 1-17 to 0-1
    df['is_early_season'] = (df['week'] <= 5).astype(int)
    df['is_mid_season'] = ((df['week'] > 5) & (df['week'] <= 12)).astype(int)
    df['is_late_season'] = (df['week'] > 12).astype(int)

    # Fill NaN values
    # Forward-fill within each QB only, so one QB's values never carry into
    # the next QB's rows; anything still missing (e.g. a first game) becomes 0.
    numeric_columns = df.select_dtypes(include=[np.number]).columns.drop('qb_id')
    df[numeric_columns] = df.groupby('qb_id')[numeric_columns].ffill().fillna(0)

    return df


def prepare_features_for_backtest(df, feature_names):
    """
    Prepare feature matrix for backtesting using the same features as training.
    """
    # Filter to only columns that exist in both df and feature_names
    available_features = [col for col in feature_names if col in df.columns]
    missing_features = [col for col in feature_names if col not in df.columns]

    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with default values
        for feat in missing_features:
            df[feat] = 0.0

    # Prepare X
    X = df[feature_names].copy()
    X = X.fillna(0)  # Fill any remaining NaN

    return X


def get_historical_qb_data(session, start_year, end_year):
    """
    Get historical QB game data for backtesting.
    """
    query = (
        session.query(
            QBGameStat,
            Game.game_date,
            Team.team_name.label('qb_team_name'),
            Team.conference.label('qb_conference'),
            Team.division.label('qb_division'),
        )
        .join(Game, QBGameStat.game_id == Game.game_id)
        .join(Team, QBGameStat.team_id == Team.team_id)
        .filter(QBGameStat.season >= start_year)
        .filter(QBGameStat.season <= end_year)
        .filter(QBGameStat.attempts > 0)  # Only QBs with passing attempts
        .order_by(QBGameStat.player_id, QBGameStat.game_date)
    )

    results = query.all()

    if not results:
        raise ValueError("No historical data found for backtesting.")

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            # Target variable
            'passing_yards': r.QBGameStat.passing_yards,

            # QB basic info
            'game_id': r.QBGameStat.game_id,
            'qb_id': r.QBGameStat.player_id,
            'qb_name': r.QBGameStat.player_name,
            'team_id': r.QBGameStat.team_id,
            'opponent_team_id': r.QBGameStat.opponent_team_id,
            'game_date': r.game_date,
            'season': r.QBGameStat.season,
            'week': r.QBGameStat.week,
            'home_away': r.QBGameStat.home_away,

            # QB stats from this game
            'passing_tds': r.QBGameStat.passing_tds,
            'passing_ints': r.QBGameStat.passing_ints,
            'completions': r.QBGameStat.completions,
            'attempts': r.QBGameStat.attempts,
            'completion_pct': r.QBGameStat.completion_pct,
            'yards_per_attempt': r.QBGameStat.yards_per_attempt,
            'rushing_yards': r.QBGameStat.rushing_yards,
            'rushing_tds': r.QBGameStat.rushing_tds,
            'rushing_attempts': r.QBGameStat.rushing_attempts,

            # Team info
            'qb_team_name': r.qb_team_name,
            'qb_conference': r.qb_conference,
            'qb_division': r.qb_division,

            # Derived features
            'is_home': 1 if r.QBGameStat.home_away == 'home' else 0,
        }
        for r in results
        if r.QBGameStat.passing_yards is not None
    ])

    return df


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


def load_training_end(model_dir='models'):
    """Last training game date saved by train.py (None for models trained before it saved one)."""
    try:
        return pd.Timestamp(joblib.load(f"{model_dir}/training_info.pkl")['train_end'])
    except FileNotFoundError:
        return None


def load_xgboost_reg(model_dir='models'):
    """The model saved by scripts/xgboost_reg.py, with its feature list and training cutoff."""
    path = f"{model_dir}/xgboost_reg.pkl"
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        print(f"Warning: {path} not found (run: python scripts/xgboost_reg.py)")
        return None
    print(f"Loaded xgboost_reg model from {path} "
          f"(trained through {pd.Timestamp(bundle['train_end']).date()})")
    return bundle


def predict_xgboost_reg(bundle, bets):
    """
    Predictions from scripts/xgboost_reg.py's model, with features built by
    that script's own code so they always match how it was trained.
    Returns (predictions, has_features); games where the QB lacks the prior
    history that script requires get NaN and has_features=False.
    """
    from scripts.xgboost_reg import build_features, has_history, load_qb_games

    qb = build_features(load_qb_games())
    qb = qb[has_history(qb)]

    features = bundle['features']
    merged = bets[['game_id', 'qb_id']].merge(
        qb[['game_id', 'player_id'] + features],
        left_on=['game_id', 'qb_id'],
        right_on=['game_id', 'player_id'],
        how='left',
    )

    has_features = merged['player_id'].notna().to_numpy()
    predictions = np.full(len(bets), np.nan)
    if has_features.any():
        predictions[has_features] = np.maximum(
            0, bundle['model'].predict(merged.loc[has_features, features])
        )
    return predictions, has_features


def main():
    parser = argparse.ArgumentParser(description="Backtest QB Passing Yards Models against stored closing lines")
    parser.add_argument("--start-year", type=int, default=2015,
                        help="First season of QB history to load (rolling features need prior games)")
    parser.add_argument("--end-year", type=int, default=datetime.now().year,
                        help="Last season to load")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument("--edge-threshold", type=float, default=0.0,
                        help="Minimum |prediction - line| in yards to place a bet")

    args = parser.parse_args()

    print("QB Passing Yards Model Backtesting")
    print("=" * 50)

    # Initialize database
    init_db()
    session = SessionLocal()

    try:
        # Load models: train.py's four (plus their ensemble) and xgboost_reg.py's
        print(f"Loading models from {args.model_dir}/...")
        models, feature_names = load_models(args.model_dir)
        xgb_reg = load_xgboost_reg(args.model_dir)

        if not models and xgb_reg is None:
            print("ERROR: No models loaded. Train first: python -m scripts.train "
                  "and/or python scripts/xgboost_reg.py")
            return 1

        # Load QB history and build features (history is needed even for
        # games without lines, because features are rolling averages)
        print(f"\nLoading QB history ({args.start_year}-{args.end_year})...")
        df = get_historical_qb_data(session, args.start_year, args.end_year)
        print(f"Loaded {len(df)} QB game logs")

        print("\nEngineering features...")
        df_features = engineer_qb_features(df)

        # Keep only QB games that have a stored closing line
        lines = load_closing_prop_lines(session)
        bets = (
            df_features.merge(
                lines,
                left_on=['game_id', 'qb_id'],
                right_on=['game_id', 'player_id'],
                how='inner',
            )
            if not lines.empty else df_features.iloc[0:0]
        )

        if bets.empty:
            print("\nNo QB games have both a stored pregame prop line and a final stat line yet.")
            print("Lines are captured by: python -m src.ingestion.run_ingestion --props")
            print("Results appear once CFBD QB stats for those games are ingested.")
            return 0

        print(f"\n{len(bets)} QB games with a closing line "
              f"({bets['game_date'].min().date()} to {bets['game_date'].max().date()})")

        # Each model may only score games after its last training game, and
        # all models are compared on the same games.
        print("\nMaking predictions...")
        all_predictions = {}
        eligible = np.ones(len(bets), dtype=bool)

        if models:
            X = prepare_features_for_backtest(bets, feature_names)
            for name, model in models.items():
                try:
                    all_predictions[name] = np.maximum(0, model.predict(X))  # No negative passing yards
                except Exception as e:
                    print(f"  Error predicting with {name}: {e}")
            if all_predictions:
                all_predictions['ensemble'] = np.mean(list(all_predictions.values()), axis=0)

            train_end = load_training_end(args.model_dir)
            if train_end is None:
                print("  WARNING: train.py models have no saved training cutoff; retrain them "
                      "(python -m scripts.train) so games they trained on can be excluded")
            else:
                after = (bets['game_date'] > train_end).to_numpy()
                print(f"  train.py models: trained through {train_end.date()}; "
                      f"{after.sum()} of {len(bets)} games are after that")
                eligible &= after

        if xgb_reg is not None:
            predictions, has_features = predict_xgboost_reg(xgb_reg, bets)
            all_predictions['xgboost_reg'] = predictions
            train_end = pd.Timestamp(xgb_reg['train_end'])
            after = (bets['game_date'] > train_end).to_numpy()
            print(f"  xgboost_reg: trained through {train_end.date()}; {after.sum()} games are after "
                  f"that; {has_features.sum()} have enough QB history (>= 10 attempts, 3 prior games)")
            eligible &= after & has_features

        if not all_predictions:
            print("ERROR: No valid predictions generated")
            return 1

        bets = bets[eligible].reset_index(drop=True)
        all_predictions = {name: pred[eligible] for name, pred in all_predictions.items()}

        if bets.empty:
            print("\nNo games are left that every model can score fairly (they're all inside a "
                  "model's training period or lack QB history). More weeks of lines will fix this.")
            return 0

        print(f"\nScoring {len(bets)} games (avg {bets['books'].mean():.1f} books per line)")
        if len(bets) < 200:
            print("NOTE: small sample; win rate and ROI are noisy until more weeks accumulate.")

        y = bets['passing_yards'].to_numpy(dtype=float)

        # Accuracy, with the sportsbook line itself as the bar to beat
        print("\nAccuracy vs actual passing yards:")
        print("-" * 50)
        rows = list(all_predictions.items()) + [('sportsbook line', bets['line'].to_numpy(dtype=float))]
        for name, pred in rows:
            mae = np.mean(np.abs(y - pred))
            rmse = np.sqrt(np.mean((y - pred) ** 2))
            print(f"{name:20} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f}")

        # Betting simulation
        print("\nBetting simulation (flat 1 unit, real closing prices):")
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

        print("\n" + "=" * 50)
        print("BACKTESTING COMPLETE")
        print("=" * 50)
        print("Notes:")
        print("- Lines are each book's last snapshot before kickoff, so how close")
        print("  they are to the true closing line depends on snapshot timing")
        print("- Only games after every model's training cutoff are scored")
        print("- xgboost_reg skips QBs with < 10 attempts in the game, so all models")
        print("  are scored only on games where the QB actually threw 10+ passes")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
