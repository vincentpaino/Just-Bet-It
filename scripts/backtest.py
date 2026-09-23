"""
QB Passing Yards Model Backtesting Script.
Evaluates model performance on historical data and simulates betting strategy.
"""
import argparse
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.db.models import QBGameStat, Game, Team, Odds
from src.db.session import SessionLocal, init_db
from sqlalchemy import and_, or_


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

    # Sort by QB and date for rolling calculations
    df = df.sort_values(['qb_id', 'game_date']).reset_index(drop=True)

    # QB-specific rolling features (last 3 games)
    df['qb_passing_yards_3game_avg'] = df.groupby('qb_id')['passing_yards'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['qb_completion_pct_3game_avg'] = df.groupby('qb_id')['completion_pct'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['qb_yards_per_attempt_3game_avg'] = df.groupby('qb_id')['yards_per_attempt'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['qb_td_rate_3game_avg'] = df.groupby('qb_id').apply(
        lambda x: (x['passing_tds'] / x['attempts'].replace(0, np.nan)).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df['qb_int_rate_3game_avg'] = df.groupby('qb_id').apply(
        lambda x: (x['passing_ints'] / x['attempts'].replace(0, np.nan)).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

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
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0)

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


def get_historical_qb_data(session, start_year=2015, end_year=2022):
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
        }
        for r in results
    ])

    return df


def simulate_betting_strategy(predictions, actual_yards, lines, kelly_fraction=0.5):
    """
    Simulate betting strategy using Kelly Criterion.

    Args:
        predictions: Array of predicted passing yards
        actual_yards: Array of actual passing yards
        lines: Array of betting lines (over/under)
        kelly_fraction: Fraction of Kelly to bet (0.5 = half-Kelly)

    Returns:
        Dictionary with betting performance metrics
    """
    predictions = np.array(predictions)
    actual_yards = np.array(actual_yards)
    lines = np.array(lines)

    # Calculate edge (prediction - line)
    edges = predictions - lines

    # Only bet on positive edge bets (threshold can be adjusted)
    bet_threshold = 0.0  # Can be made configurable
    bet_mask = edges > bet_threshold

    if not np.any(bet_mask):
        return {
            'total_bets': 0,
            'winning_bets': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'roi': 0.0,
            'avg_edge': 0.0
        }

    # For simplicity, assume fixed bet size (in practice, use Kelly)
    # We'll simulate betting 1 unit on each positive edge bet
    bet_sizes = np.ones(np.sum(bet_mask))

    # Calculate wins/losses
    # Win if actual goes OVER the line when we predicted OVER (prediction > line)
    # Actually, simpler: we win if (actual - line) has same sign as (prediction - line)
    # Or even simpler for over/under: we bet OVER if prediction > line
    # We win if actual > line

    # For our simulation: we bet OVER when prediction > line
    # We win if actual > line
    predicted_over = predictions[bet_mask] > lines[bet_mask]
    actual_over = actual_yards[bet_mask] > lines[bet_mask]
    wins = (predicted_over == actual_over)

    # Simplified profit calculation: assume -110 odds (bet 1.1 to win 1.0)
    # In reality, odds vary by sportsbook
    bet_amount = 1.1  # Amount risked per bet
    win_amount = 1.0  # Amount won per winning bet

    profits = np.where(wins, win_amount, -bet_amount)
    total_profit = np.sum(profits)
    total_bet = np.sum(bet_sizes * bet_amount)

    # Calculate metrics
    total_bets = np.sum(bet_mask)
    winning_bets = np.sum(wins)
    win_rate = winning_bets / total_bets if total_bets > 0 else 0
    roi = total_profit / total_bet if total_bet > 0 else 0
    avg_edge = np.mean(edges[bet_mask]) if total_bets > 0 else 0

    return {
        'total_bets': int(total_bets),
        'winning_bets': int(winning_bets),
        'win_rate': float(win_rate),
        'total_profit': float(total_profit),
        'roi': float(roi),
        'avg_edge': float(avg_edge)
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest QB Passing Yards Models")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for backtesting")
    parser.add_argument("--end-year", type=int, default=2022, help="End year for backtesting")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument("--kelly-fraction", type=float, default=0.5, help="Kelly fraction to use (0.5 = half-Kelly)")
    parser.add_argument("--edge-threshold", type=float, default=0.0, help="Minimum edge to place bet")

    args = parser.parse_args()

    print("QB Passing Yards Model Backtesting")
    print("=" * 50)

    # Initialize database
    init_db()
    session = SessionLocal()

    try:
        # Load models
        print(f"Loading models from {args.model_dir}/...")
        models, feature_names = load_models(args.model_dir)

        if not models:
            print("ERROR: No models loaded. Please train models first using train.py")
            return 1

        print(f"Loaded {len(models)} models")
        if feature_names:
            print(f"Using {len(feature_names)} features")

        # Load historical data
        print(f"\nLoading historical data ({args.start_year}-{args.end_year})...")
        df = get_historical_qb_data(session, args.start_year, args.end_year)
        print(f"Loaded {len(df)} QB game logs")

        if len(df) == 0:
            print("ERROR: No data loaded for backtesting.")
            return 1

        # Engineer features
        print("\nEngineering features...")
        df_features = engineer_qb_features(df)
        print(f"Feature engineering complete. Shape: {df_features.shape}")

        # Prepare features and target
        print("\nPreparing features...")
        X, y, _ = prepare_features_for_backtest(df_features, feature_names)
        print(f"Features: {X.shape[1]}")

        # Time-based split (use all data for backtesting - no train/test split)
        # In practice, you'd want to walk-forward test, but for simplicity we'll use all data
        # and make predictions using models trained on previous data
        # For this demo, we'll just use all data and note this is in-sample testing

        print("\nMaking predictions...")

        # Make predictions with each model
        all_predictions = {}
        for name, model in models.items():
            try:
                print(f"  Predicting with {name}...")
                preds = model.predict(X)
                all_predictions[name] = np.maximum(0, preds)  # No negative passing yards
            except Exception as e:
                print(f"  Error predicting with {name}: {e}")
                all_predictions[name] = None

        # Create ensemble prediction
        valid_predictions = [pred for pred in all_predictions.values() if pred is not None]
        if valid_predictions:
            ensemble_pred = np.mean(valid_predictions, axis=0)
            all_predictions['ensemble'] = np.maximum(0, ensemble_pred)
        else:
            print("ERROR: No valid predictions generated")
            return 1

        # For backtesting simulation, we need betting lines
        # Since we don't have historical odds in our current schema, we'll simulate
        # In a real implementation, you'd join with the odds table
        print("\nSimulating betting lines...")
        # Create simulated lines based on actual yards + some noise
        # In reality, you'd pull actual lines from the odds API/historical odds
        np.random.seed(42)  # For reproducible results
        line_noise = np.random.normal(0, 5, len(y))  # +/- 5 yards typical line movement
        simulated_lines = y + line_noise

        # Ensure lines are reasonable (typically between 150-400 for QB passing yards)
        simulated_lines = np.clip(simulated_lines, 150, 400)

        print(f"Actual yards stats: Mean={y.mean():.1f}, Std={y.std():.1f}")
        print(f"Simulated lines stats: Mean={simulated_lines.mean():.1f}, Std={simulated_lines.std():.1f}")

        # Evaluate model accuracy
        print("\nModel Accuracy Metrics:")
        print("-" * 40)
        for name, pred in all_predictions.items():
            if pred is not None:
                mae = np.mean(np.abs(y - pred))
                rmse = np.sqrt(np.mean((y - pred) ** 2))
                # Calculate R²
                ss_res = np.sum((y - pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                print(f"{name:20} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R²: {r2:6.4f}")

        # Simulate betting strategy
        print("\nBetting Strategy Simulation:")
        print("-" * 40)
        print(f"Using Kelly fraction: {args.kelly_fraction}")
        print(f"Minimum edge threshold: {args.edge_threshold}")

        betting_results = {}
        for name, pred in all_predictions.items():
            if pred is not None:
                results = simulate_betting_strategy(
                    pred,
                    y,
                    simulated_lines,
                    kelly_fraction=args.kelly_fraction
                )
                betting_results[name] = results

                print(f"\n{name.upper()}:")
                print(f"  Bets placed: {results['total_bets']}")
                print(f"  Winning bets: {results['winning_bets']}")
                print(f"  Win rate: {results['win_rate']:.1%}")
                print(f"  Total profit: {results['total_profit']:.2f}")
                print(f"  ROI: {results['roi']:.1%}")
                print(f"  Average edge: {results['avg_edge']:.2f} yards")

        # Summary
        print("\n" + "=" * 50)
        print("BACKTESTING COMPLETE")
        print("=" * 50)
        print("Notes:")
        print("- This backtest uses simulated betting lines")
        print("- For production backtesting, join with actual historical odds")
        print("- Model performance may be overfitted (in-sample testing)")
        print("- Consider walk-forward validation for more realistic estimates")
        print("- Feature engineering uses simplified rolling averages")
        print("- In production, ensure proper defensive stats are included")

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