"""
QB Passing Yards Prediction Script.
Loads trained models and makes predictions for upcoming games.
"""
import argparse
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.db.models import QBGameStat, Game, Team
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
        return {}, []


def engineer_qb_features_for_prediction(df):
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


def prepare_features_for_prediction(df, feature_names):
    """
    Prepare feature matrix for prediction using the same features as training.
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


def get_recent_qb_stats(session, qb_id, games_back=5):
    """
    Get recent game stats for a QB to use for feature engineering.
    """
    recent_games = (
        session.query(QBGameStat)
        .filter(QBGameStat.player_id == qb_id)
        .order_by(QBGameStat.game_date.desc())
        .limit(games_back)
        .all()
    )

    return recent_games


def predict_qb_passing_yards(session, models, feature_names, qb_id, opponent_team_id, game_date, is_home=True):
    """
    Predict passing yards for a specific QB in an upcoming game.
    """
    # Get the QB's recent stats to build features
    recent_stats = get_recent_qb_stats(session, qb_id, games_back=5)

    if not recent_stats:
        print(f"No recent stats found for QB ID {qb_id}")
        return None

    # Use the most recent game as base, then adjust for upcoming game
    latest_game = recent_stats[0]

    # Create a feature dictionary for the upcoming game
    # In a real implementation, you'd want to properly calculate rolling averages
    # based on the actual recent games
    features = {
        'qb_passing_yards_3game_avg': np.mean([g.passing_yards for g in recent_stats if g.passing_yards is not None]) if any(g.passing_yards for g in recent_stats) else 0,
        'qb_completion_pct_3game_avg': np.mean([g.completion_pct for g in recent_stats if g.completion_pct is not None]) if any(g.completion_pct for g in recent_stats) else 0,
        'qb_yards_per_attempt_3game_avg': np.mean([g.yards_per_attempt for g in recent_stats if g.yards_per_attempt is not None]) if any(g.yards_per_attempt for g in recent_stats) else 0,
        'qb_td_rate_3game_avg': np.mean([(g.passing_tds or 0) / max(g.attempts or 1, 1) for g in recent_stats]) if recent_stats else 0,
        'qb_int_rate_3game_avg': np.mean([(g.passing_ints or 0) / max(g.attempts or 1, 1) for g in recent_stats]) if recent_stats else 0,

        'is_home_game': 1 if is_home else 0,
        'days_since_last_game': (game_date - latest_game.game_date).days if latest_game.game_date else 7,
        'week_normalized': 0.5,  # Placeholder - would need actual week
        'is_early_season': 0,
        'is_mid_season': 1,
        'is_late_season': 0,

        'rolling_win_pct_home': 0.5,  # Placeholder
        'rolling_win_pct_away': 0.5,  # Placeholder
        'elo_home': 1500,             # Placeholder
        'elo_away': 1500,             # Placeholder
        'rest_days_home': 7,          # Placeholder
        'rest_days_away': 7,          # Placeholder
        'home_away_split': 0.0,       # Placeholder
        'sos_home': 0.5,              # Placeholder
        'sos_away': 0.5,              # Placeholder

        'pass_yards_vs_opp_allowed': 0.0,   # Placeholder
        'yards_per_attempt_vs_opp_allowed': 0.0, # Placeholder

        'opp_pass_yards_allowed_avg': 240.0,  # League average
        'opp_sack_rate_avg': 6.5,             # League average
        'opp_pass_yards_per_attempt_allowed': 6.8, # League average
    }

    # Convert to DataFrame for feature engineering
    df = pd.DataFrame([features])

    # Apply feature engineering (though for single prediction, this is limited)
    df_features = engineer_qb_features_for_prediction(df)

    # Prepare features
    X = prepare_features_for_prediction(df_features, feature_names)

    # Make predictions with each model
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(X)[0]
            predictions[name] = max(0, pred)  # Passing yards can't be negative
        except Exception as e:
            print(f"Error predicting with {name}: {e}")
            predictions[name] = None

    # Create ensemble prediction (average of available predictions)
    valid_preds = [p for p in predictions.values() if p is not None]
    if valid_preds:
        ensemble_pred = np.mean(valid_preds)
        predictions['ensemble'] = max(0, ensemble_pred)
    else:
        predictions['ensemble'] = None

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Predict QB Passing Yards")
    parser.add_argument("--qb-id", type=int, required=True, help="CFBD Player ID of the QB")
    parser.add_argument("--opponent-team-id", type=int, required=True, help="Team ID of the opponent")
    parser.add_argument("--game-date", type=str, required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--is-home", action="store_true", help="Is the QB playing at home?")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument("--list-recent", action="store_true", help="List recent games for the QB")

    args = parser.parse_args()

    print("QB Passing Yards Prediction")
    print("-" * 30)

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

        if args.list_recent:
            # List recent games for the QB
            recent_stats = get_recent_qb_stats(session, args.qb_id, games_back=10)
            print(f"\nRecent games for QB ID {args.qb_id}:")
            print("-" * 80)
            for i, stat in enumerate(recent_stats):
                print(f"{i+1:2}. {stat.game_date} | {stat.passcing_yards:3.0f} yds | "
                      f"{stat.completions:2.0f}/{stat.attempts:2.0f} | "
                      f"{stat.passing_tds:1.0f} TD | {stat.passing_ints:1.0f} INT")
            return 0

        # Parse game date
        try:
            game_date = datetime.strptime(args.game_date, "%Y-%m-%d").date()
        except ValueError:
            print("ERROR: Game date must be in YYYY-MM-DD format")
            return 1

        # Make prediction
        print(f"\nMaking prediction for QB ID {args.qb_id}")
        print(f" vs Opponent Team ID {args.opponent_team_id}")
        print(f" on {game_date} ({'Home' if args.is_home else 'Away'})")
        print("-" * 50)

        predictions = predict_qb_passing_yards(
            session,
            models,
            feature_names,
            args.qb_id,
            args.opponent_team_id,
            game_date,
            args.is_home
        )

        if predictions is None:
            print("ERROR: Unable to make prediction")
            return 1

        # Display predictions
        print("PREDICTIONS:")
        for model_name, pred in predictions.items():
            if pred is not None:
                print(f"  {model_name:20}: {pred:6.1f} passing yards")
            else:
                print(f"  {model_name:20}: Failed")

        # Additional context
        print("\n" + "-" * 50)
        print("NOTE: This is a simplified prediction. For production use:")
        print("- Ensure rolling averages are properly calculated")
        print("- Include actual defensive opponent stats")
        print("- Use correct week/season context")
        print("- Consider weather, injuries, game script, etc.")

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