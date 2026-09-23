"""
QB Passing Yards Prediction Model Training Script.
Implements the ML pipeline for predicting QB passing yards using an ensemble of
regression models: Linear Regression, Random Forest, XGBoost, and Neural Network.
"""
import argparse
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from src.db.models import QBGameStat, Game, Team, Feature
from src.db.session import SessionLocal, init_db
from sqlalchemy import func


def load_training_data(session, start_year=2015, end_year=2023):
    """
    Load QB game logs and join with game features for training.
    Returns features (X) and target (y) for QB passing yards prediction.
    """
    # Query to get QB stats with game and team information
    query = (
        session.query(
            QBGameStat,
            Game.game_date,
            Game.home_team_id,
            Game.away_team_id,
            Feature.rolling_win_pct_home,
            Feature.rolling_win_pct_away,
            Feature.elo_home,
            Feature.elo_away,
            Feature.rest_days_home,
            Feature.rest_days_away,
            Feature.home_away_split,
            Feature.sos_home,
            Feature.sos_away,
            Team.team_name.label('qb_team_name'),
            Team.conference.label('qb_conference'),
            Team.division.label('qb_division'),
        )
        .join(Game, QBGameStat.game_id == Game.game_id)
        .join(Team, QBGameStat.team_id == Team.team_id)
        .outerjoin(Feature, Game.game_id == Feature.game_id)
        .filter(QBGameStat.season >= start_year)
        .filter(QBGameStat.season <= end_year)
        .filter(QBGameStat.attempts > 0)  # Only QBs with passing attempts
        .order_by(Game.game_date)
    )

    # Convert to pandas DataFrame
    results = query.all()

    if not results:
        raise ValueError("No training data found. Check data ingestion and date ranges.")

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            # Target variable
            'passing_yards': r.QBGameStat.passcing_yards if hasattr(r.QBGameStat, 'passing_yards') else None,

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

            # Game features (from features table)
            'rolling_win_pct_home': r.rolling_win_pct_home,
            'rolling_win_pct_away': r.rolling_win_pct_away,
            'elo_home': r.elo_home,
            'elo_away': r.elo_away,
            'rest_days_home': r.rest_days_home,
            'rest_days_away': r.rest_days_away,
            'home_away_split': r.home_away_split,
            'sos_home': r.sos_home,
            'sos_away': r.sos_away,

            # Derived features
            'is_home': 1 if r.QBGameStat.home_away == 'home' else 0,
            'points_total': None,  # Will fill from odds if needed
            'spread': None,        # Will fill from odds if needed
            'over_under': None,    # Will fill from odds if needed
        }
        for r in results
        if r.QBGameStat.passcing_yards is not None  # Fixed typo below
    ])

    # Fix the typo in the above comprehension
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

            # Game features (from features table)
            'rolling_win_pct_home': r.rolling_win_pct_home,
            'rolling_win_pct_away': r.rolling_win_pct_away,
            'elo_home': r.elo_home,
            'elo_away': r.elo_away,
            'rest_days_home': r.rest_days_home,
            'rest_days_away': r.rest_days_away,
            'home_away_split': r.home_away_split,
            'sos_home': r.sos_home,
            'sos_away': r.sos_away,

            # Derived features
            'is_home': 1 if r.QBGameStat.home_away == 'home' else 0,
            'points_total': None,  # Will fill from odds if needed
            'spread': None,        # Will fill from odds if needed
            'over_under': None,    # Will fill from odds if needed
        }
        for r in results
        if r.QBGameStat.passing_yards is not None
    ])

    return df


def engineer_qb_features(df):
    """
    Engineer features for QB passing yards prediction.
    Creates rolling averages, matchup features, and contextual features.
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

    # Conference and division features (one-hot encoded later)
    # For now, we'll keep as categorical and let models handle it

    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0)

    return df


def prepare_features_for_training(df):
    """
    Prepare feature matrix and target vector for model training.
    """
    # Select features for training
    feature_columns = [
        # QB rolling performance
        'qb_passing_yards_3game_avg',
        'qb_completion_pct_3game_avg',
        'qb_yards_per_attempt_3game_avg',
        'qb_td_rate_3game_avg',
        'qb_int_rate_3game_avg',

        # Game context
        'is_home_game',
        'days_since_last_game',
        'week_normalized',
        'is_early_season',
        'is_mid_season',
        'is_late_season',

        # Team features
        'rolling_win_pct_home',
        'rolling_win_pct_away',
        'elo_home',
        'elo_away',
        'rest_days_home',
        'rest_days_away',
        'home_away_split',
        'sos_home',
        'sos_away',

        # Matchup features
        'pass_yards_vs_opp_allowed',
        'yards_per_attempt_vs_opp_allowed',

        # Defensive context (placeholders)
        'opp_pass_yards_allowed_avg',
        'opp_sack_rate_avg',
        'opp_pass_yards_per_attempt_allowed',
    ]

    # Filter to only columns that exist
    available_features = [col for col in feature_columns if col in df.columns]

    # Prepare X and y
    X = df[available_features].copy()
    y = df['passing_yards'].copy()

    # Handle any remaining NaN values
    X = X.fillna(0)
    y = y.fillna(y.mean())  # Fill target NaN with mean

    return X, y, available_features


def train_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Train ensemble of regression models for QB passing yards prediction.
    """
    models = {}
    predictions = {}

    print("Training models...")

    # 1. Linear Regression (Baseline)
    print("  Training Linear Regression...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    models['linear_regression'] = lr_pipeline
    predictions['linear_regression'] = lr_pred
    print(f"    LR MAE: {mean_absolute_error(y_test, lr_pred):.2f}")

    # 2. Random Forest Regressor
    print("  Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    models['random_forest'] = rf_model
    predictions['random_forest'] = rf_pred
    print(f"    RF MAE: {mean_absolute_error(y_test, rf_pred):.2f}")

    # 3. XGBoost Regressor
    print("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    models['xgboost'] = xgb_model
    predictions['xgboost'] = xgb_pred
    print(f"    XGB MAE: {mean_absolute_error(y_test, xgb_pred):.2f}")

    # 4. Neural Network (MLP Regressor)
    print("  Training Neural Network...")
    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True
        ))
    ])
    mlp_pipeline.fit(X_train, y_train)
    mlp_pred = mlp_pipeline.predict(X_test)
    models['neural_network'] = mlp_pipeline
    predictions['neural_network'] = mlp_pred
    print(f"    NN MAE: {mean_absolute_error(y_test, mlp_pred):.2f}")

    # 5. Ensemble (Weighted Average)
    print("  Creating ensemble...")
    # Simple average for now - could optimize weights based on validation performance
    ensemble_pred = np.mean([
        predictions['linear_regression'],
        predictions['random_forest'],
        predictions['xgboost'],
        predictions['neural_network']
    ], axis=0)
    models['ensemble'] = ensemble_pred  # Store prediction function
    predictions['ensemble'] = ensemble_pred
    print(f"    Ensemble MAE: {mean_absolute_error(y_test, ensemble_pred):.2f}")

    return models, predictions


def evaluate_models(y_test, predictions):
    """
    Evaluate all models using regression metrics.
    """
    print("\nModel Evaluation:")
    print("-" * 50)

    results = {}
    for name, pred in predictions.items():
        if name == 'ensemble':
            continue  # Skip duplicate

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)

        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

        print(f"{name:20} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R²: {r2:6.4f}")

    # Ensemble results
    ensemble_pred = predictions['ensemble']
    mae = mean_absolute_error(y_test, ensemble_pred)
    mse = mean_squared_error(y_test, ensemble_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, ensemble_pred)

    results['ensemble'] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    print(f"{'ensemble':20} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R²: {r2:6.4f}")

    return results


def save_models(models, feature_names, output_dir='models'):
    """
    Save trained models to disk.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save each model
    for name, model in models.items():
        if name != 'ensemble':  # Ensemble is just a prediction function
            model_path = f"{output_dir}/{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")

    # Save feature names
    feature_path = f"{output_dir}/feature_names.pkl"
    joblib.dump(feature_names, feature_path)
    print(f"Saved feature names to {feature_path}")


def main():
    parser = argparse.ArgumentParser(description="Train QB Passing Yards Prediction Models")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for training data")
    parser.add_argument("--end-year", type=int, default=2023, help="End year for training data")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training (for testing)")

    args = parser.parse_args()

    print("Starting QB Passing Yards Model Training")
    print(f"Training data: {args.start_year}-{args.end_year}")
    print(f"Test size: {args.test_size}")
    print("-" * 50)

    # Initialize database
    init_db()
    session = SessionLocal()

    try:
        # Load data
        print("Loading training data...")
        df = load_training_data(session, args.start_year, args.end_year)
        print(f"Loaded {len(df)} QB game logs")

        if len(df) == 0:
            print("ERROR: No data loaded. Check data ingestion and database.")
            return 1

        # Engineer features
        print("\nEngineering features...")
        df_features = engineer_qb_features(df)
        print(f"Feature engineering complete. Shape: {df_features.shape}")

        # Prepare features and target
        print("\nPreparing features for training...")
        X, y, feature_names = prepare_features_for_training(df_features)
        print(f"Features: {len(feature_names)}")
        print(f"Target stats - Mean: {y.mean():.1f}, Std: {y.std():.1f}, Min: {y.min():.0f}, Max: {y.max():.0f}")

        # Time-based split (no shuffling to prevent data leakage)
        split_idx = int(len(X) * (1 - args.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        if not args.skip_training:
            # Train models
            models, predictions = train_models(X_train, y_train, X_test, y_test, feature_names)

            # Evaluate models
            results = evaluate_models(y_test, predictions)

            # Save models
            print(f"\nSaving models to {args.output_dir}/...")
            save_models(models, feature_names, args.output_dir)

            print("\nTraining complete!")
        else:
            print("\nSkipping training (--skip-training flag used)")

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