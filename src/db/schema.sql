-- Just Bet It — PostgreSQL 15 schema
-- Matches the ERD in Project Doc II, Section 2.4
-- Run with: psql -U <user> -d just_bet_it -f schema.sql

BEGIN;

-- ---------------------------------------------------------------------------
-- TEAMS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team_id      SERIAL PRIMARY KEY,
    team_name    VARCHAR(100) NOT NULL UNIQUE,
    conference   VARCHAR(100),
    division     VARCHAR(100)
);

-- ---------------------------------------------------------------------------
-- GAMES
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS games (
    game_id       SERIAL PRIMARY KEY,
    home_team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    game_date     DATE NOT NULL,
    home_score    INTEGER,
    away_score    INTEGER,
    home_win      BOOLEAN,
    CONSTRAINT chk_different_teams CHECK (home_team_id <> away_team_id)
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games (game_date);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games (home_team_id);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games (away_team_id);

-- ---------------------------------------------------------------------------
-- FEATURES  (one row per game, engineered downstream of raw ingestion)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    feature_id           SERIAL PRIMARY KEY,
    game_id              INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    rolling_win_pct_home NUMERIC(5,4),
    rolling_win_pct_away NUMERIC(5,4),
    elo_home             NUMERIC(7,2),
    elo_away             NUMERIC(7,2),
    rest_days_home       INTEGER,
    rest_days_away       INTEGER,
    home_away_split      NUMERIC(5,4),
    sos_home             NUMERIC(5,4),
    sos_away             NUMERIC(5,4),
    UNIQUE (game_id)
);

-- ---------------------------------------------------------------------------
-- ODDS  (multiple rows per game — one per sportsbook)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds (
    odds_id           SERIAL PRIMARY KEY,
    game_id           INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    sportsbook        VARCHAR(100) NOT NULL,
    home_moneyline    NUMERIC(8,2),
    away_moneyline    NUMERIC(8,2),
    spread            NUMERIC(5,2),
    over_under        NUMERIC(5,2),
    implied_prob_home NUMERIC(5,4),
    implied_prob_away NUMERIC(5,4),
    fetched_at        TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (game_id, sportsbook, fetched_at)
);

CREATE INDEX IF NOT EXISTS idx_odds_game ON odds (game_id);
CREATE INDEX IF NOT EXISTS idx_odds_book ON odds (sportsbook);

-- ---------------------------------------------------------------------------
-- USERS  (★ added per Doc II feedback)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id       SERIAL PRIMARY KEY,
    username      VARCHAR(50) NOT NULL UNIQUE,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,   -- bcrypt hash, never plaintext
    created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- USER_PREDICTIONS  (★ added per Doc II feedback)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_predictions (
    prediction_id    SERIAL PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    game_id          INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    predicted_winner INTEGER REFERENCES teams(team_id),
    confidence       NUMERIC(5,4),
    bet_amount       NUMERIC(10,2),
    outcome          VARCHAR(20),   -- 'win' | 'loss' | 'push' | 'pending'
    roi_result       NUMERIC(8,4),
    created_at       TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_user ON user_predictions (user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_game ON user_predictions (game_id);

-- ---------------------------------------------------------------------------
-- INGESTION LOG  (not in the original ERD, but Doc II Section 3.2 calls for
-- logging every ingestion run — timestamp, record count, errors)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_log (
    log_id        SERIAL PRIMARY KEY,
    source        VARCHAR(50) NOT NULL,   -- 'odds_api' | 'sportsipy'
    run_at        TIMESTAMP NOT NULL DEFAULT NOW(),
    record_count  INTEGER NOT NULL DEFAULT 0,
    success       BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT
);

COMMIT;
