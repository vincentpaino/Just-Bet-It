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
    cfbd_game_id   BIGINT UNIQUE,
    start_time     TIMESTAMPTZ,
    start_time_tbd BOOLEAN,
    season         INTEGER,
    week           INTEGER,
    season_type    VARCHAR(20),
    neutral_site   BOOLEAN,
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
    source        VARCHAR(50) NOT NULL,   -- 'odds_api' | 'cfbd'
    run_at        TIMESTAMP NOT NULL DEFAULT NOW(),
    record_count  INTEGER NOT NULL DEFAULT 0,
    success       BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT
);

-- ---------------------------------------------------------------------------
-- QB GAME STATS  (one row per QB per game, from CFBD)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS qb_game_stats (
    id                 SERIAL PRIMARY KEY,
    game_id            INTEGER NOT NULL REFERENCES games(game_id),
    player_id          BIGINT NOT NULL,   -- CFBD IDs can exceed 32-bit INTEGER
    player_name        VARCHAR(100) NOT NULL,
    team_id            INTEGER NOT NULL REFERENCES teams(team_id),
    opponent_team_id   INTEGER REFERENCES teams(team_id),
    passing_yards      INTEGER,
    passing_tds        INTEGER,
    passing_ints       INTEGER,
    completions        INTEGER,
    attempts           INTEGER,
    completion_pct     NUMERIC(5,2),
    yards_per_attempt  NUMERIC(4,2),
    rushing_yards      INTEGER,
    rushing_tds        INTEGER,
    rushing_attempts   INTEGER,
    game_date          DATE,
    season             INTEGER,
    week               INTEGER,
    home_away          VARCHAR(10),
    created_at         TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_qb_game_player UNIQUE (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_qb_stats_game ON qb_game_stats (game_id);
CREATE INDEX IF NOT EXISTS idx_qb_stats_player_date ON qb_game_stats (player_id, game_date);

-- ---------------------------------------------------------------------------
-- QB GAME FEATURES  (one row per QB per game: pregame features + target)
-- All feature columns must use only games BEFORE game_date.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS qb_game_features (
    id                        SERIAL PRIMARY KEY,
    game_id                   INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player_id                 BIGINT NOT NULL,
    player_name               VARCHAR(100) NOT NULL,
    team_id                   INTEGER NOT NULL REFERENCES teams(team_id),
    opponent_team_id          INTEGER REFERENCES teams(team_id),
    season                    INTEGER,
    game_date                 DATE NOT NULL,
    is_home                   BOOLEAN,
    rest_days                 INTEGER,
    passing_yards             INTEGER,        -- target; NULL for upcoming games
    games_played_prior        INTEGER,
    avg_pass_yds_last3        NUMERIC(6,2),
    avg_pass_yds_last5        NUMERIC(6,2),
    avg_pass_yds_season       NUMERIC(6,2),
    avg_attempts_last5        NUMERIC(5,2),
    avg_ypa_last5             NUMERIC(4,2),
    avg_comp_pct_last5        NUMERIC(5,2),
    avg_rush_yds_last5        NUMERIC(6,2),
    opp_pass_yds_allowed_avg  NUMERIC(6,2),
    team_pass_att_avg         NUMERIC(5,2),
    created_at                TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_qb_feature_game_player UNIQUE (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_qb_features_player_date ON qb_game_features (player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_qb_features_season ON qb_game_features (season);

-- ---------------------------------------------------------------------------
-- QB PROP LINES  (sportsbook passing-yards lines; snapshots over time)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS qb_prop_lines (
    id           SERIAL PRIMARY KEY,
    game_id      INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player_id    BIGINT,                  -- NULL until matched to a CFBD ID
    player_name  VARCHAR(100) NOT NULL,
    sportsbook   VARCHAR(100) NOT NULL,
    line         NUMERIC(6,1) NOT NULL,   -- e.g. 245.5
    over_odds    NUMERIC(8,2),
    under_odds   NUMERIC(8,2),
    fetched_at   TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_qb_prop_snapshot UNIQUE (game_id, player_id, sportsbook, fetched_at)
);

CREATE INDEX IF NOT EXISTS idx_qb_props_game ON qb_prop_lines (game_id);

-- ---------------------------------------------------------------------------
-- GAME LINES  (CFBD historical/current game lines: one row per game per book)
-- spread is from the HOME team's side (negative = home favored). Rows are
-- updated as books move the line, so after kickoff they hold the closing line.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS game_lines (
    id               SERIAL PRIMARY KEY,
    game_id          INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    provider         VARCHAR(100) NOT NULL,
    spread           NUMERIC(5,1),
    spread_open      NUMERIC(5,1),
    over_under       NUMERIC(5,1),
    over_under_open  NUMERIC(5,1),
    home_moneyline   NUMERIC(8,2),
    away_moneyline   NUMERIC(8,2),
    updated_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_game_line_provider UNIQUE (game_id, provider)
);

CREATE INDEX IF NOT EXISTS idx_game_lines_game ON game_lines (game_id);

COMMIT;
