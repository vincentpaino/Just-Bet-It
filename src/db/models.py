"""
SQLAlchemy ORM models for Just Bet It.
Mirrors db/schema.sql but in object, code form — keep the two in sync.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"

    team_id = Column(Integer, primary_key=True)
    team_name = Column(String(100), nullable=False, unique=True)
    conference = Column(String(100))
    division = Column(String(100))


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        CheckConstraint(
            "home_team_id <> away_team_id",
            name="chk_different_teams",
        ),
    )

    game_id = Column(Integer, primary_key=True)
    home_team_id = Column(
        Integer,
        ForeignKey("teams.team_id"),
        nullable=False,
    )
    away_team_id = Column(
        Integer,
        ForeignKey("teams.team_id"),
        nullable=False,
    )
    game_date = Column(Date, nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_win = Column(Boolean)

    cfbd_game_id = Column(BigInteger, unique=True)
    start_time = Column(DateTime(timezone=True))
    start_time_tbd = Column(Boolean)
    season = Column(Integer)
    week = Column(Integer)
    season_type = Column(String(20))
    neutral_site = Column(Boolean)

    home_team = relationship(
        "Team",
        foreign_keys=[home_team_id],
    )
    away_team = relationship(
        "Team",
        foreign_keys=[away_team_id],
    )
    features = relationship(
        "Feature",
        back_populates="game",
        uselist=False,
    )
    odds = relationship(
        "Odds",
        back_populates="game",
    )
    qb_stats = relationship(
        "QBGameStat",
        back_populates="game",
    )
    qb_game_features = relationship(
        "QBGameFeature",
        back_populates="game",
    )
    qb_prop_lines = relationship(
        "QBPropLine",
        back_populates="game",
    )


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (
        UniqueConstraint(
            "game_id",
            name="uq_feature_game",
        ),
    )

    feature_id = Column(Integer, primary_key=True)
    game_id = Column(
        Integer,
        ForeignKey(
            "games.game_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    rolling_win_pct_home = Column(Numeric(5, 4))
    rolling_win_pct_away = Column(Numeric(5, 4))
    elo_home = Column(Numeric(7, 2))
    elo_away = Column(Numeric(7, 2))
    rest_days_home = Column(Integer)
    rest_days_away = Column(Integer)
    home_away_split = Column(Numeric(5, 4))
    sos_home = Column(Numeric(5, 4))
    sos_away = Column(Numeric(5, 4))

    game = relationship(
        "Game",
        back_populates="features",
    )


class Odds(Base):
    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint(
            "game_id",
            "sportsbook",
            "fetched_at",
            name="uq_odds_snapshot",
        ),
    )

    odds_id = Column(Integer, primary_key=True)
    game_id = Column(
        Integer,
        ForeignKey(
            "games.game_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    sportsbook = Column(
        String(100),
        nullable=False,
    )

    home_moneyline = Column(Numeric(8, 2))
    away_moneyline = Column(Numeric(8, 2))
    spread = Column(Numeric(5, 2))
    over_under = Column(Numeric(5, 2))
    implied_prob_home = Column(Numeric(5, 4))
    implied_prob_away = Column(Numeric(5, 4))

    fetched_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    game = relationship(
        "Game",
        back_populates="odds",
    )


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    username = Column(
        String(50),
        nullable=False,
        unique=True,
    )
    email = Column(
        String(255),
        nullable=False,
        unique=True,
    )
    password_hash = Column(
        String(255),
        nullable=False,
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    last_login = Column(DateTime)


class UserPrediction(Base):
    __tablename__ = "user_predictions"

    prediction_id = Column(Integer, primary_key=True)
    user_id = Column(
        Integer,
        ForeignKey(
            "users.user_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    game_id = Column(
        Integer,
        ForeignKey(
            "games.game_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    predicted_winner = Column(
        Integer,
        ForeignKey("teams.team_id"),
    )
    confidence = Column(Numeric(5, 4))
    bet_amount = Column(Numeric(10, 2))
    outcome = Column(String(20))
    roi_result = Column(Numeric(8, 4))
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )


class IngestionLog(Base):
    __tablename__ = "ingestion_log"

    log_id = Column(Integer, primary_key=True)
    source = Column(
        String(50),
        nullable=False,
    )
    run_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    record_count = Column(
        Integer,
        nullable=False,
        default=0,
    )
    success = Column(
        Boolean,
        nullable=False,
        default=True,
    )
    error_message = Column(Text)


class QBGameStat(Base):
    __tablename__ = "qb_game_stats"

    id = Column(Integer, primary_key=True)

    game_id = Column(
        Integer,
        ForeignKey("games.game_id"),
        nullable=False,
    )

    # CFBD player IDs can exceed PostgreSQL's
    # 32-bit INTEGER range.
    player_id = Column(
        BigInteger,
        nullable=False,
    )

    player_name = Column(
        String(100),
        nullable=False,
    )

    team_id = Column(
        Integer,
        ForeignKey("teams.team_id"),
        nullable=False,
    )

    opponent_team_id = Column(
        Integer,
        ForeignKey("teams.team_id"),
    )

    # Passing statistics
    passing_yards = Column(Integer)
    passing_tds = Column(Integer)
    passing_ints = Column(Integer)
    completions = Column(Integer)
    attempts = Column(Integer)
    completion_pct = Column(Numeric(5, 2))
    yards_per_attempt = Column(Numeric(4, 2))

    # Rushing statistics (for dual-threat QBs)
    rushing_yards = Column(Integer)
    rushing_tds = Column(Integer)
    rushing_attempts = Column(Integer)

    # Game context
    game_date = Column(Date)
    season = Column(Integer)
    week = Column(Integer)
    home_away = Column(String(10))

    # Timestamps
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationships
    game = relationship(
        "Game",
        back_populates="qb_stats",
    )

    team = relationship(
        "Team",
        foreign_keys=[team_id],
    )

    opponent_team = relationship(
        "Team",
        foreign_keys=[opponent_team_id],
    )

    class QBGameFeature(Base):
        """
        One row per QB per game: pregame features + the passing-yards target.
        Every feature column must be computed using only games BEFORE this one.
        """
        __tablename__ = "qb_game_features"
        __table_args__ = (
            UniqueConstraint(
                "game_id",
                "player_id",
                name="uq_qb_feature_game_player",
            ),
        )

        id = Column(Integer, primary_key=True)

        game_id = Column(
            Integer,
            ForeignKey(
                "games.game_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        )
        player_id = Column(BigInteger, nullable=False)
        player_name = Column(String(100), nullable=False)
        team_id = Column(
            Integer,
            ForeignKey("teams.team_id"),
            nullable=False,
        )
        opponent_team_id = Column(
            Integer,
            ForeignKey("teams.team_id"),
        )

        # Context
        season = Column(Integer)
        game_date = Column(Date, nullable=False)
        is_home = Column(Boolean)
        rest_days = Column(Integer)

        # Target (NULL for upcoming games)
        passing_yards = Column(Integer)

        # Pregame QB features (prior games only)
        games_played_prior = Column(Integer)
        avg_pass_yds_last3 = Column(Numeric(6, 2))
        avg_pass_yds_last5 = Column(Numeric(6, 2))
        avg_pass_yds_season = Column(Numeric(6, 2))
        avg_attempts_last5 = Column(Numeric(5, 2))
        avg_ypa_last5 = Column(Numeric(4, 2))
        avg_comp_pct_last5 = Column(Numeric(5, 2))
        avg_rush_yds_last5 = Column(Numeric(6, 2))

        # Pregame opponent / team features (prior games only)
        opp_pass_yds_allowed_avg = Column(Numeric(6, 2))
        team_pass_att_avg = Column(Numeric(5, 2))

        created_at = Column(
            DateTime,
            nullable=False,
            default=datetime.utcnow,
        )

        game = relationship(
            "Game",
            back_populates="qb_game_features",
        )

    class QBPropLine(Base):
        """
        Sportsbook passing-yards line for a QB in a game.
        Snapshot table: the same line can move, so fetched_at is part of the key.
        """
        __tablename__ = "qb_prop_lines"
        __table_args__ = (
            UniqueConstraint(
                "game_id",
                "player_id",
                "sportsbook",
                "fetched_at",
                name="uq_qb_prop_snapshot",
            ),
        )

        id = Column(Integer, primary_key=True)

        game_id = Column(
            Integer,
            ForeignKey(
                "games.game_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        )
        player_id = Column(BigInteger)  # may be NULL until names are matched to CFBD IDs
        player_name = Column(String(100), nullable=False)
        sportsbook = Column(String(100), nullable=False)

        line = Column(Numeric(6, 1), nullable=False)  # e.g. 245.5
        over_odds = Column(Numeric(8, 2))
        under_odds = Column(Numeric(8, 2))

        fetched_at = Column(
            DateTime,
            nullable=False,
            default=datetime.utcnow,
        )

        game = relationship(
            "Game",
            back_populates="qb_prop_lines",
        )