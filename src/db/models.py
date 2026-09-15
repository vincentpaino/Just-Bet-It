"""
SQLAlchemy ORM models for Just Bet It.
Mirrors db/schema.sql — keep the two in sync.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean, CheckConstraint, Column, Date, DateTime, ForeignKey,
    Integer, Numeric, String, Text, UniqueConstraint
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
        CheckConstraint("home_team_id <> away_team_id", name="chk_different_teams"),
    )

    game_id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    game_date = Column(Date, nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_win = Column(Boolean)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    features = relationship("Feature", back_populates="game", uselist=False)
    odds = relationship("Odds", back_populates="game")


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (UniqueConstraint("game_id", name="uq_feature_game"),)

    feature_id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.game_id", ondelete="CASCADE"), nullable=False)
    rolling_win_pct_home = Column(Numeric(5, 4))
    rolling_win_pct_away = Column(Numeric(5, 4))
    elo_home = Column(Numeric(7, 2))
    elo_away = Column(Numeric(7, 2))
    rest_days_home = Column(Integer)
    rest_days_away = Column(Integer)
    home_away_split = Column(Numeric(5, 4))
    sos_home = Column(Numeric(5, 4))
    sos_away = Column(Numeric(5, 4))

    game = relationship("Game", back_populates="features")


class Odds(Base):
    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint("game_id", "sportsbook", "fetched_at", name="uq_odds_snapshot"),
    )

    odds_id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.game_id", ondelete="CASCADE"), nullable=False)
    sportsbook = Column(String(100), nullable=False)
    home_moneyline = Column(Numeric(8, 2))
    away_moneyline = Column(Numeric(8, 2))
    spread = Column(Numeric(5, 2))
    over_under = Column(Numeric(5, 2))
    implied_prob_home = Column(Numeric(5, 4))
    implied_prob_away = Column(Numeric(5, 4))
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    game = relationship("Game", back_populates="odds")


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_login = Column(DateTime)


class UserPrediction(Base):
    __tablename__ = "user_predictions"

    prediction_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.game_id", ondelete="CASCADE"), nullable=False)
    predicted_winner = Column(Integer, ForeignKey("teams.team_id"))
    confidence = Column(Numeric(5, 4))
    bet_amount = Column(Numeric(10, 2))
    outcome = Column(String(20))
    roi_result = Column(Numeric(8, 4))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class IngestionLog(Base):
    __tablename__ = "ingestion_log"

    log_id = Column(Integer, primary_key=True)
    source = Column(String(50), nullable=False)
    run_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    record_count = Column(Integer, nullable=False, default=0)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text)
