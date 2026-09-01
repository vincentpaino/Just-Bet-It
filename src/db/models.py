"""
SQLAlchemy models matching the project's ERD (six tables):
games, teams, features, odds, users, user_predictions.

These are starting-point stubs — fill in columns per your finalized ERD,
then generate the first Alembic migration:
    alembic revision --autogenerate -m "initial schema"
    alembic upgrade head
"""

from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import relationship

from src.db.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    game_date = Column(DateTime, nullable=False)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    # add engineered feature columns here (rolling averages, efficiency metrics, etc.)


class Odds(Base):
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    sportsbook = Column(String, nullable=False)
    market = Column(String, nullable=False)  # moneyline / spread / over_under
    line = Column(Float)
    price = Column(Float)
    polled_at = Column(DateTime, nullable=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    password_hash = Column(String, nullable=False)  # bcrypt hash — never plaintext
    is_active = Column(Boolean, default=True)


class UserPrediction(Base):
    __tablename__ = "user_predictions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    model_probability = Column(Float)
    implied_probability = Column(Float)
    edge = Column(Float)
    recommended_stake = Column(Float)  # Kelly-sized amount
    outcome = Column(String, nullable=True)  # filled in post-game
