"""
Database engine + session factory. Reads connection settings from environment
variables so credentials never live in source code (see Doc I, Section 6.4 —
Security requirements).

Expected env vars (see .env.example):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base


def get_database_url() -> str:
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "just_bet_it")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


engine = create_engine(get_database_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables if they don't exist yet (dev convenience;
    prefer schema.sql + migrations for anything past local dev)."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """Yield a session, closing it afterward. Use as a context manager:
        with get_session() as session:
            ...
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
