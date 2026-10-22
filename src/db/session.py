"""
Database engine + session factory. Reads connection settings from environment
variables so credentials never live in source code (see Doc I, Section 6.4 —
Security requirements).

Expected env vars (see .env.example), either:
    DATABASE_URL                 the shared team database (Neon); used if set
or, for a local database:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Every script connects through this module, so this is the only place that
needs to know where the database lives.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import sessionmaker

from .models import Base


# Load variables from .env
load_dotenv()


def get_database_url() -> URL:
    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        # Neon hands out a plain libpq URL (postgresql://...?sslmode=require),
        # which psql/pg_dump also accept; point SQLAlchemy at psycopg2.
        # Query options such as sslmode are passed through to libpq.
        return make_url(database_url).set(drivername="postgresql+psycopg2")

    # URL.create escapes special characters (@, /, :) in the password
    return URL.create(
        drivername="postgresql+psycopg2",
        username=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "") or None,
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        database=os.environ.get("DB_NAME", "just_bet_it"),
    )


# pool_pre_ping replaces connections the server closed while idle (Neon
# suspends its compute after a few idle minutes).
engine = create_engine(get_database_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables if they don't exist yet (dev convenience;
    prefer schema.sql + migrations for anything past local dev)."""
    Base.metadata.create_all(bind=engine)


@contextmanager
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