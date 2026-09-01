"""
Central database connection module.

Every other module (ingestion, feature engineering, betting strategy engine)
should import `engine` or `SessionLocal` from here rather than opening its
own connection. Swapping from local Postgres to RDS/Cloud SQL is then just
a matter of changing DATABASE_URL in .env — no code changes needed.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")



engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_session():
    """Yield a session, closing it automatically when done."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
