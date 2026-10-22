"""
Back up the database (the shared DATABASE_URL, or the local DB_* one) to
data/raw/backups/ with pg_dump, keeping the most recent KEEP files. The weekly
snapshot job runs this after ingestion; see Doc I/III, Section 5.4.

Run from the project root:
    python -m scripts.backup_db

Restore into an empty database with:
    pg_restore --no-owner --no-privileges -d <target> data/raw/backups/<file>.dump
"""

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from src.db.session import engine, get_database_url

BACKUP_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "backups"
KEEP = 8

# Homebrew installs Postgres tools outside the default PATH (and launchd
# jobs get a minimal PATH), so check the usual locations too.
PG_DUMP_CANDIDATES = [
    shutil.which("pg_dump"),
    "/opt/homebrew/opt/postgresql@15/bin/pg_dump",
    "/usr/local/opt/postgresql@15/bin/pg_dump",
]

# URL query options that map onto libpq environment variables
QUERY_TO_ENV = {
    "sslmode": "PGSSLMODE",
    "channel_binding": "PGCHANNELBINDING",
    "options": "PGOPTIONS",
}


def find_pg_dump() -> str:
    for path in PG_DUMP_CANDIDATES:
        if path and os.access(path, os.X_OK):
            return path
    sys.exit("pg_dump not found; install PostgreSQL 15 client tools (brew install postgresql@15)")


def major_version(version_text: str) -> int:
    return int(re.search(r"(\d+)", version_text).group(1))


def libpq_env(url) -> dict:
    """Connection settings as PG* environment variables (keeps the password off the command line)."""
    env = dict(os.environ)
    for var, value in {
        "PGHOST": url.host,
        "PGPORT": url.port,
        "PGUSER": url.username,
        "PGPASSWORD": url.password,
        "PGDATABASE": url.database,
    }.items():
        if value is not None:
            env[var] = str(value)
    for key, var in QUERY_TO_ENV.items():
        if key in url.query:
            env[var] = str(url.query[key])
    return env


def main():
    pg_dump = find_pg_dump()

    # pg_dump refuses to dump a server newer than itself, so check up front.
    with engine.connect() as conn:
        server = major_version(conn.execute(text("SHOW server_version")).scalar())
    # e.g. "pg_dump (PostgreSQL) 15.19 (Homebrew)"
    client = major_version(subprocess.run(
        [pg_dump, "--version"], capture_output=True, text=True, check=True,
    ).stdout.split("(PostgreSQL)")[-1])
    if client < server:
        sys.exit(f"pg_dump {client} can't back up a PostgreSQL {server} server; "
                 f"install postgresql@{server} client tools")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUP_DIR / f"just_bet_it_{datetime.now():%Y%m%d_%H%M}.dump"

    subprocess.run(
        [pg_dump, "--format=custom", "--no-owner", "--no-privileges", "--file", str(target)],
        env=libpq_env(get_database_url()),
        check=True,
    )

    backups = sorted(BACKUP_DIR.glob("just_bet_it_*.dump"))
    for old in backups[:-KEEP]:
        old.unlink()

    print(f"[backup] wrote {target.relative_to(BACKUP_DIR.parent.parent.parent)} "
          f"({target.stat().st_size / 1e6:.1f} MB); keeping the last {KEEP}")


if __name__ == "__main__":
    main()
