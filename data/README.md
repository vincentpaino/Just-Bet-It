# Data

`raw/` and `processed/` are gitignored — large sports data files should not
live in git history.

## Regenerating raw data

Everything lives in the Postgres database, not in files here:

- NCAA football games, game lines and QB stats: CollegeFootballData.com,
  via `python -m src.ingestion.run_ingestion` (see `src/ingestion/`)
- QB passing-yards prop lines: The Odds API, via the weekly job
  (`scripts/snapshot_odds.sh`)

`raw/` holds database backups (`raw/backups/`, from `scripts/backup_db.py`)
and caches that scripts rebuild on their own.
