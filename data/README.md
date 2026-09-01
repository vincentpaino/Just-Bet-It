# Data

`raw/` and `processed/` are gitignored — large sports data files should not
live in git history.

## Regenerating raw data

- Historical NCAA football stats (2015–present): pulled via `sportsipy`, see `src/data/`
- Betting odds (40+ sportsbooks): pulled via The Odds API, see `src/data/`

Run the ingestion scripts in `scripts/` to repopulate `raw/` and `processed/`
locally, or pull directly from the Postgres database once ingestion has run.
