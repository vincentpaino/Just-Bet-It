# Just Bet It

NCAA football QB passing-yards prop predictions. Linear regression, random
forest, XGBoost and a neural net (plus their average), compared against real
sportsbook lines, with +EV detection, line shopping and Kelly Criterion bet
sizing.

**Team:** Zach Burns (Backend/Systems), Vince Paino (ML), Cody Shouse (Data)
**Mentor:** Brooks Noonan

## Project layout

```
app.py          Streamlit dashboard
src/
  ingestion/    CFBD and Odds API ingestion (run_ingestion.py is the entry point)
  betting/      odds math, win probabilities, EV, Kelly sizing, line shopping
  db/           schema.sql, SQLAlchemy models, connection (session.py)
scripts/        features + training (xgboost_reg.py, train.py), predictions,
                backtests, weekly report, scheduled jobs (*.sh)
configs/        tuned XGBoost hyperparameters
reports/        weekly results and experiment write-ups
tests/          unit tests (python -m pytest)
data/           backups and caches (gitignored, see data/README.md)
models/         trained models (gitignored; python -m scripts.train)
```

## Setup

1. `python -m venv venv && source venv/bin/activate` (or use PyCharm's interpreter setup)
2. `pip install -r requirements.txt`
3. `cp src/.env.example .env` and fill in `DATABASE_URL`, `CFBD_API_KEY` and
   (only for the weekly props job) `ODDS_API_KEY`
4. On the shared Neon database the tables already exist. For a local
   Postgres, create them with `psql -d just_bet_it -f src/db/schema.sql`
   (see TEAMMATE_SETUP.md).
## Dashboard

A web UI over the same models and data: this week's picks vs. sportsbook
lines, a QB lookup, the backtest, and model/database status (with a
retrain button). From the project root, after `python -m scripts.train`:

```
streamlit run app.py
```

It opens at http://localhost:8501. Code is in `app.py`; theme and bundled
fonts are in `.streamlit/config.toml` and `static/fonts/`.

The Backtest page and the QB lookup's past predictions use walk-forward
predictions (every week predicted by models trained only on earlier games),
cached in `data/raw/walkforward_preds_pre2024.csv`, with XGBoost settings
tuned only on games before 2024 so nothing it scores was used to choose them.
The file is gitignored, so build it once (a few minutes); without it, those
pages only score games played after the last training run:
`python -m scripts.vs_closing_lines --params configs/xgboost_reg_params_pre2024.json --cache data/raw/walkforward_preds_pre2024.csv`

## Betting (`src/betting/`)

Turns a passing-yards prediction into a bet decision:

- `odds.py`: American odds math (payouts, implied probability, the book's
  margin, no-vig market probabilities).
- `probability.py`: P(over / under / push) from the model's real misses on
  held-out games (saved by `scripts/train.py`), with a normal-curve fallback.
  A calibrator (e.g. isotonic) can replace it later with the same interface.
- `kelly.py`: expected value and Kelly Criterion sizing (handles pushes).
- `strategy.py`: line shopping across books, EV filtering, fractional Kelly
  with caps (5% per bet, 30% of bankroll per day by default, see `BetConfig`)
  and a date-ordered bankroll simulation.

`scripts/predict_upcoming.py` shows a stake for each bet that clears the
filters, `scripts/backtest.py` adds a Kelly bankroll simulation, and the
dashboard uses the same code. Retrain after pulling
(`python -m scripts.train`) so the models file includes the misses.

Tests: `python -m pytest` from the project root.

## Database

PostgreSQL. The tables in use: `teams`, `games`, `game_lines` (CFBD spreads
and totals), `qb_game_stats`, `team_game_stats`, `odds`, `qb_prop_lines`
(Friday prop snapshots), `live_picks` and `qb_prop_closing_lines` (live
closing-line value) and `ingestion_log`. The schema is `src/db/schema.sql`,
with matching SQLAlchemy models in `src/db/models.py`; new tables are created
by `init_db()` when ingestion runs.
