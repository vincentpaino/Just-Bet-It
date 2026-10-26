# Just Bet It

AI-powered NCAA football sports betting prediction system. Ensemble of
Logistic Regression, Random Forest, XGBoost, and a Neural Net, with
isotonic calibration, +EV detection, and Kelly Criterion bet sizing.

**Team:** Zach Burns (Backend/Systems), Vince Paino (ML), Cody Shouse (Data)
**Mentor:** Brooks Noonan

## Project layout

```
data/           raw/processed data (gitignored — see data/README.md)
notebooks/      exploratory analysis
src/
  data/         ingestion (Odds API, Sportsipy) + preprocessing
  models/       LR, RF, XGBoost, NN training code
  calibration/  isotonic calibration
  ensemble/     weighted ensemble logic
  betting/      Kelly sizing, EV filtering, line shopping
  db/           SQLAlchemy models + connection (database.py, models.py)
  utils/        shared helpers
models/         trained model artifacts (gitignored)
tests/          unit tests
configs/        hyperparameters, paths
scripts/        CLI entry points (train.py, predict.py, backtest.py)
```

## Setup

1. `python -m venv venv && source venv/bin/activate` (or use PyCharm's interpreter setup)
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in `DATABASE_URL` and `ODDS_API_KEY`
4. Set up Postgres locally (or via Docker), then:
   ```
   alembic upgrade head
   ```
## Dashboard

A web UI over the same models and data: this week's picks vs. sportsbook
lines, a QB lookup, the backtest, and model/database status (with a
retrain button). From the project root, after `python -m scripts.train`:

```
streamlit run app.py
```

It opens at http://localhost:8501. Code is in `app.py`; theme and bundled
fonts are in `.streamlit/config.toml` and `static/fonts/`.

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

PostgreSQL, six tables: `games`, `teams`, `features`, `odds`, `users`,
`user_predictions`. Models live in `src/db/models.py`; migrations are
managed with Alembic (`alembic/` directory).
