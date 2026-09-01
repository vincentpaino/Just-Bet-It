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

## Database

PostgreSQL, six tables: `games`, `teams`, `features`, `odds`, `users`,
`user_predictions`. Models live in `src/db/models.py`; migrations are
managed with Alembic (`alembic/` directory).
