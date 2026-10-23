# Just Bet It — New Teammate Setup

This walks through getting your environment running: Python deps, the
shared team database, and the data source connections (CFBD, Odds API).
Follow it in order — each step depends on the last one working.

## 1. Clone the repo and set up Python

```bash
git clone <repo-url>
cd Just-Bet-It
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All our code lives under `src/`, but run Python commands from the
**project root** (`python -m src.ingestion.run_ingestion ...`), since the
code imports itself as `src.…`.

## 2. Database: use the shared team database (Neon)

The team shares one hosted PostgreSQL 15 database on [Neon](https://neon.com),
so everyone sees the same games, stats, odds and prop lines. You don't need
Postgres installed to use it, but the `psql`/`pg_dump` tools are handy:

```bash
brew install postgresql@15       # client tools only; no need to start the server
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

(Intel Macs: `/usr/local/opt/postgresql@15/bin` instead.)

Ask Cody for the connection string (it's shared through a password
manager, never in git or plain chat). It looks like
`postgresql://user:password@ep-....neon.tech/neondb?sslmode=require`.

Things to know about the shared database:
- You need an internet connection. After a few idle minutes Neon goes to
  sleep, so the first query can take an extra second or so.
- Training, backtests and exploring data are always fine. **Don't run
  ingestion yourself**: the weekly job on Cody's machine does it (step 12),
  and two ingestion runs at once can collide.
- Check what's in it with `psql "$DATABASE_URL" -c "\dt"` (after
  `export DATABASE_URL=...`, or paste the string in quotes).

## 3. (Optional) A local database instead

For offline work, or to experiment without touching team data, run your own
copy. Start the server and load the schema:

```bash
brew services start postgresql@15
createdb just_bet_it
psql -d just_bet_it -f src/db/schema.sql
```

To fill it with the team's data, restore a backup (see step 12):
`pg_restore --no-owner --no-privileges -d just_bet_it data/raw/backups/<file>.dump`.

**No password needed for local dev**: Homebrew's Postgres trusts your
macOS login, so `DB_PASSWORD` stays blank.

## 4. Set up your `.env`

Copy the template and fill in your own values:

```bash
cp src/.env.example src/.env
```

Then edit `src/.env`. For the shared database, set `DATABASE_URL` (it takes
priority over the `DB_*` lines):

```
CFBD_API_KEY=<get this in step 6>
DATABASE_URL=<connection string from Cody>
```

For a local database instead, leave `DATABASE_URL` out and set:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=just_bet_it
DB_USER=<run `whoami` in Terminal to get this>
DB_PASSWORD=
```

Every script connects through `src/db/session.py`, so this is the only
place you configure the database.

**Never commit `.env`**: it's already in `.gitignore`, but double-check
before you push anything.

## 5. Get an Odds API key

**Most of the team doesn't need one.** The Odds API is only used by the
weekly props job (step 12), which runs on Cody's machine with his key.
If you do need one, sign up free at [the-odds-api.com](https://the-odds-api.com)
(500 credits/month) and put it in `ODDS_API_KEY` in your `.env`. Each key
has its own limit; never share or copy someone else's key.

## 6. Get a CFBD API key (replaces Sportsipy)

**Important change:** we originally planned to use Sportsipy for
historical NCAA stats, per Doc I/II. During Sprint 1 we confirmed
Sports-Reference.com (which Sportsipy scrapes) now returns a 403
Forbidden on scraping requests — this is dead, not a bug in our code.
We've replaced it with **CollegeFootballData.com (CFBD)**, a real,
actively-maintained REST API instead of a scraper.

Sign up free at [collegefootballdata.com](https://collegefootballdata.com)
(1,000 requests/month free tier — better than the old plan anyway).
Copy your key into `CFBD_API_KEY` in your `.env`.

`sportsipy_ingest.py` is still in the repo for reference, but
`run_ingestion.py` now imports from `cfbd_ingest.py` instead. If you're
picking up Sprint 2 historical-stats work, start from `cfbd_ingest.py`.

## 7. Install the `cfbd` package

Already listed in `requirements.txt`, but if you installed dependencies
before this update:

```bash
pip install cfbd
```

## 8. Fix SSL certificate errors (macOS, if you hit them)

If you get `SSLCertVerificationError` when testing CFBD, run:

```bash
pip install --upgrade certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

Add that `export` line to your `~/.zshrc` to make it permanent instead
of re-running it every new terminal session.

## 9. Check your connection

From the project root (this only reads data):

```bash
python -c "from src.db.session import engine; from sqlalchemy import text; \
print(engine.connect().execute(text('SELECT COUNT(*) FROM qb_game_stats')).scalar())"
```

You should get a count in the thousands. If you're on the shared database,
you're done with setup. The model scripts, in order:

```bash
python -m scripts.train              # train all 4 models + ensemble -> models/qb_models.pkl (~10s)
python -m scripts.predict_upcoming   # predictions vs. sportsbook lines for upcoming games
python -m scripts.predict --qb-id 5078810 --game-date 2026-10-24   # one QB, one game
python -m scripts.backtest           # score the models against stored real lines
python scripts/xgboost_reg.py        # XGBoost analysis: baselines, feature importance
python scripts/xgboost_reg_tune.py   # re-tune XGBoost (Optuna, ~3 min) -> configs/
```

Trained models aren't in git, so run `scripts.train` first after cloning.
Every model uses the same real, pregame-only features from
`scripts/xgboost_reg.py`; nothing in them is made up or a placeholder.

## 10. Test ingestion (local database only)

**Skip this on the shared database**: ingestion writes to it, and the weekly
job already keeps it up to date. On your own local database:

```bash
python -m src.ingestion.run_ingestion --year 2023            # CFBD games + QB stats
python -m src.ingestion.run_ingestion --odds --skip-stats --skip-qb   # game odds: costs 3 Odds API credits
```

Odds API steps only run when you ask for them (`--odds`, `--props`), since
the free tier is just 500 credits/month.

## 11. Merge old duplicate teams (old local databases only)

Odds ingestion used to store Odds API names ("Ohio State Buckeyes") as
separate teams from CFBD's ("Ohio State"), so odds never attached to CFBD
games. That's fixed (`src/ingestion/team_names.py`) and the shared database
is already cleaned up. Only if you have an old local database with odds in
it, run the one-time merge from the project root:

```bash
python -m scripts.backup_db                   # back up first
python -m scripts.merge_odds_teams            # dry run: shows what would change
python -m scripts.merge_odds_teams --apply    # do it
```

## 12. Weekly QB props snapshot and backups (Cody's machine)

The Odds API only serves current lines, so backtest history is built by
saving snapshots as we go. `scripts/snapshot_odds.sh` runs once a week
(Friday 9am, scheduled with launchd on Cody's Mac). It ingests CFBD games
and QB stats, saves QB passing-yards props for games kicking off in the
next 48h (Friday night through the Saturday slate), then backs up the
database to `data/raw/backups/` (last 8 kept, gitignored). Weeknight games
are missed; that's the trade-off for staying inside the credit limit.

Credits: only games with props posted cost anything (1 each, roughly 50-65
a week, so ~250/month of the 500). Built-in guards:
- games already snapshotted in the last 20h are skipped, so an accidental
  second run costs nothing,
- a run stops before remaining credits drop below 50 (`--props-credit-reserve`),
- at most 100 paid requests per run (`--props-max-games`),
- every run logs credits used and remaining (`logs/snapshot_odds.log`).

**Only this job collects props.** Don't run `--props` or `--odds` yourself
unless you mean to spend credits.

To move the job to another Mac (only one machine should ever have it):
`scripts/install_snapshot_schedule.sh --remove` on the old one, then
`scripts/install_snapshot_schedule.sh` on the new one. If the Mac is asleep
at 9am Friday, the job runs when it wakes.

`scripts/backtest.py` scores the models against each book's last line before
kickoff, at the real prices. It reports "no QB games" until lines have been
captured for games that have since been played and ingested.

Neon's free plan can only restore to a point in the last 6 hours, so the
weekly backups are the real safety net. To make one by hand:
`python -m scripts.backup_db`.

## Common gotchas we already hit (save yourself the time)

- **`ModuleNotFoundError: No module named 'src'`** — you're running the
  command from inside `src/`. Run it from the project root instead.
- **`.env` not loading / API key errors** — `run_ingestion.py` calls
  `load_dotenv()` explicitly near the top; if you pulled an older copy
  without that line, pull the latest from the repo.
- **`dquote>` stuck prompt in Terminal** — you pasted a command with
  curly/smart quotes instead of straight ones. Ctrl+C to cancel, retype
  manually.
- **Sportsipy is dead** — don't spend time debugging it. Sports-Reference
  blocks the scraper with a 403. Use `cfbd_ingest.py` instead.
- **CFBD `ApiTypeError: unexpected keyword argument 'division'`** — the
  installed `cfbd` package version uses `classification` as the
  parameter name, not `division`. Already fixed in `cfbd_ingest.py`, but
  worth knowing if you're calling the CFBD client directly elsewhere.
- **CFBD `SSLCertVerificationError`** — see step 8 above.
- **CFBD response has camelCase fields** (`homeTeam`, `startDate`, etc.),
  not snake_case — already handled in `cfbd_ingest.py`'s rename map, but
  worth knowing if you extend it to pull more fields.

## Questions?

Ping Cody — he's already been through every error above once.