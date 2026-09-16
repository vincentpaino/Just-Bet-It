# Just Bet It — New Teammate Setup

This walks through getting your local environment running: Python deps,
PostgreSQL, and both data source connections (Odds API + CFBD).
Follow it in order — each step depends on the last one working.

## 1. Clone the repo and set up Python

```bash
git clone <repo-url>
cd Just-Bet-It
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All our code lives under `src/` — most commands below should be run
from inside that folder (`cd src`) unless noted otherwise.

## 2. Install PostgreSQL (macOS via Homebrew)

```bash
brew install postgresql@15
brew services start postgresql@15
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

(Intel Macs: use `/usr/local/opt/postgresql@15/bin` instead of
`/opt/homebrew/...` in the PATH line above.)

Not on macOS? Install Postgres 15 however's normal for your OS, then
skip to step 3 — everything after this is the same regardless of OS.

## 3. Create the database and load the schema

```bash
createdb just_bet_it
psql -d just_bet_it -f src/db/schema.sql
```

Verify it worked:

```bash
psql -d just_bet_it -c "\dt"
```

You should see 7 tables: `teams`, `games`, `features`, `odds`, `users`,
`user_predictions`, `ingestion_log`.

**No password needed for local dev** — Homebrew's Postgres trusts your
local macOS login by default, so `DB_PASSWORD` stays blank in your `.env`.

## 4. Set up your `.env`

Copy the template and fill in your own values — **do not** copy anyone
else's `.env` file, since `DB_USER` needs to match your own machine's
username:

```bash
cd src
cp .env.example .env
```

Then edit `.env`:

```
ODDS_API_KEY=<get this in step 5>
CFBD_API_KEY=<get this in step 6>
DB_HOST=localhost
DB_PORT=5432
DB_NAME=just_bet_it
DB_USER=<run `whoami` in Terminal to get this>
DB_PASSWORD=
```

**Never commit `.env`** — it's already in `.gitignore`, but double-check
before you push anything.

## 5. Get an Odds API key

Sign up free at [the-odds-api.com](https://the-odds-api.com) (500
requests/month free tier). Copy your key from the dashboard into
`ODDS_API_KEY` in your `.env`.

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

## 9. Test the Odds API ingestion

From inside `src/`:

```bash
python -m ingestion.run_ingestion --skip-stats
```

Expect output like:
```
[odds_api] requests remaining this period: 497
[odds_api] ingested 556 odds rows
```

Verify the data landed:

```bash
psql -d just_bet_it -c "SELECT * FROM odds LIMIT 5;"
```

## 10. Test the CFBD ingestion

```bash
python -m ingestion.run_ingestion --skip-odds --year 2023
```

Expect output like:
```
[cfbd] ingested 910 game records for 2023
```

Verify:

```bash
psql -d just_bet_it -c "SELECT COUNT(*) FROM teams;"
```

## Common gotchas we already hit (save yourself the time)

- **`ModuleNotFoundError: No module named 'ingestion'`** — you're running
  the command from the project root instead of `src/`. `cd src` first.
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