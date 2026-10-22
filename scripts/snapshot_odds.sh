#!/usr/bin/env bash
# Weekly QB props snapshot that builds up real lines for scripts/backtest.py.
# Runs Friday at 9am (see TEAMMATE_SETUP.md): it first ingests CFBD games and
# QB stats (so last week's props can be scored and new starters can be
# matched), then saves QB passing-yards props for games kicking off in the
# next 48h, i.e. Friday night through the whole Saturday slate.
#
# Odds API credits: only games with props posted cost anything (1 credit
# each, ~50-65 a week). Games already captured in the last 20h are skipped,
# so running it twice by accident costs nothing extra, and a run never takes
# remaining credits below 50.
#
# Afterwards it backs up the database to data/raw/backups/ (last 8 kept).
#
# Extra arguments are passed to run_ingestion and override the defaults
# below, e.g. a cheaper test run:  scripts/snapshot_odds.sh --props-window-hours 24
#
# Logs go to logs/snapshot_odds.log.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

PY=.venv/bin/python
export SSL_CERT_FILE="$("$PY" -c 'import certifi; print(certifi.where())')"

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    "$PY" -m src.ingestion.run_ingestion --props --props-window-hours 48 "$@"
    "$PY" -m scripts.backup_db
} >> logs/snapshot_odds.log 2>&1
