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
# Then it grades the previous weekend's picks into reports/weekly_results.md
# (see scripts/weekly_report.py), retrains the models on all games so far
# (graded first, so each week is graded by the model that made its picks),
# and backs up the database to data/raw/backups/ (last 8 kept).
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

# launchd can start the job during a brief "dark wake" from sleep, before
# Wi-Fi is back. Wait (up to 15 min) until the database and the Odds API
# both answer; nothing is spent or written until then.
wait_for_network() {
    for attempt in $(seq 1 30); do
        if "$PY" - <<'PY' >/dev/null 2>&1
import socket
from src.db.session import engine
socket.getaddrinfo("api.the-odds-api.com", 443)
engine.connect().close()
PY
        then
            [ "$attempt" -gt 1 ] && echo "network up after $(( (attempt - 1) * 30 ))s"
            return 0
        fi
        sleep 30
    done
    echo "ERROR: no network after 15 minutes; nothing was run (no credits spent)"
    return 1
}

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    wait_for_network
    # caffeinate keeps the Mac from going back to sleep until the job finishes
    /usr/bin/caffeinate -i "$PY" -m src.ingestion.run_ingestion --props --props-window-hours 48 "$@"
    # Grade last weekend's picks into reports/ (a failure here shouldn't skip the backup)
    /usr/bin/caffeinate -i "$PY" -m scripts.weekly_report || echo "[report] FAILED; job continues"
    # THEN retrain on everything, including last weekend (order matters: grading
    # first means each week is graded by the model that made its picks).
    # If training fails, the previous models stay in place.
    /usr/bin/caffeinate -i "$PY" -m scripts.train || echo "[train] FAILED; keeping previous models"
    /usr/bin/caffeinate -i "$PY" -m scripts.backup_db
} >> logs/snapshot_odds.log 2>&1
