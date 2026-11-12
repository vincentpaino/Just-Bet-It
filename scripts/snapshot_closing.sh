#!/usr/bin/env bash
# Near-kickoff line snapshot for this week's confident picks (live closing-line
# value; see scripts/snapshot_closing.py). Runs Friday 4pm and Saturday 10am,
# 1pm, 4pm and 7pm Central (scripts/install_snapshot_schedule.sh). Each run only
# covers confident picks kicking off before the next run that don't have a
# snapshot yet: 1 Odds API credit per game, never below 50 remaining.
#
# Logs go to logs/snapshot_closing.log.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

PY=.venv/bin/python
export SSL_CERT_FILE="$("$PY" -c 'import certifi; print(certifi.where())')"

# Wait (up to 5 min) for the database and the Odds API after a wake from sleep
wait_for_network() {
    for attempt in $(seq 1 10); do
        if "$PY" - <<'PY' >/dev/null 2>&1
import socket
from src.db.session import engine
socket.getaddrinfo("api.the-odds-api.com", 443)
engine.connect().close()
PY
        then
            return 0
        fi
        sleep 30
    done
    echo "ERROR: no network after 5 minutes; nothing was run (no credits spent)"
    return 1
}

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    wait_for_network
    /usr/bin/caffeinate -is "$PY" -m scripts.snapshot_closing
} >> logs/snapshot_closing.log 2>&1
