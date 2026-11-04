#!/usr/bin/env bash
# Install (or reinstall) the scheduled jobs on macOS with launchd:
#   com.justbetit.snapshot-odds     scripts/snapshot_odds.sh every Friday at 9:00
#   com.justbetit.snapshot-closing  scripts/snapshot_closing.sh Friday 16:00 and
#                                   Saturday 10:00, 13:00, 16:00, 19:00
# If the Mac is asleep at a scheduled time, launchd runs the job as soon as
# the Mac wakes. Only ONE machine on the team should have these installed
# (they spend Odds API credits).
#
#   scripts/install_snapshot_schedule.sh            # install
#   scripts/install_snapshot_schedule.sh --remove   # uninstall

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
DOMAIN="gui/$(id -u)"

# calendar_entry WEEKDAY HOUR  (launchd weekdays: 0 = Sunday ... 5 = Friday, 6 = Saturday)
calendar_entry() {
    printf '        <dict><key>Weekday</key><integer>%s</integer><key>Hour</key><integer>%s</integer><key>Minute</key><integer>0</integer></dict>\n' "$1" "$2"
}

# install_job LABEL SCRIPT CALENDAR_ENTRIES
install_job() {
    local label="$1" script="$2" entries="$3"
    local plist="$HOME/Library/LaunchAgents/$label.plist"

    # Unload any existing copy first so reinstalling picks up changes.
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    if [ "$REMOVE" = 1 ]; then
        rm -f "$plist"
        echo "Removed $label"
        return
    fi

    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$PROJECT/scripts/$script</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
$entries
    </array>
    <!-- Errors from before the script starts logging (e.g. a bad path) -->
    <key>StandardOutPath</key>
    <string>$PROJECT/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT/logs/launchd.log</string>
</dict>
</plist>
EOF
    plutil -lint "$plist" >/dev/null
    launchctl bootstrap "$DOMAIN" "$plist"
    echo "Installed $label: $PROJECT/scripts/$script"
}

REMOVE=0
[ "${1:-}" = "--remove" ] && REMOVE=1
mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT/logs"

install_job com.justbetit.snapshot-odds snapshot_odds.sh "$(calendar_entry 5 9)"
install_job com.justbetit.snapshot-closing snapshot_closing.sh "$(
    calendar_entry 5 16; calendar_entry 6 10; calendar_entry 6 13; calendar_entry 6 16; calendar_entry 6 19)"

if [ "$REMOVE" = 0 ]; then
    echo "Friday 9:00 picks job log: $PROJECT/logs/snapshot_odds.log"
    echo "Near-kickoff snapshots (Fri 16:00; Sat 10, 13, 16, 19:00) log: $PROJECT/logs/snapshot_closing.log"
fi
