#!/usr/bin/env bash
# Install (or reinstall) the weekly snapshot job on macOS with launchd:
# scripts/snapshot_odds.sh every Friday at 9:00. If the Mac is asleep then,
# launchd runs it as soon as the Mac wakes. Only ONE machine on the team
# should have this installed (it spends Odds API credits).
#
#   scripts/install_snapshot_schedule.sh            # install
#   scripts/install_snapshot_schedule.sh --remove   # uninstall

set -euo pipefail

LABEL="com.justbetit.snapshot-odds"
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

# Unload any existing copy first so reinstalling picks up changes.
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true

if [ "${1:-}" = "--remove" ]; then
    rm -f "$PLIST"
    echo "Removed $LABEL"
    exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT/logs"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$PROJECT/scripts/snapshot_odds.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>5</integer>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <!-- Errors from before the script starts logging (e.g. a bad path) -->
    <key>StandardOutPath</key>
    <string>$PROJECT/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT/logs/launchd.log</string>
</dict>
</plist>
EOF

plutil -lint "$PLIST" >/dev/null
launchctl bootstrap "$DOMAIN" "$PLIST"

echo "Installed $LABEL: runs $PROJECT/scripts/snapshot_odds.sh every Friday at 9:00"
echo "Log: $PROJECT/logs/snapshot_odds.log"
