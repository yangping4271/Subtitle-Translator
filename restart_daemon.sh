#!/bin/bash

# Subtitle Translator Server Restart Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Restarting Subtitle Translator Server..."

"$SCRIPT_DIR/stop_daemon.sh"
sleep 2
"$SCRIPT_DIR/daemon_server.sh"
