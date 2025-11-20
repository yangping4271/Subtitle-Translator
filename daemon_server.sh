#!/bin/bash

# Subtitle Translator Server Daemon Script
# Keeps the server running in the background

set -e  # Exit on error

echo "🚀 Subtitle Translator Server Daemon Starting..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOGS_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOGS_DIR/subtitle_server.pid"

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️  Server is already running (PID: $OLD_PID)"
        echo "   Use './stop_daemon.sh' to stop it first"
        exit 1
    else
        echo "🧹 Cleaning up stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Check for uv
if command -v uv &> /dev/null; then
    echo "🐍 Using uv environment"
    CMD="uv run python -m subtitle_translator.cli serve"
else
    echo "❌ uv not found. Please install uv."
    exit 1
fi

echo "🌐 Starting background daemon..."
echo "📂 Working Dir: $PROJECT_ROOT"
echo "📋 Log File: $LOGS_DIR/subtitle_server.log"
echo "🆔 PID File: $PID_FILE"
echo ""

# Start server with nohup
nohup $CMD \
    > "$LOGS_DIR/subtitle_server.log" \
    2> "$LOGS_DIR/subtitle_server_error.log" &

# Record PID
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait for startup
sleep 3

# Verify startup
if kill -0 "$SERVER_PID" 2>/dev/null; then
    # Test response
    if curl -s "http://127.0.0.1:8888/health" >/dev/null 2>&1; then
        echo "✅ Daemon started successfully!"
        echo "🆔 PID: $SERVER_PID"
        echo "🌐 Address: http://127.0.0.1:8888"
        echo "📋 Logs: tail -f $LOGS_DIR/subtitle_server.log"
        echo ""
        echo "🎉 Server is running in background."
    else
        echo "❌ Server process started but not responding."
        echo "📋 Error Log: cat $LOGS_DIR/subtitle_server_error.log"
        kill "$SERVER_PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        exit 1
    fi
else
    echo "❌ Server failed to start."
    echo "📋 Error Log: cat $LOGS_DIR/subtitle_server_error.log"
    rm -f "$PID_FILE"
    exit 1
fi
