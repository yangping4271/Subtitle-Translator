#!/bin/bash

# Subtitle Translator Server Stop Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOGS_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOGS_DIR/subtitle_server.pid"

echo "🛑 Stopping Subtitle Translator Server..."

if [ -f "$PID_FILE" ]; then
    SERVER_PID=$(cat "$PID_FILE")
    echo "🆔 PID: $SERVER_PID"
    
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        echo "✅ Stop signal sent."
        
        # Wait for process to exit
        for i in {1..5}; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "⚠️  Process did not stop gracefully, forcing kill..."
            kill -9 "$SERVER_PID"
        fi
        
        rm -f "$PID_FILE"
        echo "✅ Server stopped."
    else
        echo "⚠️  Process not running."
        rm -f "$PID_FILE"
    fi
else
    echo "⚠️  PID file not found."
    
    # Try to find by name
    PIDS=$(pgrep -f "subtitle_translator.cli serve")
    if [ ! -z "$PIDS" ]; then
        echo "⚠️  Found unmanaged process(es): $PIDS"
        echo "❓ Kill them? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
            kill $PIDS
            echo "✅ Killed processes."
        fi
    else
        echo "✅ No server processes found."
    fi
fi
