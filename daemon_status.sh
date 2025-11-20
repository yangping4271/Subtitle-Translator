#!/bin/bash

# Subtitle Translator Server Status Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOGS_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOGS_DIR/subtitle_server.pid"

echo "📊 Subtitle Translator Server Status"
echo "════════════════════════════════════════"
echo ""

# Check PID file
if [ -f "$PID_FILE" ]; then
    SERVER_PID=$(cat "$PID_FILE")
    echo "🆔 PID File: $SERVER_PID"
    
    # Check process
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "✅ Status: Running"
        
        # Process info
        echo "📋 Process Info:"
        ps -p "$SERVER_PID" -o pid,ppid,pcpu,pmem,etime,comm,args 2>/dev/null || echo "   Cannot get process info"
        
    else
        echo "❌ Status: Stopped (Stale PID file)"
        echo "🧹 Cleaning up..."
        rm -f "$PID_FILE"
    fi
else
    echo "📭 PID File: None"
    
    # Find potential processes
    PIDS=$(pgrep -f "subtitle_translator.cli serve")
    if [ ! -z "$PIDS" ]; then
        echo "⚠️  Found unmanaged process(es): $PIDS"
    else
        echo "❌ Status: Not Running"
    fi
fi

echo ""

# Check Port
echo "🌐 Port Check:"
if lsof -Pi :8888 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✅ Port 8888: Listening"
else
    echo "❌ Port 8888: Not Listening"
fi

echo ""

# Check Health
echo "🔍 Health Check:"
if curl -s --connect-timeout 3 "http://127.0.0.1:8888/health" >/dev/null 2>&1; then
    echo "✅ HTTP Health: OK"
    
    # Get info
    SERVICE_INFO=$(curl -s "http://127.0.0.1:8888/health" 2>/dev/null)
    if [ ! -z "$SERVICE_INFO" ]; then
        echo "📊 Service Info:"
        echo "$SERVICE_INFO" | python3 -m json.tool 2>/dev/null || echo "$SERVICE_INFO"
    fi
else
    echo "❌ HTTP Health: Failed"
fi

echo ""

# Check Logs
echo "📋 Logs:"
if [ -f "$LOGS_DIR/subtitle_server.log" ]; then
    LOG_LINES=$(wc -l < "$LOGS_DIR/subtitle_server.log" 2>/dev/null || echo "0")
    echo "✅ Main Log: $LOG_LINES lines"
    echo "   Path: $LOGS_DIR/subtitle_server.log"
    echo "   Latest:"
    tail -3 "$LOGS_DIR/subtitle_server.log" 2>/dev/null | sed 's/^/      /' || echo "      (Cannot read)"
else
    echo "❌ Main Log: Not found"
fi

if [ -f "$LOGS_DIR/subtitle_server_error.log" ]; then
    ERROR_SIZE=$(stat -f%z "$LOGS_DIR/subtitle_server_error.log" 2>/dev/null || echo "0")
    if [ "$ERROR_SIZE" -gt 0 ]; then
        echo "⚠️  Error Log: Has content ($ERROR_SIZE bytes)"
        echo "   Latest Error:"
        tail -3 "$LOGS_DIR/subtitle_server_error.log" 2>/dev/null | sed 's/^/      /' || echo "      (Cannot read)"
    else
        echo "✅ Error Log: Empty"
    fi
else
    echo "❌ Error Log: Not found"
fi
