#!/bin/bash
# jB RAG Builder launcher
# Smart: if server already running → just open browser.
# If not running → start server in background, then open browser.

PROJECT="/Users/johanahlund/PROGRAMMING/RAG"
VENV_PYTHON="$PROJECT/.venv/bin/python"
URL="http://127.0.0.1:8000"
LOG="$PROJECT/.rag_server.log"

# Check if server is already up (responding to HTTP)
if curl -s --max-time 1 "$URL/api/version" > /dev/null 2>&1; then
    # Already running — just bring browser to front
    open "$URL"
    exit 0
fi

# Not running — kill anything stale on port 8000 (Python/uvicorn only)
for PID in $(lsof -ti :8000 -sTCP:LISTEN 2>/dev/null); do
    CMD=$(ps -p "$PID" -o comm= 2>/dev/null | tr '[:upper:]' '[:lower:]')
    if echo "$CMD" | grep -qE "python|uvicorn"; then
        kill -9 "$PID" 2>/dev/null
    fi
done
sleep 0.5

# Start server in background
cd "$PROJECT"
nohup "$VENV_PYTHON" run_app_server.py > "$LOG" 2>&1 &

# Wait for server to be ready (up to 15 seconds)
for i in $(seq 1 30); do
    if curl -s --max-time 1 "$URL/api/version" > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

# Open browser
open "$URL"
