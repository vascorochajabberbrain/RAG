#!/bin/bash
# jB RAG Builder DEV launcher
# Runs the latest code (worktree or main repo) on port 8001.
# Auto-detects the most recently modified Claude worktree.
# Both PROD (port 8000) and DEV (port 8001) can run simultaneously.

MAIN_PROJECT="/Users/johanahlund/PROGRAMMING/RAG"
WORKTREE_DIR="$MAIN_PROJECT/.claude/worktrees"
PORT=8001
URL="http://127.0.0.1:$PORT"

# Find the most recently modified worktree (if any exist)
PROJECT="$MAIN_PROJECT"
if [ -d "$WORKTREE_DIR" ]; then
    LATEST=$(ls -td "$WORKTREE_DIR"/*/web 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        PROJECT="$(dirname "$LATEST")"
    fi
fi

VENV_PYTHON="$MAIN_PROJECT/.venv/bin/python"
LOG="$MAIN_PROJECT/.rag_server_dev.log"

echo "DEV server using: $PROJECT"
echo "Port: $PORT"

# Check if DEV server is already running
if curl -s --max-time 1 "$URL/api/version" > /dev/null 2>&1; then
    open "$URL"
    exit 0
fi

# Kill anything stale on the DEV port (Python/uvicorn only)
for PID in $(lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null); do
    CMD=$(ps -p "$PID" -o comm= 2>/dev/null | tr '[:upper:]' '[:lower:]')
    if echo "$CMD" | grep -qE "python|uvicorn"; then
        kill -9 "$PID" 2>/dev/null
    fi
done
sleep 0.5

# Start DEV server in background with RAG_DEV_MODE flag
cd "$PROJECT"
RAG_DEV_MODE=1 nohup "$VENV_PYTHON" run_app_server.py $PORT > "$LOG" 2>&1 &

# Wait for server to be ready (up to 15 seconds)
for i in $(seq 1 30); do
    if curl -s --max-time 1 "$URL/api/version" > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

open "$URL"
