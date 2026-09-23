#!/usr/bin/env bash
# =============================================================================
# Stop llama-server gracefully
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/results/logs"
PID_FILE="$LOG_DIR/server.pid"

echo "[INFO] Stopping llama-server..."

if [ -f "$PID_FILE" ]; then
    SERVER_PID=$(cat "$PID_FILE")
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[INFO] Sending SIGTERM to PID $SERVER_PID..."
        kill "$SERVER_PID"

        # Wait up to 10 seconds for graceful shutdown
        for i in $(seq 1 10); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "[INFO] Server stopped gracefully."
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
        done

        # Force kill
        echo "[WARN] Server did not stop gracefully. Sending SIGKILL..."
        kill -9 "$SERVER_PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "[INFO] Server force-killed."
    else
        echo "[INFO] Server PID $SERVER_PID is not running."
        rm -f "$PID_FILE"
    fi
else
    echo "[INFO] No PID file found. Checking for running llama-server processes..."
    if pgrep -f "llama-server" > /dev/null 2>&1; then
        echo "[WARN] Found running llama-server. Killing..."
        pkill -f "llama-server" || true
        sleep 2
        echo "[INFO] Done."
    else
        echo "[INFO] No llama-server process found."
    fi
fi

# Brief GPU cooldown
echo "[INFO] Waiting 3s for GPU cooldown..."
sleep 3
echo "[INFO] Server stop complete."
