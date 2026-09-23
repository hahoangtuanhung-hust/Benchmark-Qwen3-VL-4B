#!/usr/bin/env bash
# =============================================================================
# Start llama-server for benchmark
# See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §10, §11
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Required arguments ---
MODEL_PATH="${1:?Usage: start_server.sh <model_path> <mmproj_path> <num_slots> [context_per_slot] [port]}"
MMPROJ_PATH="${2:?Usage: start_server.sh <model_path> <mmproj_path> <num_slots> [context_per_slot] [port]}"
NUM_SLOTS="${3:?Usage: start_server.sh <model_path> <mmproj_path> <num_slots> [context_per_slot] [port]}"

# --- Optional arguments ---
CONTEXT_PER_SLOT="${4:-8192}"
PORT="${5:-8080}"
HOST="${HOST:-127.0.0.1}"

# --- Paths ---
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_DIR/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_DIR/build/bin/llama-server}"
LOG_DIR="$PROJECT_DIR/results/logs"
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/llama_server_$(date +%Y%m%d_%H%M%S).log"

# --- Calculate total context ---
TOTAL_CONTEXT=$((CONTEXT_PER_SLOT * NUM_SLOTS))

echo "============================================="
echo " STARTING LLAMA-SERVER"
echo "============================================="
echo "  Model:    $MODEL_PATH"
echo "  mmproj:   $MMPROJ_PATH"
echo "  Slots:    $NUM_SLOTS (CCU=$NUM_SLOTS)"
echo "  Context:  $CONTEXT_PER_SLOT per slot ($TOTAL_CONTEXT total)"
echo "  Host:     $HOST:$PORT"
echo "  Log:      $SERVER_LOG"
echo "============================================="

# --- Build server command ---
SERVER_CMD=(
    "$LLAMA_SERVER"
    --model "$MODEL_PATH"
    --mmproj "$MMPROJ_PATH"
    --host "$HOST"
    --port "$PORT"
    --n-gpu-layers -1          # offload all layers (§10)
    --ctx-size "$TOTAL_CONTEXT"
    --parallel "$NUM_SLOTS"     # -np
    --batch-size 512            # (§10)
    --ubatch-size 256           # (§10)
    # Current llama.cpp expects an explicit value (on|off|auto).  Passing the
    # flag without a value makes the next option (for example --metrics) get
    # parsed as its value and causes the server to exit immediately.
    --flash-attn on             # (§10)
    --metrics                   # (§22)
    --slots                     # (§22)
    --no-cache-prompt           # (§12)
)

# Add --cache-type-k and --cache-type-v if supported
# Default: f16 for both (§20)
SERVER_CMD+=(
    --cache-type-k f16
    --cache-type-v f16
)

echo "[INFO] Server command:"
echo "  ${SERVER_CMD[*]}"
echo ""

# --- Kill any existing server ---
if pgrep -f "llama-server" > /dev/null 2>&1; then
    echo "[WARN] Existing llama-server found, killing..."
    pkill -f "llama-server" || true
    sleep 2
fi

# --- Start server in background ---
"${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[INFO] Server started with PID: $SERVER_PID"
echo "$SERVER_PID" > "$LOG_DIR/server.pid"
echo "$SERVER_LOG" > "$LOG_DIR/server_log_path.txt"

# --- Wait briefly to check for immediate crash ---
sleep 3
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ERROR] Server crashed immediately. Check log: $SERVER_LOG"
    tail -20 "$SERVER_LOG"
    exit 1
fi

echo "[INFO] Server appears to be running. Use wait_for_server.py to confirm health."
echo "[INFO] PID file: $LOG_DIR/server.pid"
