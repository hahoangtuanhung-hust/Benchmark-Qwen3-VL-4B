#!/usr/bin/env bash
# =============================================================================
# Collect Environment Information
# See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §5
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/results/environment"

mkdir -p "$ENV_DIR"

echo "============================================="
echo " COLLECTING ENVIRONMENT INFORMATION"
echo "============================================="

# --- Main environment file ---
ENV_FILE="$ENV_DIR/environment.txt"
{
    echo "========================================"
    echo " ENVIRONMENT SNAPSHOT"
    echo " Collected: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "========================================"

    echo ""
    echo "--- date ---"
    date

    echo ""
    echo "--- uname -a ---"
    uname -a || echo "N/A"

    echo ""
    echo "--- lscpu ---"
    lscpu 2>/dev/null || echo "N/A (not Linux or lscpu not found)"

    echo ""
    echo "--- free -h ---"
    free -h 2>/dev/null || echo "N/A"

    echo ""
    echo "--- nvidia-smi ---"
    nvidia-smi 2>/dev/null || echo "N/A (no NVIDIA GPU or nvidia-smi not found)"

    echo ""
    echo "--- nvcc --version ---"
    nvcc --version 2>/dev/null || echo "N/A"

    echo ""
    echo "--- cmake --version ---"
    cmake --version 2>/dev/null || echo "N/A"

    echo ""
    echo "--- gcc --version ---"
    gcc --version 2>/dev/null || echo "N/A"

    echo ""
    echo "--- python --version ---"
    python3 --version 2>/dev/null || python --version 2>/dev/null || echo "N/A"

    echo ""
    echo "--- git --version ---"
    git --version 2>/dev/null || echo "N/A"

    echo ""
    echo "--- llama.cpp git commit ---"
    LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_DIR/llama.cpp}"
    if [ -d "$LLAMA_CPP_DIR/.git" ]; then
        (cd "$LLAMA_CPP_DIR" && git rev-parse HEAD)
        (cd "$LLAMA_CPP_DIR" && git log -1 --format="%H %ci %s")
    else
        echo "N/A (llama.cpp not cloned yet)"
    fi

} > "$ENV_FILE"
echo "[INFO] Environment saved to: $ENV_FILE"

# --- nvidia-smi detailed ---
NVIDIA_FILE="$ENV_DIR/nvidia_smi.txt"
{
    echo "--- nvidia-smi ---"
    nvidia-smi 2>/dev/null || echo "N/A"
    echo ""
    echo "--- nvidia-smi -q ---"
    nvidia-smi -q 2>/dev/null || echo "N/A"
} > "$NVIDIA_FILE"
echo "[INFO] nvidia-smi saved to: $NVIDIA_FILE"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_DIR/llama.cpp}"

#echo "--- llama-server Help ---"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_DIR/build/bin/llama-server}"
if [ -f "$LLAMA_SERVER" ]; then
    "$LLAMA_SERVER" --help > "$ENV_DIR/llama_server_help.txt" 2>&1
    echo "  [OK] Saved llama_server_help.txt"
    
    echo "--- llama-server CUDA Devices ---"
    "$LLAMA_SERVER" --list-devices > "$ENV_DIR/llama_server_devices.txt" 2>&1 || true
    echo "  [OK] Saved llama_server_devices.txt"
else
    echo "  [WARN] llama-server not found at $LLAMA_SERVER"
fi

echo ""
echo "--- llama-quantize Help ---"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"
QUANTIZE_HELP_FILE="$ENV_DIR/llama_quantize_help.txt"
{
    echo "--- llama-quantize --help ---"
    "$LLAMA_QUANTIZE" --help 2>/dev/null || echo "N/A (llama-quantize not built yet)"
} > "$QUANTIZE_HELP_FILE" 2>&1
echo "[INFO] llama-quantize help saved to: $QUANTIZE_HELP_FILE"

# --- llama-server --list-devices ---
DEVICES_FILE="$ENV_DIR/llama_server_devices.txt"
{
    echo "--- llama-server --list-devices ---"
    "$LLAMA_SERVER" --list-devices 2>/dev/null || echo "N/A"
} > "$DEVICES_FILE" 2>&1
echo "[INFO] Device list saved to: $DEVICES_FILE"

echo ""
echo "[SUCCESS] Environment collection complete."
echo "  Output directory: $ENV_DIR"
ls -la "$ENV_DIR"
