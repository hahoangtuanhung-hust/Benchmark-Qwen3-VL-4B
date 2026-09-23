#!/usr/bin/env bash
# =============================================================================
# Qwen3-VL-4B Benchmark — Master Orchestrator
# See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §31, §41
#
# Usage:
#   ./run_benchmark.sh                  # Full benchmark (all phases)
#   ./run_benchmark.sh --smoke          # Smoke test only (5 requests)
#   ./run_benchmark.sh --phase 6        # Start from specific phase
#   ./run_benchmark.sh --model Q4_K_M   # Single model variant
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configuration ---
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_DIR/build/bin/llama-server}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8080}"
PORT="${PORT:-8080}"
MODELS_DIR="${MODELS_DIR:-./models}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
CONTEXT_PER_SLOT="${CONTEXT_PER_SLOT:-8192}"
GPU_MONITOR_PID=""

# --- Parse arguments ---
SMOKE_MODE=false
START_PHASE=0
SINGLE_MODEL=""
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_MODE=true; shift ;;
        --phase) START_PHASE="$2"; shift 2 ;;
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Utility functions ---
log_phase() {
    echo ""
    echo "============================================="
    echo " PHASE $1: $2"
    echo "============================================="
}

check_success() {
    if [ $? -ne 0 ]; then
        echo "[ERROR] Phase failed: $1"
        echo "[INFO] You can resume from this phase with: ./run_benchmark.sh --phase $2"
        exit 1
    fi
}

start_gpu_monitor() {
    local label="$1"
    local gpu_log="$RESULTS_DIR/gpu/gpu_${label}_$(date +%Y%m%d_%H%M%S).csv"
    mkdir -p "$RESULTS_DIR/gpu"
    python3 scripts/gpu_monitor.py --output "$gpu_log" --interval 200 &
    GPU_MONITOR_PID=$!
    echo "[INFO] GPU monitor started (PID=$GPU_MONITOR_PID, log=$gpu_log)"
}

stop_gpu_monitor() {
    if [ -n "$GPU_MONITOR_PID" ] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        echo "[INFO] GPU monitor stopped."
        GPU_MONITOR_PID=""
    fi
}

cleanup() {
    stop_gpu_monitor
    bash scripts/stop_server.sh 2>/dev/null || true
}
trap cleanup EXIT

# --- Detect available models ---
detect_models() {
    local models=()
    # Scan all subdirectories under models/ for .gguf files (excluding mmproj and base_hf_cache)
    for dir in models/*/; do
        local dirname=$(basename "$dir")
        # Skip mmproj (vision encoder) and HF cache directories
        if [ "$dirname" = "mmproj" ] || [ "$dirname" = "base_hf_cache" ]; then
            continue
        fi
        if [ -d "$dir" ]; then
            for gguf in "$dir"*.gguf; do
                if [ -f "$gguf" ]; then
                    models+=("$gguf")
                fi
            done
        fi
    done
    echo "${models[@]}"
}

# --- Get model label from path ---
get_model_label() {
    local path="$1"
    local filename=$(basename "$path" .gguf)
    # Extract quant type from filename (order matters — check longer names first)
    if echo "$filename" | grep -qi "Q5_K_M"; then echo "Q5_K_M"
    elif echo "$filename" | grep -qi "Q4_K_M"; then echo "Q4_K_M"
    elif echo "$filename" | grep -qi "Q4_K_S"; then echo "Q4_K_S"
    elif echo "$filename" | grep -qi "Q8_0"; then echo "Q8_0"
    elif echo "$filename" | grep -qi "Q6_K"; then echo "Q6_K"
    elif echo "$filename" | grep -qi "Q4_0"; then echo "Q4_0"
    elif echo "$filename" | grep -qi "Q4_1"; then echo "Q4_1"
    elif echo "$filename" | grep -qi "IQ4_XS"; then echo "IQ4_XS"
    elif echo "$filename" | grep -qi "F16"; then echo "F16"
    elif echo "$filename" | grep -qi "BF16"; then echo "BF16"
    else echo "$filename"
    fi
}

# --- Find mmproj ---
find_mmproj() {
    local mmproj_dir="models/mmproj"
    if [ -d "$mmproj_dir" ]; then
        local mmproj=$(find "$mmproj_dir" -name "*.gguf" | head -1)
        echo "$mmproj"
    fi
}

# =============================================================================
# PHASE 0 — Collect Environment (§5)
# =============================================================================
if [ "$START_PHASE" -le 0 ]; then
    log_phase 0 "Collect Environment"
    bash scripts/collect_environment.sh
    check_success "collect_environment" 0
fi

# =============================================================================
# PHASE 1 — Build llama.cpp (§6)
# =============================================================================
if [ "$START_PHASE" -le 1 ] && [ "$SKIP_BUILD" = false ]; then
    log_phase 1 "Build/Validate llama.cpp"
    if [ ! -f "$LLAMA_SERVER" ]; then
        bash scripts/build_llamacpp.sh
        check_success "build_llamacpp" 1
    else
        echo "[INFO] llama-server already built: $LLAMA_SERVER"
    fi
fi

# =============================================================================
# PHASE 2 — Download & Convert Hugging Face Models
# =============================================================================
if [ "$START_PHASE" -le 2 ]; then
    log_phase 2 "Download & Convert HF Models"
    bash scripts/download_and_convert_hf.sh "$SINGLE_MODEL"
    check_success "download_and_convert_hf" 2
fi

# =============================================================================
# PHASE 3 — Build Compatibility Matrix (§2)
# =============================================================================
if [ "$START_PHASE" -le 3 ]; then
    log_phase 3 "Build Compatibility Matrix"
    python3 scripts/inspect_model.py --project-dir .
    check_success "inspect_model" 3
fi

# =============================================================================
# PHASE 4 — Validate Model Artifacts (§7)
# =============================================================================
if [ "$START_PHASE" -le 4 ]; then
    log_phase 4 "Validate Model Artifacts"
    echo "[INFO] Model artifacts validated in Phase 3 (inspect_model.py)"
    echo "[INFO] Check results/model_manifest.csv for details."
fi

# =============================================================================
# PHASE 5 — Prepare Dataset (§8)
# =============================================================================
if [ "$START_PHASE" -le 5 ]; then
    log_phase 5 "Prepare/Freeze Benchmark Dataset"
    if [ ! -f "benchmark_data/manifest.jsonl" ]; then
        python3 scripts/prepare_dataset.py --project-dir .
        check_success "prepare_dataset" 5
    else
        echo "[INFO] Dataset already prepared. Manifest: benchmark_data/manifest.jsonl"
    fi
fi

# =============================================================================
# PHASE 6 — Freeze Config (§39)
# =============================================================================
if [ "$START_PHASE" -le 6 ]; then
    log_phase 6 "Freeze Benchmark Config"
    mkdir -p "$RESULTS_DIR"
    cp configs/benchmark.yaml "$RESULTS_DIR/config_snapshot.yaml"
    echo "BENCHMARK CONFIG FROZEN"
    echo "config_frozen_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RESULTS_DIR/config_snapshot.yaml"
fi

# =============================================================================
# PHASE 7 — Run CCU1 Benchmark (§14: S1, S2, S3)
# =============================================================================
if [ "$START_PHASE" -le 7 ]; then
    log_phase 7 "Run CCU1 Benchmark"

    MMPROJ=$(find_mmproj)
    if [ -z "$MMPROJ" ]; then
        echo "[ERROR] No mmproj GGUF found in models/mmproj/"
        echo "[INFO] Download Qwen3-VL mmproj and place in models/mmproj/"
        exit 1
    fi
    echo "[INFO] mmproj: $MMPROJ"

    MODEL_FILES=($(detect_models))
    if [ ${#MODEL_FILES[@]} -eq 0 ]; then
        echo "[ERROR] No model GGUF files found in models/"
        exit 1
    fi

    for MODEL_PATH in "${MODEL_FILES[@]}"; do
        MODEL_LABEL=$(get_model_label "$MODEL_PATH")

        # Filter if single model specified
        if [ -n "$SINGLE_MODEL" ] && [ "$MODEL_LABEL" != "$SINGLE_MODEL" ]; then
            continue
        fi

        echo ""
        echo "--- MODEL: $MODEL_LABEL ($MODEL_PATH) ---"

        # Start server with CCU1 config
        bash scripts/start_server.sh "$MODEL_PATH" "$MMPROJ" 1 "$CONTEXT_PER_SLOT" "$PORT"

        # Wait for server health
        python3 scripts/wait_for_server.py --port "$PORT" --timeout 300
        if [ $? -ne 0 ]; then
            echo "[WARN] Server failed to start for $MODEL_LABEL. Marking as FAIL."
            bash scripts/stop_server.sh
            continue
        fi

        # Start GPU monitor
        start_gpu_monitor "${MODEL_LABEL}_ccu1"

        # Collect metrics before
        python3 scripts/collect_server_metrics.py --server-url "$SERVER_URL" \
            --output-dir "$RESULTS_DIR/server_metrics" --label "${MODEL_LABEL}_ccu1_before"

        # Run benchmark
        SCENARIOS="S1 S2 S3"
        if [ "$SMOKE_MODE" = true ]; then
            SCENARIOS="S2"
        fi

        python3 scripts/benchmark_ccu1.py \
            --project-dir . \
            --server-url "$SERVER_URL" \
            --model-label "$MODEL_LABEL" \
            --actual-quant "$MODEL_LABEL" \
            --context-per-slot "$CONTEXT_PER_SLOT" \
            --scenarios $SCENARIOS \
            --output results/raw/requests.csv

        # Collect metrics after
        python3 scripts/collect_server_metrics.py --server-url "$SERVER_URL" \
            --output-dir "$RESULTS_DIR/server_metrics" --label "${MODEL_LABEL}_ccu1_after"

        # Stop
        stop_gpu_monitor
        bash scripts/stop_server.sh

        echo "[INFO] CCU1 complete for $MODEL_LABEL"
        echo ""
    done
fi

# =============================================================================
# PHASE 8 — Run CCU2 Benchmark (§14: S4)
# =============================================================================
if [ "$START_PHASE" -le 8 ]; then
    log_phase 8 "Run CCU2 Benchmark"

    MMPROJ=$(find_mmproj)
    MODEL_FILES=($(detect_models))

    for MODEL_PATH in "${MODEL_FILES[@]}"; do
        MODEL_LABEL=$(get_model_label "$MODEL_PATH")

        if [ -n "$SINGLE_MODEL" ] && [ "$MODEL_LABEL" != "$SINGLE_MODEL" ]; then
            continue
        fi

        echo ""
        echo "--- MODEL: $MODEL_LABEL (CCU2) ---"

        # Start server with CCU2 config (-np 2)
        bash scripts/start_server.sh "$MODEL_PATH" "$MMPROJ" 2 "$CONTEXT_PER_SLOT" "$PORT"

        python3 scripts/wait_for_server.py --port "$PORT" --timeout 300
        if [ $? -ne 0 ]; then
            echo "[WARN] Server failed for CCU2 $MODEL_LABEL."
            bash scripts/stop_server.sh
            continue
        fi

        start_gpu_monitor "${MODEL_LABEL}_ccu2"

        # Collect /slots to verify 2 slots (§22)
        python3 scripts/collect_server_metrics.py --server-url "$SERVER_URL" \
            --output-dir "$RESULTS_DIR/server_metrics" --label "${MODEL_LABEL}_ccu2_before"

        PAIRS=30
        if [ "$SMOKE_MODE" = true ]; then
            PAIRS=2
        fi

        python3 scripts/benchmark_ccu2.py \
            --project-dir . \
            --server-url "$SERVER_URL" \
            --model-label "$MODEL_LABEL" \
            --actual-quant "$MODEL_LABEL" \
            --context-per-slot "$CONTEXT_PER_SLOT" \
            --pairs "$PAIRS" \
            --output results/raw/requests.csv

        python3 scripts/collect_server_metrics.py --server-url "$SERVER_URL" \
            --output-dir "$RESULTS_DIR/server_metrics" --label "${MODEL_LABEL}_ccu2_after"

        stop_gpu_monitor
        bash scripts/stop_server.sh

        echo "[INFO] CCU2 complete for $MODEL_LABEL"
    done
fi

# =============================================================================
# PHASE 9 — Run Accuracy (§9, §34)
# =============================================================================
if [ "$START_PHASE" -le 9 ]; then
    log_phase 9 "Run Accuracy Benchmark"

    MMPROJ=$(find_mmproj)
    MODEL_FILES=($(detect_models))

    # GPU must be idle before accuracy (§34)
    echo "[INFO] Ensuring GPU is idle before accuracy suite..."
    sleep 5

    for MODEL_PATH in "${MODEL_FILES[@]}"; do
        MODEL_LABEL=$(get_model_label "$MODEL_PATH")

        if [ -n "$SINGLE_MODEL" ] && [ "$MODEL_LABEL" != "$SINGLE_MODEL" ]; then
            continue
        fi

        echo ""
        echo "--- ACCURACY: $MODEL_LABEL ---"

        bash scripts/start_server.sh "$MODEL_PATH" "$MMPROJ" 1 "$CONTEXT_PER_SLOT" "$PORT"
        python3 scripts/wait_for_server.py --port "$PORT" --timeout 300
        if [ $? -ne 0 ]; then
            echo "[WARN] Server failed for accuracy $MODEL_LABEL."
            bash scripts/stop_server.sh
            continue
        fi

        ACCURACY_MODE="quick"
        if [ "$SMOKE_MODE" = true ]; then
            ACCURACY_MODE="quick"
        fi

        python3 scripts/run_accuracy.py \
            --project-dir . \
            --server-url "$SERVER_URL" \
            --model-label "$MODEL_LABEL" \
            --actual-quant "$MODEL_LABEL" \
            --mode "$ACCURACY_MODE"

        bash scripts/stop_server.sh
    done
fi

# =============================================================================
# PHASE 10 — Validate Results (§27, §28)
# =============================================================================
if [ "$START_PHASE" -le 10 ]; then
    log_phase 10 "Validate Results"
    python3 scripts/validate_results.py --project-dir .
fi

# =============================================================================
# PHASE 11 — Generate Report & Charts (§35, §36)
# =============================================================================
if [ "$START_PHASE" -le 11 ]; then
    log_phase 11 "Generate Report & Charts"
    python3 scripts/summarize.py --project-dir .

    echo ""
    echo "============================================="
    echo " BENCHMARK COMPLETE"
    echo "============================================="
    echo "  Results directory: $RESULTS_DIR/"
    echo "  Report: $RESULTS_DIR/report/benchmark_summary.md"
    echo "  Summary CSV: $RESULTS_DIR/report/summary.csv"
    echo "  Charts: $RESULTS_DIR/charts/"
    echo "  Raw data: $RESULTS_DIR/raw/"
    echo "============================================="
fi
