#!/usr/bin/env bash
# =============================================================================
# Build llama.cpp with CUDA support
# See: docs/qwen3_vl_4b_llamacpp_benchmark_agent_prompt.md §6
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_common.sh" 2>/dev/null || true

# --- Configuration ---
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_DIR/llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
BUILD_DIR="$LLAMA_CPP_DIR/build"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
CUDA_ARCH="${CUDA_ARCH:-}"  # e.g., "75" for T4, empty = auto-detect

echo "============================================="
echo " BUILD LLAMA.CPP WITH CUDA"
echo "============================================="

# --- Clone if not exists ---
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "[INFO] Cloning llama.cpp..."
    git clone --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
else
    echo "[INFO] llama.cpp directory exists: $LLAMA_CPP_DIR"
fi

cd "$LLAMA_CPP_DIR"
LLAMA_COMMIT=$(git rev-parse HEAD)
echo "[INFO] llama.cpp commit: $LLAMA_COMMIT"

# --- Prepare CMake flags ---
CMAKE_FLAGS=(
    -DGGML_CUDA=ON
    -DCMAKE_BUILD_TYPE=Release
)

# Pin CUDA architecture if specified
if [ -n "$CUDA_ARCH" ]; then
    echo "[INFO] Pinning CUDA architecture: $CUDA_ARCH"
    CMAKE_FLAGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi

# --- Attempt standard build ---
echo "[INFO] Configuring CMake (standard)..."
if ! cmake -S . -B build "${CMAKE_FLAGS[@]}" 2>&1; then
    echo "[WARN] Standard build failed. Trying Kaggle compatibility mode..."

    # Kaggle compatibility mode (§6)
    CMAKE_FLAGS+=(
        -DGGML_CUDA_NO_VMM=ON
        -DGGML_CUDA_NCCL=OFF
    )

    echo "[INFO] Configuring CMake (Kaggle compatibility)..."
    rm -rf build
    cmake -S . -B build "${CMAKE_FLAGS[@]}"
fi

# --- Build ---
echo "[INFO] Building llama.cpp (jobs=$BUILD_JOBS)..."
cmake --build build \
    --target llama-server llama-cli llama-quantize \
    -j "$BUILD_JOBS"

# --- Verify binaries ---
echo ""
echo "[INFO] Verifying built binaries..."
BINARIES=(
    "$BUILD_DIR/bin/llama-server"
    "$BUILD_DIR/bin/llama-cli"
    "$BUILD_DIR/bin/llama-quantize"
)

ALL_OK=true
for bin in "${BINARIES[@]}"; do
    if [ -f "$bin" ]; then
        echo "  [OK] $bin"
    else
        echo "  [FAIL] $bin NOT FOUND"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = true ]; then
    echo ""
    echo "[SUCCESS] llama.cpp build completed."
    echo "  Commit: $LLAMA_COMMIT"
    echo "  Build dir: $BUILD_DIR"
else
    echo ""
    echo "[ERROR] Some binaries were not built. Check build log."
    exit 1
fi

# --- Save build info ---
mkdir -p "$PROJECT_DIR/results/environment"
cat > "$PROJECT_DIR/results/environment/build_info.txt" <<EOF
build_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
llama_cpp_dir=$LLAMA_CPP_DIR
llama_cpp_commit=$LLAMA_COMMIT
cmake_flags=${CMAKE_FLAGS[*]}
build_jobs=$BUILD_JOBS
cuda_arch=$CUDA_ARCH
EOF

echo "[INFO] Build info saved to results/environment/build_info.txt"
