#!/usr/bin/env bash
# =============================================================================
# Download & Convert Hugging Face Models to GGUF
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo " DOWNLOAD & CONVERT HUGGING FACE MODELS"
echo "============================================="

# Ensure huggingface_hub is installed
if ! python3 -c "import huggingface_hub" &> /dev/null; then
    echo "[INFO] Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

SINGLE_MODEL="${1:-}"

# Run the python script
if [ -n "$SINGLE_MODEL" ]; then
    python3 "$SCRIPT_DIR/download_and_convert_hf.py" --model "$SINGLE_MODEL"
else
    python3 "$SCRIPT_DIR/download_and_convert_hf.py"
fi

echo "[SUCCESS] Download & Convert phase completed."
