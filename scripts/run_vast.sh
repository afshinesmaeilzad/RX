#!/usr/bin/env bash
#
# One-command runner for a Vast.ai PyTorch template (NO Docker).
#
# A Vast.ai instance is already a container, so running the app directly on the
# PyTorch template is simpler and more reliable than Docker-in-Docker. This
# script installs the extra Python deps (torch already ships with the template)
# and runs the comparison.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   # dataset already available at DATA_DIR (local copy or rclone mount):
#   DATA_DIR=/data N_IMAGES=2 ./scripts/run_vast.sh
#
# Common overrides (env vars):
#   DATA_DIR   path to dataset root (contains grounded_reports_*.json + Padchest_GR_files/)
#   N_IMAGES   number of images (default 1)
#   MODELS     subset, e.g. "cure" or "cure,maira2" (default all three)
#   USE_4BIT   1 = 4-bit (fits 16GB); 0 = bf16 (needs 24GB). Default 1.
#   DEVICE     cuda | cpu (default cuda)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

export DEVICE="${DEVICE:-cuda}"
export USE_4BIT="${USE_4BIT:-1}"
export N_IMAGES="${N_IMAGES:-1}"
export DATA_DIR="${DATA_DIR:-/data}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "[ERROR] python3 not found. Install Python 3 or use a PyTorch template."
    exit 1
fi

echo "=================================================="
echo " RX CXR comparison — Vast.ai (PyTorch, no Docker)"
echo "=================================================="

# 1. Sanity checks -----------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "[ERROR] HF_TOKEN is not set. Run:  export HF_TOKEN=hf_xxx"
    echo "        (needed to download the gated MAIRA-2 / MedGemma models)"
    exit 1
fi

if [ ! -e "${DATA_DIR}/grounded_reports_20240819.json" ]; then
    echo "[WARN] ${DATA_DIR}/grounded_reports_20240819.json not found."
    echo "       Set DATA_DIR to your dataset root, or mount Google Drive first:"
    echo "         GDRIVE_SUBPATH=\"PadChest/extracted/BIMCV-Padchest-GR\" ./scripts/setup_gdrive.sh"
fi

# 2. Show GPU ----------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# 3. Install deps (torch already present in the template) ---------------------
echo ""
echo "Installing Python dependencies (this may take a few minutes)..."

# Install a CUDA 12-compatible torch first so that pip does not pull in
# a CPU-only or CUDA 13 build when resolving the other deps.
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  → Installing CUDA 12 compatible torch..."
    "$PYTHON" -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124 \
        --quiet
fi

"$PYTHON" -m pip install --upgrade pip 2>/dev/null || true
"$PYTHON" -m pip install -r requirements.txt

# 4. Run ---------------------------------------------------------------------
echo ""
echo "Running comparison: DEVICE=$DEVICE USE_4BIT=$USE_4BIT N_IMAGES=$N_IMAGES MODELS=${MODELS:-cure,maira2,medgemma15}"
echo ""
"$PYTHON" compare_models.py

echo ""
echo "Done. See outputs/compare/report.md and outputs/compare/plots/"
