#!/usr/bin/env bash
# Entrypoint for the CXR comparison container.
# All configuration is passed through environment variables (see .env.example).
#
# NOTE: the recommended path is the Vast.ai PyTorch template WITHOUT Docker
# (see README). This entrypoint just delegates to scripts/run_vast.sh, which
# builds the two model venvs (CURE and MAIRA-2 need conflicting transformers
# versions) and runs select -> run maira2 -> run cure -> report in bf16.
set -euo pipefail

echo "=================================================="
echo " CXR model comparison (CURE vs MAIRA-2, bf16)"
echo "=================================================="
echo "DEVICE     = ${DEVICE:-cuda}"
echo "N_IMAGES   = ${N_IMAGES:-200}"
echo "MODELS     = ${MODELS:-cure,maira2}"
echo "DATA_DIR   = ${DATA_DIR:-/data}"
echo "OUTPUT_DIR = ${OUTPUT_DIR:-/app/RX/outputs/compare}"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "[WARN] No HF_TOKEN set. Gated models (MAIRA-2, MedGemma-4B base) will fail to download."
    echo "       Pass it with:  -e HF_TOKEN=hf_xxx   (or via .env / env_file)"
else
    echo "HF token   = detected"
fi

if [ ! -d "${DATA_DIR:-/data}" ] || [ -z "$(ls -A "${DATA_DIR:-/data}" 2>/dev/null)" ]; then
    echo "[WARN] DATA_DIR '${DATA_DIR:-/data}' is missing or empty."
    echo "       Mount your dataset there (local copy or rclone Google Drive mount)."
fi

# `exec "$@"` lets you override the command, e.g. `docker run ... bash`.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec bash /app/RX/scripts/run_vast.sh
