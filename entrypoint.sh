#!/usr/bin/env bash
# Entrypoint for the CXR comparison container.
# All configuration is passed through environment variables (see .env.example).
set -euo pipefail

echo "=================================================="
echo " CXR model comparison (CURE / MAIRA-2 / MedGemma)"
echo "=================================================="
echo "DEVICE     = ${DEVICE:-cuda}"
echo "USE_4BIT   = ${USE_4BIT:-1}"
echo "N_IMAGES   = ${N_IMAGES:-1}"
echo "MODELS     = ${MODELS:-cure,maira2,medgemma15}"
echo "DATA_DIR   = ${DATA_DIR:-/data}"
echo "OUTPUT_DIR = ${OUTPUT_DIR:-/app/outputs}"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "[WARN] No HF_TOKEN set. Gated models (MAIRA-2, MedGemma) will fail to download."
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

exec python /app/compare_models.py
