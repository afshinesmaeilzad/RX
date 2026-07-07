# CUDA-enabled image for comparing CURE vs MAIRA-2 on chest X-rays (bf16).
# The base image ships a CUDA 12.1 build of PyTorch. CURE and MAIRA-2 need
# conflicting transformers versions, so the two model venvs are built AT RUNTIME
# by scripts/run_vast.sh (using --system-site-packages to reuse this torch).
#
# The recommended path is the Vast.ai PyTorch template without Docker (see README).
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    MPLCONFIGDIR=/tmp/mpl \
    TOKENIZERS_PARALLELISM=false

# System libraries: git (HF downloads), libGL/glib (opencv), ca-certificates.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the whole RX project (compare_models.py, requirements-*.txt, scripts/).
COPY . /app/RX
RUN chmod +x /app/RX/entrypoint.sh /app/RX/scripts/run_vast.sh

# Defaults; override at runtime with -e / env_file. bf16 only (no 4-bit).
ENV DEVICE=cuda \
    N_IMAGES=200 \
    SHUFFLE_SEED=42 \
    MODELS=cure,maira2 \
    DATA_DIR=/data \
    OUTPUT_DIR=/app/RX/outputs/compare

ENTRYPOINT ["/app/RX/entrypoint.sh"]
