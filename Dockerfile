# CUDA-enabled image for comparing CURE / MAIRA-2 / MedGemma 1.5 on chest X-rays.
# The base image already provides a CUDA 12.1 build of PyTorch, so we only add
# the Python deps in requirements.txt (which deliberately excludes torch).
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

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt

COPY compare_models.py /app/compare_models.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Defaults; override at runtime with -e / env_file.
ENV DEVICE=cuda \
    USE_4BIT=1 \
    N_IMAGES=1 \
    DATA_DIR=/data \
    OUTPUT_DIR=/app/outputs

ENTRYPOINT ["/app/entrypoint.sh"]
