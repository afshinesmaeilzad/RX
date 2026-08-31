# RX: the CURE serving + continual-learning API, on a CUDA host.
#
# The image installs the CURE pin set (transformers==4.55.4 + peft==0.17.1 —
# the versions the adapter was saved with) directly, because the API only ever
# serves CURE. The MAIRA-2 comparison still needs its own conflicting
# transformers, so the `benchmark` compose profile builds that venv at runtime
# via scripts/run_vast.sh.
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

WORKDIR /app/RX

# Dependencies first, so a code change does not re-install torch's neighbours.
COPY requirements-base.txt requirements-cure.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-cure.txt -r requirements-server.txt

# Then the project (rxapi/, compare_models.py, scripts/, outputs/).
COPY . /app/RX
RUN chmod +x /app/RX/entrypoint.sh /app/RX/scripts/run_vast.sh /app/RX/serve.py

# Defaults; override at runtime with -e / env_file. bf16 only (no 4-bit).
ENV DEVICE=cuda \
    DATA_DIR=/data \
    RX_VAR=/var/rx \
    OUTPUT_DIR=/app/RX/outputs/compare \
    N_IMAGES=200 \
    SHUFFLE_SEED=42 \
    MODELS=cure,maira2

EXPOSE 8077

# The API by default; the benchmark profile overrides this with entrypoint.sh.
CMD ["python3", "serve.py", "--host", "0.0.0.0", "--port", "8077"]
