#!/usr/bin/env bash
#
# One-command runner for a Vast.ai PyTorch template (NO Docker).
#
# Compares CURE vs MAIRA-2 on a fixed set of PadChest-GR images, in bf16 FULL
# PRECISION (no 4-bit). Because the two models need conflicting transformers
# versions, each runs in its own venv:
#
#   .venv-maira2  transformers==4.51.3            (requirements-maira2.txt)
#   .venv-cure    transformers==4.55.4 + peft     (requirements-cure.txt)
#
# Both venvs are created with --system-site-packages so they reuse the template's
# CUDA torch instead of reinstalling it.
#
# Flow:
#   1. select   -> outputs/compare/image_list.json          (shared, once)
#   2. run maira2 in .venv-maira2 -> per_model/maira2.json
#   3. run cure   in .venv-cure   -> per_model/cure.json
#   4. report                     -> report.md + plots + CSV
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   DATA_DIR=/data N_IMAGES=200 ./scripts/run_vast.sh
#
# Env overrides:
#   DATA_DIR   dataset root (grounded_reports_*.json + Padchest_GR_files/)  [default /data]
#   N_IMAGES   number of images                                            [default 200]
#   SHUFFLE_SEED  selection seed                                           [default 42]
#   MODELS     subset to run, e.g. "cure" or "maira2"                      [default cure,maira2]
#   DEVICE     cuda | cpu                                                  [default cuda]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

export DEVICE="${DEVICE:-cuda}"
export N_IMAGES="${N_IMAGES:-200}"
export SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
export DATA_DIR="${DATA_DIR:-/data}"
MODELS="${MODELS:-cure,maira2}"

VENV_MAIRA=".venv-maira2"
VENV_CURE=".venv-cure"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "[ERROR] python3 not found. Use a PyTorch template."
    exit 1
fi

echo "=================================================="
echo " RX CXR comparison — CURE vs MAIRA-2 (bf16, Vast.ai)"
echo "=================================================="

# 1. Sanity checks ----------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "[ERROR] HF_TOKEN is not set. Run:  export HF_TOKEN=hf_xxx"
    echo "        (needed for gated MAIRA-2, MedGemma-4B base, and the CURE adapter)"
    exit 1
fi

if [ ! -e "${DATA_DIR}/grounded_reports_20240819.json" ]; then
    echo "[WARN] ${DATA_DIR}/grounded_reports_20240819.json not found."
    echo "       Set DATA_DIR to your dataset root."
fi

# 2. GPU info + VRAM warning ------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]' || echo 0)"
    if [ -n "${VRAM_MB}" ] && [ "${VRAM_MB}" -lt 24000 ] 2>/dev/null; then
        echo "[WARN] GPU has ${VRAM_MB} MB VRAM. bf16 full precision wants >=24 GB;"
        echo "       MAIRA-2 (~7B) may OOM. Use a >=24 GB GPU (e.g. RTX 4090)."
    fi
fi

# Helper: make a venv (reusing template torch) and install its requirements.
make_venv () {
    local venv_dir="$1" req_file="$2"
    if [ ! -d "$venv_dir" ]; then
        echo "  → Creating $venv_dir (--system-site-packages)"
        "$PYTHON" -m venv --system-site-packages "$venv_dir"
    else
        echo "  → Reusing existing $venv_dir"
    fi
    "$venv_dir/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || true
    # Ensure a CUDA torch exists (only installs if the template lacks one).
    if ! "$venv_dir/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "  → Installing CUDA 12 torch into $venv_dir"
        "$venv_dir/bin/python" -m pip install torch torchvision \
            --index-url https://download.pytorch.org/whl/cu124 --quiet
    fi
    "$venv_dir/bin/python" -m pip install -r "$req_file"
}

want_model () { case ",$MODELS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# 3. Select the shared image list (once) ------------------------------------
echo ""
echo "== Selecting image list (N_IMAGES=$N_IMAGES, seed=$SHUFFLE_SEED) =="
"$PYTHON" compare_models.py select

# 4. MAIRA-2 -----------------------------------------------------------------
if want_model maira2; then
    echo ""
    echo "== MAIRA-2 (transformers==4.51.3, bf16) =="
    make_venv "$VENV_MAIRA" requirements-maira2.txt
    "$VENV_MAIRA/bin/python" compare_models.py run --model maira2
fi

# 5. CURE --------------------------------------------------------------------
if want_model cure; then
    echo ""
    echo "== CURE (transformers==4.55.4 + peft==0.17.1, bf16) =="
    make_venv "$VENV_CURE" requirements-cure.txt
    "$VENV_CURE/bin/python" compare_models.py run --model cure
fi

# 6. Report ------------------------------------------------------------------
echo ""
echo "== Aggregating report =="
# The report step needs no transformers; prefer the cure venv, else maira, else base python.
if [ -x "$VENV_CURE/bin/python" ]; then
    REPORT_PY="$VENV_CURE/bin/python"
elif [ -x "$VENV_MAIRA/bin/python" ]; then
    REPORT_PY="$VENV_MAIRA/bin/python"
else
    REPORT_PY="$PYTHON"
fi
"$REPORT_PY" compare_models.py report

echo ""
echo "Done. See outputs/compare/report.md, comparison_summary.csv, and plots/"
