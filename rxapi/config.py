"""Server configuration — everything from the environment, nothing hard-coded.

The box this runs on is rented and rebuilt often, so state lives under one
directory (`RX_VAR`) that can be mounted as a volume and survives the instance.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # RX/

# --------------------------------------------------------------------------
# Identity of the model we serve
# --------------------------------------------------------------------------

# The base checkpoint never changes. The adapter does: training produces a new
# *version of the same adapter*, it never stacks a second one on top.
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "google/medgemma-4b-it")
SEED_ADAPTER_ID = os.environ.get("SEED_ADAPTER_ID", "pamessina/medgemma-4b-it-cure")

DEVICE = os.environ.get("DEVICE", "auto")                  # auto | cuda | mps | cpu
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

VAR_DIR = Path(os.environ.get("RX_VAR", BASE_DIR / "var"))
ADAPTERS_DIR = VAR_DIR / "adapters"        # one directory per trained version
DATASET_DIR = VAR_DIR / "dataset"          # corrections uploaded by the app
JOBS_DIR = VAR_DIR / "jobs"                # job records + logs
REGISTRY_PATH = VAR_DIR / "registry.json"  # versions, lineage, active pointer
DATASET_PATH = DATASET_DIR / "corrections.jsonl"

# Benchmark data, for the evaluation gate.
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR.parent))
GT_JSON = Path(os.environ.get("JSON_PATH", DATA_DIR / "grounded_reports_20240819.json"))
BENCH_OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "outputs" / "compare"))

# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

# Training starts when this many *new* corrected cases have arrived since the
# run that produced the active version. Below it, /train refuses unless forced:
# a handful of cases cannot move a 4B model, and a run that overfits them is
# worse than no run.
MIN_NEW_EXAMPLES = int(os.environ.get("TRAIN_MIN_NEW_EXAMPLES", "100"))

# Replay against catastrophic forgetting: this many PadChest-GR examples are
# mixed in per corrected case.
REPLAY_RATIO = float(os.environ.get("TRAIN_REPLAY_RATIO", "3"))

AUTH_TOKEN = os.environ.get("RX_AUTH_TOKEN", "")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))


def ensure_dirs() -> None:
    for path in (VAR_DIR, ADAPTERS_DIR, DATASET_DIR, JOBS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def summary() -> dict:
    return {
        "base_model": BASE_MODEL_ID,
        "seed_adapter": SEED_ADAPTER_ID,
        "device_request": DEVICE,
        "var_dir": str(VAR_DIR),
        "min_new_examples": MIN_NEW_EXAMPLES,
        "replay_ratio": REPLAY_RATIO,
        "auth": bool(AUTH_TOKEN),
    }
