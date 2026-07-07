# RX — CURE vs MAIRA-2 on PadChest-GR (grounded localization)

Compare two medical vision-language models on the **PadChest-GR** dataset for
grounded chest-X-ray findings, scoring predicted bounding boxes against ground
truth in **original-image pixel space** (IoU, precision/recall/F1, a score-free
mAP-like surrogate) plus a lightweight keyword-vs-label match:

| Key | Model | HF repo |
|-----|-------|---------|
| `cure` | CURE (MedGemma-4B + LoRA) | [`pamessina/medgemma-4b-it-cure`](https://huggingface.co/pamessina/medgemma-4b-it-cure) |
| `maira2` | MAIRA-2 (~7B) | [`microsoft/maira-2`](https://huggingface.co/microsoft/maira-2) |

Both models output only **keyword findings + boxes** (the polished narrative
report is future work, assembled later by OpenAI from the stored `{keyword, box}`
findings). Everything runs in **bf16 full precision — no 4-bit / no bitsandbytes**.

> **Why two environments?** CURE needs `transformers==4.55.4` + `peft==0.17.1`,
> while MAIRA-2 needs `transformers==4.51.3`. These conflict in one process, so
> each model runs in its own venv and writes results to disk; a final
> transformers-free step aggregates them.

---

## Pipeline (three subcommands)

```
python compare_models.py select              # pin the shared image list (once)
python compare_models.py run --model maira2  # in .venv-maira2 (tf 4.51.3)
python compare_models.py run --model cure    # in .venv-cure   (tf 4.55.4 + peft)
python compare_models.py report              # aggregate -> report.md (no transformers)
```

`select` writes `outputs/compare/image_list.json` (`{seed, n_images, selected}`).
**Both `run` calls read the same file**, so the two models score identical images.
Each `run` writes `outputs/compare/per_model/<model>.json` with, per image, the
`{keyword, box}` findings and all per-image detection numbers. `report` aggregates.

`scripts/run_vast.sh` does all four steps for you (builds both venvs, orchestrates,
aggregates).

---

## What you get (in `outputs/compare/`)

- `report.md` — thesis-ready report: reproducibility metadata, headline metrics,
  precision/recall/F1 (micro + macro), 95% bootstrap CIs, a paired CURE-vs-MAIRA
  significance test, per-pathology breakdown, and embedded plots.
- `plots/` — bar charts: mean IoU (±std), P/R/F1@0.5, recall@0.3 vs 0.5,
  mAP-like + hallucination rate, keyword F1, latency.
- `comparison_results.json` — aggregated metrics + metadata + significance.
- `comparison_summary.csv` — per-model headline table.
- `per_model/<model>.json` — per-image `{keyword, box_norm, box_px}` findings
  (the payload for the later OpenAI narrative-report step) + per-image scores.
- `run_<model>.log` — full stdout/stderr transcript of each model run.
- `per_image/*.png` — optional single-model box previews (set `SAVE_FIGURES=1`).

### Metrics reported (per model)

- **Detection (pixel space):** precision/recall/F1 at IoU >= 0.3 and >= 0.5,
  micro (pooled) and macro (per-image mean ± std), over **all** predicted boxes
  with greedy one-to-one matching. Mean IoU of matched pairs. Hallucination rate.
- **mAP-like@[.5:.95]:** mean micro-F1 across IoU 0.50:0.05:0.95. This is a
  **score-free surrogate** — the generative models emit no box confidences, so a
  true COCO AP / PR curve is undefined.
- **Keyword match:** predicted finding keywords vs GT labels via stdlib `difflib`
  (precision/recall/F1, micro + macro). A sanity signal for "did it name the right
  finding", complementary to box IoU. No RadGraph/CheXbert/BLEU/ROUGE.
- **Statistics:** 95% bootstrap CIs and a paired bootstrap significance test
  (CURE vs MAIRA-2) on per-image IoU and F1@0.5.

---

## Recommended hardware (Vast.ai)

| Resource | Recommended |
|----------|-------------|
| GPU VRAM | **>= 24 GB — 1× RTX 4090** (bf16 full precision; each model loads one at a time) |
| Disk | **~80 GB** (OS + two venvs + ~31 GB model cache + dataset) |
| System RAM | **32 GB** |

`run_vast.sh` warns if it detects < 24 GB VRAM.

---

## Run on Vast.ai (PyTorch template, no Docker)

A Vast.ai instance is already a container, so running directly on the **PyTorch**
template is simplest.

### 1. Rent an instance
- Template: **PyTorch (Vast)** or **PyTorch NGC** (CUDA + PyTorch + SSH)
- GPU: **1× RTX 4090 (24 GB)**, Disk **~80 GB**

### 2. Clone + configure token + get the data
```bash
git clone https://github.com/afshinesmaeilzad/RX.git
cd RX
export HF_TOKEN=hf_xxx        # accept the model licenses on HF first (see below)
# Point DATA_DIR at your dataset root (local copy recommended for 200 images).
```

`DATA_DIR` must contain:
```
grounded_reports_20240819.json
Padchest_GR_files/            # the .png chest X-rays (or Padchest_GR_files/PadChest_GR/)
```

Accept the licenses once while logged in to Hugging Face:
- https://huggingface.co/google/medgemma-4b-it  (CURE base)
- https://huggingface.co/pamessina/medgemma-4b-it-cure  (CURE adapter)
- https://huggingface.co/microsoft/maira-2

### 3. Run it (one command)
```bash
# Small smoke test (2 images, both models):
DATA_DIR=/data N_IMAGES=2 ./scripts/run_vast.sh

# Full run:
DATA_DIR=/data N_IMAGES=200 ./scripts/run_vast.sh
```

`run_vast.sh` builds `.venv-maira2` and `.venv-cure` (both with
`--system-site-packages`, reusing the template's CUDA torch), runs each model in
its venv, then aggregates. Re-runs are idempotent (existing venvs are reused).
Results land in `outputs/compare/`.

---

## Configuration (env vars)

| Variable | Default | Meaning |
|----------|---------|---------|
| `HF_TOKEN` | — | Hugging Face read token (required for gated models) |
| `DATA_DIR` | `/data` (script) | Dataset root (`grounded_reports_*.json` + `Padchest_GR_files/`) |
| `N_IMAGES` | `200` | Number of images to evaluate |
| `SHUFFLE_SEED` | `42` | Selection seed (deterministic image list) |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `MODELS` | `cure,maira2` | Subset for `run_vast.sh`, e.g. `cure` or `maira2` |
| `SAVE_FIGURES` | `0` | `1` = save per-image box previews during `run` |
| `FORCE_RESELECT` | `0` | `1` = re-pick the image list even if it exists |

---

## Manual run (advanced)

If you prefer to drive the venvs yourself:
```bash
python3 -m venv --system-site-packages .venv-maira2
.venv-maira2/bin/pip install -r requirements-maira2.txt
python3 compare_models.py select
DEVICE=cuda .venv-maira2/bin/python compare_models.py run --model maira2

python3 -m venv --system-site-packages .venv-cure
.venv-cure/bin/pip install -r requirements-cure.txt
DEVICE=cuda .venv-cure/bin/python compare_models.py run --model cure

.venv-cure/bin/python compare_models.py report
```

---

## Notes & caveats

- IoU is computed in **original-image pixel coordinates** for both models (CURE
  cxcywh, MAIRA-2 xyxy and GT are all mapped to original W×H px first), so aspect
  ratio is honoured identically.
- `mAP-like@[.5:.95]` is a score-free surrogate (mean micro-F1 across IoU
  thresholds); true COCO AP is undefined without per-box confidences.
- Keyword matching is lexical (difflib ratio >= 0.8 or substring), a sanity signal
  rather than a clinical NLG metric.
- **CURE troubleshooting:** the adapter was saved with `transformers==4.55.4` +
  `peft==0.17.1`; other versions raise `KeyError: ...embed_tokens.weight`. Load
  with `AutoModelForImageTextToText` (CURE needs the vision encoder) and attach
  the LoRA directly — do **not** untie embeddings. Clear a corrupt cache with
  `rm -rf ~/.cache/huggingface/hub/models--pamessina--medgemma-4b-it-cure`.
- The narrative report and any report-text NLG metrics are **future work**,
  assembled by OpenAI from the stored `{keyword, box}` findings.
