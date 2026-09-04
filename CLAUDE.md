# RX — working notes for Claude Code

Two things live here:

1. **The benchmark** — `compare_models.py`, which produced the thesis numbers
   for CURE vs MAIRA-2 on PadChest-GR, plus `rescore_offline.py` for asking
   "would this change have helped?" without a GPU.
2. **The server API** — `rxapi/`, a FastAPI service that serves CURE to the
   desktop app (`cxr_gui`) and runs the continual-learning loop on a cloud GPU.

---

## Read this before touching any number

**Both CURE and MAIRA-2 were trained on PadChest-GR.**

- MAIRA-2: 52,828 ungrounded + **3,122 grounded** PadChest examples, alongside
  MIMIC-CXR and USMix.
- CURE: its paper states it fine-tunes "on the same three publicly available
  chest X-ray datasets used by our baseline (MAIRA-2)" — Chest ImaGenome,
  MS-CXR and **PadChest-GR**. Only VinDr-CXR is zero-shot for it.

PadChest-GR ships an official split in `master_table.csv` (3185 train / 455
validation / 915 test). **The first published run ignored it**: of its 200
pinned images, 129 (64%) were training images for both models. Re-scoring that
run split by split:

| subset | CURE F1@0.5 | MAIRA-2 F1@0.5 | winner |
|---|---|---|---|
| all 200 (the published number) | 0.341 | 0.245 | CURE +0.110 (p<0.001) |
| train only (leaked) | 0.422 | 0.218 | CURE +0.223 (p<0.001) |
| **test only (n=42)** | **0.181** | **0.296** | **MAIRA-2 +0.121 (p=0.003)** |

On held-out data the conclusion reverses, and CURE's hallucination rate rises
from 0.63 to 0.80 — the pattern of a model that memorised its training set.
n=42 is small, so this is a warning, not a verdict. The clean run is pinned and
waiting in `../cxr_gui/outputs/compare_test/image_list.json` (200 images, all
test split, seed 42).

**Therefore: any generalization claim must use `SPLIT=test`.** The env var
defaults to empty, which reproduces the original behaviour — that is
deliberate, so the published run stays reproducible. Never quote a
whole-dataset number as a generalization result.

---

## Metrics live in exactly one place

`compare_models.py` is the source of truth: greedy one-to-one box matching in
**original-image pixel coordinates**, an IoU sweep, `hallucination@0.5 = 1 −
precision`, a score-free mAP-like surrogate (these models emit no box
confidences, so true COCO AP is undefined), bootstrap CIs and a paired
bootstrap.

`rescore_offline.py` and `rxapi/evaluation.py` **import** those functions. Do
not reimplement them — a second copy drifts, and then no two numbers in the
thesis are comparable.

`rescore_offline.py --check` recomputes the published aggregate from the stored
`report_text` and refuses to report any delta unless it matches
`comparison_summary.csv` to four decimals. Keep that guard working.

### Published baseline (`../cxr_gui/outputs/compare/`)

200 images, seed 42, CUDA, bf16, 2026-07-07, commit `a35787e`:

| model | mean IoU | F1@0.5 | mAP-like | halluc@0.5 | keyword F1 | s/img |
|---|---|---|---|---|---|---|
| CURE | 0.369 ± 0.296 | 0.341 | 0.173 | 0.631 | 0.249 | 2.0 |
| MAIRA-2 | 0.286 ± 0.281 | 0.245 | 0.103 | 0.709 | 0.218 | 0.9 |

`per_model/*.json` keeps the raw `report_text` for every image. That is what
makes offline re-scoring possible; do not prune those files.

### Post-processing rules already measured — all negative

| rule | effect | why |
|---|---|---|
| `dedup` | F1 +0.0007, n.s. | repeated sentences cost only 2 boxes in 425 |
| `drop-negations` | keyword F1 **−0.0069**, p=0.019 | PadChest-GR annotates normal statements too, so dropping them loses true positives |
| `scale:0.75…1.10` | every factor worse | the boxes are not systematically mis-sized; the error is placement |

Do not re-run these expecting a different answer. The conclusion is that
`hallucination@0.5 = 0.63` is not cheap post-processing debt — improving it
needs the model.

---

## The server API (`rxapi/`)

FastAPI, started by `serve.py`. Install
`requirements-cure.txt` **and** `requirements-server.txt`.

```bash
RX_AUTH_TOKEN=secret python3 serve.py --host 0.0.0.0 --port 8077
```

| module | role |
|---|---|
| `config.py` | env-driven settings; all state under `RX_VAR` (mount it as a volume) |
| `registry.py` | adapter versions — one lineage, `v1` = the published Hub adapter |
| `dataset.py` | corrections uploaded by the app; what is pending; the threshold |
| `inference.py` | `CureService` — model loaded once, torch imported lazily |
| `training.py` | continue-training the adapter into the next version |
| `evaluation.py` | the adoption gate |
| `jobs.py` | one background job at a time (one GPU) |
| `app.py` | routes |

### The app's contract must not change

`cxr_gui/engine.py::RemoteEngine` already speaks this:

- `GET /health` → `{status, device, adapter_version, ...}`
- `POST /detect` — body is raw image bytes, `X-Image-Name` header, optional
  `X-Auth-Token` — returns the same payload the local worker returns.

Continual-learning routes: `POST /corrections`, `GET /corrections/stats`,
`GET /versions`, `POST /train`, `POST /evaluate`,
`POST /versions/{v}/activate`, `GET /jobs[/{id}]`.

### Continual-learning invariants

**One adapter, continued — never stacked.** `training.py` loads the active
version with `PeftModel.from_pretrained(..., is_trainable=True)` and keeps
training those weights. Without `is_trainable=True` peft loads the adapter
frozen and the run silently trains nothing; the code checks for a non-empty
trainable parameter list and refuses otherwise.

**The 100-example threshold.** `/train` refuses below
`TRAIN_MIN_NEW_EXAMPLES` unless forced. A 4B model cannot be moved by a handful
of cases, and a run that overfits them is worse than no run.

**Replay.** Every corrected case is mixed with `TRAIN_REPLAY_RATIO` (default 3)
PadChest-GR examples, rebuilt into CURE's own output format so a replay example
and a correction are indistinguishable to the trainer. Without this, a hundred
corrections overwrite what the adapter learned from thousands of cases.

**Train on `train`, evaluate on `test`.** Now that the leakage is known this is
not optional. `evaluation.run_version()` scores against the pinned list; point
it at the test-split list.

**Training needs CUDA.** `_run_training` refuses on MPS or CPU — a 4B VLM in
bf16 with optimizer state does not fit in 24 GB of unified memory.

**The training loop has never been executed.** It is written and its code path
is exercised by `dry_run=true` (dataset assembly, version naming, registry
write, no torch). Treat the first real run as debugging, not as an experiment.

---

## What is next

1. **Re-run the comparison on the test split.** The list is already pinned;
   on the GPU box:
   `OUTPUT_DIR=outputs/compare_test SPLIT=test python3 compare_models.py run --model cure`,
   then `--model maira2`, then `report`. Expect lower numbers and possibly a
   reversed ranking. That is the honest number.
2. **Oracle corrections from ground truth.** PadChest-GR is radiologist
   annotation, so the "reviewer" can be simulated: move each predicted box onto
   its matched GT box, delete unmatched predictions, add unmatched GT, relabel
   where the keyword differs. Produce them from `train` only. Call it an
   *oracle reviewer* in writing, never "radiologist corrections" — it is an
   upper bound, because a real reviewer sees less, is inconsistent, and makes
   mistakes of their own.
3. **v2 through the gate.** Pass or fail, both are results.
4. **Minimal report evaluation** — 20 generated reports, checked for findings
   invented outside the input list, with measured agreement against a human
   pass on the same 20.

---

## Environment notes

`compare_models.py` imports torch **and matplotlib** at module level, so
`.venv-cure` needs `requirements-base.txt` (which includes matplotlib), not
just the inference subset that `cxr_gui/cure/requirements-cure.txt` installs.

Paths come from env vars: `DATA_DIR` (dataset root), `JSON_PATH`
(`grounded_reports_20240819.json`), `MASTER_TABLE` (`master_table.csv`),
`OUTPUT_DIR`, `N_IMAGES`, `SHUFFLE_SEED`, `SPLIT`, `RX_VAR`, `RX_AUTH_TOKEN`.

The dataset is 36 GB and lives outside both repos. The HF cache is another
10.6 GB (`google/medgemma-4b-it` 8 GB, `pamessina/medgemma-4b-it-cure` 2.6 GB),
and both models are gated — accept the licences and set `HF_TOKEN` before the
first pull.
