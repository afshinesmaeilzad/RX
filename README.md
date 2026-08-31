# RX — CURE serving + continual learning (the cloud side)

The GPU half of CXR GroundAssist. Two jobs, one service:

1. **Serve CURE** to the desktop app over HTTP, speaking the contract the app's
   `RemoteEngine` already uses — pointing the app here is a URL change and
   nothing else.
2. **Run the continual-learning loop**: corrections come up from reviewers, a
   training run *continues the existing adapter*, the gate scores the candidate
   against the pinned benchmark, and only a version that passes can be promoted
   to the one `/detect` serves.

```
 desktop app                     RX (this box, GPU)
┌──────────────┐   POST /detect  ┌────────────────────────────────────┐
│ engine.py    │ ───────────────►│ rxapi/inference.py   base + adapter│
│ RemoteEngine │ ◄─────────────── │                      (one, always)│
└──────────────┘   {keyword,box} └────────────────────────────────────┘
       │                                        ▲
       │ POST /corrections                      │ promote, if it passes
       ▼                                        │
┌──────────────┐   POST /train   ┌──────────────┴─────────────────────┐
│ dataset.py   │ ───────────────►│ training.py  →  evaluation.py gate │
│ 100 pending? │                 │ continue v1 → v2 → v3              │
└──────────────┘                 └────────────────────────────────────┘
```

## One adapter, versioned — never a stack

CURE is MedGemma-4B plus **one** LoRA adapter. Training continues *that*
adapter and saves it as the next version; it never attaches a second LoRA on
top of the first. The registry is therefore a straight line:

| version | where | how it got there |
|---|---|---|
| `v1` | `pamessina/medgemma-4b-it-cure` (Hub) | the published seed |
| `v2` | `var/adapters/v2` | v1 continued on the first 100 corrections |
| `v3` | `var/adapters/v3` | v2 continued on the next 100 |

`PeftModel.from_pretrained(base, parent, is_trainable=True)` is the line that
makes this continuation rather than a fresh adapter — without `is_trainable`,
peft loads the weights frozen and the run silently trains nothing.

## When training starts

Not on a trickle. `TRAIN_MIN_NEW_EXAMPLES` (default **100**) is the number of
corrected cases that must have arrived *since the run that produced the active
version*. Below it `/train` returns 409 and says how many short you are. A 4B
model cannot be moved by a handful of cases, and a run that overfits them is
worse than no run.

Every corrected case is mixed with `TRAIN_REPLAY_RATIO` (default 3) PadChest-GR
examples, rebuilt into CURE's own output format so replay and correction
examples are indistinguishable to the trainer. Without replay, a hundred cases
would overwrite what the adapter learned from thousands.

## The gate

A candidate is scored on **the same pinned 200-image list** that produced the
published baseline in `outputs/compare/`, using `compare_models.py` itself —
not a reimplementation, because a second copy would drift and every promotion
decision would rest on numbers not comparable with the thesis.

Promotion is refused unless the candidate has been evaluated and did not
regress beyond `evaluation.REGRESSION_TOLERANCE`:

| metric | tolerance |
|---|---|
| `mean_iou_micro` | may not drop more than 0.005 |
| `f1@0.5_micro` | may not drop more than 0.005 |
| `hallucination@0.5` | may not rise more than 0.010 |

Deltas come with the paired bootstrap the thesis report uses. `force=true`
overrides, and is meant for debugging, not for shipping.

## API

| method | route | what it does |
|---|---|---|
| `GET` | `/health` | open, unauthenticated — device, active version, running job |
| `POST` | `/detect` | raw image bytes + `X-Image-Name` → the worker's payload |
| `POST` | `/model/load` | pull weights now instead of on the first X-ray |
| `POST` | `/corrections` | ingest the app's export (JSON array, object, or JSONL) |
| `GET` | `/corrections/stats` | totals, pending count, how far from the threshold |
| `GET` | `/versions` | the lineage, which is active, gate result per version |
| `POST` | `/versions/{v}/activate` | promote — refused unless the gate passed |
| `POST` | `/train` | continue the active adapter (`dry_run` walks it with no GPU) |
| `POST` | `/evaluate` | score a version against the baseline |
| `GET` | `/jobs`, `/jobs/{id}` | status, progress, log tail |

Auth is a shared token in `X-Auth-Token` — the header the app already sends.
It is a gate for a tunnelled or private-network deployment, **not** an
internet-facing authentication system. Do not expose this port publicly with
patient data behind it.

Training and evaluation hold the GPU, so `/detect` returns 503 while a job
runs, and only one job runs at a time.

## Run it

```bash
python3 -m venv .venv-cure && . .venv-cure/bin/activate
pip install -r requirements-cure.txt -r requirements-server.txt
# torch comes from the GPU image; if not:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

RX_AUTH_TOKEN=secret DATA_DIR=/data python3 serve.py --host 0.0.0.0 --port 8077
```

Then in the app: **Settings → Where CURE runs → Remote**, URL and the same
token. The model is not loaded at startup — `/health` answers while the ~9 GB
of weights are still coming down.

## Configuration

| variable | default | meaning |
|---|---|---|
| `RX_VAR` | `RX/var` | all mutable state; mount this as a volume |
| `RX_AUTH_TOKEN` | *(empty)* | shared token; empty disables the check |
| `BASE_MODEL_ID` | `google/medgemma-4b-it` | never changes |
| `SEED_ADAPTER_ID` | `pamessina/medgemma-4b-it-cure` | version `v1` |
| `TRAIN_MIN_NEW_EXAMPLES` | `100` | the collection threshold |
| `TRAIN_REPLAY_RATIO` | `3` | PadChest-GR examples per correction |
| `DATA_DIR` | repo parent | PadChest-GR images + `grounded_reports_*.json` |
| `OUTPUT_DIR` | `RX/outputs/compare` | the baseline run and its pinned image list |
| `DEVICE` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu` |

## What is verified, and what is not

The API is covered end to end without a GPU: routes, auth, ingestion, the
threshold refusal, the job lifecycle, the version lineage, and promotion being
blocked for an unevaluated or regressed candidate. The gate arithmetic is
checked against the real stored baseline — comparing that run with itself gives
a zero delta and reproduces `f1@0.5 = 0.3406`.

**The training loop itself has never been executed.** It needs a GPU and a
correction set that does not exist yet. `POST /train {"dry_run": true}` walks
everything around it — dataset assembly, replay mixing, version naming — and is
what the tests exercise. Treat the first real run as a bring-up, not a result:
watch the loss and the trainable-parameter count in the job log before trusting
anything it produces.

## The benchmark, still here

`compare_models.py` (CURE vs MAIRA-2 on PadChest-GR) and its published run in
`outputs/compare/` are kept: they define the metrics, and they are the baseline
the gate measures against.

```
CURE     mean IoU 0.369 ± 0.296 | F1@0.5 0.341 | hallucination@0.5 0.631 | keyword F1 0.249
MAIRA-2  mean IoU 0.286 ± 0.281 | F1@0.5 0.245 | hallucination@0.5 0.709 | keyword F1 0.218
```

`rescore_offline.py` answers "would this post-processing rule have helped?" from
the stored predictions, with no GPU. Three rules measured so far — sentence
dedup, dropping negations, box rescaling — are all neutral or harmful, which is
why the remaining lever is the model itself.

See `docs/` in the app repo for the reviewer-facing side.
