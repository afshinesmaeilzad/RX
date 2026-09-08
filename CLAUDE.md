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
| test only (n=42) | 0.181 | 0.296 | MAIRA-2 +0.121 (p=0.003) |

That was a warning at n=42, not a verdict. **The clean run has now been done**
— see `../cxr_gui/outputs/compare_test/` below. It confirms the reversal at proper sample
size, with a margin less than half what n=42 suggested.

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

### Clean test-split run (`../cxr_gui/outputs/compare_test/`) — the honest number

200 held-out test images, seed 42, CUDA, bf16, 2026-09-05, commit `3860d5d`,
torch 2.11.0+cu128. Only 16 of these 200 appear in the leaked list.

| model | mean IoU (macro±std) | F1@0.5 | mAP-like | halluc@0.5 | keyword F1 | s/img | n |
|---|---|---|---|---|---|---|---|
| CURE | 0.281 ± 0.266 | 0.210 | 0.078 | 0.769 | 0.158 | 2.6 | 200 |
| MAIRA-2 | 0.311 ± 0.287 | 0.253 | 0.109 | 0.712 | 0.231 | 1.0 | 198 |

Both tables quote **macro** mean IoU (mean of per-image means), so they compare
like with like. The CSV also carries `mean_iou_micro` (pooled over all matched
pairs, CURE 0.504 → 0.418); never mix the two across tables.

Paired bootstrap, CURE − MAIRA-2, over the 198 images both models scored:

| metric | diff | 95% CI | p |
|---|---|---|---|
| mean IoU | −0.032 | [−0.067, +0.004] | 0.091 — **not significant** |
| F1@0.5 | −0.054 | [−0.101, −0.011] | **0.017** |
| keyword F1 | −0.088 | [−0.129, −0.046] | **<0.001** |

**How to state this, and how not to.**

- The strongest evidence is not the ranking flip, it is the **asymmetry**:
  leaked → clean, CURE falls 0.341 → 0.210 (−38%) while MAIRA-2 barely moves,
  0.245 → 0.253. A model that generalises should not care which images you
  picked. CURE's hallucination rate rising 0.631 → 0.769, overtaking MAIRA-2,
  says the same thing.
- **Do not claim MAIRA-2 localises better.** The IoU difference is not
  significant (p=0.091). The defensible claim: MAIRA-2 achieves higher
  detection F1 and names findings better; the quality of the boxes each model
  *does* match is statistically indistinguishable.
- **Every metric moves against CURE and none against MAIRA-2.** Macro mean IoU
  0.369 → 0.281 for CURE, 0.286 → 0.311 for MAIRA-2; F1 0.341 → 0.210 vs
  0.245 → 0.253; hallucination 0.631 → 0.769 vs 0.709 → 0.712. No caveat
  needed — state the asymmetry plainly.
- Torch differs between the two runs (2.6.0+cu124 vs 2.11.0+cu128). Very
  unlikely to move F1 by 0.13, but disclose it rather than be asked.
- MAIRA-2 errored on 2 of the 200 images. `cmd_run` prints `{exc}` with no
  traceback and both messages were empty, so the cause is unrecoverable from
  the logs. The paired test correctly uses the 198 both models scored, so the
  statistics are unaffected — but log `repr(exc)` next time.

**CURE respected the official split.** On the leaked run, its scores by split
are train 0.422 / validation 0.180 / test 0.181 F1@0.5 — validation behaves
exactly like test, so PadChest-GR *validation* (455 studies) is genuinely
unseen data for CURE. That makes it the right source for simulated corrections,
with test kept untouched as the measuring stick.

**Results live in `cxr_gui/outputs/`, not here.** This repo's `.gitignore`
ignores `outputs/`, so anything written to `RX/outputs/` is untracked and one
`git clean` from gone. `cxr_gui/outputs/` *is* versioned — the published
baseline is committed there. Run with `OUTPUT_DIR` pointing at the cxr_gui
tree, or copy results across afterwards, and commit them. A byte-identical
duplicate currently sits in `RX/outputs/compare_test/`; delete it.

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

## Where we are (2026-09-06)

Done: the benchmark twice (leaked run, then the clean 200-image test-split
run), `SPLIT` support through selection / re-scoring / `run_vast.sh`, three
post-processing rules measured and all rejected, `rxapi/` written with its code
paths tested (training only as `dry_run`), and the 604-image evaluation list
pinned.

Decided:

- **CURE only.** MAIRA-2 stays a comparator, not a second product. CURE is a
  LoRA adapter on an open base and is the only one of the two that can actually
  be iterated on; MAIRA-2 is a 7B model under a restrictive research licence.
  Say plainly in the thesis that CURE scores lower on held-out data and that
  this is an engineering choice, not a performance claim.
- **Corrections come from the validation split** — never train (already seen by
  CURE) and never test (the measuring stick). 455 studies: enough for the
  100-case threshold and for a 50/150/300/455 learning curve.
- **Pre-registered target.** CURE-v1 scores 0.210 F1@0.5 held-out, MAIRA-2
  0.253. Success is closing that 0.043 gap. Write the target down before
  running and report whatever comes out.

Expect a null result: 455 examples against the tens of thousands CURE was
trained on, from the same distribution, against errors already shown not to be
systematic. Design for that — the oracle framing turns a null into an **upper
bound** ("even with noise-free corrections at this volume, no measurable
gain"), which is stronger than a null from noisy human corrections. Fix a
stopping rule now: if the first clean run does not move F1, write it up rather
than chasing hyperparameters.

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

1. ~~Re-run the comparison on the test split.~~ **Done 2026-09-05** —
   `../cxr_gui/outputs/compare_test/`. The reversal is confirmed; write it up from the
   "How to state this" notes above.
1b. **Run the 604-image evaluation** — the baseline the gate compares against,
   and the headline for the comparison chapter:
   ```
   DATA_DIR=/data SPLIT=test N_IMAGES=604 \
   OUTPUT_DIR=/path/to/cxr_gui/outputs/compare_test604 ./scripts/run_vast.sh
   ```
   The list is pinned already. Commit the results into `cxr_gui/outputs/`.
   Decide *before* looking that 604 is the headline and 200 was preliminary —
   choosing afterwards is cherry-picking.
2. **Oracle corrections from ground truth.** PadChest-GR is radiologist
   annotation, so the "reviewer" can be simulated: move each predicted box onto
   its matched GT box, delete unmatched predictions, add unmatched GT, relabel
   where the keyword differs. Produce them from **`validation`** only. Call it an
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
