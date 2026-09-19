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
tree, or copy results across afterwards, and commit them. `RX/outputs/` also
holds a working copy of the runs — convenient on the GPU box, but it is
gitignored, so never treat it as the record.

### Headline run (`../cxr_gui/outputs/compare_test604/`) — quote this one

**604 held-out test studies** (every test-split image with GT boxes), seed 42,
RTX 4090 48 GB, bf16, 2026-09-17, commit `ac788aa`, torch 2.11.0+cu128. This is
the comparison chapter's number and the baseline the gate measures against; the
200-image run above is preliminary and superseded.

| model | mean IoU (micro) | F1@0.5 | precision@0.5 | recall@0.5 | mAP-like | halluc@0.5 | pred boxes | s/img | n |
|---|---|---|---|---|---|---|---|---|---|
| CURE v1 | 0.416 | 0.2226 | 0.246 | 0.203 | 0.085 | 0.754 | 1255 | 4.6 | 604 |
| MAIRA-2 | 0.440 | 0.2379 | 0.258 | 0.221 | 0.100 | 0.742 | 1304 | 1.2 | 604 |

Same direction as the 200-image run, **smaller margin**: MAIRA-2 +0.015 F1@0.5,
not +0.043. Do not quote the 0.210 / 0.253 pair as the headline any more.

**MAIRA-2 scored 0.251 until its failures were fixed, and that difference is
entirely selection.** 10 of 604 generations hit `max_new_tokens=450` inside a
grounded phrase; the unclosed `<obj>` made MAIRA-2's own parser assert, the
images were dropped, and the model was being scored on the 594 it happened to
finish. `_parse_maira2_output` now keeps every complete phrase and discards
only the truncated tail, and `RETRY_ERRORS=1` re-runs just the failed images.
**Never report a model on fewer images than its comparator** — dropping the
hardest 2% flattered MAIRA-2 by 0.013 F1.

### The oracle reviewer (`scripts/simulate_corrections.py`)

Turns a finished prediction run plus PadChest-GR ground truth into the same
correction JSONL the desktop app exports, so the output POSTs straight to
`/corrections`.

```bash
python3 scripts/simulate_corrections.py --run outputs/val455/per_model/cure.json \
    --split validation --out exports/oracle_validation.jsonl
```

Four edits, derived per image:

| situation | edit |
|---|---|
| predicted box matches a GT box above `--match-iou` (0.3) | box moved to the GT box |
| the keyword does not match the GT wording | keyword replaced |
| predicted box matches nothing | deleted, recorded in `removed[]` |
| GT box matched nothing | added |

Matching uses `compare_models.greedy_match_boxes` in original pixel
coordinates, so what the oracle corrects and what the metric counts as a hit
cannot disagree.

**Unboxed sentences the model produced are left alone.** This is a deliberate
decision and the first version got it wrong: PadChest-GR often decomposes a
report into a single finding where CURE emitted five normal statements, so
rebuilding the target purely from the reference deleted all of them — training
the model to stop saying "no pleural effusion". `--fix drop-negations` already
measured that behaviour as a keyword-F1 loss. A reviewer corrects boxes, not
prose.

Run over the full validation split (2026-09-17): **308 studies** — not 455;
only 308 validation images carry GT boxes, and `select_images` takes only
those. 934 predicted boxes: 310 accepted as-is, 251 moved to GT, 185 keywords
replaced, 373 deleted, 770 missed boxes added. **40% of CURE's predicted boxes
had no ground-truth match at IoU 0.3** — a number worth quoting on its own (the
29-image pilot said 45%).

**The target must be written the way CURE writes**, i.e. every box of a finding
on one sentence — `Prominent vascular hila [b1] [b2]` — never the same sentence
repeated per box. The first version repeated it, and a pilot trained on that
output cut recall from 0.33 to 0.17 and taught the model a stock opening
sentence. `build_target` groups by keyword in order of first appearance.

Say *oracle reviewer* in writing, never "radiologist corrections": it sees every
error, never disagrees with itself, and never makes a mistake of its own, so
what it measures is an upper bound.

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

## Where we are (2026-09-17) — the loop has been run end to end

The GPU session is done. `cxr_gui/outputs/cl_oracle_run_2026-09-17/` holds the
whole record: validation predictions, the 308 oracle corrections, the registry,
every job log, and each version's gate verdict plus its 604-image predictions.
Adapter v3 (the best) is at `~/cxr_models/rx_adapters/v3` on the review Mac,
nowhere else — the rented box was destroyed.

### The learning curve — every point trained from v1, gated on the 604 images

| version | corrections | replay | F1@0.5 | precision | recall | halluc | pred boxes | kw F1 (micro) | gate |
|---|---|---|---|---|---|---|---|---|---|
| v1 | — | — | 0.2226 | 0.246 | 0.203 | 0.754 | 1255 | 0.175 | baseline |
| v2 | 50 | 150 | 0.2205 | 0.279 | 0.182 | 0.721 | 992 | 0.198 | pass |
| **v3** | 150 | 450 | **0.2321** | 0.271 | 0.203 | 0.729 | 1142 | 0.130 | pass |
| v4 | 300 | 900 | 0.2318 | 0.260 | 0.209 | 0.740 | 1223 | 0.069 | pass |
| v5 | 308 | 924 | 0.2228 | 0.292 | 0.180 | 0.708 | 939 | 0.095 | pass |

Per-image paired bootstrap vs v1 is **positive and significant at every point**
for IoU, F1@0.5 and keyword F1 (v3: IoU +0.038 [0.020, 0.057], F1 +0.031
[0.008, 0.056]). The aggregate moves much less than the per-image mean, because
micro pooling weights images with many findings.

**What to claim.**

- **The loop works and the gate works.** Corrections → training → an adoption
  gate on held-out data, with every candidate measured before it can be served.
  That is the contribution; the numbers are the evidence it functions.
- **The pre-registered target was not met.** Closing the CURE→MAIRA-2 gap was
  the stated goal; v3 closes about 60% of the (now smaller) 0.015 gap and no
  version passes MAIRA-2. Report that plainly.
- **The consistent effect is precision, not F1.** Every version hallucinates
  less than v1 (down to 0.708) and predicts fewer, better boxes. The oracle
  deletes 40% of CURE's boxes, so it teaches caution — and caution is exactly
  what a reviewer-corrected model should learn.
- **More corrections is not monotonically better.** F1 peaks at 150 and falls
  back by 308 as recall drops. An oracle at this volume moves the operating
  point, it does not add capability.
- **Aggregate keyword F1 collapses (0.175 → 0.069 at v4) while per-image
  keyword F1 rises.** Not a scoring bug: the trained versions write far longer
  reports (mean 140 → 478 characters) with many more unmatched keywords
  (FP 1322 → 7446, TP 295 → 331). The oracle adds 770 missed findings, and the
  model learns to say more. Box metrics are unaffected — the extra sentences
  mostly carry no box. Worth one paragraph in the write-up.

### Against MAIRA-2 — the gap closes (`scripts/compare_versions.py`)

Paired bootstrap, version − MAIRA-2, per image over all 604 test studies. Uses
the stored per-image payloads (no re-parsing: MAIRA-2 is xyxy, CURE cxcywh) and
refuses to report unless it reproduces every published F1@0.5 to four decimals.
Output: `cxr_gui/outputs/cl_oracle_run_2026-09-17/vs_maira2.{md,json}`.

| version | Δ F1@0.5 [95% CI] | p | Δ IoU [95% CI] | p | Δ keyword F1 [95% CI] | p |
|---|---|---|---|---|---|---|
| v1 | −0.031 [−0.056, −0.005] | **0.017** | −0.017 [−0.038, +0.003] | 0.103 | −0.056 [−0.077, −0.035] | **<0.001** |
| v2 | −0.021 [−0.047, +0.003] | 0.089 | −0.004 [−0.025, +0.018] | 0.731 | −0.027 [−0.050, −0.005] | **0.011** |
| **v3** | **+0.001 [−0.027, +0.028]** | **0.997** | +0.021 [−0.001, +0.041] | 0.064 | −0.025 [−0.045, −0.005] | **0.009** |
| v4 | +0.001 [−0.025, +0.028] | 0.938 | +0.019 [−0.001, +0.041] | 0.068 | −0.030 [−0.051, −0.010] | **0.006** |
| v5 | −0.005 [−0.029, +0.019] | 0.671 | +0.016 [−0.006, +0.037] | 0.146 | −0.013 [−0.036, +0.008] | 0.221 |

**This is the thesis's central quantitative result — state it exactly.**

- v1 is significantly behind MAIRA-2 on detection F1 (p=0.017). After 150
  oracle corrections (v3) **the difference is no longer significant** (+0.001,
  p=0.997), and it stays that way at 300 and 308.
- Say "*no longer significantly different*", never "matches" or "beats": a
  non-significant difference is not evidence of equality. The CI (±0.027) is
  the honest statement of how close.
- IoU leans towards CURE from v3 on (p≈0.06) — worth one sentence, not a claim.
- **Keyword naming remains significantly worse than MAIRA-2** for v1–v4; the
  corrections fix *where* CURE draws boxes more than *what it calls them*.
- Against the pre-registered target: the aggregate gap is not fully closed
  (0.2321 vs 0.2379), but the paired per-image test — the stronger analysis —
  finds no remaining difference. Report both; do not pick the kinder one.

### What the first real run cost — bugs the pilot caught

A 20-image / 30-correction pilot was run before the full one. It paid for
itself several times over; do the same before any future full run.

| commit | bug |
|---|---|
| `8d93c08` | the gate read the pinned list from `pinned["images"]`; the key is `selected`, so `/evaluate` iterated the dict's field names |
| `0b02466` | `run_vast.sh` ran `select` with the template python, which has no `cv2` |
| `3b3d25d` | `setup_hf_auth()` called interactive `login()` and ignored a stored `hf auth login` |
| `717f02c` | `DEVICE` defaulted to `cpu`; the API server exports nothing, so `/train/smoke` refused on a 48 GB GPU |
| `526c6d0` | `build_keyword_findings(pred, orig_size)` — missing `box_format`, so **every** `CureService.detect` raised, the gate scored v2 on 0 images and called it a total regression. The same call is what the desktop app's `RemoteEngine` uses |
| `a0eb02b` | training targets repeated a sentence per box, and replay drew normal studies that neither the corrections nor the benchmark contain |
| `ac788aa` | MAIRA-2 truncated-output failures dropped 10 images |

**The gate is only as trustworthy as its parity with the benchmark.** After
`526c6d0`, v1 was re-scored *through the gate* and reproduced
`compare_test604` exactly — 20/20 identical `report_text`, all deltas 0.0000.
Do that check again after any change to `inference.py` or `compare_models.py`;
a gate that cannot reproduce its own baseline can only mislead.

### Decided (unchanged)

- **CURE only.** MAIRA-2 stays a comparator. CURE is a LoRA adapter on an open
  base and the only one of the two that can be iterated on; say plainly that it
  scores lower on held-out data and that this is an engineering choice.
- **Corrections come from validation**, never train (seen by CURE) and never
  test (the measuring stick).
- **Stopping rule, honoured:** the first clean run did not close the gap, so it
  gets written up. No hyperparameter search.

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

**Replay draws train-split studies that have GT boxes — the same population as
the corrections and the benchmark.** Replaying normal studies ("No significant
findings", which `select_images` never picks) pulled a pilot towards saying
nothing: recall 0.33 → 0.17. The pool is 2096 studies, all of which must be on
disk.

**Targets are written the way CURE writes.** One sentence carries all of its
boxes. See the oracle section — this is the single change that turned a broken
pilot into a healthy one.

**Train on `train`, evaluate on `test`.** Now that the leakage is known this is
not optional. `evaluation.run_version()` scores against the pinned list; point
it at the test-split list.

**Training needs CUDA.** `_run_training` refuses on MPS or CPU — a 4B VLM in
bf16 with optimizer state does not fit in 24 GB of unified memory.

**The loop has been executed** (2026-09-17): smoke test, four training runs and
four gates, all on a 4090 48 GB, peak 36.9 GB with 1.38 B trainable parameters
(the LoRA tensors plus `modules_to_save`, held in fp32 under bf16 autocast).
`dry_run=true` still walks dataset assembly, version naming and the registry
write without torch — use it to check what a run *would* train on.

**`unload()` frees CUDA memory** (`gc` + `empty_cache`). Evaluation and
training both load a full copy; without it the second load meets a GPU that is
still holding the first.

---

## Re-running the loop on a rented box

This was run on 2026-09-17 (vast.ai RTX 4090 48 GB, ~6 h). `full.sh` and
`loop.sh` from that session are not in the repo — they are three-line wrappers
around what follows, and each stage skips if its output already exists, which
is what makes the run survive an instance stop.

**Prepare.** Dataset at `/data` (`grounded_reports_20240819.json`,
`master_table.csv`, `Padchest_GR_files/`). Replay needs **every train-split
study with GT boxes** on disk — 2096 of them; with fewer, `replay_examples`
refuses rather than silently shrinking the pool. Store the HF token with
`hf auth login` (never `export HF_TOKEN`, which the API server does not see).

```bash
# 1. headline benchmark (~1 h)   both models, 604 test studies
DATA_DIR=/data OUTPUT_DIR=…/cxr_gui/outputs/compare_test604 ./scripts/run_vast.sh
# 2. validation predictions (~25 min)   CURE only
DATA_DIR=/data SPLIT=validation N_IMAGES=455 MODELS=cure OUTPUT_DIR=outputs/val455 ./scripts/run_vast.sh
# 3. oracle corrections (seconds, no GPU)
python3 scripts/simulate_corrections.py --run outputs/val455/per_model/cure.json \
    --split validation --out exports/oracle_validation.jsonl
# 4. serve, upload, smoke, then one train+gate per curve point
RX_VAR=/workspace/rxvar OUTPUT_DIR=…/compare_test604 RX_AUTH_TOKEN=$T python3 serve.py --port 8077 &
curl -H "X-Auth-Token: $T" --data-binary @exports/oracle_validation.jsonl :8077/corrections
curl -H "X-Auth-Token: $T" -H 'Content-Type: application/json' -d '{"steps":30}' :8077/train/smoke
curl … -d '{"epochs":1,"lr":2e-5,"force":true,"parent_version":"v1","max_corrections":150,"seed":0}' :8077/train
curl … -d '{"version":"v3"}' :8077/evaluate
```

**Always `parent_version: "v1"`** on a curve point, or "300 corrections"
secretly means 150 then 300. **Always the smoke test first**: 30 steps on one
example must drive the loss to ~0 (it went 4.8 → 0.0000, weights moved 5.6e-2,
peak 37 GB). A trainer that cannot memorise one example turns a real run into a
null result that looks like a finding.

**Timings on a 4090 48 GB.** Benchmark 604 × 2 models ≈ 1 h; validation 308 ≈
25 min; training ≈ 1.2 s/step (150 corrections + 450 replay = 600 steps ≈
12 min; 308 + 924 = 1232 steps ≈ 26 min); **each gate ≈ 40–90 min**, and it
grows with the version — trained models write longer reports (v4 8.9 s/img vs
v1 4.6). Four curve points cost more in gating than in training.

**A pilot first, always.** 20 test images + 30 validation corrections is ~25
minutes and caught seven bugs (table above), including one that made every
prediction fail and one that halved recall.

**Step 6 — report evaluation (no GPU).** 20 generated reports checked for
findings invented outside the input list, with agreement measured against your
own pass on the same 20. Needs a fresh OpenAI key in the desktop app. Still to
do.

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
