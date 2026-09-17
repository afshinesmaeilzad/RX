"""Continual learning — continue training the CURE adapter, in place.

The structural point: this loads an adapter version with `is_trainable=True`
and keeps training *those* weights, then saves them as the next version. It
never calls `add_adapter` and never stacks a second LoRA on the first.

Guards against the failure modes of a small correction set:

* **replay** — each correction is mixed with `TRAIN_REPLAY_RATIO` examples
  drawn from the PadChest-GR **train split only**. Replay rehearses what the
  adapter was trained on; drawing it from validation or test would leak the
  measuring stick into training and make the gate meaningless.
* **the threshold** — `/train` refuses below `TRAIN_MIN_NEW_EXAMPLES` unless
  forced, because a run that overfits a handful of cases is worse than none.

Three failure modes this file exists to make *loud*, because each one produces
a plausible loss curve and a model that did not learn anything:

* **bf16 update underflow.** bfloat16 has ~0.8% relative precision. An AdamW
  step at lr 2e-5 on a weight near 0.02 is a ~0.1% change — below resolution,
  so it rounds to zero and training silently does nothing. Trainable weights
  are therefore held in **fp32** (the frozen base stays bf16) and the forward
  pass runs under bf16 autocast.
* **A frozen adapter.** peft loads adapters frozen unless `is_trainable=True`.
  The run refuses to start with zero trainable parameters.
* **Weights that do not move.** A snapshot of LoRA tensors is compared before
  and after; a run whose weights did not change raises instead of saving.

Before trusting any of this on real data, run `smoke_test`: it overfits a
single example and must drive the loss down hard. A trainer that cannot memorise
one example is broken, and nothing it reports about 455 means anything.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from . import config, dataset, registry
from .registry import Version

PROMPT = "Generate a grounded report."          # identical to serving


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def load_splits() -> dict[str, str]:
    """ImageID -> official PadChest-GR split. Empty if the table is missing."""
    master = Path(config.DATA_DIR) / "master_table.csv"
    if not master.is_file():
        return {}
    with master.open() as fh:
        return {
            row["ImageID"]: (row.get("split") or "").strip().lower()
            for row in csv.DictReader(fh) if row.get("ImageID")
        }


# --------------------------------------------------------------------------
# Building the training set
# --------------------------------------------------------------------------


def replay_examples(n: int, exclude: set[str], seed: int = 0) -> list[dict[str, Any]]:
    """`n` train-split PadChest-GR cases, formatted like a correction example.

    Refuses to run without the split table rather than falling back to the
    whole dataset: a silent fallback would put test images into training.
    """
    if n <= 0:
        return []
    if not config.GT_JSON.is_file():
        raise RuntimeError(f"replay needs the ground truth at {config.GT_JSON}")
    splits = load_splits()
    if not splits:
        raise RuntimeError(
            "replay needs master_table.csv to restrict itself to the train split; "
            "without it test images could leak into training"
        )

    records = json.loads(config.GT_JSON.read_text())
    images_dir = config.DATA_DIR / "Padchest_GR_files"

    pool = []
    for record in records:
        image_id = record.get("ImageID")
        if not image_id or image_id in exclude or splits.get(image_id) != "train":
            continue
        path = images_dir / image_id
        if not path.is_file():
            continue
        sentences = []
        for finding in record.get("findings") or []:
            sentence = (finding.get("sentence_en") or "").strip().rstrip(".")
            if not sentence:
                continue
            boxes = finding.get("boxes") or []
            if boxes:
                for x1, y1, x2, y2 in boxes:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    sentences.append(
                        f"{sentence} [{cx:.2f},{cy:.2f},{x2 - x1:.2f},{y2 - y1:.2f}]"
                    )
            else:
                sentences.append(sentence)
        if not sentences:
            continue
        pool.append({
            "case_id": f"replay:{image_id}",
            "image_path": str(path),
            "prompt": PROMPT,
            "target": ". ".join(sentences) + ".",
            "origin": "replay",
        })

    pool.sort(key=lambda e: e["case_id"])          # disk order must not matter
    random.Random(seed).shuffle(pool)
    if len(pool) < n:
        raise RuntimeError(
            f"asked for {n} replay examples but only {len(pool)} train-split "
            f"images are on disk under {images_dir}"
        )
    return pool[:n]


def select_corrections(rows: list[dict[str, Any]], limit: int | None,
                       seed: int = 0) -> list[dict[str, Any]]:
    """A deterministic subset, so learning-curve points are nested and repeatable.

    The 50-correction set is a prefix of the 150 set, which is a prefix of 300:
    the curve then measures the effect of *more* corrections, not of different
    ones.
    """
    ordered = sorted(rows, key=lambda r: r.get("case_id", ""))
    random.Random(seed).shuffle(ordered)
    return ordered[:limit] if limit else ordered


def build_training_set(rows: list[dict[str, Any]], replay_ratio: float,
                       seed: int = 0) -> list[dict[str, Any]]:
    corrections = [{**r, "origin": "correction"} for r in rows]
    exclude = {Path(r.get("image_path", "")).name for r in corrections}
    replay = replay_examples(round(len(corrections) * replay_ratio), exclude, seed)
    mixed = corrections + replay
    random.Random(seed).shuffle(mixed)
    return mixed


# --------------------------------------------------------------------------
# One example -> model inputs. Module-level so it can be tested on a CPU.
# --------------------------------------------------------------------------


def build_example(processor, item: dict[str, Any]) -> dict[str, Any]:
    """Tokenise prompt + target through the chat template, masking the prompt.

    Going through the template for the whole conversation is what gets the
    image tokens, `token_type_ids`, the attention mask and the closing
    `<end_of_turn>` right. The previous version concatenated a separately
    tokenised target onto the prompt: it passed no attention mask or token
    type ids, and the target had no end-of-turn token, so the model would never
    have learned where to stop.
    """
    import torch

    import compare_models as cm

    image = cm.load_xray_as_rgb_cure(item["image_path"])
    user = {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": item.get("prompt") or PROMPT},
    ]}
    assistant = {"role": "assistant", "content": [
        {"type": "text", "text": item["target"]},
    ]}

    full = processor.apply_chat_template(
        [user, assistant], tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=False,
    )
    prompt = processor.apply_chat_template(
        [user], tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    )

    n_prompt = prompt["input_ids"].shape[-1]
    if not torch.equal(full["input_ids"][0, :n_prompt], prompt["input_ids"][0]):
        raise RuntimeError(
            "the prompt is not a prefix of the full conversation; the label mask "
            "would cover the wrong tokens"
        )
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    if int((labels != -100).sum()) == 0:
        raise RuntimeError(f"no target tokens for {item.get('case_id')}")

    batch = dict(full)
    batch["labels"] = labels
    return batch


# --------------------------------------------------------------------------
# Model setup, shared by training and the smoke test
# --------------------------------------------------------------------------


def _load_trainable(log, parent: Version):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import compare_models as cm

    cfg = cm.resolve_device_config()
    if cfg.device.type != "cuda":
        raise RuntimeError(
            f"training needs CUDA; this box reports {cfg.device.type}. The trainable "
            "adapter in fp32 with AdamW state needs ~35 GB — a 48 GB card."
        )

    log(f"loading {config.BASE_MODEL_ID} (bf16, frozen)")
    processor = AutoProcessor.from_pretrained(config.BASE_MODEL_ID)
    processor.tokenizer.padding_side = "right"
    base = AutoModelForImageTextToText.from_pretrained(
        config.BASE_MODEL_ID, torch_dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True,
    )
    base.config.use_cache = False

    log(f"attaching adapter {parent.version} as trainable")
    # is_trainable=True is the whole point — without it peft loads the adapter
    # frozen and every optimizer step is a no-op.
    model = PeftModel.from_pretrained(base, parent.path_or_id(), is_trainable=True)
    cm._verify_cure_adapter_loaded(model)

    trainable, n_params = [], 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()        # fp32 master weights: see docstring
            trainable.append(param)
            n_params += param.numel()
    if not trainable:
        raise RuntimeError("adapter loaded frozen — nothing to train")
    log(f"{n_params:,} trainable parameters, held in fp32")

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        log("gradient checkpointing on")
    except (AttributeError, ValueError) as exc:
        log(f"gradient checkpointing unavailable ({exc}); memory will be higher")

    model.train()
    return model, processor, trainable, n_params


def _snapshot(model, limit: int = 8) -> dict[str, Any]:
    """Copies of a few LoRA tensors spread across the network.

    LoRA matrices, not the embedding copy: an embedding row only moves when its
    token appears in a target, so an unchanged embedding slice proves nothing.
    """
    names = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    if not names:
        return {}
    step = max(1, len(names) // limit)
    chosen = set(names[::step][:limit])
    return {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters() if n in chosen
    }


def _relative_change(model, before: dict[str, Any]) -> float:
    if not before:
        return 0.0
    import torch

    worst = 0.0
    for name, param in model.named_parameters():
        if name not in before:
            continue
        old = before[name]
        new = param.detach().float().cpu()
        denom = float(old.norm()) or 1.0
        worst = max(worst, float((new - old).norm()) / denom)
    return worst


def _step(model, batch, trainable, optimizer) -> float:
    import torch

    device = next(p for p in trainable).device
    inputs = {}
    for key, value in batch.items():
        if not hasattr(value, "to"):
            continue
        value = value.to(device)
        if key == "pixel_values":
            value = value.to(torch.bfloat16)       # the vision tower is bf16
        inputs[key] = value

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs)
    loss = out.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


# --------------------------------------------------------------------------
# The smoke test — run this first, every time the trainer changes
# --------------------------------------------------------------------------


def smoke_test(log: Callable[[str], None], *, steps: int = 30, lr: float = 1e-4,
               parent_version: str | None = None) -> dict[str, Any]:
    """Overfit one example. Passing means the trainer can learn at all.

    Uses a correction if one is uploaded, otherwise a replay example. Never
    saves weights and never touches the registry.
    """
    import torch

    parent = registry.get(parent_version) if parent_version else registry.active()
    if parent is None:
        raise ValueError(f"unknown version: {parent_version}")

    rows = dataset.load_all()
    if rows:
        item = select_corrections(rows, 1)[0]
    else:
        item = replay_examples(1, set())[0]
    log(f"smoke test on {item['case_id']} — {steps} steps at lr {lr}")

    model, processor, trainable, n_params = _load_trainable(log, parent)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    batch = build_example(processor, item)
    before = _snapshot(model)

    losses = []
    for i in range(steps):
        losses.append(_step(model, batch, trainable, optimizer))
        if i % 5 == 0 or i == steps - 1:
            log(f"step {i + 1}/{steps}  loss {losses[-1]:.4f}")

    change = _relative_change(model, before)
    passed = losses[-1] < 0.5 * losses[0] and change > 0
    log(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, weights moved {change:.2e} — "
        + ("PASSED" if passed else "FAILED: the trainer cannot overfit one example"))

    del model
    torch.cuda.empty_cache()
    return {
        "passed": passed,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
        "weight_change": change,
        "trainable_params": n_params,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def train(log: Callable[[str], None], *, epochs: int = 1, lr: float = 2e-5,
          replay_ratio: float | None = None, include_all: bool = False,
          force: bool = False, dry_run: bool = False,
          max_corrections: int | None = None, parent_version: str | None = None,
          seed: int = 0, job_id: str | None = None) -> dict[str, Any]:
    """Continue an adapter version on corrections. Returns a receipt.

    For a learning curve, train every point from the same parent
    (`parent_version="v1"`) with `max_corrections` set: chaining v3 off v2 would
    make "300 corrections" secretly mean 150 + 300.
    """
    replay_ratio = config.REPLAY_RATIO if replay_ratio is None else replay_ratio

    rows = dataset.training_rows(include_all=include_all)
    rows = select_corrections(rows, max_corrections, seed)
    if len(rows) < config.MIN_NEW_EXAMPLES and not force:
        raise RuntimeError(
            f"{len(rows)} corrected case(s); the threshold is "
            f"{config.MIN_NEW_EXAMPLES}. Collect more, or pass force=true."
        )
    if not rows:
        raise RuntimeError("no corrections to train on")

    parent = registry.get(parent_version) if parent_version else registry.active()
    if parent is None:
        raise ValueError(f"unknown parent version: {parent_version}")
    name = registry.next_version_name()
    out_dir = config.ADAPTERS_DIR / name
    log(f"continuing adapter {parent.version} -> {name}")

    examples = build_training_set(rows, replay_ratio, seed)
    n_replay = sum(1 for e in examples if e["origin"] == "replay")
    log(f"{len(rows)} correction(s) + {n_replay} replay (train split) = "
        f"{len(examples)} example(s)")

    if dry_run:
        log("dry run — no model loaded, no weights written")
        return {
            "dry_run": True,
            "parent": parent.version,
            "would_create": name,
            "corrections": len(rows),
            "replay": n_replay,
            "examples": len(examples),
            "case_ids": [r["case_id"] for r in rows],
        }

    stats = _run_training(log, examples, parent, out_dir,
                          epochs=epochs, lr=lr, seed=seed)

    version = Version(
        version=name,
        source=str(out_dir),
        parent=parent.version,
        trained_on=len(rows),
        replay_examples=n_replay,
        job_id=job_id,
        metrics={
            "case_ids": [r["case_id"] for r in rows],
            "training": {**stats, "lr": lr, "epochs": epochs, "seed": seed,
                         "replay_ratio": replay_ratio},
        },
        notes=(f"continued from {parent.version} on {len(rows)} corrections; "
               f"lr={lr}, epochs={epochs}, seed={seed}"),
    )
    registry.register(version)
    log(f"saved {name} -> {out_dir}")
    return {"version": name, **asdict(version)}


def _run_training(log, examples: list[dict[str, Any]], parent: Version,
                  out_dir: Path, *, epochs: int, lr: float, seed: int) -> dict[str, Any]:
    """The fine-tune itself. Needs a 48 GB GPU; never run on the review laptop."""
    import torch

    model, processor, trainable, n_params = _load_trainable(log, parent)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    before = _snapshot(model)

    step, window, first_window, losses = 0, [], None, []
    order = list(range(len(examples)))
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(order)
        for index in order:
            loss = _step(model, build_example(processor, examples[index]),
                         trainable, optimizer)
            step += 1
            window.append(loss)
            losses.append(loss)
            if len(window) == 20:
                mean = sum(window) / len(window)
                first_window = mean if first_window is None else first_window
                log(f"epoch {epoch + 1}/{epochs}  step {step}/{epochs * len(order)}  "
                    f"loss {mean:.4f}")
                window = []

    change = _relative_change(model, before)
    log(f"LoRA weights moved {change:.2e} (relative)")
    if change == 0.0:
        raise RuntimeError(
            "adapter weights did not change — training did nothing. Not saving: a "
            "version from this run would be scored as a real null result."
        )

    # Back to bf16 for saving, to match the seed adapter's dtype and size.
    for param in trainable:
        param.data = param.data.to(torch.bfloat16)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))            # the adapter only, not the base
    processor.save_pretrained(str(out_dir))

    last = losses[-20:] if losses else []
    stats = {
        "steps": step,
        "trainable_params": n_params,
        "first_loss": first_window,
        "last_loss": (sum(last) / len(last)) if last else None,
        "weight_change": change,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    log(f"wrote {out_dir.name}: {stats}")
    return stats
