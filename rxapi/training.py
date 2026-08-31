"""Continual learning — continue training the CURE adapter, in place.

The important structural point: this loads the **active adapter version** with
`is_trainable=True` and keeps training those same weights, then saves them as
the next version. It never calls `add_adapter`, and it never stacks a second
LoRA on the first. v2 is the same adapter as v1, further trained; v3 is the
same adapter again.

Two things guard against the failure mode of a small correction set:

* **replay** — every corrected case is mixed with `TRAIN_REPLAY_RATIO`
  PadChest-GR examples, so the run cannot simply overwrite what the adapter
  learned from thousands of cases with what it saw in a hundred.
* **the threshold** — `/train` refuses below `TRAIN_MIN_NEW_EXAMPLES`, because
  a 4B model cannot be moved by a handful of cases and a run that overfits them
  is worse than no run at all.

Nothing here has been executed against a GPU yet. `dry_run` walks the whole
path — dataset assembly, version naming, registry write — without importing
torch, which is what the code-path test exercises.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from . import config, dataset, registry
from .registry import Version

PROMPT = "Generate a grounded report."          # identical to serving


# --------------------------------------------------------------------------
# Building the training set
# --------------------------------------------------------------------------


def replay_examples(n: int, exclude: set[str], seed: int = 0) -> list[dict[str, Any]]:
    """`n` PadChest-GR cases, formatted exactly like a correction example.

    The ground-truth report is rebuilt in CURE's own output format, so a replay
    example and a correction example are indistinguishable to the trainer.
    """
    if n <= 0 or not config.GT_JSON.is_file():
        return []
    records = json.loads(config.GT_JSON.read_text())
    images_dir = config.DATA_DIR / "Padchest_GR_files"

    pool = []
    for record in records:
        image_id = record.get("ImageID")
        if not image_id or image_id in exclude:
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

    random.Random(seed).shuffle(pool)
    return pool[:n]


def build_training_set(rows: list[dict[str, Any]], replay_ratio: float,
                       seed: int = 0) -> list[dict[str, Any]]:
    corrections = [{**r, "origin": "correction"} for r in rows]
    exclude = {Path(r.get("image_path", "")).name for r in corrections}
    replay = replay_examples(int(len(corrections) * replay_ratio), exclude, seed)
    mixed = corrections + replay
    random.Random(seed).shuffle(mixed)
    return mixed


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def train(log: Callable[[str], None], *, epochs: int = 1, lr: float = 2e-5,
          replay_ratio: float | None = None, include_all: bool = False,
          force: bool = False, dry_run: bool = False,
          job_id: str | None = None) -> dict[str, Any]:
    """Continue the active adapter on the pending corrections. Returns a receipt."""
    replay_ratio = config.REPLAY_RATIO if replay_ratio is None else replay_ratio

    rows = dataset.training_rows(include_all=include_all)
    if len(rows) < config.MIN_NEW_EXAMPLES and not force:
        raise RuntimeError(
            f"{len(rows)} new corrected case(s); the threshold is "
            f"{config.MIN_NEW_EXAMPLES}. Collect more, or pass force=true."
        )
    if not rows:
        raise RuntimeError("no corrections to train on")

    parent = registry.active()
    name = registry.next_version_name()
    out_dir = config.ADAPTERS_DIR / name
    log(f"continuing adapter {parent.version} -> {name}")

    examples = build_training_set(rows, replay_ratio)
    n_replay = sum(1 for e in examples if e["origin"] == "replay")
    log(f"{len(rows)} correction(s) + {n_replay} replay = {len(examples)} example(s)")

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

    _run_training(log, examples, parent, out_dir, epochs=epochs, lr=lr)

    version = Version(
        version=name,
        source=str(out_dir),
        parent=parent.version,
        trained_on=len(rows),
        replay_examples=n_replay,
        job_id=job_id,
        metrics={"case_ids": [r["case_id"] for r in rows]},
        notes=f"continued from {parent.version}; lr={lr}, epochs={epochs}",
    )
    registry.register(version)
    log(f"saved {name} -> {out_dir}")
    return {"version": name, **asdict(version)}


def _run_training(log, examples: list[dict[str, Any]], parent: Version,
                  out_dir: Path, *, epochs: int, lr: float) -> None:
    """The actual fine-tune. Requires a GPU; never run on the review laptop."""
    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import compare_models as cm

    cfg = cm.resolve_device_config()
    if cfg.device.type != "cuda":
        raise RuntimeError(
            f"training needs CUDA; this box reports {cfg.device.type}. "
            "A 4B VLM in bf16 with optimizer state does not fit on MPS."
        )

    log(f"loading {config.BASE_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(config.BASE_MODEL_ID)
    processor.tokenizer.padding_side = "right"      # training, not generation
    base = AutoModelForImageTextToText.from_pretrained(
        config.BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True,
    )

    log(f"attaching adapter {parent.version} for continued training")
    # is_trainable=True is the whole point: without it peft loads the adapter
    # frozen and the run silently trains nothing.
    model = PeftModel.from_pretrained(base, parent.path_or_id(), is_trainable=True)
    cm._verify_cure_adapter_loaded(model)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"{sum(p.numel() for p in trainable):,} trainable parameters")
    if not trainable:
        raise RuntimeError("adapter loaded frozen — nothing to train")

    class Examples(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int):
            return examples[index]

    def collate(batch):
        messages, targets = [], []
        for item in batch:
            image = cm.load_xray_as_rgb_cure(item["image_path"])
            messages.append([{
                "role": "user",
                "content": [{"type": "image", "image": image},
                            {"type": "text", "text": item.get("prompt", PROMPT)}],
            }])
            targets.append(item["target"])
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
        labels = processor.tokenizer(
            targets, return_tensors="pt", padding=True, add_special_tokens=False
        ).input_ids
        return inputs, labels

    loader = DataLoader(Examples(), batch_size=1, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    step = 0
    for epoch in range(epochs):
        for inputs, labels in loader:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)
            merged = torch.cat([inputs["input_ids"], labels], dim=1)
            mask = torch.cat([
                torch.full_like(inputs["input_ids"], -100), labels
            ], dim=1)
            out = model(input_ids=merged, labels=mask,
                        pixel_values=inputs.get("pixel_values"))
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0:
                log(f"epoch {epoch + 1}/{epochs}  step {step}  loss {out.loss.item():.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))          # the adapter only, not the base
    processor.save_pretrained(str(out_dir))
    log(f"wrote adapter weights after {step} step(s)")
