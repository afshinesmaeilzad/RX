"""The correction store — what the reviewers send up, and when it is enough.

The app exports one JSON object per reviewed case
(`cxr_gui/scripts/export_corrections.py`); this appends them to a JSONL, keyed
by `case_id` so the same case can be re-sent after further correction and
simply replaces its earlier copy.

Training does not start on a trickle. `pending()` counts the cases that arrived
*after* the run which produced the active adapter version, and `/train` refuses
below `TRAIN_MIN_NEW_EXAMPLES` (100) unless it is forced: a 4B model cannot be
moved by a handful of cases, and a run that overfits them is worse than none.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

from . import config, registry

_lock = threading.Lock()

REQUIRED_FIELDS = ("case_id", "image_path", "target")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_all() -> list[dict[str, Any]]:
    if not config.DATASET_PATH.is_file():
        return []
    out = []
    for line in config.DATASET_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write_all(rows: Iterable[dict[str, Any]]) -> None:
    config.ensure_dirs()
    tmp = config.DATASET_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tmp.replace(config.DATASET_PATH)


def validate(example: dict[str, Any]) -> str | None:
    """Why this example cannot be trained on, or None if it can."""
    missing = [f for f in REQUIRED_FIELDS if not example.get(f)]
    if missing:
        return f"missing {', '.join(missing)}"
    if not example.get("image_size"):
        return "missing image_size"
    return None


def ingest(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Add or replace examples, keyed by case_id. Returns a small receipt."""
    accepted, rejected = [], []
    for example in examples:
        reason = validate(example)
        if reason:
            rejected.append({"case_id": example.get("case_id"), "reason": reason})
            continue
        example = dict(example)
        example["received_at"] = _now()
        accepted.append(example)

    with _lock:
        existing = {row.get("case_id"): row for row in load_all()}
        replaced = sum(1 for e in accepted if e["case_id"] in existing)
        for e in accepted:
            existing[e["case_id"]] = e
        _write_all(existing.values())

    return {
        "accepted": len(accepted),
        "replaced": replaced,
        "added": len(accepted) - replaced,
        "rejected": rejected,
        "total": len(existing),
    }


# --------------------------------------------------------------------------
# How much is waiting
# --------------------------------------------------------------------------


def trained_case_ids() -> set[str]:
    """Every case already used by some version in the active lineage."""
    used: set[str] = set()
    for name in registry.lineage(registry.active().version):
        version = registry.get(name)
        if version is not None:
            used.update(version.metrics.get("case_ids", []) or [])
    return used


def pending() -> list[dict[str, Any]]:
    """Cases the active adapter has never been trained on."""
    used = trained_case_ids()
    return [row for row in load_all() if row.get("case_id") not in used]


def stats() -> dict[str, Any]:
    rows = load_all()
    waiting = pending()
    totals = {"moved": 0, "relabelled": 0, "added": 0, "removed": 0, "kept": 0,
              "repeats": 0}
    for row in rows:
        for key, value in (row.get("corrections") or {}).items():
            if key in totals and isinstance(value, (int, float)):
                totals[key] += int(value)
    return {
        "total_cases": len(rows),
        "pending_cases": len(waiting),
        "threshold": config.MIN_NEW_EXAMPLES,
        "ready_to_train": len(waiting) >= config.MIN_NEW_EXAMPLES,
        "short_by": max(0, config.MIN_NEW_EXAMPLES - len(waiting)),
        "active_version": registry.active().version,
        "corrections": totals,
    }


def training_rows(include_all: bool = False) -> list[dict[str, Any]]:
    """The examples a run should train on: pending only, unless told otherwise."""
    return load_all() if include_all else pending()
