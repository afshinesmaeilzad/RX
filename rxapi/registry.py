"""Adapter versions — one lineage, never a stack.

CURE is MedGemma-4B plus **one** LoRA adapter. Continual learning continues
training *that* adapter and writes a new version of it; it never attaches a
second adapter on top of the first. So the registry is a straight line:

    v1  pamessina/medgemma-4b-it-cure   (the published seed, on the Hub)
    v2  var/adapters/v2                 trained from v1 on 100 corrections
    v3  var/adapters/v3                 trained from v2 on the next 100
    ...

Each version records where it came from, which examples it saw, and — once the
gate has run — how it scored. `active` is the one `/detect` serves; it only
moves when someone promotes a version that passed.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import config

SEED_VERSION = "v1"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Version:
    version: str
    source: str                       # "hub" for the seed, else a local path
    parent: str | None = None         # the version this one continued from
    created_at: str = field(default_factory=_now)
    trained_on: int = 0               # corrected cases in the run
    replay_examples: int = 0
    job_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)   # filled by the gate
    notes: str = ""

    @property
    def is_seed(self) -> bool:
        return self.source == "hub"

    def path_or_id(self) -> str:
        """What `PeftModel.from_pretrained` should be given."""
        return config.SEED_ADAPTER_ID if self.is_seed else self.source


def _blank() -> dict[str, Any]:
    seed = Version(version=SEED_VERSION, source="hub",
                   notes="published CURE adapter; the starting point")
    return {"active": SEED_VERSION, "versions": {SEED_VERSION: asdict(seed)}}


def _read() -> dict[str, Any]:
    config.ensure_dirs()
    if not config.REGISTRY_PATH.is_file():
        return _blank()
    try:
        raw = json.loads(config.REGISTRY_PATH.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return _blank()
    if "versions" not in raw or SEED_VERSION not in raw.get("versions", {}):
        return _blank()
    return raw


def _write(state: dict[str, Any]) -> None:
    config.ensure_dirs()
    tmp = config.REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(config.REGISTRY_PATH)


# --------------------------------------------------------------------------


def all_versions() -> list[Version]:
    state = _read()
    out = [Version(**v) for v in state["versions"].values()]
    out.sort(key=lambda v: int(v.version.lstrip("v")))
    return out


def get(version: str) -> Version | None:
    state = _read()
    raw = state["versions"].get(version)
    return Version(**raw) if raw else None


def active() -> Version:
    state = _read()
    raw = state["versions"].get(state.get("active") or SEED_VERSION)
    return Version(**(raw or state["versions"][SEED_VERSION]))


def next_version_name() -> str:
    numbers = [int(v.version.lstrip("v")) for v in all_versions()]
    return f"v{max(numbers) + 1}"


def register(version: Version) -> Version:
    with _lock:
        state = _read()
        state["versions"][version.version] = asdict(version)
        _write(state)
    return version


def set_metrics(version: str, metrics: dict[str, Any]) -> None:
    with _lock:
        state = _read()
        if version in state["versions"]:
            state["versions"][version]["metrics"] = metrics
            _write(state)


def activate(version: str) -> Version:
    """Promote a version to the one `/detect` serves."""
    with _lock:
        state = _read()
        if version not in state["versions"]:
            raise KeyError(version)
        state["active"] = version
        _write(state)
    return active()


def lineage(version: str) -> list[str]:
    """v3 -> ['v1', 'v2', 'v3'] — the chain of continued training."""
    chain: list[str] = []
    seen: set[str] = set()
    current: str | None = version
    while current and current not in seen:
        seen.add(current)
        chain.append(current)
        node = get(current)
        current = node.parent if node else None
    return list(reversed(chain))
