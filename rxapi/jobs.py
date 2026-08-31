"""Background jobs — training and evaluation take hours, HTTP requests do not.

A job is a thread plus a JSON record on disk. Deliberately in-process and
single-slot: this server owns one GPU, and a second training run alongside the
first would simply run both out of memory. `submit()` refuses while another job
is running.

Records survive a restart (the thread does not): a job left `running` by a
crash is reported as `interrupted` rather than lying about progress.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from . import config

PENDING, RUNNING, DONE, FAILED, INTERRUPTED = (
    "pending", "running", "done", "failed", "interrupted"
)

_lock = threading.Lock()
_current: dict[str, Any] | None = None       # the job this process is running
_thread: threading.Thread | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(job_id: str):
    return config.JOBS_DIR / f"{job_id}.json"


def _save(record: dict[str, Any]) -> None:
    config.ensure_dirs()
    tmp = _path(record["job_id"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(_path(record["job_id"]))


def get(job_id: str) -> dict[str, Any] | None:
    if _current is not None and _current["job_id"] == job_id:
        return dict(_current)
    path = _path(job_id)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    # A record still marked running with no live thread lost its process.
    if record.get("status") == RUNNING and (
        _current is None or _current["job_id"] != job_id
    ):
        record["status"] = INTERRUPTED
    return record


def recent(limit: int = 20) -> list[dict[str, Any]]:
    config.ensure_dirs()
    records = []
    for path in sorted(config.JOBS_DIR.glob("*.json"), reverse=True):
        record = get(path.stem)
        if record:
            records.append(record)
        if len(records) >= limit:
            break
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


def busy() -> bool:
    return _thread is not None and _thread.is_alive()


def current() -> dict[str, Any] | None:
    return dict(_current) if _current is not None else None


def submit(kind: str, params: dict[str, Any],
           work: Callable[[Callable[[str], None], dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Start `work(log, record)` in a thread. Raises RuntimeError if busy.

    `work` receives a `log` callable and the mutable record; whatever dict it
    returns is stored as the job's `result`.
    """
    global _current, _thread

    with _lock:
        if busy():
            raise RuntimeError(
                f"a {_current['kind']} job is already running ({_current['job_id']})"
            )
        record = {
            "job_id": uuid.uuid4().hex[:12],
            "kind": kind,
            "params": params,
            "status": RUNNING,
            "created_at": _now(),
            "finished_at": None,
            "progress": "",
            "log": [],
            "result": None,
            "error": None,
        }
        _current = record
        _save(record)

    def log(message: str) -> None:
        record["progress"] = str(message)
        record["log"].append(f"{_now()}  {message}")
        del record["log"][:-400]          # keep the tail bounded
        _save(record)

    def run() -> None:
        global _current
        try:
            record["result"] = work(log, record)
            record["status"] = DONE
        except Exception as exc:                              # noqa: BLE001
            record["status"] = FAILED
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["log"].append(traceback.format_exc())
        finally:
            record["finished_at"] = _now()
            _save(record)
            _current = None

    _thread = threading.Thread(target=run, name=f"rx-{kind}", daemon=True)
    _thread.start()
    return dict(record)
