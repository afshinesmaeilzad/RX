"""The RX server API — inference for the desktop app, plus continual learning.

Two audiences, one service:

* **the app** — `GET /health` and `POST /detect` keep the exact contract the
  desktop client's `RemoteEngine` already speaks, so pointing it here is a URL
  change and nothing else.
* **the loop** — corrections come up, a training run continues the adapter, the
  gate scores the candidate against the pinned benchmark, and only then can a
  version be promoted to the one `/detect` serves.

Auth is a shared token in `X-Auth-Token`, the same header the app sends. It is
a gate for a tunnelled or private-network deployment, not an internet-facing
authentication system — do not expose this with patient data behind it.
"""

from __future__ import annotations

import hmac
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from . import config, dataset, evaluation, jobs, registry, training
from .inference import SERVICE

app = FastAPI(
    title="RX — CURE serving + continual learning",
    version="1.0",
    description=__doc__,
)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


def require_token(x_auth_token: str = Header(default="")) -> None:
    if not config.AUTH_TOKEN:
        return
    if not hmac.compare_digest(x_auth_token, config.AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="bad or missing X-Auth-Token")


Auth = Depends(require_token)


# --------------------------------------------------------------------------
# Serving — the contract the desktop app already speaks
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """Unauthenticated on purpose: it is how you find out the box is up."""
    return {
        **SERVICE.status(),
        "job": jobs.current(),
        "dataset": {"total": dataset.stats()["total_cases"],
                    "pending": dataset.stats()["pending_cases"]},
    }


@app.post("/detect", dependencies=[Auth])
async def detect(request: Request,
                 x_image_name: str = Header(default="upload.png")) -> dict[str, Any]:
    """Raw image bytes in, the worker's payload out."""
    if jobs.busy():
        raise HTTPException(
            status_code=503,
            detail=f"a {jobs.current()['kind']} job holds the GPU; try again later",
        )
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    if len(body) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large")

    suffix = Path(x_image_name).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(body)
        tmp.flush()
        try:
            payload = SERVICE.detect(tmp.name)
        except Exception as exc:                              # noqa: BLE001
            raise HTTPException(status_code=500,
                                detail=f"{type(exc).__name__}: {exc}") from exc
    # The server only ever saw a temp copy; the client knows the real path.
    payload["image_path"] = x_image_name
    return payload


@app.post("/model/load", dependencies=[Auth])
def load_model(version: str | None = Body(default=None, embed=True)) -> dict[str, Any]:
    """Pull the weights now instead of on the first X-ray."""
    try:
        SERVICE.load(version)
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"{type(exc).__name__}: {exc}") from exc
    return SERVICE.status()


# --------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------


@app.post("/corrections", dependencies=[Auth])
async def upload_corrections(request: Request) -> dict[str, Any]:
    """Accept the app's export: a JSON array, one object, or JSONL."""
    raw = (await request.body()).decode("utf-8", errors="replace").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="empty body")

    import json
    examples: list[dict[str, Any]] = []
    try:
        parsed = json.loads(raw)
        examples = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        for line in raw.splitlines():                 # JSONL
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400,
                                    detail=f"bad JSON line: {exc}") from exc

    receipt = dataset.ingest(examples)
    return {**receipt, "stats": dataset.stats()}


@app.get("/corrections/stats", dependencies=[Auth])
def correction_stats() -> dict[str, Any]:
    return dataset.stats()


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------


@app.get("/versions", dependencies=[Auth])
def list_versions() -> dict[str, Any]:
    active = registry.active()
    return {
        "active": active.version,
        "lineage": registry.lineage(active.version),
        "versions": [
            {**v.__dict__, "active": v.version == active.version,
             "gate": (v.metrics or {}).get("gate", {}).get("passes_gate")}
            for v in registry.all_versions()
        ],
    }


@app.post("/versions/{version}/activate", dependencies=[Auth])
def activate_version(version: str, force: bool = Body(default=False, embed=True)) -> dict[str, Any]:
    target = registry.get(version)
    if target is None:
        raise HTTPException(status_code=404, detail=f"unknown version {version}")

    gate = (target.metrics or {}).get("gate")
    if not target.is_seed and not force:
        if gate is None:
            raise HTTPException(
                status_code=409,
                detail=f"{version} has not been evaluated; run /evaluate first "
                       "(or pass force=true)",
            )
        if not gate.get("passes_gate"):
            raise HTTPException(
                status_code=409,
                detail=f"{version} regressed on: {', '.join(gate.get('regressions', []))}",
            )

    registry.activate(version)
    SERVICE.unload()          # the next detection loads the promoted weights
    return {"active": registry.active().version, "reloaded": False}


# --------------------------------------------------------------------------
# Jobs: training and evaluation
# --------------------------------------------------------------------------


@app.post("/train", dependencies=[Auth])
def start_training(
    epochs: int = Body(default=1, embed=True),
    lr: float = Body(default=2e-5, embed=True),
    replay_ratio: float | None = Body(default=None, embed=True),
    include_all: bool = Body(default=False, embed=True),
    force: bool = Body(default=False, embed=True),
    dry_run: bool = Body(default=False, embed=True),
) -> dict[str, Any]:
    """Continue the active adapter on the pending corrections.

    Refuses below the collection threshold unless forced — see
    `TRAIN_MIN_NEW_EXAMPLES`.
    """
    stats = dataset.stats()
    if not stats["ready_to_train"] and not force and not dry_run:
        raise HTTPException(
            status_code=409,
            detail=(f"{stats['pending_cases']} new case(s); "
                    f"{stats['short_by']} short of the {stats['threshold']} "
                    "needed. Pass force=true to override."),
        )

    def work(log, record):
        return training.train(
            log, epochs=epochs, lr=lr, replay_ratio=replay_ratio,
            include_all=include_all, force=force or dry_run, dry_run=dry_run,
            job_id=record["job_id"],
        )

    try:
        return jobs.submit("train", {"epochs": epochs, "lr": lr,
                                     "dry_run": dry_run}, work)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/evaluate", dependencies=[Auth])
def start_evaluation(
    version: str = Body(embed=True),
    limit: int | None = Body(default=None, embed=True),
) -> dict[str, Any]:
    """Score a version against the baseline on the pinned image list."""
    if registry.get(version) is None:
        raise HTTPException(status_code=404, detail=f"unknown version {version}")

    def work(log, record):
        return evaluation.evaluate(log, version, limit=limit)

    try:
        return jobs.submit("evaluate", {"version": version, "limit": limit}, work)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/jobs", dependencies=[Auth])
def list_jobs(limit: int = 20) -> dict[str, Any]:
    return {"running": jobs.current(), "jobs": jobs.recent(limit)}


@app.get("/jobs/{job_id}", dependencies=[Auth])
def get_job(job_id: str) -> dict[str, Any]:
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return record


@app.get("/config", dependencies=[Auth])
def show_config() -> dict[str, Any]:
    return config.summary()


@app.exception_handler(RuntimeError)
def runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})
