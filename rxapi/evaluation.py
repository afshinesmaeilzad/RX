"""The adoption gate — a version is served only if it earned it.

Two questions, both answered against the pinned 200-image PadChest-GR list that
produced the published baseline:

1. did the benchmark regress?   (mean IoU, F1@0.5, hallucination@0.5)
2. is the difference real?      (paired bootstrap over the same images)

Scoring reuses `compare_models.py` — the module that produced the baseline in
`outputs/compare/`. A second implementation would drift, and then every
promotion decision would rest on numbers that are not comparable to the thesis.

`compare()` needs no GPU: it re-scores stored predictions. `run_version()` does,
because it has to generate them first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import config, registry

BASELINE_RUN = config.BENCH_OUTPUT_DIR / "per_model" / "cure.json"

# A candidate may not lose more than this on the headline metrics.
REGRESSION_TOLERANCE = {
    "mean_iou_micro": -0.005,
    "f1@0.5_micro": -0.005,
    "hallucination@0.5": 0.010,        # this one is better when it goes down
}


def _load_gt() -> dict[str, Any]:
    import compare_models as cm
    return cm.load_gt_index(str(config.GT_JSON))


def _headline(metrics: dict[str, Any]) -> dict[str, float]:
    t5 = metrics["thresholds"].get("0.50", {})
    return {
        "n_images": metrics["n_images"],
        "mean_iou_micro": round(metrics["mean_iou_micro"], 4),
        "f1@0.5_micro": round(t5.get("f1_micro", 0.0), 4),
        "precision@0.5": round(t5.get("precision_micro", 0.0), 4),
        "recall@0.5": round(t5.get("recall_micro", 0.0), 4),
        "hallucination@0.5": round(t5.get("hallucination_rate", 0.0), 4),
        "map_like_50_95": round(metrics["map_like_50_95"], 4),
        "keyword_f1_micro": round(metrics["keyword"]["f1_micro"], 4),
    }


def score_run(run_path: Path) -> dict[str, Any]:
    """Aggregate a stored run (predictions already generated) — no GPU."""
    import compare_models as cm

    run = json.loads(Path(run_path).read_text())
    gt = _load_gt()
    per_image = {}
    for image_id, payload in run["per_image"].items():
        if payload.get("error"):
            per_image[image_id] = payload
            continue
        entry = dict(payload)
        findings = cm.parse_grounded_report_cxcywh(entry.get("report_text", ""))
        gt_entry = gt.get(image_id)
        if gt_entry is None:
            entry["error"] = "no ground truth"
            per_image[image_id] = entry
            continue
        entry["detection"] = cm.evaluate_detection(
            findings, gt_entry, entry.get("box_format", "cxcywh"),
            tuple(entry["orig_size"]),
        )
        entry["keyword"] = cm.evaluate_keywords(findings, gt_entry)
        per_image[image_id] = entry
    return cm.aggregate_model(per_image)


def compare(candidate_run: Path,
            baseline_run: Path = BASELINE_RUN) -> dict[str, Any]:
    """Candidate vs baseline on the same images, with significance."""
    import compare_models as cm

    base = score_run(baseline_run)
    cand = score_run(candidate_run)
    base_head, cand_head = _headline(base), _headline(cand)

    deltas = {
        key: round(cand_head[key] - base_head[key], 4)
        for key in base_head if key != "n_images"
    }

    common = sorted(set(base["per_image"]) & set(cand["per_image"]))
    significance = {}
    for metric in ("iou", "f1@0.5", "kw_f1"):
        a = [cand["per_image"][i].get(metric, 0.0) for i in common]
        b = [base["per_image"][i].get(metric, 0.0) for i in common]
        stat = cm._paired_bootstrap(a, b)
        stat["significant"] = bool(stat["ci_low"] > 0 or stat["ci_high"] < 0)
        significance[metric] = stat

    regressions = [
        key for key, limit in REGRESSION_TOLERANCE.items()
        if (deltas.get(key, 0.0) < limit if limit < 0 else deltas.get(key, 0.0) > limit)
    ]
    return {
        "baseline": base_head,
        "candidate": cand_head,
        "delta": deltas,
        "significance": significance,
        "n_paired_images": len(common),
        "regressions": regressions,
        "passes_gate": not regressions,
    }


def run_version(log: Callable[[str], None], version: str,
                limit: int | None = None) -> Path:
    """Generate predictions for one adapter version over the pinned list (GPU)."""
    import compare_models as cm

    from .inference import CureService

    target = registry.get(version)
    if target is None:
        raise ValueError(f"unknown version: {version}")

    image_list_path = config.BENCH_OUTPUT_DIR / "image_list.json"
    if not image_list_path.is_file():
        raise RuntimeError(
            f"no pinned image list at {image_list_path}. The candidate must be "
            "scored on exactly the images the baseline used."
        )
    pinned = json.loads(image_list_path.read_text())
    image_ids = pinned["images"] if isinstance(pinned, dict) and "images" in pinned else pinned
    if limit:
        image_ids = image_ids[:limit]

    service = CureService()
    log(f"loading adapter {version}")
    service.load(version)

    gt = _load_gt()
    images_dir = Path(cm.IMAGES_DIR)
    per_image: dict[str, Any] = {}
    for i, image_id in enumerate(image_ids, 1):
        path = images_dir / image_id
        try:
            payload = service.detect(str(path))
            findings = cm.parse_grounded_report_cxcywh(payload["report_text"])
            entry = {
                "report_text": payload["report_text"],
                "box_format": "cxcywh",
                "orig_size": payload["image_size"],
                "keyword_findings": payload["findings"],
                "latency_s": payload["latency_s"],
                "error": None,
            }
            gt_entry = gt.get(image_id)
            if gt_entry is not None:
                entry["detection"] = cm.evaluate_detection(
                    findings, gt_entry, "cxcywh", tuple(payload["image_size"])
                )
                entry["keyword"] = cm.evaluate_keywords(findings, gt_entry)
            per_image[image_id] = entry
        except Exception as exc:                              # noqa: BLE001
            per_image[image_id] = {"error": f"{type(exc).__name__}: {exc}"}
        if i % 10 == 0:
            log(f"{i}/{len(image_ids)} images")

    service.unload()
    out_path = config.VAR_DIR / "evals" / f"{version}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"model": "cure", "adapter_version": version,
         "n_images": len(per_image), "per_image": per_image}, indent=2
    ))
    log(f"predictions written to {out_path}")
    return out_path


def evaluate(log: Callable[[str], None], version: str,
             limit: int | None = None) -> dict[str, Any]:
    """Generate, score, compare, and record the verdict on the version."""
    run_path = evaluation_run_path(version)
    if not run_path.is_file():
        run_path = run_version(log, version, limit=limit)
    log("scoring against the baseline")
    result = compare(run_path)
    metrics = dict(registry.get(version).metrics or {})
    metrics.update({"gate": result})
    registry.set_metrics(version, metrics)
    log("passed the gate" if result["passes_gate"]
        else f"regressed on: {', '.join(result['regressions'])}")
    return result


def evaluation_run_path(version: str) -> Path:
    return config.VAR_DIR / "evals" / f"{version}.json"
