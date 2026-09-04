#!/usr/bin/env python3
"""Re-score a finished run from its stored text — no model, no GPU.

`outputs/compare/per_model/<model>.json` keeps the raw `report_text` for every
image. That is enough to answer "would this post-processing rule have helped?"
without touching a GPU: re-parse the stored text, apply the rule, and score the
result with the *same* functions `compare_models.py` uses, on the same pinned
image list.

    python3 rescore_offline.py --check                  # reproduce the published numbers
    python3 rescore_offline.py --fix dedup              # measure one rule
    python3 rescore_offline.py --fix dedup,drop-negations

`--check` is the guard: with no rule applied the aggregate must equal
`comparison_summary.csv` to four decimals. If it does not, this harness is
lying and nothing it reports about a rule can be trusted.

The rules here come from what reviewers actually corrected in the app (see
`cxr_gui/scripts/export_corrections.py`): CURE repeats sentences, and it emits
normal-finding statements that carry no box and cannot be matched.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_models as cm      # noqa: E402  (path set above)

NEGATION_MARKERS = (
    "no ", "without", "not present", "does not", "is not", "are not",
    "no evidence", "unremarkable", "normal", "no alterations", "no significant",
)


# --------------------------------------------------------------------------
# Post-processing rules, each: findings -> findings
# --------------------------------------------------------------------------


def rule_dedup(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeated sentences, and repeated boxes inside one sentence.

    CURE restates the same finding several times ("Suggestive of a distended
    gastric chamber" three times on one film). Every repeat is a fresh
    predicted box, so each one is a false positive that cannot match anything —
    the ground truth has only one.
    """
    seen_sentences: set[str] = set()
    out: list[dict[str, Any]] = []
    for f in findings:
        key = cm._norm_text(f.get("sentence", ""))
        if key and key in seen_sentences:
            continue
        if key:
            seen_sentences.add(key)
        boxes, seen_boxes = [], set()
        for b in f.get("boxes") or []:
            bkey = tuple(round(v, 3) for v in b)
            if bkey in seen_boxes:
                continue
            seen_boxes.add(bkey)
            boxes.append(b)
        out.append({**f, "boxes": boxes})
    return out


def rule_drop_negations(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop "no pleural effusion" style statements.

    They describe absent findings, carry no box, and PadChest-GR only annotates
    what *is* there — so they can only ever be keyword false positives. This
    cannot change box metrics; it is here to show what it does to keyword F1.
    """
    out = []
    for f in findings:
        text = cm._norm_text(f.get("sentence", ""))
        if not (f.get("boxes") or []) and any(text.startswith(m) or f" {m}" in text
                                              for m in NEGATION_MARKERS):
            continue
        out.append(f)
    return out


def make_scale_rule(factor: float) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Scale every box about its own centre.

    Matched boxes average IoU 0.50, so the geometry is roughly right and the
    size is not. If CURE is systematically generous, one constant shrink lifts
    IoU everywhere — the cheapest possible calibration, and measurable here
    without re-running anything. Boxes are cxcywh, so this is two multiplies.
    """
    def rule(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for f in findings:
            boxes = []
            for cx, cy, w, h in f.get("boxes") or []:
                boxes.append([cx, cy, w * factor, h * factor])
            out.append({**f, "boxes": boxes})
        return out
    return rule


RULES: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "dedup": rule_dedup,
    "drop-negations": rule_drop_negations,
}


def resolve_rule(name: str) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """`dedup`, or a parametrised rule such as `scale:0.9`."""
    if name in RULES:
        return RULES[name]
    if name.startswith("scale:"):
        return make_scale_rule(float(name.split(":", 1)[1]))
    raise KeyError(name)


# --------------------------------------------------------------------------


def load_splits(master_csv: Path) -> dict[str, str]:
    """ImageID -> official PadChest-GR split (train / validation / test)."""
    import csv
    splits: dict[str, str] = {}
    if not master_csv.is_file():
        return splits
    with master_csv.open() as fh:
        for row in csv.DictReader(fh):
            image_id = row.get("ImageID")
            if image_id:
                splits[image_id] = (row.get("split") or "").strip()
    return splits


def rescore(run: dict[str, Any], gt_by_id: dict[str, Any],
            rules: list[str], keep_ids: set[str] | None = None) -> dict[str, Any]:
    """Re-parse every stored prediction, apply the rules, score it again."""
    per_image: dict[str, dict[str, Any]] = {}
    for image_id, payload in run["per_image"].items():
        if keep_ids is not None and image_id not in keep_ids:
            continue
        entry = copy.deepcopy(payload)
        if entry.get("error") or "orig_size" not in entry:
            per_image[image_id] = entry
            continue

        findings = cm.parse_grounded_report_cxcywh(entry.get("report_text", ""))
        for name in rules:
            findings = resolve_rule(name)(findings)

        gt_entry = gt_by_id.get(image_id)
        if gt_entry is None:
            entry["error"] = "no ground truth"
            per_image[image_id] = entry
            continue

        orig_size = tuple(entry["orig_size"])
        entry["detection"] = cm.evaluate_detection(
            findings, gt_entry, entry.get("box_format", "cxcywh"), orig_size
        )
        entry["keyword"] = cm.evaluate_keywords(findings, gt_entry)
        per_image[image_id] = entry
    return cm.aggregate_model(per_image)


def headline(metrics: dict[str, Any]) -> dict[str, float]:
    t5 = metrics["thresholds"].get("0.50", {})
    return {
        "mean_iou_micro": round(metrics["mean_iou_micro"], 4),
        "f1@0.5_micro": round(t5.get("f1_micro", 0.0), 4),
        "precision@0.5": round(t5.get("precision_micro", 0.0), 4),
        "recall@0.5": round(t5.get("recall_micro", 0.0), 4),
        "hallucination@0.5": round(t5.get("hallucination_rate", 0.0), 4),
        "map_like_50_95": round(metrics["map_like_50_95"], 4),
        "keyword_f1_micro": round(metrics["keyword"]["f1_micro"], 4),
        "total_pred_boxes": metrics["total_pred_boxes"],
    }


def published(summary_csv: Path, model: str) -> dict[str, str] | None:
    if not summary_csv.is_file():
        return None
    import csv
    with summary_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("model") == model:
                return row
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path,
                    default=Path(cm.PER_MODEL_DIR) / "cure.json")
    ap.add_argument("--gt", type=Path, default=Path(cm.JSON_PATH))
    ap.add_argument("--fix", default="",
                    help="comma-separated rules: " + ", ".join(RULES))
    ap.add_argument("--check", action="store_true",
                    help="only verify the harness reproduces the published run")
    ap.add_argument("--split", default="",
                    help="restrict to an official PadChest-GR split: train|validation|test. "
                         "Both CURE and MAIRA-2 were trained on PadChest-GR, so only "
                         "'test' is a generalization estimate.")
    ap.add_argument("--master-csv", type=Path,
                    default=Path(cm.DATA_DIR) / "master_table.csv")
    args = ap.parse_args()

    if not args.run.is_file():
        print(f"no run at {args.run}", file=sys.stderr)
        return 1
    if not args.gt.is_file():
        print(f"no ground truth at {args.gt}", file=sys.stderr)
        return 1

    run = json.loads(args.run.read_text())
    gt_by_id = cm.load_gt_index(str(args.gt))
    model = run.get("model", args.run.stem)

    keep_ids = None
    if args.split:
        splits = load_splits(args.master_csv)
        if not splits:
            print(f"no split table at {args.master_csv}", file=sys.stderr)
            return 1
        keep_ids = {i for i, s in splits.items() if s == args.split}
        print(f"restricted to the '{args.split}' split "
              f"({len(keep_ids & set(run['per_image']))} of "
              f"{len(run['per_image'])} scored images)")

    base = rescore(run, gt_by_id, [], keep_ids)
    base_head = headline(base)

    # ------------------------------------------------------ the guard
    # The published summary covers the whole run; a split subset cannot match it.
    ref = None if args.split else published(
        args.run.parent.parent / "comparison_summary.csv", model)
    print(f"re-scored {base['n_images']} image(s) from {args.run.name}\n")
    if ref:
        print("harness check — recomputed vs published")
        ok = True
        for key in ("mean_iou_micro", "f1@0.5_micro", "hallucination@0.5",
                    "map_like_50_95", "keyword_f1_micro"):
            mine, theirs = base_head[key], round(float(ref[key]), 4)
            match = abs(mine - theirs) < 1e-4
            ok &= match
            print(f"  {key:20s} {mine:.4f}  vs  {theirs:.4f}   "
                  f"{'ok' if match else 'MISMATCH'}")
        if not ok:
            print("\nThe harness does not reproduce the run — do not trust any "
                  "delta it reports.", file=sys.stderr)
            return 2
        print()
    else:
        print("(no comparison_summary.csv next to the run; skipping the check)\n")

    if args.check or not args.fix:
        return 0

    # ------------------------------------------------------ the rules
    rules = [r.strip() for r in args.fix.split(",") if r.strip()]
    try:
        for r in rules:
            resolve_rule(r)
    except (KeyError, ValueError):
        print(f"unknown rule in {rules}. Known: {list(RULES)} or scale:<factor>",
              file=sys.stderr)
        return 1

    fixed = rescore(run, gt_by_id, rules, keep_ids)
    fixed_head = headline(fixed)

    print(f"effect of: {' + '.join(rules)}\n")
    print(f"  {'metric':20s} {'baseline':>10s} {'fixed':>10s} {'delta':>10s}")
    for key in base_head:
        b, f = base_head[key], fixed_head[key]
        delta = f - b
        arrow = "" if abs(delta) < 1e-9 else ("  +" if delta > 0 else "  ")
        print(f"  {key:20s} {b:>10.4f} {f:>10.4f} {arrow}{delta:>8.4f}")

    # Paired over the same images, exactly as the thesis report does it.
    common = sorted(set(base["per_image"]) & set(fixed["per_image"]))
    print(f"\npaired bootstrap over {len(common)} image(s) (fixed - baseline)")
    for metric in ("iou", "f1@0.5", "kw_f1"):
        a = [fixed["per_image"][i].get(metric, 0.0) for i in common]
        b = [base["per_image"][i].get(metric, 0.0) for i in common]
        st = cm._paired_bootstrap(a, b)
        verdict = "significant" if (st["ci_low"] > 0 or st["ci_high"] < 0) else "not significant"
        print(f"  {metric:8s} diff {st['diff']:+.4f}  "
              f"95% CI [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}]  "
              f"p={st['p_value']:.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
