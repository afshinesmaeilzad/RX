#!/usr/bin/env python3
"""Simulate an oracle reviewer, from PadChest-GR ground truth.

A radiologist reviewing CURE's output would move a box that is in roughly the
right place, delete one that is not there, add one the model missed, and fix a
wrong keyword. PadChest-GR already contains a radiologist's answer for every
image, so those four edits can be *derived* rather than collected — turning
thousands of annotated studies into correction examples without a reviewer.

Call the result an **oracle reviewer**, never "radiologist corrections". It is
an upper bound and differs from a real reviewer in three ways worth stating in
writing: it sees every error, it is perfectly consistent, and it never makes a
mistake of its own. A null result from it is therefore strong — even flawless
corrections at this volume did not help.

    python3 scripts/simulate_corrections.py --run outputs/.../per_model/cure.json \\
        --split validation --out exports/oracle_validation.jsonl

Output is the JSONL the desktop app's exporter produces, so it can be POSTed
straight to the RX server's /corrections endpoint.

**Which split.** Default `validation`. CURE was trained on PadChest-GR *train*
(re-training there deepens memorisation) and *test* is the measuring stick.
Validation is the only split that is both unseen by the model and not the
yardstick — on the leaked run CURE scored 0.180 F1 there against 0.181 on test
and 0.422 on train, which is how we know it was held out.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_models as cm      # noqa: E402  (path set above)

PROMPT = cm.GROUNDED_REPORT_PROMPT


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def load_splits(master_csv: Path) -> dict[str, str]:
    splits: dict[str, str] = {}
    if not master_csv.is_file():
        return splits
    with master_csv.open() as fh:
        for row in csv.DictReader(fh):
            image_id = row.get("ImageID")
            if image_id:
                splits[image_id] = (row.get("split") or "").strip().lower()
    return splits


def gt_entries(gt_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Ground truth flattened to one entry per box, plus the box-less findings.

    A GT finding with no box is still part of the report CURE should produce —
    PadChest-GR annotates normal statements too — so it is kept with a null box
    rather than dropped.
    """
    out: list[dict[str, Any]] = []
    for finding in gt_entry.get("findings") or []:
        sentence = (finding.get("sentence_en") or "").strip().rstrip(".")
        if not sentence:
            continue
        boxes = finding.get("boxes") or []
        if not boxes:
            out.append({"sentence": sentence, "box_norm": None})
            continue
        for box in boxes:
            out.append({"sentence": sentence, "box_norm": [float(v) for v in box]})
    return out


def predicted_entries(report_text: str) -> list[dict[str, Any]]:
    """Model output flattened the same way: one entry per predicted box."""
    out: list[dict[str, Any]] = []
    for finding in cm.parse_grounded_report_cxcywh(report_text):
        sentence = (finding.get("sentence") or "").strip().rstrip(".")
        boxes = finding.get("boxes") or []
        if not boxes:
            if sentence:
                out.append({"sentence": sentence, "box_norm": None})
            continue
        for box in boxes:
            out.append({
                "sentence": sentence,
                "box_norm": cm.cxcywh_to_xyxy([float(v) for v in box]),
            })
    return out


def to_px(box_norm: list[float] | None, size: tuple[int, int]) -> list[float] | None:
    return cm.norm_xyxy_to_px(box_norm, size) if box_norm else None


def cxcywh_str(box_norm: list[float]) -> str:
    x1, y1, x2, y2 = box_norm
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return f"[{cx:.2f},{cy:.2f},{w:.2f},{h:.2f}]"


def build_target(entries: list[dict[str, Any]]) -> str:
    """The corrected report, in the format CURE itself emits."""
    sentences = []
    for entry in entries:
        keyword = (entry.get("keyword") or "").strip().rstrip(".")
        if not keyword:
            continue
        box = entry.get("box_norm")
        sentences.append(f"{keyword} {cxcywh_str(box)}" if box else keyword)
    if not sentences:
        return "No relevant findings"
    return ". ".join(sentences) + "."


# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------


def correct_one(report_text: str, gt_entry: dict[str, Any],
                size: tuple[int, int], match_iou: float) -> dict[str, Any]:
    """Diff one prediction against ground truth into reviewer-style edits.

    Matching is the benchmark's own greedy one-to-one IoU matcher in original
    pixel coordinates, so "what the oracle corrected" and "what the metric
    counted as a hit" can never disagree.
    """
    pred = predicted_entries(report_text)
    truth = gt_entries(gt_entry)

    pred_boxed = [p for p in pred if p["box_norm"]]
    pred_prose = [p for p in pred if not p["box_norm"]]
    gt_boxed = [g for g in truth if g["box_norm"]]

    pred_px = [to_px(p["box_norm"], size) for p in pred_boxed]
    gt_px = [to_px(g["box_norm"], size) for g in gt_boxed]

    matches = cm.greedy_match_boxes(pred_px, gt_px)
    # Below the threshold the reviewer would not call it the same finding: they
    # would delete the prediction and draw the real one instead.
    pairs = {i: (j, iou) for i, j, iou in matches if iou >= match_iou}
    matched_gt = {j for j, _ in pairs.values()}

    findings: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    moved = relabelled = added = kept = 0
    boxed_findings: list[dict[str, Any]] = []

    for i, p in enumerate(pred_boxed):
        pair = pairs.get(i)
        if pair is None:
            removed.append({
                "keyword": p["sentence"],
                "box_norm": p["box_norm"],
                "box_px": [round(v, 2) for v in (pred_px[i] or [])] or None,
                "removed_reason": "no ground-truth box matched",
            })
            continue
        j, iou = pair
        g = gt_boxed[j]
        same_words = cm._keyword_match(cm._norm_text(p["sentence"]),
                                       cm._norm_text(g["sentence"]))
        keyword = p["sentence"] if same_words else g["sentence"]
        box_changed = iou < 0.999
        if box_changed:
            moved += 1
        if not same_words:
            relabelled += 1
        if not box_changed and same_words:
            kept += 1
        boxed_findings.append({
            "keyword": keyword,
            "original_label": p["sentence"],
            "box_norm": [round(v, 6) for v in g["box_norm"]],
            "original_box": [round(v, 2) for v in (pred_px[i] or [])] or None,
            "box_px": [round(v, 2) for v in (gt_px[j] or [])] or None,
            "source": "corrected" if (box_changed or not same_words) else "model",
            "status": "Approved",
            "iou_before": round(iou, 4),
        })

    for j, g in enumerate(gt_boxed):
        if j in matched_gt:
            continue
        added += 1
        boxed_findings.append({
            "keyword": g["sentence"],
            "original_label": None,
            "box_norm": [round(v, 6) for v in g["box_norm"]],
            "original_box": None,
            "box_px": [round(v, 2) for v in (gt_px[j] or [])] or None,
            "source": "reviewer",
            "status": "Approved",
            "iou_before": 0.0,
        })

    # Unboxed sentences the model produced are LEFT ALONE. A reviewer corrects
    # boxes; they do not delete "no pleural effusion" from the prose. Dropping
    # them would also teach the model to stop describing normal findings, and
    # `rescore_offline.py --fix drop-negations` already showed that costs
    # keyword F1 (-0.0069, p=0.019). The reference is often terser than the
    # model here — one GT finding against five normal statements is common —
    # and that terseness is an artefact of how PadChest-GR decomposed the
    # report, not an error to train away.
    said = [cm._norm_text(p["sentence"]) for p in pred_prose]
    for p in pred_prose:
        kept += 1
        findings.append({
            "keyword": p["sentence"],
            "original_label": p["sentence"],
            "box_norm": None,
            "original_box": None,
            "box_px": None,
            "source": "model",
            "status": "Approved",
            "iou_before": None,
        })

    # Findings the reference names without localising, that the model did not
    # already say, are added.
    for g in truth:
        if g["box_norm"] is not None:
            continue
        if any(cm._keyword_match(cm._norm_text(g["sentence"]), s) for s in said):
            continue
        added += 1
        findings.append({
            "keyword": g["sentence"],
            "original_label": None,
            "box_norm": None,
            "original_box": None,
            "box_px": None,
            "source": "reviewer",
            "status": "Approved",
            "iou_before": None,
        })

    findings.extend(boxed_findings)      # prose first, then the localised findings

    ious = [f["iou_before"] for f in findings
            if f.get("iou_before") not in (None, 0.0)]
    return {
        "findings": findings,
        "removed": removed,
        "corrections": {
            "moved": moved,
            "relabelled": relabelled,
            "added": added,
            "removed": len(removed),
            "kept": kept,
            "mean_iou_before_after": round(sum(ious) / len(ious), 4) if ious else None,
        },
    }


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="a finished run's per_model/<model>.json")
    ap.add_argument("--gt", type=Path, default=Path(cm.JSON_PATH))
    ap.add_argument("--master-csv", type=Path,
                    default=Path(cm.DATA_DIR) / "master_table.csv")
    ap.add_argument("--images-dir", type=Path, default=Path(cm.IMAGES_DIR))
    ap.add_argument("--split", default="validation",
                    help="which official split to draw from (default validation; "
                         "train is already seen by CURE and test is the yardstick)")
    ap.add_argument("--match-iou", type=float, default=0.3,
                    help="below this a prediction is deleted rather than moved")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("exports/oracle_corrections.jsonl"))
    args = ap.parse_args()

    for path in (args.run, args.gt):
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            return 1

    run = json.loads(args.run.read_text())
    gt_by_id = cm.load_gt_index(str(args.gt))
    splits = load_splits(args.master_csv)
    if args.split and not splits:
        print(f"no split table at {args.master_csv}", file=sys.stderr)
        return 1

    examples, skipped = [], Counter()
    for image_id, payload in run.get("per_image", {}).items():
        if args.split and splits.get(image_id) != args.split:
            skipped["wrong split"] += 1
            continue
        if payload.get("error") or "orig_size" not in payload:
            skipped["failed prediction"] += 1
            continue
        gt_entry = gt_by_id.get(image_id)
        if gt_entry is None:
            skipped["no ground truth"] += 1
            continue
        image_path = args.images_dir / image_id
        if not image_path.is_file():
            skipped["image missing"] += 1
            continue

        size = tuple(payload["orig_size"])
        result = correct_one(payload.get("report_text", ""), gt_entry,
                             size, args.match_iou)
        examples.append({
            "case_id": f"oracle:{image_id}",
            "image_path": str(image_path),
            "image_size": list(size),
            "prompt": PROMPT,
            "target": build_target(result["findings"]),
            "model_output": payload.get("report_text", ""),
            "status": "Approved",
            "origin": "oracle",
            "split": args.split or "all",
            "note": "corrections derived from PadChest-GR reference annotations",
            "review": "",
            "findings": result["findings"],
            "removed": result["removed"],
            "corrections": result["corrections"],
        })
        if args.limit and len(examples) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for example in examples:
            fh.write(json.dumps(example) + "\n")

    total = Counter()
    for e in examples:
        for key, value in e["corrections"].items():
            if isinstance(value, int):
                total[key] += value
    boxes = total["kept"] + total["moved"] + total["removed"]

    print(f"\nwrote {len(examples)} oracle example(s) -> {args.out}")
    for reason, n in skipped.items():
        print(f"  skipped {n} ({reason})")
    if not examples:
        print("\nNothing produced. Is --split right for this run?")
        return 0

    print(f"\nWhat the oracle changed  ({len(examples)} case(s), "
          f"{boxes} predicted boxes)")
    print(f"  accepted as-is        {total['kept']:5d}")
    print(f"  box moved to GT       {total['moved']:5d}")
    print(f"  keyword replaced      {total['relabelled']:5d}")
    print(f"  deleted (false pos.)  {total['removed']:5d}")
    print(f"  added (missed)        {total['added']:5d}")
    if boxes:
        print(f"\n  {total['removed'] / boxes:.0%} of predicted boxes had no "
              f"ground-truth match at IoU {args.match_iou} and were deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
