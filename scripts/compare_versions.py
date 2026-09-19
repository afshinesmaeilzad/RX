#!/usr/bin/env python3
"""Paired comparison of stored runs — every CURE version against MAIRA-2.

The gate answers "is vN better than v1?". The thesis also has to answer "how
does the best version stand against the comparator?", on the same images, with
the same statistics. This does that from files already on disk — no GPU.

    python3 scripts/compare_versions.py \\
        --ref maira2=../cxr_gui/outputs/compare_test604/per_model/maira2.json \\
        --run v1=../cxr_gui/outputs/compare_test604/per_model/cure.json \\
        --run v3=../cxr_gui/outputs/cl_oracle_run_2026-09-17/rxvar/evals/v3.json \\
        --out ../cxr_gui/outputs/cl_oracle_run_2026-09-17/vs_maira2

Each run's per-image `detection` / `keyword` payloads are used as stored. They
are *not* re-parsed from `report_text`: MAIRA-2's text is xyxy and CURE's is
cxcywh, so one parser cannot serve both, and the stored payloads are what every
published number was computed from.

Guard: the aggregate recomputed here must equal the published headline
(`--expect label=f1`) to four decimals, or nothing is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_models as cm      # noqa: E402  (path set above)

METRICS = (("iou", "mean IoU"), ("f1@0.5", "F1@0.5"), ("kw_f1", "keyword F1"))


def load(path: Path) -> dict[str, Any]:
    run = json.loads(path.read_text())
    return cm.aggregate_model(run["per_image"])


def headline(agg: dict[str, Any]) -> dict[str, float]:
    t5 = agg["thresholds"].get("0.50", {})
    return {
        "n_images": agg["n_images"],
        "f1@0.5_micro": round(t5.get("f1_micro", 0.0), 4),
        "precision@0.5": round(t5.get("precision_micro", 0.0), 4),
        "recall@0.5": round(t5.get("recall_micro", 0.0), 4),
        "hallucination@0.5": round(t5.get("hallucination_rate", 0.0), 4),
        "mean_iou_micro": round(agg["mean_iou_micro"], 4),
    }


def pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """a − b over the images both scored."""
    common = sorted(set(a["per_image"]) & set(b["per_image"]))
    out: dict[str, Any] = {"n_paired": len(common)}
    for key, _ in METRICS:
        st = cm._paired_bootstrap([a["per_image"][i].get(key, 0.0) for i in common],
                                  [b["per_image"][i].get(key, 0.0) for i in common])
        st["significant"] = bool(st["ci_low"] > 0 or st["ci_high"] < 0)
        out[key] = st
    return out


def parse_pairs(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected label=path, got {item!r}")
        out.append((label, Path(path)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="label=path of the comparator run")
    ap.add_argument("--run", action="append", default=[], help="label=path, repeatable")
    ap.add_argument("--expect", action="append", default=[],
                    help="label=f1 — published F1@0.5 the recomputation must match")
    ap.add_argument("--out", type=Path, help="write <out>.json and <out>.md")
    args = ap.parse_args()

    (ref_label, ref_path), = parse_pairs([args.ref])
    runs = parse_pairs(args.run)

    aggs = {ref_label: load(ref_path)}
    for label, path in runs:
        aggs[label] = load(path)
    heads = {label: headline(agg) for label, agg in aggs.items()}

    ok = True
    for item in args.expect:
        label, _, value = item.partition("=")
        mine, theirs = heads[label]["f1@0.5_micro"], round(float(value), 4)
        match = abs(mine - theirs) < 1e-4
        ok &= match
        print(f"check {label}: F1@0.5 {mine:.4f} vs published {theirs:.4f} "
              f"{'ok' if match else 'MISMATCH'}")
    if not ok:
        print("recomputation does not match the published numbers — not reporting",
              file=sys.stderr)
        return 2

    results = {label: pair(aggs[label], aggs[ref_label]) for label, _ in runs}

    lines = [f"# CURE versions vs {ref_label} — paired bootstrap", "",
             f"Difference = version − {ref_label}, per image, over the images both "
             "scored. 2000 resamples, 95% CI, two-sided p. Aggregates are micro "
             "(pooled); the paired test is on per-image values, so the two can "
             "disagree in sign.", "",
             "| version | n | F1@0.5 | precision | recall | halluc | "
             "Δ IoU [95% CI] | Δ F1@0.5 [95% CI] | Δ keyword F1 [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    r = heads[ref_label]
    lines.append(f"| **{ref_label}** | {r['n_images']} | {r['f1@0.5_micro']:.4f} | "
                 f"{r['precision@0.5']:.3f} | {r['recall@0.5']:.3f} | "
                 f"{r['hallucination@0.5']:.3f} | — | — | — |")
    for label, _ in runs:
        h, p = heads[label], results[label]
        cells = []
        for key, _ in METRICS:
            st = p[key]
            star = " *" if st["significant"] else ""
            cells.append(f"{st['diff']:+.3f} [{st['ci_low']:+.3f}, {st['ci_high']:+.3f}] "
                         f"p={st['p_value']:.3f}{star}")
        lines.append(f"| {label} | {p['n_paired']} | {h['f1@0.5_micro']:.4f} | "
                     f"{h['precision@0.5']:.3f} | {h['recall@0.5']:.3f} | "
                     f"{h['hallucination@0.5']:.3f} | " + " | ".join(cells) + " |")
    lines += ["", "`*` = 95% CI excludes zero."]
    report = "\n".join(lines)
    print("\n" + report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".json").write_text(json.dumps(
            {"reference": ref_label, "headline": heads, "paired": results}, indent=2))
        args.out.with_suffix(".md").write_text(report + "\n")
        print(f"\nwrote {args.out.with_suffix('.md')} and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
