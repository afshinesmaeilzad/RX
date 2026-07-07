#!/usr/bin/env python3
"""
Compare CURE and MAIRA-2 on PadChest-GR grounded finding localization.

Each model runs in its OWN process/venv (their transformers versions conflict):
    - MAIRA-2 : transformers==4.51.3            (requirements-maira2.txt)
    - CURE    : transformers==4.55.4 + peft     (requirements-cure.txt)

Everything runs in bf16 FULL PRECISION (no 4-bit / no bitsandbytes), so a single
GPU with >=24 GB VRAM is recommended (each model is loaded one at a time).

The pipeline is three subcommands so the heavy model load happens in the right venv:

    python compare_models.py select                 # pin the shared image list (once)
    python compare_models.py run --model maira2      # in .venv-maira2
    python compare_models.py run --model cure        # in .venv-cure
    python compare_models.py report                  # aggregate + report (no transformers)

The `select` step writes outputs/compare/image_list.json ({seed, n_images, selected}).
Both `run` invocations read that SAME file, so the two models score identical images.
Each `run` writes outputs/compare/per_model/<model>.json with, per image, the simple
{keyword, box} findings (the payload a later OpenAI step will turn into a narrative
report) plus all per-image detection numbers. `report` aggregates them.

Scope (per project decision):
    - Models output only KEYWORD findings + boxes, not a full narrative report.
    - Metrics are detection (box IoU in original pixel space, greedy multi-box matching,
      P/R/F1 @ 0.3/0.5, a score-free mAP-like sweep) + a lightweight keyword-vs-GT-label
      match (stdlib difflib). NO RadGraph/CheXbert/BLEU/ROUGE.
    - Bootstrap CIs + a paired significance test back the CURE-vs-MAIRA claims.

Env vars:
    DATA_DIR      dataset root (grounded_reports_*.json + Padchest_GR_files/)  [default: script dir]
    JSON_PATH     override GT json path
    IMAGES_DIR    override images dir
    OUTPUT_DIR    output dir           [default: <script>/outputs/compare]
    N_IMAGES      how many images      [default: 200]
    SHUFFLE_SEED  selection seed       [default: 42]
    DEVICE        cuda | mps | cpu     [default: cpu]
    MODELS        subset for `run`/`report` order [default: cure,maira2]
    SAVE_FIGURES  1 = save per-image preview pngs in `run` [default: 0]
    HF_TOKEN      HuggingFace token (gated MAIRA-2 / MedGemma-4B base + CURE adapter)
"""

from __future__ import annotations

import argparse
import csv
import difflib
import gc
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _maybe_mount_drive() -> None:
    """Mount Google Drive when running in Colab with MOUNT_DRIVE=1."""
    if os.environ.get("MOUNT_DRIVE", "0") != "1":
        return
    try:
        from google.colab import drive  # type: ignore

        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
            print("Mounted Google Drive at /content/drive")
    except Exception as exc:  # pragma: no cover - only runs in Colab
        print(f"[WARN] Could not mount Google Drive: {exc}")


def _dir_has_images(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        for fn in os.listdir(path):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                return True
    except OSError:
        return False
    return False


def _resolve_images_dir(data_dir: str) -> str:
    """
    Locate the folder holding the PNGs. On Google Drive the dataset is nested as
    Padchest_GR_files/PadChest_GR/, while a local extract keeps the PNGs directly
    under Padchest_GR_files/. Prefer the candidate that actually contains images.
    """
    candidates = [
        os.path.join(data_dir, "Padchest_GR_files", "PadChest_GR"),
        os.path.join(data_dir, "Padchest_GR_files"),
    ]
    for candidate in candidates:
        if _dir_has_images(candidate):
            return candidate
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[-1]


_maybe_mount_drive()

DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
JSON_PATH = os.environ.get("JSON_PATH") or os.path.join(DATA_DIR, "grounded_reports_20240819.json")
IMAGES_DIR = os.environ.get("IMAGES_DIR") or _resolve_images_dir(DATA_DIR)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(BASE_DIR, "outputs", "compare")

PER_MODEL_DIR = os.path.join(OUTPUT_DIR, "per_model")
IMAGE_LIST_PATH = os.path.join(OUTPUT_DIR, "image_list.json")

N_IMAGES = int(os.environ.get("N_IMAGES", "200"))
SHUFFLE_SEED = int(os.environ["SHUFFLE_SEED"]) if os.environ.get("SHUFFLE_SEED") else 42
DEFAULT_VERIFY_IMAGE = "106997070894779966614346591942916625787_fsxv2a.png"

ALL_MODELS = ["cure", "maira2"]
MODELS = [
    m.strip().lower()
    for m in os.environ.get("MODELS", "cure,maira2").split(",")
    if m.strip()
]

CURE_BASE_ID = "google/medgemma-4b-it"
CURE_ADAPTER_ID = "pamessina/medgemma-4b-it-cure"
MAIRA2_ID = "microsoft/maira-2"

CURE_IMAGE_SIZE = 448
CURE_CLAHE_CLIP_LIMIT = 3.0
CURE_CLAHE_TILE_GRID = (8, 8)

MODEL_COLORS = {
    "cure": "#ff5252",
    "maira2": "#448aff",
}

# Simple prompt (per project decision). CURE emits short grounded findings + boxes.
GROUNDED_REPORT_PROMPT = "Generate a grounded report."

# Detection IoU thresholds. Headline reporting uses 0.3 and 0.5; the sweep
# 0.50:0.05:0.95 feeds a score-free mAP-like surrogate (models emit no box scores).
HEADLINE_THRESHOLDS = (0.3, 0.5)
SWEEP_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50 .. 0.95
ALL_THRESHOLDS = tuple(sorted({0.3, *SWEEP_THRESHOLDS}))

# Keyword-match acceptance ratio (stdlib difflib).
KEYWORD_MATCH_RATIO = 0.8


def _tk(t: float) -> str:
    """Canonical string key for a threshold (e.g. 0.5 -> '0.50')."""
    return f"{t:.2f}"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeviceConfig:
    device: torch.device
    dtype: torch.dtype


@dataclass
class ImageSample:
    image_id: str
    image_path: str
    gt_entry: dict[str, Any]
    raw_image: Image.Image | None = None
    cure_image: Image.Image | None = None
    orig_size: tuple[int, int] = (0, 0)  # (W, H) of the ORIGINAL image, for pixel-space IoU
    gt_boxes_xyxy: list[list[float]] = field(default_factory=list)
    gt_sentences: list[str] = field(default_factory=list)


@dataclass
class ModelRunResult:
    model_key: str
    report_text: str
    pred_findings: list[dict[str, Any]]
    box_format: str
    latency_s: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Auth + device
# ---------------------------------------------------------------------------


def setup_hf_auth() -> None:
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)
    else:
        login()


def resolve_device_config() -> DeviceConfig:
    """bf16 full precision only. No quantization."""
    requested = os.environ.get("DEVICE", os.environ.get("CURE_DEVICE", "cpu")).lower()

    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[WARN] MPS is experimental; use DEVICE=cpu if outputs are empty.")
    else:
        device = torch.device("cpu")
        if requested == "cuda":
            print("[WARN] CUDA requested but unavailable; falling back to CPU.")

    dtype = torch.bfloat16
    print(f"Device: {device}  dtype: {dtype}  (bf16 full precision, no 4-bit)")
    return DeviceConfig(device=device, dtype=dtype)


def free_model(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Image loading / geometry / drawing
# ---------------------------------------------------------------------------


def load_raw_xray_rgb(path: str) -> Image.Image:
    img_np = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_np is None:
        raise IOError(f"cv2.imread failed to load image: {path}")

    if img_np.dtype == np.uint16:
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_np)


def load_xray_as_rgb_cure(path: str) -> Image.Image:
    """Official CURE pipeline: CLAHE + resize 448x448."""
    img_np = np.array(load_raw_xray_rgb(path))
    clahe = cv2.createCLAHE(
        clipLimit=CURE_CLAHE_CLIP_LIMIT,
        tileGridSize=CURE_CLAHE_TILE_GRID,
    )
    channels = [clahe.apply(c) for c in cv2.split(img_np)]
    img_np = cv2.merge(channels)
    img_np = cv2.resize(
        img_np,
        (CURE_IMAGE_SIZE, CURE_IMAGE_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    return Image.fromarray(img_np)


def cxcywh_to_xyxy(box: list[float]) -> list[float]:
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_xyxy_boxes(image: Image.Image, boxes: list[list[float]], color: str, width: int = 4) -> Image.Image:
    """Draw normalized [0,1] xyxy boxes."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=color, width=width)
    return img


def draw_cxcywh_boxes(image: Image.Image, boxes: list[list[float]], color: str, width: int = 4) -> Image.Image:
    return draw_xyxy_boxes(image, [cxcywh_to_xyxy(b) for b in boxes], color=color, width=width)


def finding_boxes_to_xyxy_norm(box: list[float], box_format: str) -> list[float]:
    """Normalize a single predicted box (cxcywh or xyxy) to normalized xyxy [0,1]."""
    return cxcywh_to_xyxy(box) if box_format == "cxcywh" else [float(x) for x in box]


def norm_xyxy_to_px(box_norm: list[float], size: tuple[int, int]) -> list[float]:
    w, h = size
    return [box_norm[0] * w, box_norm[1] * h, box_norm[2] * w, box_norm[3] * h]


# ---------------------------------------------------------------------------
# Parsing (CURE grounded output -> findings)
# ---------------------------------------------------------------------------


def parse_grounded_report_cxcywh(text: str) -> list[dict[str, Any]]:
    bbox_pat = re.compile(
        r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,"
        r"\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
    )
    findings = []
    for raw_sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        s = raw_sent.strip().rstrip(".")
        if not s:
            continue
        boxes = [[float(x) for x in m.groups()] for m in bbox_pat.finditer(s)]
        sentence_clean = bbox_pat.sub("", s).strip(" ,.")
        findings.append({"sentence": sentence_clean, "boxes": boxes})
    return findings


def findings_to_report_text(findings: list[dict[str, Any]], box_format: str) -> str:
    parts = []
    for f in findings:
        sentence = f.get("sentence", "").strip()
        boxes = f.get("boxes") or []
        if boxes:
            box_str = ", ".join(str(list(b)) for b in boxes)
            parts.append(f"{sentence} [{box_str}]")
        else:
            parts.append(sentence)
    return ". ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Evaluation: detection (pixel-space) + keyword match
# ---------------------------------------------------------------------------


def greedy_match_boxes(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU. Returns (pred_idx, gt_idx, iou)."""
    candidates: list[tuple[float, int, int]] = []
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou = iou_xyxy(pb, gb)
            if iou > 0:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)

    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, iou))
    return matches


def evaluate_detection(
    pred_findings: list[dict[str, Any]],
    gt_entry: dict[str, Any],
    box_format: str,
    orig_size: tuple[int, int],
) -> dict[str, Any]:
    """
    IoU is computed in ORIGINAL-image PIXEL coordinates for every model, so aspect
    ratio is honoured identically (CURE cxcywh, MAIRA xyxy and GT are all mapped to
    original W x H px first). Matching is greedy one-to-one over ALL predicted boxes.
    """
    pred_px: list[list[float]] = []
    for f in pred_findings:
        for b in f.get("boxes") or []:
            xyxy = finding_boxes_to_xyxy_norm(b, box_format)
            pred_px.append(norm_xyxy_to_px(xyxy, orig_size))

    gt_px: list[list[float]] = []
    gt_meta: list[dict[str, Any]] = []
    for f in gt_entry.get("findings", []) or []:
        for box in f.get("boxes") or []:  # GT boxes are normalized xyxy
            gt_px.append(norm_xyxy_to_px([float(x) for x in box], orig_size))
            gt_meta.append({"labels": f.get("labels") or [], "sentence": f.get("sentence_en", "")})

    matches = greedy_match_boxes(pred_px, gt_px)
    matched_ious = [iou for _, _, iou in matches]

    gt_best = [0.0] * len(gt_px)
    for _, j, iou in matches:
        gt_best[j] = iou

    n_pred = len(pred_px)
    n_gt = len(gt_px)
    per_threshold: dict[str, dict[str, int]] = {}
    for t in ALL_THRESHOLDS:
        tp = sum(1 for iou in matched_ious if iou >= t)
        per_threshold[_tk(t)] = {"tp": tp, "fp": n_pred - tp, "fn": n_gt - tp}

    return {
        "n_pred_boxes": n_pred,
        "n_gt_boxes": n_gt,
        "matched_ious": matched_ious,
        "mean_iou": (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0,
        "recall@0.3": (sum(1 for i in matched_ious if i >= 0.3) / n_gt) if n_gt else 0.0,
        "recall@0.5": (sum(1 for i in matched_ious if i >= 0.5) / n_gt) if n_gt else 0.0,
        "per_threshold": per_threshold,
        "gt_detail": [
            {"labels": m["labels"], "best_iou": b} for m, b in zip(gt_meta, gt_best)
        ],
    }


def _norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _keyword_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= KEYWORD_MATCH_RATIO


def evaluate_keywords(
    pred_findings: list[dict[str, Any]],
    gt_entry: dict[str, Any],
) -> dict[str, Any]:
    """Lightweight keyword-vs-GT-label match (stdlib difflib, no NLG libs)."""
    preds = [_norm_text(f.get("sentence", "")) for f in pred_findings]
    preds = [p for p in preds if p]

    gts: list[str] = []
    for f in gt_entry.get("findings", []) or []:
        labels = [_norm_text(l) for l in (f.get("labels") or []) if _norm_text(l)]
        if labels:
            gts.extend(labels)
        else:
            s = _norm_text(f.get("sentence_en", ""))
            if s:
                gts.append(s)

    used_g: set[int] = set()
    tp = 0
    for p in preds:
        for j, g in enumerate(gts):
            if j in used_g:
                continue
            if _keyword_match(p, g):
                used_g.add(j)
                tp += 1
                break
    fp = len(preds) - tp
    fn = len(gts) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "n_pred": len(preds), "n_gt": len(gts)}


# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------


def has_gt_box(entry: dict[str, Any]) -> bool:
    for f in entry.get("findings", []) or []:
        if f.get("boxes"):
            return True
    return False


def load_gt_index(json_path: str) -> dict[str, dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["ImageID"]: entry for entry in data if entry.get("ImageID")}


def select_images(
    images_dir: str,
    gt_by_id: dict[str, dict[str, Any]],
    n_images: int,
    shuffle_seed: int | None,
    prefer_image: str | None = None,
) -> list[str]:
    all_files = sorted(
        fn for fn in os.listdir(images_dir)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    candidates = [fn for fn in all_files if fn in gt_by_id and has_gt_box(gt_by_id[fn])]

    selected: list[str] = []
    if prefer_image and prefer_image in candidates:
        selected.append(prefer_image)
        candidates = [c for c in candidates if c != prefer_image]

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(candidates)

    for fn in candidates:
        if len(selected) >= n_images:
            break
        selected.append(fn)

    if not selected:
        raise RuntimeError("No images with ground-truth boxes found.")
    return selected[:n_images]


def build_sample(image_id: str, gt_by_id: dict[str, dict[str, Any]], which: str) -> ImageSample:
    """Build one sample, loading only the image variant the model needs."""
    image_path = os.path.join(IMAGES_DIR, image_id)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing image: {image_path}")

    gt_entry = gt_by_id[image_id]
    gt_boxes, gt_sentences = [], []
    for f in gt_entry.get("findings", []) or []:
        gt_sentences.append(f.get("sentence_en", ""))
        gt_boxes.extend(f.get("boxes") or [])

    # Original (W, H) from the file header (cheap, no full decode) - CURE resizes to a
    # 448 square, but IoU must be evaluated in ORIGINAL pixel space for both models.
    with Image.open(image_path) as _im:
        orig_size = _im.size  # (W, H)

    sample = ImageSample(
        image_id=image_id,
        image_path=image_path,
        gt_entry=gt_entry,
        orig_size=orig_size,
        gt_boxes_xyxy=gt_boxes,
        gt_sentences=gt_sentences,
    )
    if which == "cure":
        sample.cure_image = load_xray_as_rgb_cure(image_path)
    else:
        sample.raw_image = load_raw_xray_rgb(image_path)
    return sample


# ---------------------------------------------------------------------------
# Model runners (bf16 full precision)
# ---------------------------------------------------------------------------


def _verify_cure_adapter_loaded(model: Any) -> None:
    """Fail fast if CURE LoRA / embed weights did not load (base-model prose output)."""
    modules_to_save = [n for n, _ in model.named_parameters() if "modules_to_save" in n]
    lora = [n for n, _ in model.named_parameters() if "lora_" in n]
    if not lora:
        raise RuntimeError(
            "[cure] No LoRA weights after adapter load. Re-install peft==0.17.1 and "
            "clear ~/.cache/huggingface/hub/models--pamessina--medgemma-4b-it-cure"
        )
    if not modules_to_save:
        raise RuntimeError(
            "[cure] embed_tokens/lm_head (modules_to_save) missing - the adapter is not "
            "active and output will have no [cx,cy,w,h] boxes."
        )
    norms = [
        p.detach().float().norm().item()
        for n, p in model.named_parameters()
        if "modules_to_save" in n
    ]
    if not any(n > 0 for n in norms):
        raise RuntimeError(
            "[cure] modules_to_save weights are all zero - adapter cache may be corrupt. "
            "Run: rm -rf ~/.cache/huggingface/hub/models--pamessina--medgemma-4b-it-cure"
        )
    print(
        f"[cure] Adapter loaded: {len(lora)} LoRA tensors, "
        f"{len(modules_to_save)} modules_to_save tensors"
    )


def run_cure_chat(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    decoded = processor.decode(generation[0], skip_special_tokens=True)
    if "model\n" in decoded:
        return decoded.split("model\n")[-1].strip()
    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()


def load_cure_model(cfg: DeviceConfig) -> tuple[Any, Any]:
    """Official CURE recipe (transformers==4.55.4 + peft==0.17.1), bf16 full precision.

    The tied MedGemma-4B base is loaded directly and the CURE LoRA adapter is attached
    with PeftModel.from_pretrained. We do NOT untie embeddings or clone lm_head - that
    breaks peft's modules_to_save mapping (KeyError on ...embed_tokens.weight).
    """
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(CURE_BASE_ID)
    processor.tokenizer.padding_side = "left"

    on_cuda = cfg.device.type == "cuda"
    base_model = AutoModelForImageTextToText.from_pretrained(
        CURE_BASE_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto" if on_cuda else None,
        low_cpu_mem_usage=True,
    )
    if not on_cuda:
        base_model = base_model.to(cfg.device)

    model = PeftModel.from_pretrained(base_model, CURE_ADAPTER_ID)
    if not on_cuda:
        model = model.to(cfg.device)
    model.eval()
    _verify_cure_adapter_loaded(model)
    return model, processor


def infer_cure(sample: ImageSample, model: Any, processor: Any) -> ModelRunResult:
    t0 = time.time()
    report_text = run_cure_chat(
        model,
        processor,
        sample.cure_image,
        GROUNDED_REPORT_PROMPT,
        max_new_tokens=512,
    )
    pred_findings = parse_grounded_report_cxcywh(report_text)
    return ModelRunResult(
        model_key="cure",
        report_text=report_text,
        pred_findings=pred_findings,
        box_format="cxcywh",
        latency_s=time.time() - t0,
    )


def load_maira2_model(cfg: DeviceConfig) -> tuple[Any, Any]:
    """MAIRA-2 (transformers==4.51.3), bf16 full precision."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        MAIRA2_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MAIRA2_ID, trust_remote_code=True)
    model = model.to(cfg.device)
    model.eval()
    return model, processor


def infer_maira2(sample: ImageSample, model: Any, processor: Any) -> ModelRunResult:
    t0 = time.time()
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=sample.raw_image,
        current_lateral=None,
        prior_frontal=None,
        indication=None,
        technique=None,
        comparison=None,
        prior_report=None,
        return_tensors="pt",
        get_grounding=True,
    )
    processed_inputs = {k: v.to(model.device) for k, v in processed_inputs.items()}

    with torch.inference_mode():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=450,
            use_cache=True,
        )

    prompt_length = processed_inputs["input_ids"].shape[-1]
    raw_prediction = processor.decode(
        output_decoding[0][prompt_length:],
        skip_special_tokens=True,
    ).lstrip()

    parsed = processor.convert_output_to_plaintext_or_grounded_sequence(raw_prediction)
    pred_findings: list[dict[str, Any]] = []

    if isinstance(parsed, list):
        w, h = sample.raw_image.size
        for sentence, boxes in parsed:
            adj_boxes = []
            if boxes:
                for box in boxes:
                    # Returns box normalized [0,1] w.r.t. the ORIGINAL image.
                    adjusted = processor.adjust_box_for_original_image_size(box, w, h)
                    adj_boxes.append([float(x) for x in adjusted])
            pred_findings.append({"sentence": sentence.strip(), "boxes": adj_boxes})

    report_text = findings_to_report_text(pred_findings, box_format="xyxy")
    return ModelRunResult(
        model_key="maira2",
        report_text=report_text or raw_prediction,
        pred_findings=pred_findings,
        box_format="xyxy",
        latency_s=time.time() - t0,
    )


MODEL_REGISTRY = {
    "cure": (load_cure_model, infer_cure),
    "maira2": (load_maira2_model, infer_maira2),
}


# ---------------------------------------------------------------------------
# Per-model result serialization (keyword + box payload for the later OpenAI step)
# ---------------------------------------------------------------------------


def build_keyword_findings(
    pred_findings: list[dict[str, Any]],
    box_format: str,
    orig_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Flatten model output into {keyword, box_norm, box_px} entries."""
    out: list[dict[str, Any]] = []
    for f in pred_findings:
        keyword = f.get("sentence", "")
        boxes = f.get("boxes") or []
        if not boxes:
            out.append({"keyword": keyword, "box_norm": None, "box_px": None})
            continue
        for b in boxes:
            xyxy = finding_boxes_to_xyxy_norm(b, box_format)
            out.append({
                "keyword": keyword,
                "box_norm": [round(v, 6) for v in xyxy],
                "box_px": [round(v, 2) for v in norm_xyxy_to_px(xyxy, orig_size)],
            })
    return out


def make_per_image_payload(
    sample: ImageSample,
    result: ModelRunResult,
    orig_size: tuple[int, int],
) -> dict[str, Any]:
    if result.error is not None:
        return {"error": result.error, "latency_s": result.latency_s}

    det = evaluate_detection(result.pred_findings, sample.gt_entry, result.box_format, orig_size)
    kw = evaluate_keywords(result.pred_findings, sample.gt_entry)
    return {
        "report_text": result.report_text,
        "box_format": result.box_format,
        "orig_size": list(orig_size),
        "keyword_findings": build_keyword_findings(result.pred_findings, result.box_format, orig_size),
        "detection": det,
        "keyword": kw,
        "latency_s": result.latency_s,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Aggregation + statistics (report step; no transformers)
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via non-parametric bootstrap of the mean."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap on per-image differences (a - b). Two-sided p-value."""
    if not a or not b or len(a) != len(b):
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": min(len(a), len(b))}
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return {
        "diff": float(d.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": min(p, 1.0),
        "n": len(d),
    }


def _per_image_ok(p: dict[str, Any]) -> bool:
    """True when a per-image payload has the fields needed for aggregation."""
    if p.get("error"):
        return False
    return "detection" in p and "keyword" in p


def aggregate_model(per_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one model's per-image payloads into thesis metrics + per-image arrays."""
    valid_items = [(iid, p) for iid, p in per_image.items() if _per_image_ok(p)]
    n_errors = len(per_image) - len(valid_items)

    metrics: dict[str, Any] = {
        "n_images": len(valid_items),
        "n_errors": n_errors,
        "thresholds": {},
        "sweep_f1_micro": {},
        "per_image": {},  # image_id -> {iou, f1@0.5, kw_f1} for stats/significance
    }

    all_pair_ious: list[float] = []
    per_img_mean_iou: list[float] = []
    latencies: list[float] = []
    total_pred = total_gt = 0

    for iid, p in valid_items:
        det = p["detection"]
        ious = det["matched_ious"]
        all_pair_ious.extend(ious)
        img_iou = (sum(ious) / len(ious)) if ious else 0.0
        per_img_mean_iou.append(img_iou)
        latencies.append(p.get("latency_s", 0.0))
        total_pred += det["n_pred_boxes"]
        total_gt += det["n_gt_boxes"]
        metrics["per_image"][iid] = {"iou": img_iou}

    metrics["mean_iou_micro"] = (sum(all_pair_ious) / len(all_pair_ious)) if all_pair_ious else 0.0
    mean_iou, iou_lo, iou_hi = _bootstrap_ci(per_img_mean_iou)
    metrics["mean_iou_macro"] = mean_iou
    metrics["mean_iou_std"] = statistics.pstdev(per_img_mean_iou) if len(per_img_mean_iou) > 1 else 0.0
    metrics["mean_iou_ci"] = [iou_lo, iou_hi]
    metrics["avg_latency_s"] = statistics.mean(latencies) if latencies else 0.0
    metrics["total_pred_boxes"] = total_pred
    metrics["total_gt_boxes"] = total_gt

    # Detection P/R/F1 for every threshold (headline + sweep).
    for t in ALL_THRESHOLDS:
        key = _tk(t)
        tp = fp = fn = 0
        per_img_p: list[float] = []
        per_img_r: list[float] = []
        per_img_f1: list[float] = []
        for iid, p in valid_items:
            cell = p["detection"]["per_threshold"][key]
            i_tp, i_fp, i_fn = cell["tp"], cell["fp"], cell["fn"]
            tp += i_tp
            fp += i_fp
            fn += i_fn
            p_i = _safe_div(i_tp, i_tp + i_fp)
            r_i = _safe_div(i_tp, i_tp + i_fn)
            per_img_p.append(p_i)
            per_img_r.append(r_i)
            f1_i = _f1(p_i, r_i)
            per_img_f1.append(f1_i)
            if abs(t - 0.5) < 1e-9:
                metrics["per_image"][iid]["f1@0.5"] = f1_i

        precision_micro = _safe_div(tp, tp + fp)
        recall_micro = _safe_div(tp, tp + fn)
        f1_micro = _f1(precision_micro, recall_micro)
        metrics["sweep_f1_micro"][key] = f1_micro
        if t in HEADLINE_THRESHOLDS:
            metrics["thresholds"][key] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "precision_macro": statistics.mean(per_img_p) if per_img_p else 0.0,
                "recall_macro": statistics.mean(per_img_r) if per_img_r else 0.0,
                "f1_macro": statistics.mean(per_img_f1) if per_img_f1 else 0.0,
                "recall_macro_std": statistics.pstdev(per_img_r) if len(per_img_r) > 1 else 0.0,
                "hallucination_rate": 1.0 - precision_micro,
            }

    # Score-free mAP-like surrogate = mean micro F1 across the 0.50:0.05:0.95 sweep.
    sweep_vals = [metrics["sweep_f1_micro"][_tk(t)] for t in SWEEP_THRESHOLDS]
    metrics["map_like_50_95"] = statistics.mean(sweep_vals) if sweep_vals else 0.0
    metrics["ap50_f1"] = metrics["sweep_f1_micro"].get(_tk(0.5), 0.0)

    # Keyword-vs-label match.
    ktp = kfp = kfn = 0
    kw_p: list[float] = []
    kw_r: list[float] = []
    kw_f1: list[float] = []
    for iid, p in valid_items:
        kw = p["keyword"]
        ktp += kw["tp"]
        kfp += kw["fp"]
        kfn += kw["fn"]
        p_i = _safe_div(kw["tp"], kw["tp"] + kw["fp"])
        r_i = _safe_div(kw["tp"], kw["tp"] + kw["fn"])
        kw_p.append(p_i)
        kw_r.append(r_i)
        f1_i = _f1(p_i, r_i)
        kw_f1.append(f1_i)
        metrics["per_image"][iid]["kw_f1"] = f1_i
    kw_prec_micro = _safe_div(ktp, ktp + kfp)
    kw_rec_micro = _safe_div(ktp, ktp + kfn)
    kw_f1_mean, kw_f1_lo, kw_f1_hi = _bootstrap_ci(kw_f1)
    metrics["keyword"] = {
        "tp": ktp,
        "fp": kfp,
        "fn": kfn,
        "precision_micro": kw_prec_micro,
        "recall_micro": kw_rec_micro,
        "f1_micro": _f1(kw_prec_micro, kw_rec_micro),
        "precision_macro": statistics.mean(kw_p) if kw_p else 0.0,
        "recall_macro": statistics.mean(kw_r) if kw_r else 0.0,
        "f1_macro": kw_f1_mean,
        "f1_macro_std": statistics.pstdev(kw_f1) if len(kw_f1) > 1 else 0.0,
        "f1_ci": [kw_f1_lo, kw_f1_hi],
    }

    # Per-pathology (label) breakdown.
    label_stats: dict[str, dict[str, Any]] = {}
    for iid, p in valid_items:
        for g in p["detection"].get("gt_detail", []):
            for label in g.get("labels", []) or []:
                ls = label_stats.setdefault(label, {"n_gt": 0, "matched_05": 0, "iou_sum": 0.0})
                ls["n_gt"] += 1
                ls["iou_sum"] += g.get("best_iou", 0.0)
                if g.get("best_iou", 0.0) >= 0.5:
                    ls["matched_05"] += 1
    per_label = []
    for label, ls in sorted(label_stats.items(), key=lambda kv: -kv[1]["n_gt"]):
        per_label.append({
            "label": label,
            "n_gt": ls["n_gt"],
            "recall@0.5": _safe_div(ls["matched_05"], ls["n_gt"]),
            "mean_best_iou": _safe_div(ls["iou_sum"], ls["n_gt"]),
        })
    metrics["per_label"] = per_label

    return metrics


def compute_significance(thesis_metrics: dict[str, Any]) -> dict[str, Any]:
    """Paired bootstrap CURE vs MAIRA-2 on per-image IoU, F1@0.5, keyword F1."""
    if "cure" not in thesis_metrics or "maira2" not in thesis_metrics:
        return {}
    cure_pi = thesis_metrics["cure"].get("per_image", {})
    maira_pi = thesis_metrics["maira2"].get("per_image", {})
    common = sorted(set(cure_pi) & set(maira_pi))
    if not common:
        return {}

    out: dict[str, Any] = {"n_paired_images": len(common)}
    for metric_key in ("iou", "f1@0.5", "kw_f1"):
        a = [cure_pi[i].get(metric_key, 0.0) for i in common]
        b = [maira_pi[i].get(metric_key, 0.0) for i in common]
        out[metric_key] = _paired_bootstrap(a, b)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _bar_plot(
    model_keys: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    out_path: str,
    errors: dict[str, list[float]] | None = None,
) -> None:
    x = np.arange(len(model_keys))
    n_series = len(series)
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(model_keys) * n_series), 4.5))
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        err = errors.get(name) if errors else None
        bars = ax.bar(x + offset, vals, width, label=name, yerr=err, capsize=4)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in model_keys])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if n_series > 1:
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def generate_plots(thesis_metrics: dict[str, Any], plots_dir: str) -> list[str]:
    os.makedirs(plots_dir, exist_ok=True)
    model_keys = [k for k in ALL_MODELS if k in thesis_metrics and thesis_metrics[k]["n_images"] > 0]
    if not model_keys:
        return []

    paths: list[str] = []

    iou_path = os.path.join(plots_dir, "mean_iou.png")
    _bar_plot(
        model_keys,
        {"mean IoU (macro)": [thesis_metrics[k]["mean_iou_macro"] for k in model_keys]},
        "Mean IoU of matched boxes (macro avg +/- std, pixel space)",
        "IoU",
        iou_path,
        errors={"mean IoU (macro)": [thesis_metrics[k]["mean_iou_std"] for k in model_keys]},
    )
    paths.append(iou_path)

    prf_path = os.path.join(plots_dir, "precision_recall_f1_at_0.5.png")
    _bar_plot(
        model_keys,
        {
            "precision": [thesis_metrics[k]["thresholds"]["0.50"]["precision_micro"] for k in model_keys],
            "recall": [thesis_metrics[k]["thresholds"]["0.50"]["recall_micro"] for k in model_keys],
            "F1": [thesis_metrics[k]["thresholds"]["0.50"]["f1_micro"] for k in model_keys],
        },
        "Detection performance @ IoU>=0.5 (micro)",
        "score",
        prf_path,
    )
    paths.append(prf_path)

    rec_path = os.path.join(plots_dir, "recall_at_thresholds.png")
    _bar_plot(
        model_keys,
        {
            "recall@0.3": [thesis_metrics[k]["thresholds"]["0.30"]["recall_micro"] for k in model_keys],
            "recall@0.5": [thesis_metrics[k]["thresholds"]["0.50"]["recall_micro"] for k in model_keys],
        },
        "Recall at IoU thresholds (micro)",
        "recall",
        rec_path,
    )
    paths.append(rec_path)

    map_path = os.path.join(plots_dir, "map_like_and_hallucination.png")
    _bar_plot(
        model_keys,
        {
            "mAP-like@[.5:.95]": [thesis_metrics[k]["map_like_50_95"] for k in model_keys],
            "hallucination@0.5": [thesis_metrics[k]["thresholds"]["0.50"]["hallucination_rate"] for k in model_keys],
        },
        "mAP-like surrogate (mean F1 over IoU 0.5:0.95) and hallucination rate",
        "score",
        map_path,
    )
    paths.append(map_path)

    kw_path = os.path.join(plots_dir, "keyword_f1.png")
    _bar_plot(
        model_keys,
        {
            "keyword precision": [thesis_metrics[k]["keyword"]["precision_micro"] for k in model_keys],
            "keyword recall": [thesis_metrics[k]["keyword"]["recall_micro"] for k in model_keys],
            "keyword F1": [thesis_metrics[k]["keyword"]["f1_micro"] for k in model_keys],
        },
        "Keyword vs GT-label match (micro)",
        "score",
        kw_path,
    )
    paths.append(kw_path)

    lat_path = os.path.join(plots_dir, "latency.png")
    _bar_plot(
        model_keys,
        {"sec/image": [thesis_metrics[k]["avg_latency_s"] for k in model_keys]},
        "Average inference latency per image (bf16)",
        "seconds",
        lat_path,
    )
    paths.append(lat_path)

    return paths


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def write_markdown_report(
    report_path: str,
    metadata: dict[str, Any],
    thesis_metrics: dict[str, Any],
    significance: dict[str, Any],
    plot_paths: list[str],
    output_dir: str,
) -> None:
    model_keys = [k for k in ALL_MODELS if k in thesis_metrics]
    lines: list[str] = []

    def rel(path: str) -> str:
        return os.path.relpath(path, output_dir)

    lines.append("# CXR Grounded Finding Localization: CURE vs MAIRA-2")
    lines.append("")
    lines.append(
        "Comparison of **CURE** and **MAIRA-2** on PadChest-GR. Each model outputs "
        "short keyword findings + bounding boxes (the narrative report is future work, "
        "assembled by OpenAI from these `{keyword, box}` findings). Boxes are matched to "
        "ground truth by IoU in **original-image pixel coordinates** (greedy, one-to-one, "
        "over all predicted boxes). Both models run in **bf16 full precision (no 4-bit)**."
    )
    lines.append("")

    lines.append("## 1. Run metadata (reproducibility)")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for key in [
        "timestamp_utc", "git_commit", "device", "dtype",
        "n_images", "shuffle_seed", "models",
        "python", "platform", "torch",
        "maira2_transformers", "cure_transformers", "cure_peft",
    ]:
        if key in metadata:
            val = metadata[key]
            if isinstance(val, list):
                val = ", ".join(val)
            lines.append(f"| `{key}` | {val} |")
    lines.append("")

    lines.append("## 2. Headline metrics")
    lines.append("")
    lines.append(
        "IoU in pixel space; micro = pooled over boxes, macro = mean over per-image "
        "scores (+/- std). `mAP-like@[.5:.95]` is the mean micro-F1 over IoU 0.50:0.05:0.95 "
        "(a **score-free surrogate** for COCO mAP - these generative models emit no box "
        "confidences, so true AP is undefined). Keyword F1 matches predicted finding "
        "keywords to GT labels (difflib)."
    )
    lines.append("")
    header = (
        "| Model | N | mean IoU (macro±std) | P@0.5 | R@0.5 | F1@0.5 | R@0.3 | "
        "mAP-like | Halluc.@0.5 | keyword F1 | sec/img | errors |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 12)
    for k in model_keys:
        m = thesis_metrics[k]
        t5 = m["thresholds"].get("0.50", {})
        t3 = m["thresholds"].get("0.30", {})
        lines.append(
            f"| **{k.upper()}** | {m['n_images']} | "
            f"{_fmt(m['mean_iou_macro'])} ± {_fmt(m['mean_iou_std'])} | "
            f"{_fmt(t5.get('precision_micro', 0))} | {_fmt(t5.get('recall_micro', 0))} | "
            f"{_fmt(t5.get('f1_micro', 0))} | {_fmt(t3.get('recall_micro', 0))} | "
            f"{_fmt(m['map_like_50_95'])} | {_fmt(t5.get('hallucination_rate', 0))} | "
            f"{_fmt(m['keyword']['f1_micro'])} | {_fmt(m['avg_latency_s'], 1)} | {m['n_errors']} |"
        )
    lines.append("")
    lines.append(
        "Box totals: "
        + "; ".join(
            f"{k.upper()} pred={thesis_metrics[k]['total_pred_boxes']}, "
            f"gt={thesis_metrics[k]['total_gt_boxes']}"
            for k in model_keys
        )
    )
    lines.append("")

    lines.append("## 3. Precision / Recall / F1 (micro and macro)")
    lines.append("")
    for t in HEADLINE_THRESHOLDS:
        key = _tk(t)
        lines.append(f"### IoU threshold {t}")
        lines.append("")
        lines.append("| Model | P micro | R micro | F1 micro | P macro | R macro | F1 macro | TP | FP | FN |")
        lines.append("|" + "---|" * 10)
        for k in model_keys:
            td = thesis_metrics[k]["thresholds"].get(key, {})
            lines.append(
                f"| {k.upper()} | {_fmt(td.get('precision_micro', 0))} | "
                f"{_fmt(td.get('recall_micro', 0))} | {_fmt(td.get('f1_micro', 0))} | "
                f"{_fmt(td.get('precision_macro', 0))} | {_fmt(td.get('recall_macro', 0))} | "
                f"{_fmt(td.get('f1_macro', 0))} | {td.get('tp', 0)} | {td.get('fp', 0)} | "
                f"{td.get('fn', 0)} |"
            )
        lines.append("")

    lines.append("## 4. Confidence intervals (95% bootstrap, macro per image)")
    lines.append("")
    lines.append("| Model | mean IoU [95% CI] | keyword F1 [95% CI] |")
    lines.append("|---|---|---|")
    for k in model_keys:
        m = thesis_metrics[k]
        iou_ci = m.get("mean_iou_ci", [0, 0])
        kw_ci = m["keyword"].get("f1_ci", [0, 0])
        lines.append(
            f"| {k.upper()} | {_fmt(m['mean_iou_macro'])} "
            f"[{_fmt(iou_ci[0])}, {_fmt(iou_ci[1])}] | "
            f"{_fmt(m['keyword']['f1_macro'])} [{_fmt(kw_ci[0])}, {_fmt(kw_ci[1])}] |"
        )
    lines.append("")

    if significance:
        lines.append("## 5. Significance: CURE vs MAIRA-2 (paired bootstrap)")
        lines.append("")
        lines.append(
            f"Paired over {significance.get('n_paired_images', 0)} images. "
            "diff = CURE - MAIRA-2 (positive favours CURE). Significant at 95% if the CI excludes 0."
        )
        lines.append("")
        lines.append("| Metric | diff (CURE-MAIRA) | 95% CI | p-value | significant |")
        lines.append("|---|---|---|---|---|")
        label_map = {"iou": "mean IoU", "f1@0.5": "F1@0.5", "kw_f1": "keyword F1"}
        for mk, label in label_map.items():
            s = significance.get(mk)
            if not s:
                continue
            sig = "yes" if (s["ci_low"] > 0 or s["ci_high"] < 0) else "no"
            lines.append(
                f"| {label} | {_fmt(s['diff'])} | "
                f"[{_fmt(s['ci_low'])}, {_fmt(s['ci_high'])}] | "
                f"{_fmt(s['p_value'])} | {sig} |"
            )
        lines.append("")

    if plot_paths:
        lines.append("## 6. Plots")
        lines.append("")
        for p in plot_paths:
            name = os.path.splitext(os.path.basename(p))[0].replace("_", " ")
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"![{name}]({rel(p)})")
            lines.append("")

    lines.append("## 7. Per-pathology breakdown (recall@0.5 / mean best IoU)")
    lines.append("")
    for k in model_keys:
        per_label = thesis_metrics[k].get("per_label", [])
        if not per_label:
            continue
        lines.append(f"### {k.upper()}")
        lines.append("")
        lines.append("| Pathology | GT boxes | recall@0.5 | mean best IoU |")
        lines.append("|---|---|---|---|")
        for row in per_label:
            lines.append(
                f"| {row['label']} | {row['n_gt']} | "
                f"{_fmt(row['recall@0.5'])} | {_fmt(row['mean_best_iou'])} |"
            )
        lines.append("")

    lines.append("## 8. Caveats")
    lines.append("")
    lines.append(
        "- IoU is computed in original-image pixel coordinates for every model "
        "(CURE cxcywh, MAIRA-2 xyxy and GT are all mapped to original W x H px first), "
        "so aspect ratio is honoured identically."
    )
    lines.append(
        "- `mAP-like@[.5:.95]` is a score-free surrogate (mean micro-F1 across IoU "
        "thresholds); the generative models produce no per-box confidence, so a true "
        "COCO AP / PR curve is not defined."
    )
    lines.append(
        "- Keyword matching is lexical (difflib ratio >= 0.8 or substring); it is a "
        "sanity signal for 'did the model name the right finding', not a clinical NLG metric."
    )
    lines.append(
        "- Both models run in bf16 full precision; numbers therefore differ from any "
        "published 4-bit results."
    )
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Logging (tee stdout+stderr to a log file)
# ---------------------------------------------------------------------------


class _Tee:
    def __init__(self, log_path: str):
        self._log = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._log.flush()

    def close(self) -> None:
        try:
            self._log.close()
        finally:
            sys.stdout = self._stdout
            sys.stderr = self._stderr


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_select(_: argparse.Namespace) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(IMAGE_LIST_PATH) and os.environ.get("FORCE_RESELECT", "0") != "1":
        with open(IMAGE_LIST_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(f"image_list.json already exists ({len(state.get('selected', []))} images).")
        print("Reusing it so both models score the same set. Set FORCE_RESELECT=1 to re-pick.")
        return

    if not os.path.exists(JSON_PATH) or not os.path.exists(IMAGES_DIR):
        raise FileNotFoundError(f"Missing GT json ({JSON_PATH}) or images dir ({IMAGES_DIR}).")

    gt_by_id = load_gt_index(JSON_PATH)
    prefer = DEFAULT_VERIFY_IMAGE if N_IMAGES == 1 else None
    selected = select_images(IMAGES_DIR, gt_by_id, N_IMAGES, SHUFFLE_SEED, prefer_image=prefer)

    state = {
        "seed": SHUFFLE_SEED,
        "n_images": N_IMAGES,
        "n_selected": len(selected),
        "selected": selected,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(IMAGE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"Selected {len(selected)} image(s) (seed={SHUFFLE_SEED}) -> {IMAGE_LIST_PATH}")
    for i, iid in enumerate(selected[:10], start=1):
        print(f"  {i:>2}. {iid}")
    if len(selected) > 10:
        print(f"  ... (+{len(selected) - 10} more)")


def _load_image_list() -> list[str]:
    if not os.path.exists(IMAGE_LIST_PATH):
        raise FileNotFoundError(
            f"{IMAGE_LIST_PATH} not found. Run `python compare_models.py select` first."
        )
    with open(IMAGE_LIST_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    selected = state.get("selected") or []
    if not selected:
        raise RuntimeError("image_list.json has no 'selected' images.")
    return selected


def cmd_run(args: argparse.Namespace) -> None:
    model_key = args.model
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Choose from: {list(MODEL_REGISTRY)}")

    os.makedirs(PER_MODEL_DIR, exist_ok=True)
    tee = _Tee(os.path.join(OUTPUT_DIR, f"run_{model_key}.log"))
    sys.stdout = tee
    sys.stderr = tee

    try:
        print(f"Run [{model_key}] started (UTC): "
              f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        setup_hf_auth()
        cfg = resolve_device_config()

        selected = _load_image_list()
        gt_by_id = load_gt_index(JSON_PATH)
        print(f"Scoring {len(selected)} images for {model_key.upper()}.")

        import transformers  # noqa: F401  (record version)

        loader, infer_fn = MODEL_REGISTRY[model_key]
        print(f"Loading {model_key.upper()} (bf16)...")
        model, processor = loader(cfg)

        per_image: dict[str, dict[str, Any]] = {}
        save_figures = os.environ.get("SAVE_FIGURES", "0") == "1"
        fig_dir = os.path.join(OUTPUT_DIR, "per_image")
        if save_figures:
            os.makedirs(fig_dir, exist_ok=True)

        for idx, image_id in enumerate(selected, start=1):
            try:
                sample = build_sample(image_id, gt_by_id, which=model_key)
                result = infer_fn(sample, model, processor)
                payload = make_per_image_payload(sample, result, sample.orig_size)
                per_image[image_id] = payload

                det = payload.get("detection", {})
                print(
                    f"[{model_key}] {idx}/{len(selected)} {image_id}  "
                    f"IoU={det.get('mean_iou', 0.0):.3f} "
                    f"R@0.5={det.get('recall@0.5', 0.0):.2f} "
                    f"pred={det.get('n_pred_boxes', 0)} gt={det.get('n_gt_boxes', 0)} "
                    f"({payload.get('latency_s', 0.0):.1f}s)"
                )

                if save_figures:
                    _save_preview(sample, result, model_key, os.path.join(
                        fig_dir, f"{model_key}_{os.path.splitext(image_id)[0]}.png"))
            except Exception as exc:  # noqa: BLE001
                print(f"[{model_key}] {idx}/{len(selected)} {image_id}  [ERROR] {exc}")
                per_image[image_id] = {"error": str(exc), "latency_s": 0.0}

        free_model(model, processor)

        out = {
            "model": model_key,
            "transformers": transformers.__version__,
            "n_images": len(per_image),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "per_image": per_image,
        }
        out_path = os.path.join(PER_MODEL_DIR, f"{model_key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved {out_path}")
    finally:
        tee.close()


def _save_preview(sample: ImageSample, result: ModelRunResult, model_key: str, out_path: str) -> None:
    base = sample.cure_image if model_key == "cure" else sample.raw_image
    if base is None:
        return
    vis = draw_xyxy_boxes(base, sample.gt_boxes_xyxy, color="lime", width=3)
    pred_boxes = [b for f in result.pred_findings for b in (f.get("boxes") or [])]
    if result.box_format == "cxcywh":
        vis = draw_cxcywh_boxes(vis, pred_boxes, color=MODEL_COLORS[model_key], width=3)
    else:
        vis = draw_xyxy_boxes(vis, pred_boxes, color=MODEL_COLORS[model_key], width=3)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(vis)
    ax.axis("off")
    ax.set_title(f"{model_key.upper()} ({sample.image_id})", fontsize=9)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _collect_report_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": os.environ.get("DEVICE", "cpu"),
        "dtype": "bfloat16",
        "n_images": N_IMAGES,
        "shuffle_seed": SHUFFLE_SEED,
        "models": [k for k in ALL_MODELS],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    # Pull each model's recorded transformers version from its per-model JSON.
    for mk in ALL_MODELS:
        p = os.path.join(PER_MODEL_DIR, f"{mk}.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta[f"{mk}_transformers"] = json.load(f).get("transformers", "n/a")
            except Exception:
                pass
    meta["cure_peft"] = "0.17.1 (pinned)"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, stderr=subprocess.DEVNULL
        )
        meta["git_commit"] = commit.decode().strip()
    except Exception:
        meta["git_commit"] = "n/a"
    return meta


def cmd_report(_: argparse.Namespace) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loaded: dict[str, dict[str, Any]] = {}
    for mk in ALL_MODELS:
        p = os.path.join(PER_MODEL_DIR, f"{mk}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                loaded[mk] = json.load(f)
        else:
            print(f"[WARN] Missing {p} - {mk} will be absent from the report.")

    if not loaded:
        raise SystemExit("No per-model results found. Run `run --model ...` first.")

    for mk, data in loaded.items():
        per_image = data.get("per_image", {})
        skipped = [iid for iid, p in per_image.items() if not _per_image_ok(p)]
        if skipped:
            print(
                f"[WARN] {mk}: skipping {len(skipped)}/{len(per_image)} image(s) "
                f"without full detection metrics (errors or incomplete payloads)."
            )

    thesis_metrics = {mk: aggregate_model(data.get("per_image", {})) for mk, data in loaded.items()}
    significance = compute_significance(thesis_metrics)
    metadata = _collect_report_metadata()

    # Console summary.
    print("\n================ CURE vs MAIRA-2 SUMMARY ================")
    header = f"{'model':<8} {'N':>4} {'IoU':>7} {'F1@0.5':>7} {'mAP~':>7} {'kwF1':>7} {'sec':>7} {'err':>4}"
    print(header)
    print("-" * len(header))
    for k in [m for m in ALL_MODELS if m in thesis_metrics]:
        m = thesis_metrics[k]
        t5 = m["thresholds"].get("0.50", {})
        print(
            f"{k:<8} {m['n_images']:>4} {m['mean_iou_macro']:>7.3f} "
            f"{t5.get('f1_micro', 0.0):>7.3f} {m['map_like_50_95']:>7.3f} "
            f"{m['keyword']['f1_micro']:>7.3f} {m['avg_latency_s']:>7.1f} {m['n_errors']:>4}"
        )
    print("=" * len(header))

    # JSON payload (metrics only; per-image detail already on disk per model).
    metrics_for_json = {}
    for mk, m in thesis_metrics.items():
        m2 = {kk: vv for kk, vv in m.items() if kk != "per_image"}
        metrics_for_json[mk] = m2
    json_payload = {
        "metadata": metadata,
        "significance": significance,
        "metrics": metrics_for_json,
    }
    json_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path}")

    # CSV.
    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    csv_fields = [
        "model", "n_images",
        "mean_iou_micro", "mean_iou_macro", "mean_iou_std",
        "precision@0.5_micro", "recall@0.5_micro", "f1@0.5_micro",
        "recall@0.3_micro", "map_like_50_95", "hallucination@0.5",
        "keyword_precision_micro", "keyword_recall_micro", "keyword_f1_micro",
        "avg_latency_s", "total_pred_boxes", "total_gt_boxes", "errors",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for k in [m for m in ALL_MODELS if m in thesis_metrics]:
            m = thesis_metrics[k]
            t5 = m["thresholds"].get("0.50", {})
            t3 = m["thresholds"].get("0.30", {})
            kw = m["keyword"]
            writer.writerow({
                "model": k,
                "n_images": m["n_images"],
                "mean_iou_micro": round(m["mean_iou_micro"], 4),
                "mean_iou_macro": round(m["mean_iou_macro"], 4),
                "mean_iou_std": round(m["mean_iou_std"], 4),
                "precision@0.5_micro": round(t5.get("precision_micro", 0.0), 4),
                "recall@0.5_micro": round(t5.get("recall_micro", 0.0), 4),
                "f1@0.5_micro": round(t5.get("f1_micro", 0.0), 4),
                "recall@0.3_micro": round(t3.get("recall_micro", 0.0), 4),
                "map_like_50_95": round(m["map_like_50_95"], 4),
                "hallucination@0.5": round(t5.get("hallucination_rate", 0.0), 4),
                "keyword_precision_micro": round(kw["precision_micro"], 4),
                "keyword_recall_micro": round(kw["recall_micro"], 4),
                "keyword_f1_micro": round(kw["f1_micro"], 4),
                "avg_latency_s": round(m["avg_latency_s"], 2),
                "total_pred_boxes": m["total_pred_boxes"],
                "total_gt_boxes": m["total_gt_boxes"],
                "errors": m["n_errors"],
            })
    print(f"Saved CSV: {csv_path}")

    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    plot_paths = generate_plots(thesis_metrics, plots_dir)
    for p in plot_paths:
        print(f"Saved plot: {p}")

    report_path = os.path.join(OUTPUT_DIR, "report.md")
    write_markdown_report(
        report_path=report_path,
        metadata=metadata,
        thesis_metrics=thesis_metrics,
        significance=significance,
        plot_paths=plot_paths,
        output_dir=OUTPUT_DIR,
    )
    print(f"Saved report: {report_path}")
    print(f"All outputs in: {OUTPUT_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CURE and MAIRA-2 on PadChest-GR grounded localization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_select = sub.add_parser("select", help="Pin the shared image list (once).")
    p_select.set_defaults(func=cmd_select)

    p_run = sub.add_parser("run", help="Run ONE model over the pinned image list.")
    p_run.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="Aggregate per-model results into report.md.")
    p_report.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
