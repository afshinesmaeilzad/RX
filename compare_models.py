#!/usr/bin/env python3
"""
Compare CURE, MAIRA-2, and MedGemma 1.5 on Padchest-GR grounded report generation.

Install (pin transformers for MAIRA-2 + MedGemma compatibility):
    pip install "transformers==4.51.3" accelerate peft bitsandbytes huggingface_hub \\
        safetensors sentencepiece pillow opencv-python-headless matplotlib protobuf

Hugging Face gating (accept terms before first run):
    - https://huggingface.co/pamessina/medgemma-4b-it-cure
    - https://huggingface.co/microsoft/maira-2
    - https://huggingface.co/google/medgemma-1.5-4b-it

Auth:
    export HF_TOKEN="hf_..."   # or: huggingface-cli login

Mac CPU verify (1 image, slow but correct for CURE bf16):
    DEVICE=cpu N_IMAGES=1 python compare_models.py

Read the dataset from a local extract (default) or a Google Drive folder:
    # Local: dataset next to this script (Padchest_GR_files/, grounded_reports_*.json)
    DEVICE=cpu N_IMAGES=1 python compare_models.py
    # Google Drive on Colab (auto-mounts, nested Padchest_GR_files/PadChest_GR/):
    MOUNT_DRIVE=1 \\
    DATA_DIR="/content/drive/MyDrive/PadChest/extracted/BIMCV-Padchest-GR" \\
    DEVICE=cuda N_IMAGES=1 python compare_models.py
    # Or point directly at any mounted path:
    DATA_DIR="/path/to/BIMCV-Padchest-GR" python compare_models.py

GPU server (Vast.ai RTX 4090 24GB recommended):
    DEVICE=cuda N_IMAGES=10 USE_4BIT=1 python compare_models.py

Run a subset of models:
    MODELS=cure,maira2 python compare_models.py

Caveats:
    - Models are loaded sequentially (MAIRA-2 is 7B; all three won't fit together).
    - CURE boxes are normalized to its 448x448 CLAHE input; MAIRA-2/MedGemma use
      original-image coordinates. IoU scores are comparable to Padchest-GR GT (xyxy
      on original image) for MAIRA-2/MedGemma; CURE IoU is approximate.
    - MedGemma 1.5 box output format is less standardized; parsing is best-effort.
    - Rotate any HF token that was ever committed to a notebook or repo.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import login
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
    # Fall back to the first existing directory, else the last candidate.
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[-1]


_maybe_mount_drive()

# Where the dataset lives. Defaults to the script dir (local extract). To read
# from a mounted Google Drive, set:
#   DATA_DIR="/content/drive/MyDrive/PadChest/extracted/BIMCV-Padchest-GR" MOUNT_DRIVE=1
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)

JSON_PATH = os.environ.get("JSON_PATH") or os.path.join(DATA_DIR, "grounded_reports_20240819.json")
IMAGES_DIR = os.environ.get("IMAGES_DIR") or _resolve_images_dir(DATA_DIR)
# Outputs always go to a local, writable directory (not onto Drive by default).
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(BASE_DIR, "outputs", "compare")

N_IMAGES = int(os.environ.get("N_IMAGES", "1"))
SHUFFLE_SEED = int(os.environ["SHUFFLE_SEED"]) if os.environ.get("SHUFFLE_SEED") else 42
DEFAULT_VERIFY_IMAGE = "106997070894779966614346591942916625787_fsxv2a.png"

MODELS = [
    m.strip().lower()
    for m in os.environ.get("MODELS", "cure,maira2,medgemma15").split(",")
    if m.strip()
]

CURE_BASE_ID = "google/medgemma-4b-it"
CURE_ADAPTER_ID = "pamessina/medgemma-4b-it-cure"
MAIRA2_ID = "microsoft/maira-2"
MEDGEMMA15_ID = "google/medgemma-1.5-4b-it"

CURE_IMAGE_SIZE = 448
CURE_CLAHE_CLIP_LIMIT = 3.0
CURE_CLAHE_TILE_GRID = (8, 8)

MODEL_COLORS = {
    "cure": "#ff5252",
    "maira2": "#448aff",
    "medgemma15": "#ffab40",
}

GROUNDED_REPORT_PROMPT = "Generate a grounded report."
MEDGEMMA15_GROUNDED_PROMPT = (
    "Generate a radiology findings report for this chest X-ray. "
    "For each finding, include a normalized bounding box as "
    "[x_min, y_min, x_max, y_max] with values between 0 and 1."
)


@dataclass
class DeviceConfig:
    device: torch.device
    dtype: torch.dtype
    use_4bit: bool


@dataclass
class ImageSample:
    image_id: str
    image_path: str
    gt_entry: dict[str, Any]
    raw_image: Image.Image
    cure_image: Image.Image
    gt_boxes_xyxy: list[list[float]] = field(default_factory=list)
    gt_sentences: list[str] = field(default_factory=list)


@dataclass
class ModelRunResult:
    model_key: str
    report_text: str
    pred_findings: list[dict[str, Any]]
    box_format: str
    latency_s: float
    eval_summary: dict[str, Any] | None = None
    display_image: Image.Image | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Auth + device
# ---------------------------------------------------------------------------


def setup_hf_auth() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)
    else:
        login()


def resolve_device_config() -> DeviceConfig:
    requested = os.environ.get("DEVICE", os.environ.get("CURE_DEVICE", "cpu")).lower()
    use_4bit_env = os.environ.get("USE_4BIT")

    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        use_4bit = use_4bit_env != "0" if use_4bit_env is not None else True
    elif requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.bfloat16
        use_4bit = False
        print("[WARN] MPS is experimental; use DEVICE=cpu if outputs are empty.")
    else:
        device = torch.device("cpu")
        dtype = torch.bfloat16
        use_4bit = False
        if requested == "cuda":
            print("[WARN] CUDA requested but unavailable; falling back to CPU.")

    print(f"Device: {device}  dtype: {dtype}  use_4bit: {use_4bit}")
    return DeviceConfig(device=device, dtype=dtype, use_4bit=use_4bit)


def free_model(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Image loading / drawing helpers
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


def xyxy_to_cxcywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1 + w / 2, y1 + h / 2, w, h]


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
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=color, width=width)
    return img


def draw_cxcywh_boxes(image: Image.Image, boxes: list[list[float]], color: str, width: int = 4) -> Image.Image:
    return draw_xyxy_boxes(image, [cxcywh_to_xyxy(b) for b in boxes], color=color, width=width)


def extract_boxes_from_text(text: str) -> list[list[float]]:
    pattern = (
        r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,"
        r"\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
    )
    return [[float(x) for x in m.groups()] for m in re.finditer(pattern, text)]


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


def parse_grounded_report_xyxy(text: str) -> list[dict[str, Any]]:
    findings = parse_grounded_report_cxcywh(text)
    for finding in findings:
        finding["boxes"] = [cxcywh_to_xyxy(b) if _looks_like_cxcywh(b) else b for b in finding["boxes"]]
    return findings


def _looks_like_cxcywh(box: list[float]) -> bool:
    """Heuristic: cxcywh boxes usually have all coords <= 1 and w/h <= 1."""
    if len(box) != 4:
        return False
    _, _, third, fourth = box
    return third <= 1.0 and fourth <= 1.0 and third > 0 and fourth > 0


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


def evaluate_grounded_report(
    pred_findings: list[dict[str, Any]],
    gt_entry: dict[str, Any],
    box_format: str = "cxcywh",
    iou_thresholds: tuple[float, ...] = (0.3, 0.5),
) -> dict[str, Any]:
    gt_findings = []
    for f in gt_entry.get("findings", []) or []:
        for box in f.get("boxes", []) or []:
            gt_findings.append({
                "sentence": f.get("sentence_en", ""),
                "labels": f.get("labels", []) or [],
                "box_xyxy": box,
                "matched": False,
            })

    rows = []
    for p in pred_findings:
        if not p.get("boxes"):
            rows.append({"pred": p.get("sentence", ""), "iou": None, "matched_gt": None})
            continue

        box = p["boxes"][0]
        pred_xyxy = cxcywh_to_xyxy(box) if box_format == "cxcywh" else list(box)

        best = (-1.0, None)
        for i, g in enumerate(gt_findings):
            if g["matched"]:
                continue
            score = iou_xyxy(g["box_xyxy"], pred_xyxy)
            if score > best[0]:
                best = (score, i)

        if best[1] is not None and best[0] > 0:
            gt_findings[best[1]]["matched"] = True
            rows.append({
                "pred": p.get("sentence", ""),
                "iou": best[0],
                "matched_gt": gt_findings[best[1]]["sentence"],
            })
        else:
            rows.append({"pred": p.get("sentence", ""), "iou": 0.0, "matched_gt": None})

    valid_ious = [r["iou"] for r in rows if r["iou"] is not None]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    n_gt = len(gt_findings)
    recalls = {
        f"recall@{t}": (
            sum(
                1
                for g in gt_findings
                if g["matched"]
                and any(
                    r["matched_gt"] == g["sentence"]
                    and r["iou"] is not None
                    and r["iou"] >= t
                    for r in rows
                )
            )
            / n_gt
            if n_gt
            else 0.0
        )
        for t in iou_thresholds
    }

    return {
        "rows": rows,
        "mean_iou": mean_iou,
        "n_predicted": len(pred_findings),
        "n_predicted_with_box": sum(1 for p in pred_findings if p.get("boxes")),
        "n_gt_boxes": n_gt,
        "missed_gt": [g["sentence"] for g in gt_findings if not g["matched"]],
        **recalls,
    }


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
    if prefer_image and prefer_image in gt_by_id and has_gt_box(gt_by_id[prefer_image]):
        if prefer_image in candidates:
            selected.append(prefer_image)
            candidates = [c for c in candidates if c != prefer_image]

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(candidates)

    for fn in candidates:
        if len(selected) >= n_images:
            break
        selected.append(fn)

    if not selected:
        raise RuntimeError("No images with ground-truth boxes found.")
    return selected[:n_images]


def build_samples(selected_ids: list[str], gt_by_id: dict[str, dict[str, Any]]) -> list[ImageSample]:
    samples = []
    for image_id in selected_ids:
        image_path = os.path.join(IMAGES_DIR, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")

        gt_entry = gt_by_id[image_id]
        gt_boxes, gt_sentences = [], []
        for f in gt_entry.get("findings", []) or []:
            gt_sentences.append(f.get("sentence_en", ""))
            gt_boxes.extend(f.get("boxes") or [])

        samples.append(
            ImageSample(
                image_id=image_id,
                image_path=image_path,
                gt_entry=gt_entry,
                raw_image=load_raw_xray_rgb(image_path),
                cure_image=load_xray_as_rgb_cure(image_path),
                gt_boxes_xyxy=gt_boxes,
                gt_sentences=gt_sentences,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------


def run_cure_chat(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 256,
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
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(CURE_BASE_ID)
    processor.tokenizer.padding_side = "left"

    quant_config = None
    model_kwargs: dict[str, Any] = {
        "torch_dtype": cfg.dtype,
        "low_cpu_mem_usage": True,
    }
    if cfg.use_4bit and cfg.device.type == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["device_map"] = None

    base_model = AutoModelForImageTextToText.from_pretrained(CURE_BASE_ID, **model_kwargs)
    if not cfg.use_4bit or cfg.device.type != "cuda":
        base_model.to(cfg.device)

    model = PeftModel.from_pretrained(base_model, CURE_ADAPTER_ID)
    if not cfg.use_4bit or cfg.device.type != "cuda":
        model.to(cfg.device)
    model.eval()
    return model, processor


def infer_cure(sample: ImageSample, model: Any, processor: Any) -> ModelRunResult:
    t0 = time.time()
    report_text = run_cure_chat(
        model,
        processor,
        sample.cure_image,
        GROUNDED_REPORT_PROMPT,
        max_new_tokens=256,
    )
    pred_findings = parse_grounded_report_cxcywh(report_text)
    eval_summary = evaluate_grounded_report(pred_findings, sample.gt_entry, box_format="cxcywh")

    pred_boxes_cxcywh = [b for f in pred_findings for b in f.get("boxes", [])]
    vis = draw_xyxy_boxes(sample.cure_image, sample.gt_boxes_xyxy, color="lime", width=3)
    vis = draw_cxcywh_boxes(vis, pred_boxes_cxcywh, color=MODEL_COLORS["cure"], width=3)

    return ModelRunResult(
        model_key="cure",
        report_text=report_text,
        pred_findings=pred_findings,
        box_format="cxcywh",
        latency_s=time.time() - t0,
        eval_summary=eval_summary,
        display_image=vis,
    )


def load_maira2_model(cfg: DeviceConfig) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if cfg.use_4bit and cfg.device.type == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["torch_dtype"] = cfg.dtype

    model = AutoModelForCausalLM.from_pretrained(MAIRA2_ID, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MAIRA2_ID, trust_remote_code=True)

    if not (cfg.use_4bit and cfg.device.type == "cuda"):
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
    pred_findings = []
    pred_boxes_xyxy: list[list[float]] = []

    if isinstance(parsed, list):
        for sentence, boxes in parsed:
            adj_boxes = []
            if boxes:
                w, h = sample.raw_image.size
                for box in boxes:
                    adjusted = processor.adjust_box_for_original_image_size(box, w, h)
                    adj = [float(x) for x in adjusted]
                    adj_boxes.append(adj)
                    pred_boxes_xyxy.append(adj)
            pred_findings.append({"sentence": sentence.strip(), "boxes": adj_boxes})

    report_text = findings_to_report_text(pred_findings, box_format="xyxy")
    eval_summary = evaluate_grounded_report(pred_findings, sample.gt_entry, box_format="xyxy")

    vis = draw_xyxy_boxes(sample.raw_image, sample.gt_boxes_xyxy, color="lime", width=3)
    vis = draw_xyxy_boxes(vis, pred_boxes_xyxy, color=MODEL_COLORS["maira2"], width=3)

    return ModelRunResult(
        model_key="maira2",
        report_text=report_text or raw_prediction,
        pred_findings=pred_findings,
        box_format="xyxy",
        latency_s=time.time() - t0,
        eval_summary=eval_summary,
        display_image=vis,
    )


def load_medgemma15_model(cfg: DeviceConfig) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    model_kwargs: dict[str, Any] = {
        "torch_dtype": cfg.dtype,
        "low_cpu_mem_usage": True,
    }
    if cfg.use_4bit and cfg.device.type == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA15_ID, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MEDGEMMA15_ID)

    if not (cfg.use_4bit and cfg.device.type == "cuda"):
        model.to(cfg.device)
    model.eval()
    return model, processor


def infer_medgemma15(sample: ImageSample, model: Any, processor: Any) -> ModelRunResult:
    t0 = time.time()
    report_text = run_cure_chat(
        model,
        processor,
        sample.raw_image,
        MEDGEMMA15_GROUNDED_PROMPT,
        max_new_tokens=512,
    )
    pred_findings = parse_grounded_report_xyxy(report_text)
    eval_summary = evaluate_grounded_report(pred_findings, sample.gt_entry, box_format="xyxy")

    pred_boxes_xyxy = [b for f in pred_findings for b in f.get("boxes", [])]
    vis = draw_xyxy_boxes(sample.raw_image, sample.gt_boxes_xyxy, color="lime", width=3)
    vis = draw_xyxy_boxes(vis, pred_boxes_xyxy, color=MODEL_COLORS["medgemma15"], width=3)

    return ModelRunResult(
        model_key="medgemma15",
        report_text=report_text,
        pred_findings=pred_findings,
        box_format="xyxy",
        latency_s=time.time() - t0,
        eval_summary=eval_summary,
        display_image=vis,
    )


MODEL_REGISTRY = {
    "cure": (load_cure_model, infer_cure),
    "maira2": (load_maira2_model, infer_maira2),
    "medgemma15": (load_medgemma15_model, infer_medgemma15),
}


# ---------------------------------------------------------------------------
# Aggregation + outputs
# ---------------------------------------------------------------------------


def aggregate_model_metrics(results_by_model: dict[str, list[ModelRunResult]]) -> list[dict[str, Any]]:
    summary_rows = []
    for model_key, runs in results_by_model.items():
        valid = [r for r in runs if r.eval_summary is not None and r.error is None]
        if not valid:
            summary_rows.append({
                "model": model_key,
                "n_images": len(runs),
                "mean_iou_macro": 0.0,
                "recall@0.3_macro": 0.0,
                "recall@0.5_macro": 0.0,
                "avg_latency_s": 0.0,
                "errors": sum(1 for r in runs if r.error),
            })
            continue

        summary_rows.append({
            "model": model_key,
            "n_images": len(valid),
            "mean_iou_macro": sum(r.eval_summary["mean_iou"] for r in valid) / len(valid),
            "recall@0.3_macro": sum(r.eval_summary["recall@0.3"] for r in valid) / len(valid),
            "recall@0.5_macro": sum(r.eval_summary["recall@0.5"] for r in valid) / len(valid),
            "avg_latency_s": sum(r.latency_s for r in valid) / len(valid),
            "errors": sum(1 for r in runs if r.error),
        })
    return summary_rows


def save_comparison_figure(
    sample: ImageSample,
    model_results: dict[str, ModelRunResult],
    out_path: str,
) -> None:
    model_keys = [k for k in MODELS if k in model_results]
    n = len(model_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, model_key in zip(axes, model_keys):
        result = model_results[model_key]
        img = result.display_image if result.display_image is not None else sample.raw_image
        ax.imshow(img)
        ax.axis("off")
        if result.error:
            title = f"{model_key.upper()}  ERROR"
        else:
            ev = result.eval_summary or {}
            title = (
                f"{model_key.upper()}  IoU={ev.get('mean_iou', 0.0):.2f}  "
                f"R@0.3={ev.get('recall@0.3', 0.0):.2f}"
            )
        ax.set_title(title, fontsize=10)

    fig.suptitle(f"{sample.image_id}\n(green=GT, colored=predicted)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    print("\n================ MODEL COMPARISON SUMMARY ================")
    header = f"{'model':<12} {'n':>3} {'meanIoU':>8} {'R@0.3':>8} {'R@0.5':>8} {'sec/img':>8} {'err':>4}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['model']:<12} {row['n_images']:>3} "
            f"{row['mean_iou_macro']:>8.3f} {row['recall@0.3_macro']:>8.3f} "
            f"{row['recall@0.5_macro']:>8.3f} {row['avg_latency_s']:>8.1f} "
            f"{row['errors']:>4}"
        )
    print("==========================================================")


def serialize_result(result: ModelRunResult) -> dict[str, Any]:
    return {
        "model": result.model_key,
        "report_text": result.report_text,
        "box_format": result.box_format,
        "pred_findings": result.pred_findings,
        "latency_s": result.latency_s,
        "eval": result.eval_summary,
        "error": result.error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    per_image_dir = os.path.join(OUTPUT_DIR, "per_image")
    os.makedirs(per_image_dir, exist_ok=True)

    setup_hf_auth()
    cfg = resolve_device_config()

    print("\nChecking paths...")
    print("JSON exists:", os.path.exists(JSON_PATH))
    print("Images dir exists:", os.path.exists(IMAGES_DIR))
    if not os.path.exists(JSON_PATH) or not os.path.exists(IMAGES_DIR):
        raise FileNotFoundError("Missing Padchest-GR JSON or images directory.")

    gt_by_id = load_gt_index(JSON_PATH)
    prefer = DEFAULT_VERIFY_IMAGE if N_IMAGES == 1 else None
    selected_ids = select_images(
        IMAGES_DIR,
        gt_by_id,
        N_IMAGES,
        SHUFFLE_SEED,
        prefer_image=prefer,
    )
    samples = build_samples(selected_ids, gt_by_id)

    print(f"\nSelected {len(samples)} image(s):")
    for i, sample in enumerate(samples, start=1):
        print(
            f"  {i:>2}. {sample.image_id}  "
            f"(gt_boxes={len(sample.gt_boxes_xyxy)}, gt_findings={len(sample.gt_sentences)})"
        )

    print(f"\nModels to run (sequential): {', '.join(MODELS)}")

    all_results: dict[str, dict[str, ModelRunResult]] = {
        sample.image_id: {} for sample in samples
    }
    results_by_model: dict[str, list[ModelRunResult]] = {k: [] for k in MODELS}

    for model_key in MODELS:
        if model_key not in MODEL_REGISTRY:
            print(f"[WARN] Unknown model '{model_key}', skipping.")
            continue

        loader, infer_fn = MODEL_REGISTRY[model_key]
        print(f"\n{'=' * 60}\nLoading {model_key.upper()}...\n{'=' * 60}")
        try:
            model, processor = loader(cfg)
        except Exception as exc:
            print(f"[ERROR] Failed to load {model_key}: {exc}")
            for sample in samples:
                err_result = ModelRunResult(
                    model_key=model_key,
                    report_text="",
                    pred_findings=[],
                    box_format="xyxy",
                    latency_s=0.0,
                    error=str(exc),
                )
                all_results[sample.image_id][model_key] = err_result
                results_by_model[model_key].append(err_result)
            continue

        for idx, sample in enumerate(samples, start=1):
            print(f"[{model_key}] {idx}/{len(samples)}  {sample.image_id}")
            try:
                result = infer_fn(sample, model, processor)
                all_results[sample.image_id][model_key] = result
                results_by_model[model_key].append(result)
                ev = result.eval_summary or {}
                print(
                    f"  IoU={ev.get('mean_iou', 0.0):.3f}  "
                    f"R@0.3={ev.get('recall@0.3', 0.0):.2f}  "
                    f"R@0.5={ev.get('recall@0.5', 0.0):.2f}  "
                    f"({result.latency_s:.1f}s)"
                )
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                err_result = ModelRunResult(
                    model_key=model_key,
                    report_text="",
                    pred_findings=[],
                    box_format="xyxy",
                    latency_s=0.0,
                    error=str(exc),
                )
                all_results[sample.image_id][model_key] = err_result
                results_by_model[model_key].append(err_result)

        free_model(model, processor)

    summary_rows = aggregate_model_metrics(results_by_model)
    print_summary_table(summary_rows)

    json_payload = {
        "config": {
            "n_images": N_IMAGES,
            "shuffle_seed": SHUFFLE_SEED,
            "models": MODELS,
            "device": str(cfg.device),
            "use_4bit": cfg.use_4bit,
        },
        "images": [],
        "summary": summary_rows,
    }

    for sample in samples:
        image_entry = {
            "image_id": sample.image_id,
            "image_path": sample.image_path,
            "gt_sentences": sample.gt_sentences,
            "gt_boxes_xyxy": sample.gt_boxes_xyxy,
            "models": {},
        }
        fig_path = os.path.join(
            per_image_dir,
            f"compare_{os.path.splitext(sample.image_id)[0]}.png",
        )
        save_comparison_figure(sample, all_results[sample.image_id], fig_path)
        image_entry["comparison_figure"] = fig_path

        for model_key, result in all_results[sample.image_id].items():
            image_entry["models"][model_key] = serialize_result(result)

        json_payload["images"].append(image_entry)
        print(f"Saved figure: {fig_path}")

    json_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "n_images",
                "mean_iou_macro",
                "recall@0.3_macro",
                "recall@0.5_macro",
                "avg_latency_s",
                "errors",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
