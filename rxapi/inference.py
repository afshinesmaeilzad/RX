"""Serving CURE — one model in memory, one generate() at a time.

The payload is byte-for-byte the shape the desktop app already consumes from
its local worker, so `RemoteEngine` needs no change to point here:

    {"model", "device", "image_path", "image_size": [w, h], "report_text",
     "box_format", "findings": [{"keyword", "box_norm", "box_px"}], "latency_s"}

torch, transformers and peft are imported **inside** `load()`. The API must be
importable — and `/health` answerable — on a box where the model has not been
pulled yet, or where it never will be.

Preprocessing, prompt and parsing come from `compare_models.py` rather than a
private copy: that module is what produced the thesis baseline, and a second
implementation that drifts from it would make every later comparison a lie.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from . import config, registry

_lock = threading.Lock()          # one GPU, one generate()


class CureService:
    """Holds the loaded model. Reloaded only when the active version changes."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "?"
        self.version: str | None = None
        self.error: str | None = None
        self.loading = False

    # ---------------------------------------------------------------- state
    @property
    def ready(self) -> bool:
        return self.model is not None

    def status(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else ("loading" if self.loading else "idle"),
            "backend": "cure",
            "device": self.device,
            "model": config.BASE_MODEL_ID,
            "adapter_version": self.version or registry.active().version,
            "adapter": registry.active().path_or_id(),
            "error": self.error,
        }

    # ----------------------------------------------------------------- load
    def load(self, version: str | None = None) -> None:
        """Load base + the requested adapter version. Idempotent."""
        target = registry.get(version) if version else registry.active()
        if target is None:
            raise ValueError(f"unknown adapter version: {version}")
        if self.ready and self.version == target.version:
            return

        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        import compare_models as cm

        self.loading, self.error = True, None
        try:
            cfg = cm.resolve_device_config()
            device = cfg.device
            on_cuda = device.type == "cuda"

            processor = AutoProcessor.from_pretrained(config.BASE_MODEL_ID)
            processor.tokenizer.padding_side = "left"

            base = AutoModelForImageTextToText.from_pretrained(
                config.BASE_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto" if on_cuda else None,
                low_cpu_mem_usage=True,
            )
            if not on_cuda:
                base = base.to(device)

            # One adapter, whichever version — never a second one stacked on top.
            model = PeftModel.from_pretrained(base, target.path_or_id())
            if not on_cuda:
                model = model.to(device)
            model.eval()
            cm._verify_cure_adapter_loaded(model)

            self.model, self.processor = model, processor
            self.device, self.version = str(device), target.version
        except Exception as exc:                              # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.model = self.processor = None
            raise
        finally:
            self.loading = False

    def unload(self) -> None:
        self.model = self.processor = None
        self.version = None

    # --------------------------------------------------------------- detect
    def detect(self, image_path: str, max_new_tokens: int | None = None) -> dict[str, Any]:
        if not self.ready:
            self.load()
        import compare_models as cm

        started = time.time()
        raw = cm.load_raw_xray_rgb(image_path)
        cure_image = cm.load_xray_as_rgb_cure(image_path)
        orig_size = raw.size

        with _lock:
            text = cm.run_cure_chat(
                self.model, self.processor, cure_image,
                cm.GROUNDED_REPORT_PROMPT,
                max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            )

        pred = cm.parse_grounded_report_cxcywh(text)
        findings = cm.build_keyword_findings(pred, orig_size)
        return {
            "model": "cure",
            "device": self.device,
            "adapter_version": self.version,
            "image_path": image_path,
            "image_size": [orig_size[0], orig_size[1]],
            "report_text": text,
            "box_format": "cxcywh",
            "findings": findings,
            "latency_s": round(time.time() - started, 3),
        }


SERVICE = CureService()
