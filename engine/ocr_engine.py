"""PP-OCR text core for XLiteOCR.

Thin wrapper over PaddleOCR (Apache-2.0 code + weights). Loaded once and reused
(warm) so the model and MKLDNN graph stay resident. Returns per-line text, the
detector's quad polygon, and a confidence score.

CPU tuning for the 32-core host (per compliance/perf review):
  enable_mkldnn        -> True on Linux (oneDNN graph, big win on
                          high-core CPUs). False on Windows: paddle
                          3.3.1's win_amd64 build fails inside its
                          oneDNN-fused conv on the PP-OCR models
                          ("OneDnnContext does not have the input
                          Filter", operator fused_conv2d); 2.6.2
                          tolerated it.
  cpu_threads=16       -> pin to 16 to avoid hyperthreading thrashing
  use_gpu=False        -> CPU-only box, no GPU
Do NOT set OMP_NUM_THREADS in the environment; PaddleOCR's cpu_threads governs
this and the env var triggers an OpenBLAS warning / can hurt throughput.

Note on weights: PaddleOCR 2.10.0 with lang='en' serves PP-OCRv3 detection +
PP-OCRv4 English recognition (both Apache-2.0). PP-OCRv5 multilingual weights
are a config swap later; see README. The pipeline shape is identical.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
from PIL import Image

_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _get_ocr(lang: str = "en"):
    """Warm, cached PaddleOCR instance per language. Thread-safe init."""
    from paddleocr import PaddleOCR

    if os.name == "nt":
        # paddle 3.3.1 defaults oneDNN ON for CPU inference
        # (Config().mkldnn_enabled() is True before any enable call)
        # and its win_amd64 oneDNN fused_conv2d kernel is broken on
        # the PP-OCR models ("OneDnnContext does not have the input
        # Filter"). paddleocr only ever ENABLES onednn on its config,
        # so force it off on every Config before predictors build.
        from paddle import inference as _pi
        if not getattr(_pi.Config, "_vx_onednn_off", False):
            class _Config(_pi.Config):
                _vx_onednn_off = True
                def __init__(self, *args):
                    super().__init__(*args)
                    (self.disable_onednn
                     if hasattr(self, "disable_onednn")
                     else self.disable_mkldnn)()
            _pi.Config = _Config

    with _LOCK:
        return PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            use_gpu=False,
            # paddle 3.3.1 win_amd64 breaks in the oneDNN fused_conv2d
            # kernel on these models; Linux is fine both ways.
            enable_mkldnn=(os.name != "nt"),
            cpu_threads=16,
            show_log=False,
        )


def _to_ndarray(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def run(image: Image.Image, lang: str = "en") -> list[dict]:
    """Run detection + recognition on a single page image.

    Returns a list of line dicts:
        {"text": str, "box": [[x,y]x4], "confidence": float}
    Ordered as the detector returns them (roughly reading order).
    """
    ocr = _get_ocr(lang)
    arr = _to_ndarray(image)

    # PaddleOCR 2.x classic API: ocr(img, cls=False) -> [ [ [box, (text, conf)], ... ] ]
    with _LOCK:
        raw = ocr.ocr(arr, cls=False)

    lines: list[dict] = []
    if not raw or raw[0] is None:
        return lines
    for entry in raw[0]:
        box, (text, conf) = entry
        lines.append(
            {
                "text": text,
                "box": [[float(x), float(y)] for x, y in box],
                "confidence": float(conf),
            }
        )
    return lines
