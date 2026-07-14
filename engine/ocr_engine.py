"""PP-OCR text core for XLiteOCR.

Thin wrapper over PaddleOCR (Apache-2.0 code + weights). Loaded once and reused
(warm) so the model and MKLDNN graph stay resident. Returns per-line text, the
detector's quad polygon, and a confidence score.

CPU tuning for the 32-core host (per compliance/perf review):
  enable_mkldnn=True   -> MKLDNN graph, big win on high-core CPUs
  cpu_threads=16       -> pin to 16 to avoid hyperthreading thrashing
  use_gpu=False        -> CPU-only box, no GPU
Do NOT set OMP_NUM_THREADS in the environment; PaddleOCR's cpu_threads governs
this and the env var triggers an OpenBLAS warning / can hurt throughput.

Note on weights: PaddleOCR 2.10.0 with lang='en' serves PP-OCRv3 detection +
PP-OCRv4 English recognition (both Apache-2.0). PP-OCRv5 multilingual weights
are a config swap later; see README. The pipeline shape is identical.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
from PIL import Image

_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _get_ocr(lang: str = "en"):
    """Warm, cached PaddleOCR instance per language. Thread-safe init."""
    from paddleocr import PaddleOCR

    with _LOCK:
        return PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            use_gpu=False,
            enable_mkldnn=True,
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
