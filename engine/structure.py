"""Structured-document layer for XLiteOCR (PP-Structure, Apache-2.0).

Wraps PaddleOCR's PPStructure to produce, per page:
  - markdown   : a reading-order markdown rendering
  - blocks     : typed layout blocks [{type, bbox, ...}]
  - figures    : figure regions vectorized to SVG (engine/figures.py)

PPStructure region shape (confirmed on this build):
  {"type": "text|title|table|figure|list|...", "bbox": [x0,y0,x1,y1], "res": ...}
    - table -> res = {"html": "<table>...", "cell_bbox": [...]}
    - text/title/list -> res = [ {"text":..., "confidence":..., "text_region":...}, ... ]

Formula recognition is DISABLED by default: its LaTeX model is ~100 MB and slow,
at odds with "lightweight". Enable via XLITE_FORMULA=1 if needed.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
from PIL import Image

from . import figures as figures_mod

_LOCK = threading.Lock()
_FORMULA = os.environ.get("XLITE_FORMULA", "0") == "1"


@lru_cache(maxsize=2)
def _get_structure(lang: str = "en"):
    from paddleocr import PPStructure

    if os.name == "nt":
        # Same guard as ocr_engine._get_ocr: paddle 3.3.1 defaults oneDNN
        # ON for CPU inference and its win_amd64 oneDNN fused_conv2d
        # kernel is broken on the PP models; paddleocr only ever ENABLES
        # onednn, so force it off on every Config before predictors build.
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
        return PPStructure(
            lang=lang,
            show_log=False,
            use_gpu=False,
            # broken win_amd64 oneDNN kernels; Linux is fine both ways
            enable_mkldnn=(os.name != "nt"),
            cpu_threads=16,
            recovery=False,
            formula=_FORMULA,
        )


def _region_text(res) -> str:
    """Join the OCR lines of a text/title/list region into one string."""
    if not isinstance(res, list):
        return ""
    parts = []
    for line in res:
        t = line.get("text") if isinstance(line, dict) else None
        if t:
            parts.append(t)
    return " ".join(parts)


def _block_to_markdown(block: dict) -> str:
    t = block.get("type", "text")
    if t == "title":
        return f"# {block.get('text', '').strip()}"
    if t == "table":
        return block.get("html", "")
    if t == "figure":
        return "![figure](figure)"
    if t in ("header", "footer"):
        return ""
    return block.get("text", "").strip()


def parse(image: Image.Image, lang: str = "en", with_figures: bool = True) -> dict:
    """Return {"markdown": str, "blocks": [...], "figures": [...]} for one page."""
    eng = _get_structure(lang)
    arr = np.asarray(image.convert("RGB"))
    with _LOCK:
        regions = eng(arr)

    # Sort top-to-bottom, then left-to-right for stable reading order.
    regions = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    blocks: list[dict] = []
    fig_out: list[dict] = []
    for r in regions:
        rtype = r.get("type", "text")
        bbox = [float(v) for v in r.get("bbox", [0, 0, 0, 0])]
        res = r.get("res")
        block: dict = {"type": rtype, "bbox": bbox}

        if rtype == "table" and isinstance(res, dict):
            block["html"] = res.get("html", "")
        elif rtype == "figure":
            if with_figures:
                try:
                    svg = figures_mod.figure_to_svg(image, bbox)
                except Exception as e:  # vectorization is best-effort
                    svg = ""
                    block["svg_error"] = str(e)
                block["svg"] = svg
                fig_out.append({"box": bbox, "type": "figure", "svg": svg})
        else:
            block["text"] = _region_text(res)

        blocks.append(block)

    markdown = "\n\n".join(
        md for md in (_block_to_markdown(b) for b in blocks) if md
    )
    return {"markdown": markdown, "blocks": blocks, "figures": fig_out}
