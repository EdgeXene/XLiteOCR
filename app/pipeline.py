"""One page, from a decoded image to its JSON block.

Extracted from app/server.py so the document worker can do the work without
importing the FastAPI application. The worker runs in a separate process and
has no HTTP surface; pulling in the app there would build a router, install
middleware and load the web stack for nothing.
"""

from __future__ import annotations

from app import limits
from engine import color_detect, ocr_engine, structure


def process_page(image, page_index: int, structured: bool, figure_budget: int) -> dict:
    lines = ocr_engine.run(image)
    out_lines = []
    for ln in lines:
        color = color_detect.region_color(image, ln["box"])
        out_lines.append(
            {
                "text": ln["text"],
                "box": ln["box"],
                "confidence": ln["confidence"],
                "color": color,
            }
        )
    page = {
        "page": page_index,
        "lines": out_lines,
        "full_text": "\n".join(l["text"] for l in out_lines),
    }
    warnings: list[str] = []
    if structured:
        s = structure.parse(image)
        page["markdown"] = s["markdown"]
        page["blocks"] = s["blocks"]
        figures = s["figures"]
        allowed = max(0, min(limits.MAX_FIGURES_PER_PAGE, figure_budget))
        if len(figures) > allowed:
            warnings.append(
                f"page has {len(figures)} figures; {allowed} returned "
                "(figure limit reached)"
            )
            figures = figures[:allowed]
        page["figures"] = figures
    if warnings:
        page["warnings"] = warnings
    return page


