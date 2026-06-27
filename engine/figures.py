"""Figure -> SVG for XLiteOCR.

PRIMARY path (v1, used for ALL figures): VTracer (MIT) traces a raster figure
crop into geometric SVG paths. Deterministic, in-scope, commercial-safe. Great
on logos / line-art / diagrams; approximate on photographs (documented honestly).

SECONDARY path (deferred, NOT v1): native PDF vector extraction via
pdfminer.six (MIT). PDFium has no high-level sub-region->SVG API, so true vector
recovery means parsing raw content streams. Stubbed here for later.

Charts: the structured layer additionally returns PP-Chart2Table data alongside
the VTracer SVG (see engine/structure.py).
"""

from __future__ import annotations

import vtracer
from PIL import Image


def raster_to_svg(image: Image.Image, *, color_precision: int = 6,
                  filter_speckle: int = 4) -> str:
    """Trace a raster crop into an SVG string via VTracer.

    Uses convert_pixels_to_svg over RGBA pixels so we never touch disk.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    w, h = image.size
    pixels = list(image.getdata())  # list of (r,g,b,a) tuples
    svg = vtracer.convert_pixels_to_svg(
        pixels,
        size=(w, h),
        colormode="color",
        filter_speckle=filter_speckle,
        color_precision=color_precision,
    )
    return svg


def figure_to_svg(image: Image.Image, box) -> str:
    """Crop the figure region from a page image and vectorize it.

    box: (x0, y0, x1, y1) in page pixels.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = image.size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = image.crop((x0, y0, x1, y1))
    return raster_to_svg(crop)


def native_vector_extract_stub(*_args, **_kwargs):  # pragma: no cover
    """Deferred secondary path (pdfminer.six). Not implemented in v1."""
    raise NotImplementedError(
        "Native PDF vector extraction is a deferred enhancement; "
        "v1 uses the VTracer raster path for all figures."
    )
