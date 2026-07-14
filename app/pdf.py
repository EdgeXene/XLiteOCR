"""PDF -> page images via pypdfium2 (PDFium, BSD-3, V8-disabled build).

We deliberately do NOT use pdf2image/poppler (GPL). PDFium rasterizes each page
to a PIL image at a chosen DPI for the OCR + structure pipeline.
"""

from __future__ import annotations

import io
import math

import pypdfium2 as pdfium
from PIL import Image

from app.limits import MAX_IMAGE_PIXELS, check_pixel_budget

DEFAULT_DPI = 200


def is_pdf(data: bytes, filename: str | None = None) -> bool:
    if filename and filename.lower().endswith(".pdf"):
        return True
    return data[:5] == b"%PDF-"


def _page_scale(page, dpi: int) -> float:
    """Effective render scale, clamped so no page exceeds the pixel budget.

    A tiny PDF can declare an enormous MediaBox; rendering at the nominal scale
    would rasterize a multi-GB bitmap (this path never touches Image.open, so
    Pillow's guard does not apply). Clamp the scale instead of rejecting so a
    legitimate large-format drawing still OCRs, just at reduced resolution.
    """
    scale = dpi / 72.0
    w_pt, h_pt = page.get_size()
    area_pt = w_pt * h_pt
    if area_pt <= 0:
        return scale
    # 0.999 margin: PDFium rounds pixel dimensions up from scale*points, so the
    # rendered bitmap can land a hair over an exact-fit scale. Stay just under.
    max_scale = math.sqrt(MAX_IMAGE_PIXELS / area_pt) * 0.999
    return min(scale, max_scale)


def render_pages(data: bytes, dpi: int = DEFAULT_DPI, max_pages: int = 50) -> list[Image.Image]:
    """Rasterize each PDF page to an RGB PIL image.

    scale = dpi / 72 (PDF user-space is 72 units/inch), clamped per page so a
    crafted MediaBox cannot exhaust the worker's memory.
    """
    pdf = pdfium.PdfDocument(data)
    try:
        pages: list[Image.Image] = []
        n = min(len(pdf), max_pages)
        for i in range(n):
            page = pdf[i]
            bitmap = page.render(scale=_page_scale(page, dpi))
            pil = bitmap.to_pil().convert("RGB")
            pages.append(pil)
        return pages
    finally:
        pdf.close()


def load_image(data: bytes) -> Image.Image:
    """Load a raster image (PNG/JPEG/etc.) into RGB.

    Reads the header dimensions first (Image.open is lazy and does not decode
    pixels) and rejects anything over the pixel budget before allocating.
    """
    img = Image.open(io.BytesIO(data))
    check_pixel_budget(*img.size)
    return img.convert("RGB")
