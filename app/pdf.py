"""PDF -> page images via pypdfium2 (PDFium, BSD-3, V8-disabled build).

We deliberately do NOT use pdf2image/poppler (GPL). PDFium rasterizes each page
to a PIL image at a chosen DPI for the OCR + structure pipeline.
"""

from __future__ import annotations

import io

import pypdfium2 as pdfium
from PIL import Image

DEFAULT_DPI = 200


def is_pdf(data: bytes, filename: str | None = None) -> bool:
    if filename and filename.lower().endswith(".pdf"):
        return True
    return data[:5] == b"%PDF-"


def render_pages(data: bytes, dpi: int = DEFAULT_DPI, max_pages: int = 50) -> list[Image.Image]:
    """Rasterize each PDF page to an RGB PIL image.

    scale = dpi / 72 (PDF user-space is 72 units/inch).
    """
    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(data)
    try:
        pages: list[Image.Image] = []
        n = min(len(pdf), max_pages)
        for i in range(n):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil().convert("RGB")
            pages.append(pil)
        return pages
    finally:
        pdf.close()


def load_image(data: bytes) -> Image.Image:
    """Load a raster image (PNG/JPEG/etc.) into RGB."""
    return Image.open(io.BytesIO(data)).convert("RGB")
