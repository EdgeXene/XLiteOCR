"""PDF -> page images via pypdfium2 (PDFium, BSD-3, V8-disabled build).

We deliberately do NOT use pdf2image/poppler (GPL). PDFium rasterizes each page
to a PIL image at a chosen DPI for the OCR + structure pipeline.
"""

from __future__ import annotations

import io
import math

import pypdfium2 as pdfium
from PIL import Image

from app import limits
from app.limits import check_pixel_budget

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
    # Read the limit at call time. Importing it by name bound the value at
    # import, so the worker adopting a configured limit from the parent left
    # the scaler using the old ceiling: a page would be scaled for 40 Mpx and
    # then refused by a budget that had been lowered.
    max_scale = math.sqrt(limits.MAX_IMAGE_PIXELS / area_pt) * 0.999
    return min(scale, max_scale)


def page_count(data: bytes) -> int:
    """How many pages the PDF declares, without rendering any of them."""
    pdf = pdfium.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def iter_pdf_pages(data: bytes, budget, dpi: int = DEFAULT_DPI):
    """Yield one rendered page at a time, refusing rather than truncating.

    The previous implementation was::

        n = min(len(pdf), max_pages)

    which rendered the first 50 pages of a longer document and returned them
    with no indication that the rest existed. A caller could not tell a
    50-page document from the first 50 pages of a 400-page one. Over-long
    documents are now refused before any page is rendered.

    Pages are yielded one at a time and the caller is expected to finish with
    each before asking for the next, so peak memory is one page rather than
    the whole document. The aggregate pixel budget is charged per page, so a
    document that is within the page count but not within the decode budget
    stops partway with an explicit error instead of exhausting the worker.
    """
    pdf = pdfium.PdfDocument(data)
    try:
        limits.check_page_count(len(pdf))
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                scale = _page_scale(page, dpi)
                w_pt, h_pt = page.get_size()
                budget.charge(int(w_pt * scale) + 1, int(h_pt * scale) + 1)
                bitmap = page.render(scale=scale)
                pil = bitmap.to_pil().convert("RGB")
            finally:
                page.close()
            yield pil
            pil.close()
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


def frame_count(image: Image.Image) -> int:
    """Frames in a multipage raster (TIFF, GIF). 1 for an ordinary image."""
    return getattr(image, "n_frames", 1)


def iter_image_frames(data: bytes, budget):
    """Yield every frame of a raster image, not just the first.

    `Image.open(...).convert("RGB")` silently returns frame zero, so a
    multipage TIFF, which is a normal scanner output, was OCR'd one page deep
    and reported as if it were the whole document. Frames are counted and
    charged like PDF pages, and an over-long one is refused the same way.
    """
    with Image.open(io.BytesIO(data)) as img:
        limits.check_page_count(frame_count(img))
        for index in range(frame_count(img)):
            img.seek(index)
            check_pixel_budget(*img.size)
            budget.charge(*img.size)
            yield img.convert("RGB")


def source_page_count(data: bytes, filename: str | None = None) -> int:
    """Pages the document declares, before any decoding work."""
    if is_pdf(data, filename):
        return page_count(data)
    with Image.open(io.BytesIO(data)) as img:
        return frame_count(img)


def iter_document_pages(data: bytes, filename: str | None, budget):
    """One iterator over any accepted document, PDF or raster."""
    if is_pdf(data, filename):
        yield from iter_pdf_pages(data, budget)
    else:
        yield from iter_image_frames(data, budget)
