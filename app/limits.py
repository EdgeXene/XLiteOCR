"""Upload and decode limits (defense-in-depth against resource-exhaustion).

A deployment will normally cap request bodies at its reverse proxy, but the
service also binds 127.0.0.1:3011 directly, so any local caller bypasses those
caps. These app-level limits protect the single uvicorn worker regardless of
who calls it.

Stdlib + Pillow only — no new dependency, no license impact.
"""

from __future__ import annotations

import warnings

from PIL import Image

# Reject the raw upload before decode. Real documents (image or PDF) are well
# under this; anything larger is almost certainly a bomb aimed at the worker.
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB

# Hard pixel ceiling for a single decoded/rasterized page. 40 Mpx comfortably
# covers A4 at 300+ DPI while blocking the decompression-bomb band that Pillow
# only warns about (its default MAX_IMAGE_PIXELS is ~89.5 Mpx and it does not
# error until ~178.9 Mpx).
MAX_IMAGE_PIXELS = 40_000_000


# Maximum pages (PDF) or frames (multipage TIFF/GIF) accepted in one document.
# Exceeding this is REFUSED, not truncated: the previous code rendered the first
# 50 pages of a 400-page PDF and returned them with no indication that 350 were
# dropped, which is a wrong answer presented as a complete one.
MAX_PAGES = 50

# Aggregate decoded pixels across every page of one document. MAX_IMAGE_PIXELS
# bounds a single page; without this, 50 pages at the per-page ceiling is 2
# gigapixels of decoded bitmap for one request.
MAX_TOTAL_PIXELS = 250_000_000

# Vectorized figures are unbounded work per page and unbounded bytes in the
# response. Both are capped.
MAX_FIGURES_PER_PAGE = 20
MAX_FIGURES_PER_DOCUMENT = 100

# Wall clock for processing one accepted document, after the body has arrived.
PROCESSING_DEADLINE_SECONDS = 300.0

# Ceiling on the serialized response. A pathological page can produce more SVG
# and text than any caller wants to receive.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class UploadTooLarge(ValueError):
    """Raised when a request body or a decoded page exceeds the limits above."""


class TooManyPages(ValueError):
    """Raised when a document declares more pages/frames than MAX_PAGES."""


class BudgetExceeded(ValueError):
    """Raised when the aggregate pixel or response budget is exhausted."""


class ProcessingTimeout(ValueError):
    """Raised when processing exceeds PROCESSING_DEADLINE_SECONDS."""


def install_pillow_guards() -> None:
    """Lower Pillow's pixel ceiling and make its bomb warning a hard error.

    Import-time side effect: calling code (server startup) invokes this once so
    the ~89.5-178.9 Mpx warn-only band is blocked too. region_color / structure
    keep working on legitimately-sized pages.
    """
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    warnings.simplefilter("error", Image.DecompressionBombWarning)


def check_upload_size(data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadTooLarge(
            f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
        )


def check_pixel_budget(width: int, height: int) -> None:
    if width * height > MAX_IMAGE_PIXELS:
        raise UploadTooLarge(
            f"image exceeds {MAX_IMAGE_PIXELS // 1_000_000} megapixel limit"
        )


def check_page_count(declared: int) -> None:
    """Refuse an over-long document up front, before rendering anything."""
    if declared > MAX_PAGES:
        raise TooManyPages(
            f"document has {declared} pages; the limit is {MAX_PAGES}. "
            "Split the document and submit the parts."
        )


class PixelBudget:
    """Aggregate decoded-pixel accounting across the pages of one document."""

    def __init__(self, total: int | None = None):
        # Read the module attribute at call time, not as a default argument.
        # A default is evaluated once at import, which silently freezes the
        # limit: a deployment that raised MAX_TOTAL_PIXELS would still get the
        # value that was current when this module was first imported.
        self.total = MAX_TOTAL_PIXELS if total is None else total
        self.used = 0

    def charge(self, width: int, height: int) -> None:
        check_pixel_budget(width, height)
        self.used += width * height
        if self.used > self.total:
            raise BudgetExceeded(
                f"document exceeds the {self.total // 1_000_000} megapixel "
                "total decode budget"
            )
