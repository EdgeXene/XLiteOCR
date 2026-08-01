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


class UploadTooLarge(ValueError):
    """Raised when a request body or a decoded page exceeds the limits above."""


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
