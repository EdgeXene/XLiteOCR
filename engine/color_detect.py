"""Per-region text color detection — XLiteOCR's differentiator.

Given a detected text polygon, isolate the foreground (the glyph strokes) from
the background and report the dominant stroke color. Pure Pillow + numpy: no
OpenCV, which keeps the dependency stack free of OpenCV's bundled-codec license
question (see COMMERCIAL-USE.md).

Algorithm per region:
  1. Crop the polygon's bounding box from the page image.
  2. Convert to grayscale; threshold (Otsu) to split dark vs. light pixels.
  3. Take the darker class as the stroke (ink on paper is the common case);
     only treat the dark class as background when it is a solid dark banner
     (dominant area AND uniform color), which makes the light class the stroke.
     This is independent of pixel-count majority, so it stays correct for
     bold/large headings where the strokes cover >50% of the crop, and it does
     not rely on edge/corner sampling (tight detector crops clip through glyphs).
  4. K-means (k=2) over the stroke pixels' RGB and return the cluster furthest
     from the background color — the solid glyph body, not the anti-alias fringe.
  5. Emit {hex, rgb, name} where name is the nearest CSS basic color.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Basic CSS color anchors for nearest-name lookup. Kept small and obvious;
# extend if callers need finer names.
_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (220, 20, 20),
    "orange": (255, 140, 0),
    "yellow": (240, 220, 20),
    "green": (20, 160, 40),
    "teal": (0, 128, 128),
    "blue": (30, 60, 220),
    "navy": (0, 0, 128),
    "purple": (128, 0, 160),
    "magenta": (220, 20, 200),
    "pink": (255, 150, 190),
    "brown": (140, 80, 40),
}


def _otsu_threshold(gray: np.ndarray) -> int:
    """Classic Otsu threshold on a 0-255 grayscale array."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    if total == 0:
        return 128
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def _kmeans_dominant(
    pixels: np.ndarray,
    k: int = 2,
    iters: int = 12,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Tiny fixed-iteration k-means over stroke pixels.

    Returns the RGB of the cluster whose center is *furthest* from the
    background color when one is given (the solid glyph body, ignoring the
    anti-aliasing fringe that sits between stroke and paper); otherwise falls
    back to the most populous cluster.
    """
    if len(pixels) == 0:
        return np.array([0, 0, 0], dtype=float)
    if len(pixels) <= k:
        return pixels.mean(axis=0)
    pts = pixels.astype(float)
    # Deterministic init: spread seeds across the luminance range.
    lum = pts @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    centers = pts[order[np.linspace(0, len(pts) - 1, k).astype(int)]].copy()
    labels = np.zeros(len(pts), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = pts[mask].mean(axis=0)
    if background is not None:
        dist = np.linalg.norm(centers - np.asarray(background, dtype=float), axis=1)
        return centers[int(dist.argmax())]
    counts = np.bincount(labels, minlength=k)
    return centers[counts.argmax()]


def _nearest_name(rgb: tuple[int, int, int]) -> str:
    arr = np.array(rgb, dtype=float)
    best, best_d = "black", float("inf")
    for name, ref in _NAMED_COLORS.items():
        d = float(np.linalg.norm(arr - np.array(ref, dtype=float)))
        if d < best_d:
            best_d, best = d, name
    return best


_DARK_BG_AREA_FRAC = 0.65   # dark class must dominate the crop to be background
_DARK_BG_MAX_STD = 25.0     # ...and be color-uniform (a solid banner, not text)


def _background_is_dark(dark: np.ndarray, light: np.ndarray, n_total: int) -> bool:
    """Decide whether the DARK Otsu class is the background (light text on dark).

    The common case is dark ink on light paper, so the stroke is the darker
    class. We only treat the dark class as background — making the LIGHT class
    the stroke — when the dark class both dominates the crop area and is
    color-uniform, i.e. a solid dark banner with lighter text on it. Pixel-count
    majority alone is not enough (bold headings make the stroke the majority),
    and edge/corner sampling is unreliable because tight detector crops clip
    through glyphs; the area+uniformity test is what holds up on real crops.
    """
    if len(dark) == 0:
        return False
    if len(light) == 0:
        return True
    dark_frac = len(dark) / n_total if n_total else 0.0
    dark_std = float(dark.std(axis=0).mean())
    return dark_frac > _DARK_BG_AREA_FRAC and dark_std < _DARK_BG_MAX_STD


def _polygon_bbox(polygon) -> tuple[int, int, int, int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def region_color(image: Image.Image, polygon) -> dict:
    """Return the dominant foreground (text stroke) color of a polygon region.

    Args:
        image: full-page PIL Image (RGB).
        polygon: list of (x, y) points (a quad from the detector), or an
                 (x0, y0, x1, y1) bbox tuple.
    Returns:
        {"hex": "#rrggbb", "rgb": [r, g, b], "name": "<css-name>"}
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    if (
        len(polygon) == 4
        and all(isinstance(v, (int, float)) for v in polygon)
    ):
        x0, y0, x1, y1 = (int(v) for v in polygon)
    else:
        x0, y0, x1, y1 = _polygon_bbox(polygon)

    w, h = image.size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return {"hex": "#000000", "rgb": [0, 0, 0], "name": "black"}

    crop = np.asarray(image.crop((x0, y0, x1, y1)), dtype=np.uint8)
    flat = crop.reshape(-1, 3)
    gray = (flat @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)

    t = _otsu_threshold(gray)
    dark = flat[gray <= t]
    light = flat[gray > t]

    # Stroke = the class the background is NOT. Default to the darker class
    # (ink on paper); only treat the dark class as background when it is a solid
    # dark banner (dominant area + uniform color). This stays correct for bold
    # headings where the stroke covers >50% of the crop, without being fooled by
    # tight crops that clip through glyphs at the edges.
    if _background_is_dark(dark, light, len(gray)):
        stroke, background = light, dark
    else:
        stroke, background = dark, light

    bg_ref = background.mean(axis=0) if len(background) else None
    dominant = _kmeans_dominant(stroke, k=2, background=bg_ref)
    rgb = tuple(int(round(c)) for c in np.clip(dominant, 0, 255))
    hexv = "#{:02x}{:02x}{:02x}".format(*rgb)
    return {"hex": hexv, "rgb": list(rgb), "name": _nearest_name(rgb)}
