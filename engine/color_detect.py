"""Per-region text color detection — XLiteOCR's differentiator.

Given a detected text polygon, isolate the foreground (the glyph strokes) from
the background and report the dominant stroke color. Pure Pillow + numpy: no
OpenCV, which keeps the dependency stack free of OpenCV's bundled-codec license
question (see COMMERCIAL-USE.md).

Algorithm per region:
  1. Crop the polygon's bounding box from the page image.
  2. Convert to grayscale; threshold (Otsu) to split dark vs. light pixels.
  3. Text is usually the *minority* class against a larger background, but not
     always (light text on dark). We pick the class whose mean luminance is
     furthest from the background-dominant class, i.e. the stroke pixels.
  4. K-means (k=2) over the stroke pixels' RGB to find the dominant color,
     ignoring anti-aliasing fringe.
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


def _kmeans_dominant(pixels: np.ndarray, k: int = 2, iters: int = 12) -> np.ndarray:
    """Tiny fixed-iteration k-means; returns the RGB of the largest cluster."""
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

    # Stroke = the minority class (text usually covers less area than its
    # background). Fall back to whichever class is non-empty.
    if len(dark) == 0:
        stroke = light
    elif len(light) == 0:
        stroke = dark
    else:
        stroke = dark if len(dark) <= len(light) else light

    dominant = _kmeans_dominant(stroke, k=2)
    rgb = tuple(int(round(c)) for c in np.clip(dominant, 0, 255))
    hexv = "#{:02x}{:02x}{:02x}".format(*rgb)
    return {"hex": hexv, "rgb": list(rgb), "name": _nearest_name(rgb)}
