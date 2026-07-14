"""Unit + smoke tests for the XLiteOCR engine and API."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import color_detect, figures, ocr_engine  # noqa: E402


def _font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _text_image(text: str, fill, size=(400, 100)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((20, 25), text, fill=fill, font=_font(44))
    return img


# ---------- color detection ----------

@pytest.mark.parametrize(
    "fill,expected",
    [((0, 0, 0), "black"), ((200, 0, 0), "red"), ((0, 0, 200), "blue"),
     ((0, 150, 0), "green")],
)
def test_color_detection(fill, expected):
    img = _text_image("SAMPLE", fill)
    # whole-image bbox region
    res = color_detect.region_color(img, (0, 0, img.width, img.height))
    assert res["name"] == expected, f"{fill} -> {res}"
    assert res["hex"].startswith("#") and len(res["hex"]) == 7


def test_color_detection_stroke_majority():
    """Bold/large text whose strokes cover >50% of a tight crop must still
    report the stroke color, not the background (regression: minority heuristic
    reported the paper color for stroke-dominant crops)."""
    # 40x40 crop, ~60% red foreground on a white border ring.
    crop = Image.new("RGB", (40, 40), "white")
    ImageDraw.Draw(crop).rectangle([6, 6, 33, 33], fill=(220, 20, 20))
    res = color_detect.region_color(crop, (0, 0, 40, 40))
    assert res["name"] == "red", res


def test_color_detection_light_on_dark():
    """White text on a solid dark banner: the dark class is background (dominant
    area + uniform), so the light class is the stroke."""
    crop = Image.new("RGB", (60, 40), (10, 10, 10))
    ImageDraw.Draw(crop).text((6, 8), "HI", fill=(255, 255, 255), font=_font(22))
    res = color_detect.region_color(crop, (0, 0, 60, 40))
    assert res["name"] == "white", res


def test_color_detection_on_synthetic_sample():
    """Regression: the shipped 'colored text' sample must report the foreground
    ink color per line (black / red / blue), not the white background. This
    guards the exact failure a browser check caught where every line read
    'white' because the background was mis-selected as the stroke."""
    from engine import ocr_engine

    sample = Path(__file__).resolve().parent.parent / "samples" / "synthetic.png"
    if not sample.exists():
        pytest.skip("synthetic.png sample not present")
    img = Image.open(sample).convert("RGB")
    lines = ocr_engine.run(img)
    names = {color_detect.region_color(img, ln["box"])["name"] for ln in lines}
    # No line should read as the background paper color.
    assert "white" not in names, names
    # The three foreground colors should all appear.
    assert {"black", "red", "blue"} <= names, names


# ---------- OCR core ----------

def test_ocr_reads_text():
    img = _text_image("HELLO", (0, 0, 0), size=(420, 110))
    lines = ocr_engine.run(img)
    assert lines, "no lines detected"
    joined = " ".join(l["text"].upper() for l in lines)
    assert "HELLO" in joined
    assert all(0.0 <= l["confidence"] <= 1.0 for l in lines)
    assert all(len(l["box"]) == 4 for l in lines)


# ---------- figure -> SVG ----------

def test_vtracer_svg():
    logo = Image.new("RGB", (120, 120), "white")
    d = ImageDraw.Draw(logo)
    d.ellipse([20, 20, 100, 100], fill=(200, 30, 30))
    svg = figures.raster_to_svg(logo)
    assert svg.strip().startswith("<?xml") or "<svg" in svg
    assert "path" in svg.lower() or "<svg" in svg.lower()


# ---------- API smoke ----------

def test_api_smoke():
    from fastapi.testclient import TestClient
    from app.server import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    img = _text_image("RED WORLD", (200, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/ocr", files={"file": ("t.png", buf, "image/png")})
    assert r.status_code == 200
    page = r.json()["pages"][0]
    assert page["lines"], "no lines from API"
    assert page["lines"][0]["color"]["name"] == "red"


# ---------- limits & error handling ----------

@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_load_image_rejects_oversized_pixels():
    """A small file declaring huge dimensions must be rejected before decode
    (decompression-bomb guard), not allocated."""
    from app import limits
    from app.pdf import load_image

    over = limits.MAX_IMAGE_PIXELS + 1
    # width*height just over the cap; a flat image so the file stays tiny.
    w = 8000
    h = over // w + 1
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    with pytest.raises(limits.UploadTooLarge):
        load_image(buf.getvalue())


def test_api_rejects_oversized_upload():
    from fastapi.testclient import TestClient
    from app import limits
    from app.server import app

    client = TestClient(app)
    payload = b"\x00" * (limits.MAX_UPLOAD_BYTES + 1)
    r = client.post("/ocr", files={"file": ("big.bin", io.BytesIO(payload), "application/octet-stream")})
    assert r.status_code == 413


def test_api_bad_input_generic_400():
    """Unreadable input returns a generic message — no library internals in the
    body."""
    from fastapi.testclient import TestClient
    from app.server import app

    client = TestClient(app)
    r = client.post("/ocr", files={"file": ("x.png", io.BytesIO(b"not an image"), "image/png")})
    assert r.status_code == 400
    assert r.json() == {"error": "could not read input"}


def test_pdf_render_scale_clamped():
    """A page with a huge MediaBox must render clamped under the pixel budget,
    not to a multi-GB bitmap."""
    from app import limits
    from app.pdf import render_pages

    # Minimal one-page PDF with a 6000x6000 MediaBox.
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 6000 6000]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n"
    )
    pages = render_pages(pdf_bytes)
    assert pages, "no pages rendered"
    for pg in pages:
        assert pg.width * pg.height <= limits.MAX_IMAGE_PIXELS, (pg.width, pg.height)
