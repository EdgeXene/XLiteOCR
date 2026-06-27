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
