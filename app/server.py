"""XLiteOCR FastAPI service.

Endpoints
  GET  /health           -> {"status":"ok", ...}
  POST /ocr              -> raw text + boxes + per-region color (fast core)
       ?structured=true  -> additionally markdown + typed blocks + figure SVGs

Response shape (Mistral-OCR-like):
  {
    "pages": [
      {
        "page": 0,
        "lines": [{"text","box","confidence","color":{hex,rgb,name}}],
        "full_text": "...",
        # when structured=true:
        "markdown": "...",
        "blocks": [{type,bbox,...}],
        "figures": [{box,type,svg}]
      }
    ]
  }
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow "engine"/"app" imports when launched as a script under pm2.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from app import limits, pdf as pdf_mod
from engine import color_detect, ocr_engine, structure

log = logging.getLogger("xlite-ocr")

# Lower Pillow's pixel ceiling and turn its decompression-bomb warning into a
# hard (catchable) error for the whole process.
limits.install_pillow_guards()

app = FastAPI(title="XLiteOCR", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "xlite-ocr", "version": "1.0.0"}


def _process_page(image, page_index: int, structured: bool) -> dict:
    lines = ocr_engine.run(image)
    out_lines = []
    for ln in lines:
        color = color_detect.region_color(image, ln["box"])
        out_lines.append(
            {
                "text": ln["text"],
                "box": ln["box"],
                "confidence": ln["confidence"],
                "color": color,
            }
        )
    page = {
        "page": page_index,
        "lines": out_lines,
        "full_text": "\n".join(l["text"] for l in out_lines),
    }
    if structured:
        s = structure.parse(image)
        page["markdown"] = s["markdown"]
        page["blocks"] = s["blocks"]
        page["figures"] = s["figures"]
    return page


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    structured: bool = Query(False),
):
    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty file"}, status_code=400)

    try:
        limits.check_upload_size(data)
    except limits.UploadTooLarge as e:
        return JSONResponse({"error": str(e)}, status_code=413)

    try:
        if pdf_mod.is_pdf(data, file.filename):
            images = pdf_mod.render_pages(data)
        else:
            images = [pdf_mod.load_image(data)]
    except limits.UploadTooLarge as e:
        return JSONResponse({"error": str(e)}, status_code=413)
    except Exception:
        # Detail (library-internal error strings) stays server-side.
        log.warning("ocr: could not read input", exc_info=True)
        return JSONResponse({"error": "could not read input"}, status_code=400)

    try:
        pages = [_process_page(img, i, structured) for i, img in enumerate(images)]
    except Exception:
        log.error("ocr: processing failed", exc_info=True)
        return JSONResponse({"error": "processing failed"}, status_code=500)
    return {"pages": pages}
