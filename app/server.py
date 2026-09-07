"""XLiteOCR FastAPI service.

Endpoints
  GET  /health           -> {"status":"ok", ...}
  POST /ocr              -> raw text + boxes + per-region color (fast core)
       ?structured=true  -> additionally markdown + typed blocks + figure SVGs

Response shape:
  {
    "schema_version": 2,
    "source_pages": 3,        # what the document declares
    "processed_pages": 3,     # what was actually OCR'd
    "complete": true,         # processed_pages == source_pages
    "pages": [
      {
        "page": 0,
        "lines": [{"text","box","confidence","color":{hex,rgb,name}}],
        "full_text": "...",
        # when structured=true:
        "markdown": "...",
        "blocks": [{type,bbox,...}],
        "figures": [{box,type,svg}],
        "warnings": ["..."]    # only when something was capped
      }
    ]
  }

`complete` exists because the previous version could not be complete and could
not say so: it rendered the first 50 pages of any PDF and returned them with no
indication that more existed, so a caller could not tell a 50-page document
from the first 50 pages of a 400-page one. Over-long documents are now refused
with 413 instead, and the field is explicit rather than implied.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Allow "engine"/"app" imports when launched as a script under pm2.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from app import ingest, limits, worker_host
from app.pipeline import process_page  # noqa: F401  (re-exported for tests)

log = logging.getLogger("xlite-ocr")

SCHEMA_VERSION = 2

# Lower Pillow's pixel ceiling and turn its decompression-bomb warning into a
# hard (catchable) error for the whole process.
limits.install_pillow_guards()

app = FastAPI(title="XLiteOCR", version="1.1.0")

# Counts body bytes as the server delivers them, before the multipart parser
# runs. This is the bound that `check_upload_size(await file.read())` only
# described after the fact.
app.add_middleware(ingest.BoundedBodyMiddleware)

# One document at a time. The engine is a single warmed model in a single
# process; a second concurrent document doubles peak memory and makes both
# slower, so callers are told to retry rather than parked on an unbounded queue.
admission = ingest.SingleDocumentAdmission()

# One child process per document. Holds no state between requests by design.
worker = worker_host.DocumentWorker()


@app.get("/health")
def health():
    return {"status": "ok", "service": "xlite-ocr", "version": "1.1.0"}


def _error(status: int, message: str, **extra):
    body = {"error": message}
    body.update(extra)
    return JSONResponse(body, status_code=status)


def _busy():
    return JSONResponse(
        {"error": "another document is being processed, retry shortly"},
        status_code=503,
        headers={"Retry-After": "5"},
    )


@app.post("/ocr")
async def ocr(request: Request, structured: bool = Query(False)):
    if admission.busy:
        return _busy()
    try:
        async with admission:
            return await _handle_ocr(request, structured)
    except ingest.Busy:
        return _busy()


async def _handle_ocr(request: Request, structured: bool):
    # 1. Exactly one file part, with every spool closed on every path.
    try:
        filename, data = await ingest.read_single_upload(request)
    except ingest.MalformedUpload as e:
        return _error(400, str(e))

    if not data:
        return _error(400, "empty file")

    try:
        limits.check_upload_size(data)
    except limits.UploadTooLarge as e:
        return _error(413, str(e))

    # 2. NO native parsing here. Counting pages meant handing the document to
    #    PDFium or Pillow inside this process, while holding admission and
    #    outside the worker's limits and deadline, which is exactly what the
    #    process boundary exists to prevent. The worker refuses an over-long
    #    document itself, before rendering any page, and reports it back.

    # 3. Hand the raw bytes to a fresh child process. Every native library
    #    that touches this document runs there: PDFium, Pillow, OpenCV, Paddle
    #    and VTracer. A crash, an out-of-memory kill or a hang costs one
    #    document instead of the service, and the wall deadline below is a real
    #    kill rather than a promise, because a synchronous native call inside
    #    this process could never have been cancelled.
    try:
        result = await worker.run(
            data,
            filename,
            structured,
            deadline_seconds=limits.PROCESSING_DEADLINE_SECONDS,
            is_disconnected=request.is_disconnected,
        )
    except worker_host.WorkerTimeout as e:
        return _error(504, str(e))
    except worker_host.WorkerUnavailable as e:
        return JSONResponse({"error": str(e)}, status_code=503,
                            headers={"Retry-After": "5"})
    except worker_host.WorkerError as e:
        log.error("ocr: worker failed: %s", e)
        return _error(500, "processing failed")

    # The worker reports its own refusals rather than raising across the
    # process boundary, so they are mapped back to status codes here.
    if "error" in result:
        kind = result.get("kind")
        if kind == "resource":
            return _error(413, result["error"])
        if kind == "response_too_large":
            return _error(413, "the result is too large to return")
        if kind == "timeout":
            return _error(504, result["error"])
        if kind in ("unreadable", "protocol"):
            return _error(400, "could not read input")
        # The kind is a class name, never a message: an exception message from
        # a decoder routinely quotes the bytes that upset it.
        log.error("ocr: worker reported %s", kind)
        return _error(500, "processing failed")

    pages = result.get("pages") or []

    # 4. A 200 means the whole document was processed. Say so explicitly.
    return {
        "schema_version": SCHEMA_VERSION,
        "source_pages": result.get("source_pages"),
        "processed_pages": result.get("processed_pages", len(pages)),
        "complete": bool(result.get("complete", False)),
        "pages": pages,
    }
