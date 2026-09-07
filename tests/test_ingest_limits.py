"""Ingest bounds for /ocr: body cap, part discipline, page budgets, honesty.

Each test names the behavior that was wrong before. The point of several of
them is not that a limit exists but that it is applied EARLY: the previous code
had a 30 MB cap that ran after the body had already been read and spooled, so
it produced the right status code and none of the protection.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import ingest, limits, pdf as pdf_mod
from app.server import app


def client():
    return TestClient(app)


def png_bytes(w=40, h=20, color=(255, 255, 255)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def multiframe_tiff(frames=3, w=40, h=20):
    images = [
        Image.new("RGB", (w, h), (10 * i, 10 * i, 10 * i)) for i in range(frames)
    ]
    buf = io.BytesIO()
    images[0].save(buf, format="TIFF", save_all=True, append_images=images[1:])
    return buf.getvalue()


def multipage_pdf(pages=3, w=80, h=60):
    images = [Image.new("RGB", (w, h), (255, 255, 255)) for _ in range(pages)]
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    return buf.getvalue()


# ----------------------------------------------------- body cap, before parsing

class _Collector:
    """An ASGI app that records whether it ever received body bytes."""

    def __init__(self):
        self.body = b""
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            self.body += message.get("body") or b""
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def run_middleware(chunks, max_bytes, content_length=None):
    """Drive BoundedBodyMiddleware over a fake ASGI transport."""
    inner = _Collector()
    mw = ingest.BoundedBodyMiddleware(inner, max_bytes=max_bytes, paths=("/ocr",))
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {"type": "http", "path": "/ocr", "method": "POST", "headers": headers}

    queue = list(chunks)
    sent = []

    async def receive():
        if queue:
            return {"type": "http.request", "body": queue.pop(0), "more_body": bool(queue)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(mw(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, inner


def test_a_body_over_the_cap_is_refused_while_streaming():
    """The cap must hold when Content-Length is absent or lies.

    Chunked transfer has no declared length, so an early header check alone
    protects nothing. The counter is what makes the bound real.
    """
    status, inner = run_middleware([b"x" * 100] * 10, max_bytes=250)
    assert status == 413
    # The inner app must not have been handed the whole body.
    assert len(inner.body) <= 250


def test_a_declared_length_over_the_cap_is_refused_without_reading_anything():
    status, inner = run_middleware([b"x" * 10], max_bytes=100, content_length=1_000_000)
    assert status == 413
    assert not inner.called, "the app ran despite an over-cap Content-Length"


def test_a_body_within_the_cap_passes_through_untouched():
    payload = [b"abc", b"def"]
    status, inner = run_middleware(payload, max_bytes=100)
    assert status == 200
    assert inner.body == b"abcdef"


def test_a_lying_content_length_does_not_raise_the_cap():
    """An understated Content-Length must not buy extra bytes."""
    status, _ = run_middleware([b"x" * 500], max_bytes=100, content_length=10)
    assert status == 413


def test_the_middleware_ignores_paths_it_does_not_guard():
    inner = _Collector()
    mw = ingest.BoundedBodyMiddleware(inner, max_bytes=10, paths=("/ocr",))
    scope = {"type": "http", "path": "/health", "method": "GET", "headers": []}
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"x" * 1000, "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(mw(scope, receive, send))
    assert inner.called


def test_the_body_cap_leaves_room_for_multipart_framing():
    """A file at exactly the file cap must fit inside its own envelope."""
    assert ingest.MAX_BODY_BYTES > limits.MAX_UPLOAD_BYTES


# ------------------------------------------------------------- part discipline

def test_a_second_file_part_is_refused():
    """FastAPI parses with max_files=1000 by default; one is the contract."""
    r = client().post("/ocr", files=[
        ("file", ("a.png", io.BytesIO(png_bytes()), "image/png")),
        ("file", ("b.png", io.BytesIO(png_bytes()), "image/png")),
    ])
    assert r.status_code == 400


def test_a_part_with_the_wrong_name_is_refused():
    r = client().post("/ocr", files={
        "document": ("a.png", io.BytesIO(png_bytes()), "image/png"),
    })
    assert r.status_code == 400
    assert "file" in r.json()["error"]


def test_an_extra_form_field_beside_the_file_is_refused():
    """max_fields=0: nothing but the one file part."""
    r = client().post(
        "/ocr",
        files={"file": ("a.png", io.BytesIO(png_bytes()), "image/png")},
        data={"extra": "value"},
    )
    assert r.status_code == 400


def test_a_body_that_is_not_multipart_is_refused_cleanly():
    r = client().post("/ocr", content=b"not multipart",
                      headers={"content-type": "text/plain"})
    assert r.status_code == 400
    assert "error" in r.json()


# --------------------------------------------------- no silent page truncation

def test_a_document_over_the_page_limit_is_refused_not_truncated(monkeypatch):
    """The defect: `n = min(len(pdf), max_pages)` returned a wrong answer.

    A 400-page PDF came back as a 200 with 50 pages and nothing to distinguish
    it from a genuinely 50-page document.
    """
    monkeypatch.setattr(limits, "MAX_PAGES", 2)
    r = client().post("/ocr", files={
        "file": ("many.pdf", io.BytesIO(multipage_pdf(pages=4)), "application/pdf"),
    })
    assert r.status_code == 413
    assert "413" not in r.json()["error"]
    assert "limit is 2" in r.json()["error"]


def test_a_document_at_the_page_limit_is_accepted_whole(monkeypatch):
    monkeypatch.setattr(limits, "MAX_PAGES", 3)
    r = client().post("/ocr", files={
        "file": ("three.pdf", io.BytesIO(multipage_pdf(pages=3)), "application/pdf"),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["source_pages"] == 3
    assert body["processed_pages"] == 3
    assert body["complete"] is True


def test_page_count_is_read_without_rendering():
    assert pdf_mod.source_page_count(multipage_pdf(pages=4), "x.pdf") == 4


# ------------------------------------------------------ multipage raster images

def test_every_frame_of_a_multipage_tiff_is_processed():
    """`Image.open(...).convert("RGB")` silently returned frame zero only.

    A multipage TIFF is ordinary scanner output, so this was a wrong answer on
    a common input, not an exotic one.
    """
    r = client().post("/ocr", files={
        "file": ("scan.tiff", io.BytesIO(multiframe_tiff(frames=3)), "image/tiff"),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["source_pages"] == 3
    assert body["processed_pages"] == 3
    assert body["complete"] is True


def test_a_multiframe_tiff_over_the_page_limit_is_refused(monkeypatch):
    monkeypatch.setattr(limits, "MAX_PAGES", 2)
    r = client().post("/ocr", files={
        "file": ("scan.tiff", io.BytesIO(multiframe_tiff(frames=5)), "image/tiff"),
    })
    assert r.status_code == 413


def test_a_single_frame_image_still_reports_one_page():
    r = client().post("/ocr", files={
        "file": ("a.png", io.BytesIO(png_bytes()), "image/png"),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["source_pages"] == 1
    assert body["processed_pages"] == 1
    assert body["complete"] is True


# ------------------------------------------------------------ aggregate budgets

def test_the_aggregate_pixel_budget_is_charged_across_pages():
    budget = limits.PixelBudget(total=1000)
    budget.charge(20, 20)
    budget.charge(20, 20)
    with pytest.raises(limits.BudgetExceeded):
        budget.charge(20, 20)


def test_a_single_page_still_cannot_exceed_the_per_page_ceiling():
    budget = limits.PixelBudget()
    with pytest.raises(limits.UploadTooLarge):
        budget.charge(limits.MAX_IMAGE_PIXELS, 2)


def test_the_aggregate_budget_stops_a_document_partway_with_an_error(monkeypatch):
    """Within the page count but not within the decode budget."""
    monkeypatch.setattr(limits, "MAX_TOTAL_PIXELS", 100)
    r = client().post("/ocr", files={
        "file": ("scan.tiff", io.BytesIO(multiframe_tiff(frames=3)), "image/tiff"),
    })
    assert r.status_code == 413


# ------------------------------------------------------------ response contract

def test_the_response_states_completeness_explicitly():
    r = client().post("/ocr", files={
        "file": ("a.png", io.BytesIO(png_bytes()), "image/png"),
    })
    body = r.json()
    for field in ("schema_version", "source_pages", "processed_pages", "complete"):
        assert field in body, f"{field} missing from the response"


def test_existing_fields_are_unchanged():
    """Compatibility: the page shape callers already store must not move."""
    r = client().post("/ocr", files={
        "file": ("a.png", io.BytesIO(png_bytes()), "image/png"),
    })
    page = r.json()["pages"][0]
    assert "page" in page and "lines" in page and "full_text" in page


# ------------------------------------------------------------------- admission

def test_a_second_concurrent_document_is_told_to_retry():
    """No queue: an unbounded queue is just a slower way to exhaust the worker."""
    from app import server

    server.admission._lock = asyncio.Lock()

    async def scenario():
        await server.admission._lock.acquire()
        try:
            assert server.admission.busy
        finally:
            server.admission._lock.release()
        assert not server.admission.busy

    asyncio.run(scenario())


def test_the_busy_response_tells_the_caller_when_to_come_back():
    from app import server

    response = server._busy()
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"


def test_the_upload_deadline_stops_applying_once_the_body_has_arrived():
    """R-007: it bounds how long a client may take to SEND, nothing more.

    Starlette's request.is_disconnected() calls receive() again during
    processing. With the timer still armed, a document that took longer than
    the 120-second UPLOAD timeout returned 408, well inside the separate
    300-second processing deadline.
    """
    inner_calls = []

    class LateReceiver:
        """Delivers the body, then a slow disconnect poll much later."""

        def __init__(self):
            self.stage = 0

        async def __call__(self, scope, receive, send):
            # Consume the body.
            while True:
                message = await receive()
                inner_calls.append(message["type"])
                if not message.get("more_body"):
                    break
            # Now poll again, as is_disconnected() does during processing.
            message = await receive()
            inner_calls.append(message["type"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    inner = LateReceiver()
    # A zero-second upload deadline: if it were still armed after the body,
    # the second receive would raise BodyReadTimeout and return 408.
    mw = ingest.BoundedBodyMiddleware(inner, max_bytes=10_000,
                                      timeout_seconds=0.05, paths=("/ocr",))
    scope = {"type": "http", "path": "/ocr", "method": "POST", "headers": []}
    queue = [{"type": "http.request", "body": b"abc", "more_body": False}]
    sent = []

    async def receive():
        if queue:
            return queue.pop(0)
        await asyncio.sleep(0.2)  # longer than the upload deadline
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(mw(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    assert status == 200, "the upload deadline fired after the body had arrived"


def test_the_upload_deadline_still_applies_while_the_body_is_arriving():
    """The other half: a slow trickle must still be cut off."""
    inner = _Collector()
    mw = ingest.BoundedBodyMiddleware(inner, max_bytes=10_000,
                                      timeout_seconds=0.05, paths=("/ocr",))
    scope = {"type": "http", "path": "/ocr", "method": "POST", "headers": []}
    sent = []

    async def receive():
        await asyncio.sleep(0.5)  # never finishes in time
        return {"type": "http.request", "body": b"x", "more_body": True}

    async def send(message):
        sent.append(message)

    asyncio.run(mw(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    assert status == 408


# ------------------------------------------------- deterministic spool cleanup

def test_the_parser_still_exposes_the_attribute_the_cleanup_depends_on():
    """app/ingest.py closes `parser._files_to_close_on_error` directly.

    That is a private attribute of an installed library. If an upgrade renames
    it, `getattr(..., ())` would quietly return an empty tuple and the cleanup
    would become a no-op that still looks like cleanup. Fail here instead.
    """
    from starlette.formparsers import MultiPartParser

    parser = MultiPartParser.__new__(MultiPartParser)
    MultiPartParser.__init__(
        parser,
        headers=type("H", (), {"get": lambda self, k, d=None: "multipart/form-data; boundary=x"})(),
        stream=None,
    )
    assert hasattr(parser, "_files_to_close_on_error"), (
        "Starlette renamed the spool list; app/ingest.py cleanup is now a no-op"
    )
    assert isinstance(parser._files_to_close_on_error, list)


def test_every_spool_is_closed_when_parsing_fails():
    """The defect: a failed parse left the spool to garbage collection.

    Starlette closes its spools for MultiPartException and OSError only, and
    request.form() discards the parser on failure so nothing else can reach
    them. ingest.py owns the parser now, so this is checkable.
    """
    import app.ingest as ingest_mod

    closed = []

    class RecordingSpool:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
            closed.append(self)

    class ExplodingParser:
        def __init__(self, *args, **kwargs):
            self._files_to_close_on_error = [RecordingSpool(), RecordingSpool()]

        async def parse(self):
            raise RuntimeError("parser blew up mid-body")

    original = ingest_mod.MultiPartParser
    ingest_mod.MultiPartParser = ExplodingParser
    try:
        class FakeRequest:
            headers = {"content-type": "multipart/form-data; boundary=x"}

            def stream(self):
                return None

        with pytest.raises(ingest.MalformedUpload):
            asyncio.run(ingest_mod.read_single_upload(FakeRequest()))
    finally:
        ingest_mod.MultiPartParser = original

    assert len(closed) == 2, "spools were left to garbage collection"
    assert all(s.closed for s in closed)


def test_spools_are_closed_when_the_body_cap_trips_mid_parse():
    """The path that used to re-raise before anything could be closed."""
    import app.ingest as ingest_mod

    closed = []

    class CappedParser:
        def __init__(self, *args, **kwargs):
            spool = type("S", (), {"close": lambda self: closed.append(self)})()
            self._files_to_close_on_error = [spool]

        async def parse(self):
            raise ingest_mod.BodyTooLarge("over the cap")

    original = ingest_mod.MultiPartParser
    ingest_mod.MultiPartParser = CappedParser
    try:
        class FakeRequest:
            headers = {"content-type": "multipart/form-data; boundary=x"}

            def stream(self):
                return None

        with pytest.raises(ingest.BodyTooLarge):
            asyncio.run(ingest_mod.read_single_upload(FakeRequest()))
    finally:
        ingest_mod.MultiPartParser = original

    assert len(closed) == 1, "the body-cap path skipped spool cleanup"


def test_spools_are_closed_when_the_request_is_cancelled():
    """Cancellation is not an Exception subclass; it used to bypass cleanup."""
    import app.ingest as ingest_mod

    closed = []

    class CancellingParser:
        def __init__(self, *args, **kwargs):
            spool = type("S", (), {"close": lambda self: closed.append(self)})()
            self._files_to_close_on_error = [spool]

        async def parse(self):
            raise asyncio.CancelledError()

    original = ingest_mod.MultiPartParser
    ingest_mod.MultiPartParser = CancellingParser
    try:
        class FakeRequest:
            headers = {"content-type": "multipart/form-data; boundary=x"}

            def stream(self):
                return None

        with pytest.raises(BaseException):
            asyncio.run(ingest_mod.read_single_upload(FakeRequest()))
    finally:
        ingest_mod.MultiPartParser = original

    assert len(closed) == 1, "a cancelled request left its spool open"
