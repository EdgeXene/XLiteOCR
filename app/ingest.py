"""Bounded request ingest for /ocr.

What was wrong
--------------
The route read the whole body first and checked its size afterwards::

    data = await file.read()          # entire body, however large
    limits.check_upload_size(data)    # 413, after the fact

By the time the check ran, Starlette's multipart parser had already consumed
the body and spooled every part over 1 MiB to a temporary file. The 30 MB cap
described a response code, not a resource bound.

Two further gaps in the installed stack, both measured rather than assumed
(starlette 1.3.1, fastapi 0.139.0):

  * FastAPI calls ``await request.form()`` with no arguments, so the parser
    runs with its defaults of ``max_files=1000`` and ``max_fields=1000``. A
    single request could open a thousand spool files. (An earlier note in this
    workspace recorded the installed defaults as ``max_files=1, max_fields=0``;
    that is not what the installed code does.)
  * ``MultiPartParser.parse`` closes its spools on ``MultiPartException`` and
    ``OSError`` only. A client disconnect, a cancellation or a timeout leaves
    them open, and ``max_part_size`` bounds field parts, not file bytes. This
    module therefore owns the parser instead of going through
    ``request.form()``, which discards it on failure, and closes every tracked
    spool itself.

What this module does
---------------------
1. ``BoundedBodyMiddleware`` counts bytes as they arrive from the ASGI server,
   before the parser sees them. It refuses a declared ``Content-Length`` over
   the cap without reading anything, and it also refuses a body that exceeds
   the cap while streaming, which is the case a lying or absent Content-Length
   (chunked transfer) creates. It carries a wall deadline for the same reason:
   a slow trickle holds the worker as effectively as a large body.

2. ``read_single_upload`` parses the form with ``max_files=1, max_fields=0``,
   requires exactly one part named ``file``, and closes every spool on every
   exit path including cancellation.

The cap allows a documented margin over the file cap for multipart framing, so
a 30 MB file inside a multipart envelope is not rejected for its own headers.
Nothing here trusts a client-supplied length for anything except an early
rejection that saves work.
"""

from __future__ import annotations

import asyncio
import logging

from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartParser
from starlette.requests import Request

from app import limits

log = logging.getLogger("xlite-ocr.ingest")

# Multipart framing around the file: boundary lines, part headers, trailer.
# Generous, and bounded, so the envelope cannot become the payload.
MULTIPART_OVERHEAD_BYTES = 64 * 1024

# Total bytes accepted for one request body.
MAX_BODY_BYTES = limits.MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES

# Wall clock for receiving the body. Generous for a 30 MB upload on a slow
# link, short enough that a trickle cannot pin the single worker.
BODY_READ_TIMEOUT_SECONDS = 120.0


class BodyTooLarge(Exception):
    """The request body exceeded MAX_BODY_BYTES."""


class BodyReadTimeout(Exception):
    """The request body did not arrive within BODY_READ_TIMEOUT_SECONDS."""


class MalformedUpload(Exception):
    """The multipart body was not exactly one file part named `file`."""


def _plain_response(status: int, payload: bytes, extra_headers=()):
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    headers.extend(extra_headers)
    return (
        {"type": "http.response.start", "status": status, "headers": headers},
        {"type": "http.response.body", "body": payload},
    )


class BoundedBodyMiddleware:
    """Pure ASGI middleware: cap body bytes and wall time before parsing.

    Deliberately not a BaseHTTPMiddleware subclass. That class materializes the
    request body to hand it to the next app, which is the behavior being fixed.
    """

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES,
                 timeout_seconds: float = BODY_READ_TIMEOUT_SECONDS,
                 paths=("/ocr",)):
        self.app = app
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self.paths = frozenset(paths)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") not in self.paths:
            return await self.app(scope, receive, send)

        # Early refusal: a declared length over the cap is rejected without
        # reading a byte. A client that lies here is caught by the counter.
        for name, value in scope.get("headers") or []:
            if name.lower() != b"content-length":
                continue
            try:
                declared = int(value)
            except (TypeError, ValueError):
                break
            if declared > self.max_bytes:
                return await self._refuse(send, 413, "upload exceeds size limit")
            break

        received = 0
        started = asyncio.get_running_loop().time()
        state = {"response_started": False, "body_complete": False}

        async def counting_receive():
            nonlocal received
            # Once the body has arrived, this deadline is spent. It bounds how
            # long a client may take to SEND, not how long the request may
            # take. Starlette's request.is_disconnected() calls receive() again
            # during processing, so leaving the timer armed turned a long but
            # legitimate document into a 408 at the upload deadline, well
            # inside the separate processing deadline.
            if state["body_complete"]:
                return await receive()

            remaining = self.timeout_seconds - (
                asyncio.get_running_loop().time() - started
            )
            if remaining <= 0:
                raise BodyReadTimeout("body read deadline exceeded")
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise BodyReadTimeout("body read deadline exceeded") from exc
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > self.max_bytes:
                    raise BodyTooLarge(
                        f"body exceeds {self.max_bytes} bytes"
                    )
                if not message.get("more_body"):
                    state["body_complete"] = True
            elif message.get("type") == "http.disconnect":
                state["body_complete"] = True
            return message

        async def tracking_send(message):
            if message.get("type") == "http.response.start":
                state["response_started"] = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except BodyTooLarge:
            log.info("ingest: body over cap after %d bytes", received)
            if state["response_started"]:
                raise
            await self._refuse(send, 413, "upload exceeds size limit")
        except BodyReadTimeout:
            log.info("ingest: body read timed out after %d bytes", received)
            if state["response_started"]:
                raise
            await self._refuse(send, 408, "upload timed out")

    async def _refuse(self, send, status, message):
        import json

        payload = json.dumps({"error": message}).encode()
        start, body = _plain_response(status, payload)
        await send(start)
        await send(body)


async def read_single_upload(request: Request):
    """Return (filename, data) for exactly one part named `file`.

    The parser is constructed here rather than through ``request.form()`` for
    one reason: ownership of the spool files.

    ``request.form()`` builds the parser internally and caches the result only
    on SUCCESS, so on any failure there is no object to close and no way to
    reach the temporary files the parser had already created. Starlette closes
    them itself for ``MultiPartException`` and ``OSError`` and no other case,
    which leaves a cancellation, a disconnect, a timeout and the body-cap
    refusal relying on garbage collection to release a file holding part of
    someone's document.

    Holding the parser makes cleanup deterministic: ``_files_to_close_on_error``
    is the list it tracks every spool in, and the ``finally`` below closes all
    of them on every exit path. That attribute is private, so
    ``tests/test_ingest_limits.py`` asserts it still exists on the installed
    Starlette; an upgrade that renames it fails the suite instead of quietly
    turning the cleanup into a no-op.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        raise MalformedUpload("expected a multipart/form-data body")

    parser = MultiPartParser(
        request.headers,
        request.stream(),
        max_files=1,
        max_fields=0,
        max_part_size=limits.MAX_UPLOAD_BYTES,
    )
    form = None
    try:
        try:
            form = await parser.parse()
        except (BodyTooLarge, BodyReadTimeout):
            raise
        except Exception as exc:  # MultiPartException and friends
            raise MalformedUpload("malformed multipart body") from exc

        keys = list(form.keys())
        if keys != ["file"]:
            raise MalformedUpload(
                "expected exactly one part named 'file', got " f"{keys or 'nothing'}"
            )
        upload = form["file"]
        if not isinstance(upload, UploadFile):
            raise MalformedUpload("part 'file' is not a file upload")

        return upload.filename, await upload.read()
    finally:
        # Every spool the parser created, on every exit path: success,
        # refusal, malformed input, disconnect, timeout and cancellation.
        for spool in getattr(parser, "_files_to_close_on_error", ()):
            try:
                spool.close()
            except Exception:
                log.warning("ingest: closing an upload spool failed")
        if form is not None:
            try:
                await form.close()
            except Exception:
                log.warning("ingest: closing the form failed")


class SingleDocumentAdmission:
    """One document in flight at a time, with no queue.

    The engine holds a warmed model and one process. Letting a second document
    in does not make it faster, it makes both slower and doubles peak memory.
    A caller that arrives while one is running is told to retry rather than
    being parked on a queue that has no bound.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def try_acquire(self):
        if self._lock.locked():
            return None
        return self._lock

    async def __aenter__(self):
        if self._lock.locked():
            raise Busy("another document is being processed")
        await self._lock.acquire()
        return self

    async def __aexit__(self, *exc):
        self._lock.release()
        return False


class Busy(Exception):
    """A document is already being processed."""
