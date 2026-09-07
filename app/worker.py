"""Document worker: the child process that touches untrusted document bytes.

Why this exists
---------------
Everything native in this service runs on attacker-supplied input: PDFium
parses the PDF, Pillow and OpenCV decode the images, Paddle runs the models,
and VTracer traces the figures. All of that used to happen inside the single
uvicorn process that also terminates HTTP. A memory-safety bug in any of those
libraries, reached through a crafted document, would land in the process
holding every other request, and a runaway allocation or an infinite loop in
one of them would take the whole service down with no way to intervene.

The boundary is the raw bytes. The parent hands the child a document and a
couple of options and gets JSON back. The child does the decoding, the OCR, the
structure work and the vectorization, then exits. Nothing is reused between
documents.

What this is, and what it is not
--------------------------------
This is RESOURCE and FAULT containment, not a sandbox:

  * Address space, CPU time and core dumps are capped with POSIX rlimits, set
    HERE, before any native library is imported, so the limits are in force
    while the model initializes rather than after.
  * A crash, an OOM kill or a hang costs one document. The parent kills and
    reaps the child and keeps serving.
  * A fresh process per document means no state, no warmed model and no heap
    carries from one caller's document to the next.

It is NOT a privilege or filesystem or network boundary. The child runs as the
same user with the same filesystem and the same network access as the parent.
Running the service as a dedicated unprivileged account, with the models
pre-provisioned read-only and network egress restricted, is a DEPLOYMENT
requirement and is documented as one. Nothing in this file establishes it.

On Windows there are no POSIX rlimits. The process boundary and the wall
deadline still apply; the memory and CPU caps do not. That platform must not be
described as equivalently contained.

Protocol (stdin -> stdout, both length-prefixed, big-endian uint32):
    in :  [4] header length, header JSON, then the raw document bytes
    out:  [4] payload length, payload JSON

Run as:  python -m app.worker
"""

from __future__ import annotations

import json
import os
import struct
import sys

# Defaults chosen from measurement, not from a round number. A recorded core
# plus structured pass in this environment reported VmPeak of 12,400,792 kB,
# so an 8 GiB address-space cap would refuse a legitimate document before it
# started. 16 GiB sits above the measured startup requirement with room, and
# still bounds a runaway allocation.
DEFAULT_ADDRESS_SPACE_BYTES = 16 * 1024 * 1024 * 1024
# A measured pass completed in 4.15 seconds of wall time. 90 seconds of CPU is
# generous for a long document and short enough to stop a spin.
DEFAULT_CPU_SECONDS = 90


def apply_resource_limits(
    address_space: int = DEFAULT_ADDRESS_SPACE_BYTES,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
) -> dict:
    """Cap this process before any native library loads. POSIX only."""
    applied = {"address_space": None, "cpu_seconds": None, "core": None,
               "platform": os.name}
    try:
        import resource
    except ImportError:  # Windows
        return applied

    def _set(which, value, key, allow_zero=False):
        # A non-positive request is not a cap. RLIM_INFINITY is -1, so passing
        # a negative value here would remove the limit and then report a number,
        # which reads like a cap was applied. Report None instead: a limit that
        # is not in force must never look like one that is.
        if value < 0 or (value == 0 and not allow_zero):
            applied[key] = None
            return
        try:
            soft, hard = resource.getrlimit(which)
            # Never RAISE a limit the environment already set lower, whether it
            # set it as a soft or a hard limit. Considering only the hard limit
            # meant an inherited 8 GiB soft cap with an unlimited hard cap was
            # quietly widened to 16 GiB.
            target = value
            for existing in (soft, hard):
                if existing != resource.RLIM_INFINITY:
                    target = min(target, existing)
            resource.setrlimit(which, (target, hard))
            applied[key] = target
        except (ValueError, OSError):
            applied[key] = None

    _set(resource.RLIMIT_AS, address_space, "address_space")
    _set(resource.RLIMIT_CPU, cpu_seconds, "cpu_seconds")
    # A core dump of a 12 GB process, written on a crash triggered by a
    # document, is both a denial of service and a copy of that document on
    # disk. Refuse to write one.
    _set(resource.RLIMIT_CORE, 0, "core", allow_zero=True)
    return applied


# A missing or unreachable FILE is never the uploaded document: the document
# arrived in memory over a pipe. These are always our installation's problem,
# and several of them subclass OSError, so they are excluded by instance check
# rather than by removing names from the tuple below.
INFRASTRUCTURE_ERRORS = (
    FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError,
    ImportError, ModuleNotFoundError, InterruptedError, BlockingIOError,
    ChildProcessError,
)

# An OSError is the operating system reporting that a syscall failed. The
# uploaded document never touches a syscall: it arrives in memory over a pipe.
# So an OSError that carries an errno is the machine, not the document, and the
# default is inverted rather than enumerated. Three rounds of review were spent
# adding errnos one at a time (ENOSPC, then EBADF and ELOOP, then ENXIO and
# ECONNREFUSED) and the list was still incomplete each time, which is the sign
# that the rule was the wrong shape.
#
# The one carve-out is the errno-less OSError, which is how Pillow reports a
# truncated or malformed image. That IS the document's problem.
def _is_infrastructure_failure(exc) -> bool:
    """True when the machine failed, not when the document was unreadable."""
    if isinstance(exc, INFRASTRUCTURE_ERRORS):
        return True
    if isinstance(exc, OSError):
        return exc.errno is not None
    return False


def _unreadable_types() -> tuple:
    """Exception types that mean "this document cannot be read", not a bug.

    Resolved at call time from the installed libraries, so a rename upstream
    shows up as a 500 that someone investigates rather than being silently
    absorbed by a broad `except RuntimeError`.
    """
    types = [ValueError, OSError, TypeError, AttributeError, IndexError,
             KeyError, StopIteration, ArithmeticError]
    try:
        from pypdfium2 import PdfiumError

        types.append(PdfiumError)
    except Exception:
        pass
    try:
        from PIL import Image, UnidentifiedImageError

        types.append(UnidentifiedImageError)
        types.append(Image.DecompressionBombError)
    except Exception:
        pass
    return tuple(types)


def read_exact(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("input ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_request(stream):
    (header_len,) = struct.unpack(">I", read_exact(stream, 4))
    header = json.loads(read_exact(stream, header_len).decode("utf-8"))
    body = read_exact(stream, int(header["body_bytes"]))
    return header, body


def write_response(stream, payload: dict, max_bytes: int) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = json.dumps(
            {
                "error": "response too large",
                "detail": f"{len(encoded)} bytes exceeds the {max_bytes} byte limit",
                "kind": "response_too_large",
            }
        ).encode("utf-8")
    stream.write(struct.pack(">I", len(encoded)))
    stream.write(encoded)
    stream.flush()


class InitializationFailed(Exception):
    """The service could not get ready. Never the document's fault."""


def process(header: dict, body: bytes) -> dict:
    """Do the whole document, in two phases that fail differently.

    The phases matter for classification. Everything before the document is
    touched is OUR installation: imports, configured limits, model loading. A
    failure there is an infrastructure failure, and calling it "could not read
    input" blames the caller for a missing model file, returns 400, and skips
    the cooldown so the next request tries again immediately.

    Only failures raised while decoding or recognizing the document are
    document failures.
    """
    # ---------------------------------------------------------- init phase
    try:
        # Imported late and deliberately: apply_resource_limits() must be in
        # force before Paddle, PDFium, Pillow or OpenCV allocate anything.
        from app import limits, pdf as pdf_mod
        from app.pipeline import process_page
        from engine import ocr_engine, structure

        # Adopt the parent's limits. Without this the child would silently use
        # its own module defaults, so a deployment that lowered a limit would
        # find it applied to the preflight in the parent and ignored by the
        # process that actually decodes the document.
        for key, attribute in (
            ("max_pages", "MAX_PAGES"),
            ("max_total_pixels", "MAX_TOTAL_PIXELS"),
            ("max_image_pixels", "MAX_IMAGE_PIXELS"),
            ("max_figures_per_document", "MAX_FIGURES_PER_DOCUMENT"),
            ("max_figures_per_page", "MAX_FIGURES_PER_PAGE"),
        ):
            if header.get(key) is not None:
                setattr(limits, attribute, int(header[key]))

        limits.install_pillow_guards()

        structured = bool(header.get("structured"))
        # Load the models BEFORE the document. A model that cannot be loaded is
        # an installation problem, and loading it here means it is reported as
        # one instead of surfacing later as an unreadable page.
        ocr_engine._get_ocr()
        if structured:
            structure._get_structure()
    except Exception as exc:
        raise InitializationFailed(f"{type(exc).__name__}") from exc

    # ------------------------------------------------------ document phase
    filename = header.get("filename")
    source_pages = pdf_mod.source_page_count(body, filename)
    limits.check_page_count(source_pages)

    budget = limits.PixelBudget()
    figures_left = limits.MAX_FIGURES_PER_DOCUMENT
    pages = []
    for index, image in enumerate(pdf_mod.iter_document_pages(body, filename, budget)):
        page = process_page(image, index, structured, figures_left)
        figures_left -= len(page.get("figures") or ())
        pages.append(page)

    return {
        "source_pages": source_pages,
        "processed_pages": len(pages),
        "complete": len(pages) == source_pages,
        "pages": pages,
    }


def main() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        header, body = read_request(stdin)
    except (EOFError, ValueError, KeyError, struct.error) as exc:
        write_response(stdout, {"error": "bad request to worker",
                                "detail": str(exc), "kind": "protocol"},
                       1 << 20)
        return 2

    applied = apply_resource_limits(
        address_space=int(header.get("address_space") or DEFAULT_ADDRESS_SPACE_BYTES),
        cpu_seconds=int(header.get("cpu_seconds") or DEFAULT_CPU_SECONDS),
    )
    max_response = int(header.get("max_response_bytes") or (32 * 1024 * 1024))

    # Fail closed. Reporting a limit as absent and then processing the document
    # anyway gives up the containment while still looking contained. On a POSIX
    # host every cap must be in force before any document is touched.
    if os.name == "posix":
        missing = [k for k in ("address_space", "cpu_seconds", "core")
                   if applied.get(k) is None]
        if missing:
            write_response(stdout, {
                "error": "resource limits unavailable; refusing to process",
                "kind": "limits_unavailable",
                "missing": missing,
            }, max_response)
            return 4

    try:
        result = process(header, body)
        result["limits_applied"] = applied
        result["worker_pid"] = os.getpid()
        write_response(stdout, result, max_response)
        return 0
    except MemoryError:
        # The address-space cap did its job.
        write_response(stdout, {"error": "document exceeded the memory limit",
                                "kind": "resource"}, max_response)
        return 3
    except BaseException as exc:
        # ONE handler. Two sibling `except Exception` blocks do not chain: a
        # bare `raise` in the first propagates out of the whole try rather than
        # falling through to the second, so the sanitized path below was
        # unreachable and every ordinary failure escaped as a traceback.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, InitializationFailed):
            # Checked BEFORE importing anything. The previous order imported
            # app.limits first, so an import failure, which is precisely what
            # this branch exists to report, broke its own handler and escaped
            # with a traceback and no response frame.
            print(f"worker: initialization failed: {exc}", file=sys.stderr, flush=True)
            write_response(stdout, {
                "error": "the service could not initialize",
                "kind": "initialization_failed",
            }, max_response)
            return 5
        try:
            from app import limits as _limits
        except Exception:
            write_response(stdout, {
                "error": "the service could not initialize",
                "kind": "initialization_failed",
            }, max_response)
            return 5

        if isinstance(exc, (_limits.UploadTooLarge, _limits.BudgetExceeded,
                            _limits.TooManyPages)):
            write_response(stdout, {"error": str(exc), "kind": "resource"},
                           max_response)
            return 3
        if isinstance(exc, _limits.ProcessingTimeout):
            write_response(stdout, {"error": str(exc), "kind": "timeout"},
                           max_response)
            return 3
        # Anything the decoders raise on a document they cannot read. The
        # library exception types are resolved rather than assumed: pypdfium2
        # raises PdfiumError, which subclasses RuntimeError and so matched
        # none of the general types below, sending every malformed PDF to a
        # 500 instead of a 400.
        # Checked on the INSTANCE, not by removing entries from the tuple:
        # NotADirectoryError and friends subclass OSError, so dropping the
        # exact names left them matching through the superclass anyway.
        unreadable = (isinstance(exc, _unreadable_types())
                      and not _is_infrastructure_failure(exc))
        # Only the exception CLASS crosses the boundary or reaches a log. A
        # decoder's message routinely quotes the bytes that upset it, and the
        # service log is a file on persistent storage.
        if os.environ.get("XLITE_DEBUG_WORKER_ERRORS") == "1":
            print(f"worker: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        else:
            print(f"worker: {type(exc).__name__}", file=sys.stderr, flush=True)
        write_response(stdout, {
            "error": "could not read input" if unreadable else "processing failed",
            "kind": "unreadable" if unreadable else type(exc).__name__,
        }, max_response)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
