"""Parent side of the document worker: spawn, feed, deadline, kill, reap.

The lifecycle rules this file keeps, and why each one is here:

  * SPAWN, never fork. `asyncio.create_subprocess_exec` execs a fresh
    interpreter. Forking the API process would hand the child a copy of a
    warmed Paddle heap, which is exactly the state that must not carry across
    documents, and forking a process with live threads and native locks is a
    reliable way to produce a child that deadlocks instead of working.

  * One document at a time. Enforced by the admission gate in server.py; this
    module holds no queue of its own. An unbounded queue is a slower way to run
    out of memory.

  * A wall deadline that is actually enforced. This is the part an in-process
    design cannot do: a synchronous native call inside the server cannot be
    cancelled, so a deadline there is a promise the code cannot keep. Here the
    deadline is a kill.

  * Kill AND reap, on every path: timeout, client disconnect, shutdown, and
    normal completion. A killed child that is never waited on becomes a zombie,
    and enough of those exhaust the process table.

  * No respawn loop. A worker that failed is not retried inside one request,
    and a cooldown after a failure stops a crash-looping input from becoming a
    fork bomb driven by a client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path

from app import limits

log = logging.getLogger("xlite-ocr.worker")

REPO_ROOT = Path(__file__).resolve().parent.parent

# After a worker fails to start or dies abnormally, refuse new work briefly.
# Without this, an input that reliably kills the child lets a client spawn
# processes as fast as it can send requests.
FAILURE_COOLDOWN_SECONDS = 5.0

# How long to wait for a killed child to actually die before giving up on a
# clean reap.
KILL_GRACE_SECONDS = 5.0


class WorkerError(Exception):
    """The worker could not produce a result."""


class WorkerTimeout(WorkerError):
    """The worker exceeded its wall deadline and was killed."""


class WorkerUnavailable(WorkerError):
    """The worker could not be started, or is in a failure cooldown."""



# The response is attacker-influenced if the child is ever compromised, and is
# malformed if the child is merely broken. JSON decoding alone accepted `null`,
# `[]` and `{}`; the last became a successful response with no page counts.
_REQUIRED_RESULT_FIELDS = ("pages", "source_pages", "processed_pages", "complete")


def _validated(payload):
    """Return the payload, or raise WorkerError if it is not the contract."""
    if not isinstance(payload, dict):
        raise WorkerError(
            f"worker returned {type(payload).__name__}, expected an object"
        )
    if "error" in payload:
        if not isinstance(payload.get("kind"), (str, type(None))):
            raise WorkerError("worker returned a malformed error frame")
        if not isinstance(payload.get("error"), str):
            raise WorkerError("worker returned a malformed error frame")
        return payload
    missing = [f for f in _REQUIRED_RESULT_FIELDS if f not in payload]
    if missing:
        raise WorkerError(f"worker response is missing {', '.join(missing)}")

    pages = payload["pages"]
    if not isinstance(pages, list):
        raise WorkerError("worker returned a non-list `pages`")
    if not all(isinstance(page, dict) for page in pages):
        raise WorkerError("worker returned a page that is not an object")

    for field in ("source_pages", "processed_pages"):
        value = payload[field]
        # bool is a subclass of int, so a JSON true passed an isinstance(int)
        # check and became a page count of 1.
        if not isinstance(value, int) or isinstance(value, bool):
            raise WorkerError(f"worker returned a non-integer `{field}`")
        if value < 0:
            raise WorkerError(f"worker returned a negative `{field}`")

    if not isinstance(payload["complete"], bool):
        raise WorkerError("worker returned a non-boolean `complete`")

    # The fields have to agree with each other. A broken or compromised child
    # could otherwise report two processed pages, an empty page list and
    # complete=true, and the service would pass it on as a successful answer.
    if payload["processed_pages"] != len(pages):
        raise WorkerError(
            "worker returned processed_pages that does not match `pages`"
        )
    if payload["processed_pages"] > payload["source_pages"]:
        raise WorkerError("worker processed more pages than the document has")
    if payload["complete"] != (payload["processed_pages"] == payload["source_pages"]):
        raise WorkerError("worker returned a `complete` that contradicts its counts")
    return payload


class DocumentWorker:
    """Runs one document per child process."""

    def __init__(self, python_executable: str | None = None,
                 cooldown_seconds: float = FAILURE_COOLDOWN_SECONDS):
        self.python = python_executable or sys.executable
        self.cooldown_seconds = cooldown_seconds
        self._cooldown_until = 0.0
        # Children we could not confirm dead. While any of these are alive the
        # worker refuses new documents: the "one document at a time" bound is
        # about resources, and a surviving child still holds its memory. A
        # timed cooldown alone would expire and admit a second child beside it.
        self._unresolved: list = []

    async def _reap_unresolved(self) -> int:
        """Try again on children a previous request could not confirm dead."""
        still_alive = []
        for process in self._unresolved:
            if process.returncode is not None:
                # Exited, but its pipes may still be open: dropping it from the
                # list without closing them leaks the descriptors that made it
                # unresolvable in the first place.
                self._close_pipes(process)
                continue
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except Exception:
                pass
            # Release the pipes BEFORE waiting, for the same reason _terminate
            # does: process.wait() also waits on the pipe transports, so a
            # child whose stdout is paused under flow control never completes
            # and recovery decides, wrongly, that it is unkillable.
            self._close_pipes(process)
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except Exception:
                still_alive.append(process)
            # A cancellation can be delivered while that wait completes
            # successfully. Recovery must not swallow it and carry on.
            self._unresolved = still_alive + self._unresolved[
                self._unresolved.index(process) + 1:]
            self._raise_if_cancelling()
        self._unresolved = still_alive
        return len(still_alive)

    @property
    def in_cooldown(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def _begin_cooldown(self) -> None:
        self._cooldown_until = time.monotonic() + self.cooldown_seconds

    def _environment(self) -> dict:
        """A rebuilt environment, not an inherited one.

        The child needs an interpreter, an import path and the model cache. It
        does not need whatever else happens to be exported into the service.
        """
        keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR",
                "XLITE_FORMULA", "HF_HOME", "PADDLE_HOME", "PADDLEOCR_HOME",
                "OMP_NUM_THREADS", "FLAGS_use_mkldnn",
                # Windows resolves side-by-side assemblies through these. A
                # rebuilt environment without them can fail the interpreter or
                # its native dependencies at startup.
                "SystemRoot", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP",
                "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
                # The worker's own opt-in for verbose failures.
                "XLITE_DEBUG_WORKER_ERRORS")
        env = {name: os.environ[name] for name in keep if name in os.environ}
        env["PYTHONPATH"] = str(REPO_ROOT)
        # PYTHONFAULTHANDLER is enabled by ANY non-empty value, so setting it
        # to "0" switched ON the fatal-signal tracebacks it looked like it was
        # switching off. Leave it unset: an unset variable is the off state.
        env.pop("PYTHONFAULTHANDLER", None)
        return env

    # Framed failures that mean the worker itself is broken, as opposed to a
    # document being refused. A refusal is a normal answer and must not put the
    # worker into cooldown; a startup failure repeated at request rate is how a
    # client turns one bad input into continuous process churn.
    WORKER_FAILURE_KINDS = frozenset({
        "limits_unavailable", "protocol", "response_too_large",
    })
    DOCUMENT_REFUSAL_KINDS = frozenset({"resource", "timeout", "unreadable"})

    async def run(self, data: bytes, filename: str | None, structured: bool,
                  deadline_seconds: float | None = None,
                  is_disconnected=None) -> dict:
        """Process one document in a fresh child. Always kills and reaps."""
        if self.in_cooldown:
            raise WorkerUnavailable(
                "the document worker is recovering from a failure, retry shortly"
            )
        if self._unresolved and await self._reap_unresolved():
            # Not a timed state: it clears when the child actually dies.
            raise WorkerUnavailable(
                "a previous document worker has not exited; refusing new work"
            )
        # Nothing new starts while a cancellation is pending on this task.
        self._raise_if_cancelling()

        deadline = deadline_seconds or limits.PROCESSING_DEADLINE_SECONDS
        # The child is a fresh interpreter, so it re-imports app.limits and
        # gets that module's defaults. The parent is where limits are
        # configured, so they travel with the document rather than being
        # duplicated in two processes that can drift apart.
        header = {
            "body_bytes": len(data),
            "filename": filename,
            "structured": bool(structured),
            "max_response_bytes": limits.MAX_RESPONSE_BYTES,
            "max_pages": limits.MAX_PAGES,
            "max_total_pixels": limits.MAX_TOTAL_PIXELS,
            "max_image_pixels": limits.MAX_IMAGE_PIXELS,
            "max_figures_per_document": limits.MAX_FIGURES_PER_DOCUMENT,
            "max_figures_per_page": limits.MAX_FIGURES_PER_PAGE,
        }
        encoded_header = json.dumps(header).encode("utf-8")

        try:
            process = await asyncio.create_subprocess_exec(
                self.python, "-m", "app.worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(REPO_ROOT),
                env=self._environment(),
                # A new session, so a signal sent to the server's process group
                # does not race with the explicit kill below.
                start_new_session=True,
            )
        except OSError as exc:
            self._begin_cooldown()
            # A cancellation racing a spawn failure outranks it; converting
            # unconditionally masked the shutdown signal permanently.
            self._raise_if_cancelling()
            raise WorkerUnavailable(f"could not start the worker: {exc}") from exc

        # Drain stderr continuously. An unread pipe fills, and then the child
        # blocks on write instead of finishing, holding admission until the
        # deadline kills it. The bytes are kept only in a small ring and are
        # never logged, because a decoder's message can quote the document.
        drain = asyncio.ensure_future(self._drain_stderr(process))
        outcome = {"reaped": False, "cancelled": False}
        try:
            result = await self._exchange(
                process, encoded_header, data, deadline, is_disconnected
            )
        finally:
            # Runs on every path INCLUDING cancellation, and does not return
            # until the child is gone. Releasing admission while a child still
            # holds memory and pipes is how "one document at a time" stops
            # being true.
            await self._await_cleanup(process, drain, outcome)
            # Raised from INSIDE the finally on purpose. A finally preserves an
            # exception that was already propagating; a check placed after it
            # is skipped entirely on that path, so a cancellation arriving
            # during cleanup was lost whenever the exchange had also failed.
            # Cancellation outranks a worker error: shutdown needs the signal.
            if outcome["cancelled"]:
                raise asyncio.CancelledError()

        if not outcome["reaped"]:
            # Do not hand back a successful payload from a request whose child
            # could not be confirmed dead. Logging it and returning 200 was the
            # previous behavior and it reported a leak as a success.
            raise WorkerError("the document worker could not be reaped")

        self._classify_framed_result(result)
        return result

    def _classify_framed_result(self, result) -> None:
        """A framed error is still an error; some of them mean cooldown."""
        if not isinstance(result, dict) or "error" not in result:
            return
        kind = result.get("kind")
        if not isinstance(kind, str):
            # `kind: []` used to raise TypeError inside the set membership
            # test, escaping as an uncaught error with no cooldown.
            self._begin_cooldown()
            return
        if kind in self.DOCUMENT_REFUSAL_KINDS:
            return  # a refused document is a normal answer
        # Everything else is the worker failing, including a refusal to start
        # because its resource limits could not be installed.
        self._begin_cooldown()

    async def _await_cleanup(self, process, drain, outcome) -> None:
        """Await termination to completion, even while being cancelled.

        `asyncio.shield` alone protects the inner task but still raises
        CancelledError in the awaiter, so the caller unwound and released
        admission while cleanup ran on independently. Looping over the shield
        keeps ownership: the cancellation is still delivered to the caller
        afterwards, because this runs inside a finally.
        """
        task = asyncio.ensure_future(self._terminate(process, drain))
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Remember it. The caller re-raises after cleanup finishes, so
                # a cancellation delivered during termination is not lost.
                outcome["cancelled"] = True
                continue
            except Exception:
                break
        try:
            outcome["reaped"] = bool(task.result())
        except BaseException:
            # BaseException, not Exception: a cancellation aimed at the
            # terminate task itself surfaces here, and treating it as "not our
            # problem" dropped a live child on the floor.
            outcome["reaped"] = False
        if not outcome["reaped"] and process is not None:
            # The terminate task may have been cancelled before it signaled
            # anything, so make one last direct attempt, finish the cleanup
            # obligations it skipped, and record the child as unresolved.
            try:
                if getattr(process, "returncode", None) is None:
                    process.kill()
            except Exception:
                pass
            if drain is not None:
                if not drain.done():
                    drain.cancel()
                try:
                    await asyncio.gather(drain, return_exceptions=True)
                except asyncio.CancelledError:
                    # Record it. Swallowing a cancellation first delivered in
                    # the fallback left run() with nothing to re-raise.
                    outcome["cancelled"] = True
                except BaseException:
                    pass
            self._close_pipes(process)
            self._unresolved.append(process)

    async def _drain_stderr(self, process, keep_bytes: int = 4096) -> bytes:
        buffered = b""
        try:
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    return buffered
                buffered = (buffered + chunk)[-keep_bytes:]
        except (asyncio.CancelledError, Exception):
            return buffered

    @staticmethod
    def _raise_if_cancelling():
        """Convert a pending cancellation into one that is actually raised.

        asyncio.wait_for can deliver an inner future's RESULT or its EXCEPTION
        while a cancellation is already requested on this task. Checking only
        the success path meant a worker error raced with a cancellation
        surfaced as the error, and shutdown never saw the signal.
        """
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            raise asyncio.CancelledError()

    async def _exchange(self, process, encoded_header, data, deadline,
                        is_disconnected):
        async def feed_and_read():
            process.stdin.write(struct.pack(">I", len(encoded_header)))
            process.stdin.write(encoded_header)
            process.stdin.write(data)
            await process.stdin.drain()
            process.stdin.close()

            raw_length = await process.stdout.readexactly(4)
            (length,) = struct.unpack(">I", raw_length)
            if length > limits.MAX_RESPONSE_BYTES:
                raise WorkerError("worker announced an oversized response")
            payload = await process.stdout.readexactly(length)
            return _validated(json.loads(payload.decode("utf-8")))

        task = asyncio.ensure_future(feed_and_read())
        started = time.monotonic()
        try:
            while True:
                remaining = deadline - (time.monotonic() - started)
                if remaining <= 0:
                    raise asyncio.TimeoutError
                # Wake up regularly so a client that has gone away stops the
                # work instead of paying for it to finish.
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(task), timeout=min(0.5, remaining)
                    )
                    # asyncio.wait_for returns the completed result when a
                    # cancellation lands after the inner future finished but
                    # before this task resumes, so a successful payload could
                    # be returned over a cancellation nobody ever saw.
                    self._raise_if_cancelling()
                    return result
                except asyncio.TimeoutError:
                    if is_disconnected is not None and await is_disconnected():
                        raise WorkerError("client disconnected")
                    continue
        except asyncio.TimeoutError as exc:
            self._begin_cooldown()
            self._raise_if_cancelling()
            raise WorkerTimeout(
                f"processing exceeded {int(deadline)}s and was stopped"
            ) from exc
        except asyncio.CancelledError:
            # Cancellation is not an application error. Turning it into a
            # WorkerError produced a 500 and hid the signal from shutdown
            # supervision, so it propagates unchanged; cleanup still runs in
            # the caller's finally.
            self._begin_cooldown()
            raise
        except WorkerError:
            # Includes the oversized-response and disconnect paths. Every
            # abnormal end starts the cooldown, so a client cannot relaunch a
            # worker in a tight loop by repeating whichever input causes it.
            self._begin_cooldown()
            self._raise_if_cancelling()
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # IncompleteRead, a reset pipe, malformed JSON or invalid UTF-8:
            # the child died or answered nonsense. A cancellation racing with
            # any of those outranks them.
            self._begin_cooldown()
            self._raise_if_cancelling()
            raise WorkerError("the document worker stopped unexpectedly") from exc
        finally:
            # The task is shielded, so a timeout or a cancellation above leaves
            # it running against a process we are about to kill.
            if not task.done():
                task.cancel()
            try:
                await asyncio.gather(task, return_exceptions=True)
            except asyncio.CancelledError:
                # A cancellation FIRST delivered at this await was being
                # discarded, and the successful result returned anyway.
                # Raising from the finally lets the cancellation win.
                raise
            except BaseException:
                pass

    async def _terminate(self, process, drain=None) -> bool:
        """Kill if running, release the pipes, then reap. True only if reaped.

        The ORDER matters and was wrong. `process.wait()` waits for the child
        AND for its pipe transports to finish. A child that filled stdout while
        the parent was not reading leaves that transport paused under flow
        control, so the wait never completed and a child that had already died
        was reported as unreapable. Releasing the pipes first is what makes the
        wait mean "the process is gone".
        """
        reaped = False
        try:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                except Exception:
                    # PermissionError and friends. Signalling failed, but the
                    # remaining cleanup obligations still have to run, so this
                    # falls through rather than abandoning them.
                    log.error("worker %s could not be signaled", process.pid)

            # Stop reading stderr, then drop every pipe, before waiting.
            if drain is not None:
                if not drain.done():
                    drain.cancel()
                try:
                    await asyncio.gather(drain, return_exceptions=True)
                except BaseException:
                    pass
            self._close_pipes(process)

            try:
                await asyncio.wait_for(process.wait(), timeout=KILL_GRACE_SECONDS)
                reaped = True
            except asyncio.TimeoutError:
                # An unreaped child is a zombie, and enough of them exhaust the
                # process table. The caller turns this into a failed request
                # and keeps the worker unavailable until it really dies.
                log.error("worker %s did not exit %ss after kill",
                          process.pid, KILL_GRACE_SECONDS)
                self._begin_cooldown()
            except Exception:
                log.error("worker %s could not be reaped", process.pid)
                self._begin_cooldown()
        except Exception:
            log.error("worker %s cleanup failed", getattr(process, "pid", "?"))
        return reaped

    @staticmethod
    def _close_pipes(process) -> None:
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name, None)
            if stream is None:
                continue
            # StreamWriter closes directly; a StreamReader has no close(), so
            # its transport is closed instead.
            for closer in (getattr(stream, "close", None),
                           getattr(getattr(stream, "_transport", None), "close", None)):
                if closer is None:
                    continue
                try:
                    closer()
                except Exception:
                    pass
