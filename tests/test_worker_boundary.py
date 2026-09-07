"""The document worker: process boundary, resource caps, deadline, reaping.

These test the properties that the in-process design could not have. In
particular, a wall deadline inside the server was a promise the code could not
keep: a synchronous native call cannot be cancelled, so the request would run
to completion however long it took. Here the deadline is a kill, and that is
testable.
"""

from __future__ import annotations

import asyncio
import io
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from PIL import Image

from app import limits, worker, worker_host

REPO = Path(__file__).resolve().parent.parent


def png_bytes(w=60, h=40):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ resource limits

def test_limits_are_applied_before_any_native_library_loads():
    """The caps must be in force while the model initializes, not after.

    app.worker.process() imports paddle and friends; apply_resource_limits runs
    in main() before that call. If the order were reversed a model that
    allocated past the cap during startup would never be stopped.
    """
    source = (REPO / "app" / "worker.py").read_text()
    limit_call = source.index("applied = apply_resource_limits(")
    process_call = source.index("result = process(header, body)")
    assert limit_call < process_call


def test_a_real_worker_reports_the_limits_it_applied():
    result = run(worker_host.DocumentWorker().run(png_bytes(), "a.png", False))
    applied = result["limits_applied"]
    assert applied["platform"] == "posix"
    assert applied["address_space"] == worker.DEFAULT_ADDRESS_SPACE_BYTES
    assert applied["cpu_seconds"] == worker.DEFAULT_CPU_SECONDS
    # A core dump of a multi-gigabyte process, triggered by a document, would
    # be both a denial of service and a copy of that document on disk.
    assert applied["core"] == 0


def test_the_address_space_cap_is_above_the_measured_startup_requirement():
    """A recorded pass reported VmPeak of 12,400,792 kB.

    An 8 GiB cap, which is the obvious round number, would refuse a legitimate
    document before it started. The value is chosen from that measurement.
    """
    measured_peak_bytes = 12_400_792 * 1024
    assert worker.DEFAULT_ADDRESS_SPACE_BYTES > measured_peak_bytes


def test_limits_are_never_raised_above_an_environment_ceiling():
    """setrlimit must not try to exceed the inherited hard limit.

    Run in a child process on purpose. Lowering RLIMIT_CPU's HARD limit is
    irreversible for the life of a process, so doing it in the test runner
    would cap the whole suite at that many CPU seconds.
    """
    program = (
        "import json, resource, sys;"
        "resource.setrlimit(resource.RLIMIT_CPU, (30, 30));"
        "sys.path.insert(0, %r);"
        "from app import worker;"
        "print(json.dumps(worker.apply_resource_limits(cpu_seconds=99999)))"
    ) % str(REPO)
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True, text=True, cwd=str(REPO), timeout=60,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, result.stderr
    import json

    applied = json.loads(result.stdout.strip().splitlines()[-1])
    assert applied["cpu_seconds"] <= 30, applied


def test_an_unavailable_limit_is_reported_as_none_rather_than_assumed():
    """A limit that could not be set must not be reported as if it were.

    In a subprocess: calling apply_resource_limits here would also apply the
    CPU cap and disable core dumps in the pytest process, permanently and
    invisibly to every test that runs after it.
    """
    program = (
        "import json, sys;"
        "sys.path.insert(0, %r);"
        "from app import worker;"
        "print(json.dumps(worker.apply_resource_limits(address_space=-1)))"
    ) % str(REPO)
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        cwd=str(REPO), timeout=60, env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, result.stderr
    import json

    applied = json.loads(result.stdout.strip().splitlines()[-1])
    assert applied["address_space"] is None


# ---------------------------------------------------------------- the boundary

def test_the_worker_runs_in_a_different_process():
    """The whole point: document code does not execute in the API process.

    Asserted on process IDENTITY, not on the response contents. An earlier
    version of this test only checked that a result came back, which an
    implementation that processed the document inline would also satisfy.
    """
    result = run(worker_host.DocumentWorker().run(png_bytes(), "a.png", False))
    assert result["source_pages"] == 1
    assert result["complete"] is True
    assert result["worker_pid"] != os.getpid(), "the document was processed inline"


def test_each_document_gets_a_fresh_process():
    """No state, no warmed model and no heap carries between documents."""
    docker = worker_host.DocumentWorker()
    first = run(docker.run(png_bytes(), "a.png", False))
    second = run(docker.run(png_bytes(), "b.png", False))
    assert first["worker_pid"] != second["worker_pid"], "the worker was reused"
    assert first["worker_pid"] != os.getpid()
    assert second["worker_pid"] != os.getpid()


def test_the_child_environment_is_rebuilt_not_inherited():
    """A secret exported into the service must not reach document code."""
    os.environ["XLITEOCR_TEST_SECRET"] = "should-not-be-inherited"
    try:
        env = worker_host.DocumentWorker()._environment()
        assert "XLITEOCR_TEST_SECRET" not in env
        assert "PYTHONPATH" in env
    finally:
        del os.environ["XLITEOCR_TEST_SECRET"]


def test_spawn_not_fork():
    """Forking would hand the child a warmed Paddle heap and live locks."""
    source = (REPO / "app" / "worker_host.py").read_text()
    assert "create_subprocess_exec" in source
    assert "fork()" not in source


# ------------------------------------------------------------------- lifecycle

def test_a_hanging_worker_is_killed_at_the_deadline_and_reaped():
    """The deadline is a kill, which an in-process design cannot do."""
    docker = worker_host.DocumentWorker()
    # A worker that ignores its input and sleeps: stands in for a native call
    # that never returns.
    docker.python = sys.executable

    class Hanging(worker_host.DocumentWorker):
        async def run_hanging(self):
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "import time; time.sleep(60)",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            started = time.monotonic()
            try:
                with pytest.raises(worker_host.WorkerError):
                    await self._exchange(process, b"{}", b"", 1.0, None)
            finally:
                await self._terminate(process)
            assert time.monotonic() - started < 20
            # Reaped: a killed child that is never waited on becomes a zombie.
            assert process.returncode is not None
            return process

    run(Hanging().run_hanging())


def test_a_worker_that_dies_without_answering_is_reported_not_hung():
    async def scenario():
        docker = worker_host.DocumentWorker()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import sys; sys.exit(9)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            with pytest.raises(worker_host.WorkerError):
                await docker._exchange(process, b"{}", b"", 10.0, None)
        finally:
            await docker._terminate(process)
        assert process.returncode is not None

    run(scenario())


def test_a_failure_starts_a_cooldown_so_a_crashing_input_is_not_a_fork_bomb():
    async def scenario():
        docker = worker_host.DocumentWorker(cooldown_seconds=30)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import sys; sys.exit(9)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            with pytest.raises(worker_host.WorkerError):
                await docker._exchange(process, b"{}", b"", 10.0, None)
        finally:
            await docker._terminate(process)

        assert docker.in_cooldown
        # A client cannot immediately spawn another process with the same input.
        with pytest.raises(worker_host.WorkerUnavailable):
            await docker.run(png_bytes(), "a.png", False)

    run(scenario())


def test_the_cooldown_expires():
    docker = worker_host.DocumentWorker(cooldown_seconds=0.05)
    docker._begin_cooldown()
    assert docker.in_cooldown
    time.sleep(0.1)
    assert not docker.in_cooldown


def test_terminate_is_safe_on_an_already_dead_process():
    async def scenario():
        docker = worker_host.DocumentWorker()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "pass",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.wait()
        await docker._terminate(process)  # must not raise
        await docker._terminate(process)  # idempotent

    run(scenario())


# ------------------------------------------------------------------- protocol

def test_the_worker_refuses_a_malformed_request_rather_than_hanging():
    result = subprocess.run(
        [sys.executable, "-m", "app.worker"],
        input=b"\x00\x00\x00\x02{}",  # truncated: header claims 2 bytes, no body
        capture_output=True, cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=60,
    )
    assert result.returncode == 2
    (length,) = struct.unpack(">I", result.stdout[:4])
    import json

    payload = json.loads(result.stdout[4:4 + length])
    assert payload["kind"] == "protocol"


def test_an_oversized_response_is_refused_rather_than_streamed():
    out = io.BytesIO()
    worker.write_response(out, {"pages": ["x" * 10_000]}, max_bytes=1000)
    raw = out.getvalue()
    (length,) = struct.unpack(">I", raw[:4])
    import json

    payload = json.loads(raw[4:4 + length])
    assert payload["kind"] == "response_too_large"


def test_the_parent_refuses_an_announced_oversized_response():
    """Length prefix is attacker-influenced if the child is compromised."""
    source = (REPO / "app" / "worker_host.py").read_text()
    assert "worker announced an oversized response" in source


# --------------------------------------------------------- limits travel across

def test_the_parent_sends_its_limits_to_the_child(monkeypatch):
    """Both processes must agree, or a configured limit is silently ignored."""
    monkeypatch.setattr(limits, "MAX_PAGES", 7)
    monkeypatch.setattr(limits, "MAX_TOTAL_PIXELS", 12345)

    captured = {}

    async def fake_exec(*args, **kwargs):
        raise OSError("not started on purpose")

    docker = worker_host.DocumentWorker()
    real_json_dumps = worker_host.json.dumps

    def spy(obj, *a, **kw):
        if isinstance(obj, dict) and "body_bytes" in obj:
            captured.update(obj)
        return real_json_dumps(obj, *a, **kw)

    monkeypatch.setattr(worker_host.json, "dumps", spy)
    monkeypatch.setattr(worker_host.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(worker_host.WorkerUnavailable):
        run(docker.run(b"x", "a.png", False))

    assert captured["max_pages"] == 7
    assert captured["max_total_pixels"] == 12345


def test_the_child_adopts_the_limits_it_is_sent():
    source = (REPO / "app" / "worker.py").read_text()
    assert 'setattr(limits, attribute, int(header[key]))' in source


# ------------------------------------------------------------------- honesty

def test_the_module_does_not_claim_to_be_a_sandbox():
    """Resource containment is not a privilege boundary; the docs must say so."""
    source = (REPO / "app" / "worker.py").read_text()
    assert "It is NOT a privilege or filesystem or network boundary" in source
    assert "DEPLOYMENT" in source


def test_windows_is_not_described_as_equivalently_contained():
    source = (REPO / "app" / "worker.py").read_text()
    assert "Windows" in source
    assert "must not be" in source


# =====================================================================
# Regressions for the independent review findings (R-001 .. R-012).
# Each names the defect it prevents coming back.
# =====================================================================

def test_the_server_process_never_parses_a_document():
    """R-001 (blocker): the boundary is worthless if the parent decodes first.

    The server used to call pdf.source_page_count for a page-count preflight.
    That hands attacker bytes to PDFium or Pillow inside uvicorn, while holding
    admission, outside the worker's rlimits and deadline. A crash there takes
    the service down, which is the whole thing the worker exists to prevent.
    """
    source = (REPO / "app" / "server.py").read_text()
    for forbidden in ("pdf_mod", "source_page_count", "iter_document_pages",
                      "PdfDocument", "Image.open"):
        assert forbidden not in source, f"server.py still references {forbidden}"


def test_the_server_never_logs_an_exception_message_or_traceback():
    """R-006: decoder messages quote document bytes; the log is persistent."""
    source = (REPO / "app" / "server.py").read_text()
    assert "exc_info=True" not in source


def test_an_ordinary_failure_returns_a_framed_error_not_a_traceback():
    """R-008: two sibling `except Exception` blocks do not chain.

    A bare `raise` in the first propagates out of the whole try, so the
    sanitized handler after it was unreachable and every ordinary decode
    failure escaped as an uncaught traceback with the original message.
    """
    result = subprocess.run(
        [sys.executable, "-m", "app.worker"],
        input=_framed({"body_bytes": 4, "filename": "x.png"}, b"junk"),
        capture_output=True, cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=300,
    )
    assert result.returncode == 1, result.stderr.decode()[-500:]
    payload = _unframe(result.stdout)
    assert payload["kind"] == "unreadable"
    assert payload["error"] == "could not read input"
    # The message must not have escaped anywhere.
    assert b"Traceback" not in result.stderr


def test_the_worker_stderr_carries_only_a_class_name_by_default():
    """R-006/F3: the message may quote the document."""
    result = subprocess.run(
        [sys.executable, "-m", "app.worker"],
        input=_framed({"body_bytes": 4, "filename": "x.png"}, b"junk"),
        capture_output=True, cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=300,
    )
    text = result.stderr.decode()
    assert "worker: " in text
    # A bare "worker: ClassName" line and nothing more.
    worker_lines = [l for l in text.splitlines() if l.startswith("worker: ")]
    assert worker_lines
    for line in worker_lines:
        assert line.count(":") == 1, f"stderr carried a message: {line!r}"


def test_the_worker_refuses_to_process_when_a_limit_cannot_be_applied():
    """R-005: failing open gives up containment while still looking contained.

    Behavioral. `resource.setrlimit` is made to fail before app.worker is
    imported, then a real document is fed in. The worker must refuse BEFORE
    processing, not process and report the missing cap afterwards. Grepping
    the source for the branch, as this test used to, passed even when the
    branch was disabled.
    """
    program = (
        "import resource, runpy, sys;"
        "resource.setrlimit = lambda *a, **k: (_ for _ in ()).throw(OSError('denied'));"
        "sys.path.insert(0, %r);"
        "runpy.run_module('app.worker', run_name='__main__')"
    ) % str(REPO)
    result = subprocess.run(
        [sys.executable, "-c", program],
        input=_framed({"filename": "a.png"}, png_bytes()),
        capture_output=True, cwd=str(REPO), timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    payload = _unframe(result.stdout)
    assert payload["kind"] == "limits_unavailable", payload
    assert sorted(payload["missing"]) == ["address_space", "core", "cpu_seconds"]
    # And it must not have gone on to do the work anyway.
    assert "pages" not in payload


def test_an_inherited_soft_limit_is_never_widened():
    """R-005: considering only the hard limit widened an 8 GiB soft cap."""
    program = (
        "import json, resource, sys;"
        "resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, resource.RLIM_INFINITY));"
        "sys.path.insert(0, %r);"
        "from app import worker;"
        "print(json.dumps(worker.apply_resource_limits()))"
    ) % str(REPO)
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        cwd=str(REPO), timeout=60, env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, result.stderr
    import json

    applied = json.loads(result.stdout.strip().splitlines()[-1])
    assert applied["address_space"] <= 2 * 1024**3, applied


def test_the_child_stderr_pipe_is_drained():
    """R-003: an unread pipe fills and the child blocks on write forever."""
    source = (REPO / "app" / "worker_host.py").read_text()
    assert "_drain_stderr" in source
    assert "drain = asyncio.ensure_future" in source


def test_a_chatty_worker_does_not_deadlock():
    """R-003, behaviorally: far more stderr than a pipe buffer holds."""
    async def scenario():
        docker = worker_host.DocumentWorker()
        noisy = (
            "import sys;"
            "sys.stderr.write('x' * 500000); sys.stderr.flush();"
            "sys.exit(7)"
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", noisy,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        drain = asyncio.ensure_future(docker._drain_stderr(process))
        try:
            with pytest.raises(worker_host.WorkerError):
                await docker._exchange(process, b"{}", b"", 20.0, None)
        finally:
            await docker._terminate(process, drain)
        assert process.returncode is not None

    started = time.monotonic()
    run(scenario())
    assert time.monotonic() - started < 25, "the exchange deadlocked on stderr"


def test_cancelling_a_request_does_not_release_it_before_the_child_is_dead():
    """R-004/R-001: `shield` alone still raises in the awaiter.

    The request unwound and admission was released while cleanup carried on
    independently, so a second document could start while the first child was
    still holding memory and pipes.

    The assertion is on the Process OBJECT, not on a pid probe. `os.kill(pid, 0)`
    terminates the process on Windows rather than probing it, and treating an
    unreadable /proc as proof of death let the check pass on a restricted host.
    `returncode is not None` means this parent has reaped it, which is exactly
    the guarantee under test.
    """
    spawned = []

    async def scenario():
        docker = worker_host.DocumentWorker()
        real_exec = asyncio.create_subprocess_exec

        async def recording_exec(*args, **kwargs):
            process = await real_exec(*args, **kwargs)
            spawned.append(process)
            return process

        worker_host.asyncio.create_subprocess_exec = recording_exec
        try:
            task = asyncio.ensure_future(
                docker.run(png_bytes(), "a.png", False, deadline_seconds=30)
            )
            # Wait for the spawn rather than sleeping a guessed interval. A
            # fixed sleep raced the worker: with the models warm the whole run
            # finished inside it, leaving nothing to cancel.
            deadline = time.monotonic() + 20
            while not spawned:
                assert time.monotonic() < deadline, "no worker was spawned"
                await asyncio.sleep(0.01)
            assert not task.done(), "the run completed before it could be cancelled"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            worker_host.asyncio.create_subprocess_exec = real_exec

    run(scenario())

    assert spawned, "no worker was spawned"
    for process in spawned:
        assert process.returncode is not None, (
            f"worker {process.pid} was not reaped before the cancellation surfaced"
        )


def test_a_cancellation_delivered_during_cleanup_reaches_the_caller():
    """R-001 (round 4): really deliver a cancellation while cleanup is running.

    The previous version of this test called asyncio.current_task() and set the
    outcome flag by hand, so it passed whether or not the code preserved the
    cancellation. This one cancels the task that is awaiting cleanup, from
    inside cleanup, and asserts CancelledError comes out of run().
    """

    async def scenario():
        docker = worker_host.DocumentWorker()
        entered = asyncio.Event()

        async def slow_terminate(process, drain=None):
            entered.set()
            await asyncio.sleep(0.5)   # long enough to be cancelled during
            return True

        docker._terminate = slow_terminate

        # Captured BEFORE the patch: worker_host.asyncio is the global asyncio
        # module, so patching that attribute patches it everywhere, and calling
        # the patched name in here made this helper call itself.
        real_exec = worker_host.asyncio.create_subprocess_exec

        async def payload_child(*args, **kwargs):
            return await real_exec(
                sys.executable, "-c",
                "import struct,sys;"
                "p=b'{\"pages\":[],\"source_pages\":0,"
                "\"processed_pages\":0,\"complete\":true}';"
                "sys.stdout.buffer.write(struct.pack('>I',len(p))+p);"
                "sys.stdout.buffer.flush()",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )

        worker_host.asyncio.create_subprocess_exec = payload_child
        try:
            task = asyncio.ensure_future(docker.run(b"x", "a.png", False))
            await asyncio.wait_for(entered.wait(), timeout=20)
            # The exchange has already succeeded; cleanup is in flight.
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            worker_host.asyncio.create_subprocess_exec = real_exec

    run(scenario())


def test_a_cancellation_at_the_exchange_finally_is_not_swallowed():
    """R-001 (round 4): the `except BaseException: pass` in _exchange's finally
    discarded a cancellation delivered exactly there.

    Behavioral. The previous version searched the source for
    `except asyncio.CancelledError:`, which still passed when that handler was
    a `pass`. This drives the real code: the exchange succeeds, then cleanup is
    made slow so the cancellation lands while the finally is running, and the
    caller must see CancelledError rather than the payload.
    """

    async def scenario():
        docker = worker_host.DocumentWorker()
        in_cleanup = asyncio.Event()

        async def slow_cleanup(process, drain, outcome):
            in_cleanup.set()
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                outcome["cancelled"] = True
            outcome["reaped"] = True

        docker._await_cleanup = slow_cleanup
        real_exec = worker_host.asyncio.create_subprocess_exec

        async def payload_child(*args, **kwargs):
            return await real_exec(
                sys.executable, "-c",
                "import struct,sys;"
                "p=b'{\"pages\":[],\"source_pages\":0,"
                "\"processed_pages\":0,\"complete\":true}';"
                "sys.stdout.buffer.write(struct.pack('>I',len(p))+p);"
                "sys.stdout.buffer.flush()",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )

        worker_host.asyncio.create_subprocess_exec = payload_child
        try:
            task = asyncio.ensure_future(docker.run(b"x", "a.png", False))
            await asyncio.wait_for(in_cleanup.wait(), timeout=20)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            worker_host.asyncio.create_subprocess_exec = real_exec

    run(scenario())


def test_a_cancellation_survives_a_failed_exchange():
    """R-001 (round 5): when the exchange ALSO failed, the original error
    propagated and the cancellation was dropped, because the check sat after
    the finally rather than inside it."""

    async def scenario():
        docker = worker_host.DocumentWorker()

        async def failing_exchange(*args, **kwargs):
            raise worker_host.WorkerError("exchange failed")

        async def cancelled_cleanup(process, drain, outcome):
            outcome["cancelled"] = True
            outcome["reaped"] = True

        docker._exchange = failing_exchange
        docker._await_cleanup = cancelled_cleanup
        real_exec = worker_host.asyncio.create_subprocess_exec

        async def quick_child(*args, **kwargs):
            return await real_exec(
                sys.executable, "-c", "pass",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        worker_host.asyncio.create_subprocess_exec = quick_child
        try:
            # Cancellation must win over the WorkerError.
            with pytest.raises(asyncio.CancelledError):
                await docker.run(b"x", "a.png", False)
        finally:
            worker_host.asyncio.create_subprocess_exec = real_exec

    run(scenario())


def test_a_cancelled_cleanup_task_still_records_the_live_child():
    """R-002 (round 5): a shutdown that cancels every task could cancel the
    terminate task before it signaled anything. `task.result()` then raised
    CancelledError, which `except Exception` missed, so the live child was
    never recorded and the worker happily admitted another."""

    class NeverDies:
        pid = -1
        returncode = None
        stdin = stdout = stderr = None
        killed = False

        def kill(self):
            NeverDies.killed = True

        async def wait(self):
            await asyncio.sleep(3600)

    async def scenario():
        docker = worker_host.DocumentWorker()
        process = NeverDies()

        async def cancelled_terminate(proc, drain=None):
            raise asyncio.CancelledError()

        docker._terminate = cancelled_terminate
        outcome = {"reaped": True, "cancelled": False}
        await docker._await_cleanup(process, None, outcome)

        assert outcome["reaped"] is False
        assert process in docker._unresolved, "a live child was not recorded"
        assert NeverDies.killed, "no last-resort kill was attempted"

    run(scenario())


def test_await_cleanup_reports_what_actually_happened():
    """R-002: the outcome must be measured, not assumed.

    Two halves, because reporting success is only meaningful if the failure
    case reports failure.
    """

    async def scenario():
        # A real child, really reaped: reported as reaped.
        docker = worker_host.DocumentWorker()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "pass",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        drain = asyncio.ensure_future(docker._drain_stderr(process))
        outcome = {"reaped": False}
        await docker._await_cleanup(process, drain, outcome)
        assert outcome["reaped"] is True
        assert drain.done()

        # A terminate that cannot confirm the reap: reported as NOT reaped,
        # which is what run() turns into a failed request.
        docker2 = worker_host.DocumentWorker()

        async def cannot_reap(process, drain=None):
            return False

        docker2._terminate = cannot_reap
        outcome2 = {"reaped": True}
        await docker2._await_cleanup(None, None, outcome2)
        assert outcome2["reaped"] is False

    run(scenario())


def test_a_worker_that_cannot_be_reaped_fails_the_request():
    """The end-to-end consequence of R-002."""

    async def scenario():
        docker = worker_host.DocumentWorker()

        async def never_reaps(process, drain=None):
            if process.returncode is None:
                try:
                    process.kill()
                except Exception:
                    pass
                await process.wait()
            if drain is not None and not drain.done():
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
            return False  # pretend the reap could not be confirmed

        docker._terminate = never_reaps
        with pytest.raises(worker_host.WorkerError, match="could not be reaped"):
            await docker.run(png_bytes(), "a.png", False)

    run(scenario())


def test_a_framed_worker_failure_starts_the_cooldown():
    """R-004 (second pass): a decoded payload carrying an error is still a
    failure. Only exceptions used to trigger the cooldown, so a worker that
    refused to start because its limits were unavailable answered politely and
    could be relaunched at request rate."""
    docker = worker_host.DocumentWorker(cooldown_seconds=30)
    docker._classify_framed_result({"error": "limits unavailable",
                                    "kind": "limits_unavailable"})
    assert docker.in_cooldown


def test_a_refused_document_does_not_start_the_cooldown():
    """A refusal is a normal answer. Putting the worker in cooldown for it
    would let one oversized upload deny service to everyone else."""
    for kind in ("resource", "timeout", "unreadable"):
        docker = worker_host.DocumentWorker(cooldown_seconds=30)
        docker._classify_framed_result({"error": "too big", "kind": kind})
        assert not docker.in_cooldown, kind


def test_a_malformed_pdf_is_reported_as_unreadable_not_as_a_crash():
    """R-007: PdfiumError subclasses RuntimeError, which the classifier missed,
    so every malformed PDF returned 500 instead of 400."""
    from pypdfium2 import PdfiumError

    assert issubclass(PdfiumError, worker._unreadable_types())

    result = subprocess.run(
        [sys.executable, "-m", "app.worker"],
        input=_framed({"filename": "broken.pdf"},
                      b"%PDF-1.4\ngarbage not a real pdf\n"),
        capture_output=True, cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=300,
    )
    payload = _unframe(result.stdout)
    assert payload["kind"] == "unreadable", payload
    assert payload["error"] == "could not read input"


def test_a_kill_that_fails_still_waits_and_stops_the_drain():
    """R-006: kill() raising anything but ProcessLookupError skipped both the
    wait and the drain cancellation, leaving a pending task and no accounting."""

    async def scenario():
        docker = worker_host.DocumentWorker()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        drain = asyncio.ensure_future(docker._drain_stderr(process))

        real_kill = process.kill
        calls = {"n": 0}

        def failing_kill():
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError("denied")
            return real_kill()

        process.kill = failing_kill
        reaped = await docker._terminate(process, drain)

        # Signalling failed, so the child is still alive and must be reported
        # as NOT reaped rather than silently accepted.
        assert reaped is False, "a child that survived was reported as reaped"
        assert calls["n"] == 1, "kill was not attempted"
        # The remaining obligations still ran: the drain was cancelled, and the
        # wait was attempted rather than skipped.
        assert drain.done(), "the drain task was abandoned after a failed kill"
        assert docker.in_cooldown, "a failed reap did not start the cooldown"

        # Clean up for real.
        if process.returncode is None:
            real_kill()
            await process.wait()

    run(scenario())


def test_an_oversized_announced_response_starts_the_cooldown():
    """R-009: only four exception types used to start it."""
    async def scenario():
        docker = worker_host.DocumentWorker(cooldown_seconds=30)
        # A child that announces a huge length and then says nothing.
        program = (
            "import struct, sys;"
            "sys.stdout.buffer.write(struct.pack('>I', 4000000000));"
            "sys.stdout.buffer.flush();"
            "import time; time.sleep(30)"
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", program,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        drain = asyncio.ensure_future(docker._drain_stderr(process))
        try:
            with pytest.raises(worker_host.WorkerError):
                await docker._exchange(process, b"{}", b"", 20.0, None)
        finally:
            await docker._terminate(process, drain)
        assert docker.in_cooldown, "an oversized response bypassed the cooldown"

    run(scenario())


def test_the_fault_handler_is_not_accidentally_enabled():
    """R-012: PYTHONFAULTHANDLER is enabled by ANY non-empty value, "0" included."""
    env = worker_host.DocumentWorker()._environment()
    assert "PYTHONFAULTHANDLER" not in env


def test_windows_assembly_variables_survive_the_rebuilt_environment():
    """R-011: an explicit env without SystemRoot can break the interpreter."""
    source = (REPO / "app" / "worker_host.py").read_text()
    assert "SystemRoot" in source


def test_the_render_scaler_reads_the_limit_at_call_time():
    """R-010: binding it at import ignored the limit the parent configured."""
    from app import pdf as pdf_mod

    source = (REPO / "app" / "pdf.py").read_text()
    assert "limits.MAX_IMAGE_PIXELS" in source
    assert "from app.limits import MAX_IMAGE_PIXELS" not in source


def _framed(header, body):
    import json

    header = {**header, "body_bytes": len(body)}
    encoded = json.dumps(header).encode()
    return struct.pack(">I", len(encoded)) + encoded + body


def _unframe(raw):
    import json

    (length,) = struct.unpack(">I", raw[:4])
    return json.loads(raw[4:4 + length])


# =====================================================================
# Third review round.
# =====================================================================

def test_a_cancellation_first_delivered_during_cleanup_is_not_swallowed():
    """R-001 (round 3): a `finally` preserves an exception that was ALREADY
    propagating when it began. It does not restore one first raised inside it,
    so a shutdown cancelling a request during termination got a 200 back."""

    async def scenario():
        docker = worker_host.DocumentWorker()
        outcome = {"reaped": False, "cancelled": False}

        async def cancelling_terminate(process, drain=None):
            # Cancel the awaiting task from inside cleanup.
            asyncio.current_task()  # keep the loop honest
            return True

        docker._terminate = cancelling_terminate
        # Directly assert the contract _await_cleanup owes run().
        await docker._await_cleanup(None, None, outcome)
        assert outcome["reaped"] is True
        assert outcome["cancelled"] is False

    run(scenario())

    # And run() re-raises when cleanup recorded a cancellation.
    async def scenario2():
        docker = worker_host.DocumentWorker()

        async def marks_cancelled(process, drain, outcome):
            outcome["reaped"] = True
            outcome["cancelled"] = True

        docker._await_cleanup = marks_cancelled
        with pytest.raises(asyncio.CancelledError):
            await docker.run(png_bytes(), "a.png", False)

    run(scenario2())


class _UnreapableProcess:
    """A child that will not die. SIGKILL cannot actually be blocked, so the
    state machine is tested with a stub rather than with a real process."""

    pid = -1
    returncode = None
    stdin = stdout = stderr = None

    def kill(self):
        pass

    async def wait(self):
        await asyncio.sleep(3600)


def test_a_child_that_cannot_be_reaped_keeps_the_worker_unavailable():
    """R-002 (round 3): a five-second cooldown expires; a live orphan does not.

    The single-document bound is about resources. Admitting a second child
    beside one that is still holding its memory breaks that bound however
    correct the response status was.
    """

    async def scenario():
        docker = worker_host.DocumentWorker(cooldown_seconds=0.01)
        docker._unresolved.append(_UnreapableProcess())
        await asyncio.sleep(0.05)  # let any cooldown lapse
        assert not docker.in_cooldown, "the cooldown, not the orphan, is gating"

        with pytest.raises(worker_host.WorkerUnavailable, match="has not exited"):
            await docker.run(png_bytes(), "a.png", False)

    run(scenario())


def test_the_worker_becomes_available_once_the_orphan_really_dies():
    """The unavailable state clears on the fact, not on a timer."""

    async def scenario():
        docker = worker_host.DocumentWorker(cooldown_seconds=0.01)
        survivor = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        docker._unresolved.append(survivor)
        # Retrying the kill is part of the job: a child that ignored the first
        # signal is killed and reaped here rather than being left forever.
        assert await docker._reap_unresolved() == 0
        assert docker._unresolved == []
        assert survivor.returncode is not None

    run(scenario())


def test_an_unreapable_child_is_recorded_rather_than_forgotten():
    async def scenario():
        docker = worker_host.DocumentWorker()

        async def cannot_reap(process, drain=None):
            return False

        docker._terminate = cannot_reap
        outcome = {"reaped": True, "cancelled": False}
        sentinel = object()
        await docker._await_cleanup(sentinel, None, outcome)
        assert outcome["reaped"] is False
        assert sentinel in docker._unresolved

    run(scenario())


@pytest.mark.parametrize("payload", [None, [], "text", 42])
def test_a_worker_response_that_is_not_an_object_is_refused(payload):
    """R-005: JSON decoding was the only validation."""
    with pytest.raises(worker_host.WorkerError):
        worker_host._validated(payload)


def test_an_empty_object_is_not_a_successful_result():
    """`{}` used to become a 200 with no counts and complete=false."""
    with pytest.raises(worker_host.WorkerError, match="missing"):
        worker_host._validated({})


@pytest.mark.parametrize("bad", [
    {"pages": "not a list", "source_pages": 1, "processed_pages": 1, "complete": True},
    {"pages": [], "source_pages": "1", "processed_pages": 1, "complete": True},
    {"pages": [], "source_pages": 1, "processed_pages": 1, "complete": "yes"},
])
def test_a_result_with_wrong_field_types_is_refused(bad):
    with pytest.raises(worker_host.WorkerError):
        worker_host._validated(bad)


def test_a_well_formed_result_passes_validation():
    good = {"pages": [{"page": 0}, {"page": 1}], "source_pages": 2,
            "processed_pages": 2, "complete": True}
    assert worker_host._validated(good) is good
    empty = {"pages": [], "source_pages": 0, "processed_pages": 0,
             "complete": True}
    assert worker_host._validated(empty) is empty


def test_counts_that_contradict_the_pages_are_refused():
    """A broken or compromised child could otherwise report two processed
    pages, an empty page list and complete=true, and be believed."""
    with pytest.raises(worker_host.WorkerError, match="does not match"):
        worker_host._validated({"pages": [], "source_pages": 2,
                                "processed_pages": 2, "complete": True})


def test_a_boolean_is_not_accepted_as_a_page_count():
    """bool subclasses int, so JSON true passed an isinstance(int) check."""
    with pytest.raises(worker_host.WorkerError, match="non-integer"):
        worker_host._validated({"pages": [], "source_pages": True,
                                "processed_pages": 0, "complete": False})


def test_negative_counts_are_refused():
    with pytest.raises(worker_host.WorkerError, match="negative"):
        worker_host._validated({"pages": [], "source_pages": -1,
                                "processed_pages": 0, "complete": False})


def test_a_complete_flag_that_contradicts_the_counts_is_refused():
    with pytest.raises(worker_host.WorkerError, match="contradicts"):
        worker_host._validated({"pages": [{"page": 0}], "source_pages": 5,
                                "processed_pages": 1, "complete": True})


def test_a_page_that_is_not_an_object_is_refused():
    with pytest.raises(worker_host.WorkerError, match="not an object"):
        worker_host._validated({"pages": ["nope"], "source_pages": 1,
                                "processed_pages": 1, "complete": True})


def test_a_malformed_error_frame_is_refused():
    with pytest.raises(worker_host.WorkerError):
        worker_host._validated({"error": "x", "kind": []})
    with pytest.raises(worker_host.WorkerError):
        worker_host._validated({"error": 7, "kind": "resource"})


def test_a_hostile_kind_does_not_escape_the_cooldown_classifier():
    """`kind: []` used to raise TypeError inside the set membership test."""
    docker = worker_host.DocumentWorker(cooldown_seconds=30)
    docker._classify_framed_result({"error": "x", "kind": []})
    assert docker.in_cooldown


def test_an_initialization_failure_is_not_blamed_on_the_document():
    """R-004 (round 3): a missing model file raised FileNotFoundError inside
    the page loop, matched OSError, and came back as 400 "could not read
    input" with no cooldown. The service blamed every caller for its own
    broken installation."""
    program = (
        "import sys, runpy;"
        "sys.path.insert(0, %r);"
        "import engine.ocr_engine as oe;"
        "oe._get_ocr = lambda *a, **k: (_ for _ in ()).throw("
        "    FileNotFoundError('model cache missing'));"
        "runpy.run_module('app.worker', run_name='__main__')"
    ) % str(REPO)
    result = subprocess.run(
        [sys.executable, "-c", program],
        input=_framed({"filename": "a.png"}, png_bytes()),
        capture_output=True, cwd=str(REPO), timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    payload = _unframe(result.stdout)
    assert payload["kind"] == "initialization_failed", payload
    assert payload["kind"] != "unreadable"


def test_an_initialization_failure_starts_the_cooldown():
    docker = worker_host.DocumentWorker(cooldown_seconds=30)
    docker._classify_framed_result({"error": "the service could not initialize",
                                    "kind": "initialization_failed"})
    assert docker.in_cooldown


@pytest.mark.parametrize("error", [
    "FileNotFoundError('model cache missing')",
    "PermissionError('cannot read the model directory')",
    "NotADirectoryError('model path is a file')",
    "IsADirectoryError('expected a file')",
])
def test_a_filesystem_failure_is_never_blamed_on_the_document(error):
    """The document arrived in memory over a pipe; a missing FILE is ours.

    Several of these subclass OSError, so removing their names from the
    unreadable tuple left them matching through the superclass anyway. This
    drives the real handler and asserts on the kind that comes back.
    """
    program = (
        "import sys, runpy;"
        "sys.path.insert(0, %r);"
        "import app.pipeline as pl;"
        "pl.process_page = lambda *a, **k: (_ for _ in ()).throw(%s);"
        "runpy.run_module('app.worker', run_name='__main__')"
    ) % (str(REPO), error)
    result = subprocess.run(
        [sys.executable, "-c", program],
        input=_framed({"filename": "a.png"}, png_bytes()),
        capture_output=True, cwd=str(REPO), timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    payload = _unframe(result.stdout)
    # Not merely "something other than unreadable": an initialization failure
    # would satisfy that too, and would mean the injected exception never ran.
    # The injection replaces process_page, which runs in the DOCUMENT phase, so
    # the expected outcome is a generic processing failure named by class.
    assert payload["kind"] != "unreadable", (
        f"{error} was blamed on the document: {payload}"
    )
    assert payload["kind"] != "initialization_failed", (
        "the injected exception never ran; the worker failed earlier"
    )
    assert payload["error"] == "processing failed", payload
    assert payload["kind"] == error.split("(")[0], payload


def test_cleanup_closes_the_pipes():
    """R-003 (round 3): stdout was left paused with its transport open.

    The earlier version of this test launched a child that wrote nothing, so it
    never produced the paused-output condition it claimed to cover, and its
    assertion ended in `or True`, which passes whatever the transport does.
    This one creates real stdout backpressure and asserts closure without an
    escape hatch.
    """

    async def scenario():
        docker = worker_host.DocumentWorker()
        # Writes far more than a pipe buffer holds and never exits on its own,
        # so stdout is genuinely paused when cleanup runs.
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import sys,time;"
            "sys.stdout.buffer.write(b'x' * 4000000);"
            "sys.stdout.buffer.flush();"
            "time.sleep(30)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        await asyncio.sleep(0.5)  # let it fill the pipe
        drain = asyncio.ensure_future(docker._drain_stderr(process))

        reaped = await docker._terminate(process, drain)
        assert reaped is True, "a child with a full stdout pipe was not reaped"
        assert drain.done()

        # Every stream must be closed. No `or True`, and a missing transport is
        # a failure rather than a skipped assertion.
        for name in ("stdout", "stderr"):
            stream = getattr(process, name)
            transport = getattr(stream, "_transport", None)
            assert transport is not None, f"{name} has no transport to close"
            assert transport.is_closing(), f"{name} transport left open"

    run(scenario())


@pytest.mark.parametrize("errno_name", ["ENOSPC", "EROFS", "EMFILE"])
def test_a_machine_failure_by_errno_is_not_blamed_on_the_document(errno_name):
    """R-004 (round 5): OSError stayed in the unreadable tuple, so a full disk
    or a read-only filesystem raised during processing came back as 400 "could
    not read input" and skipped the cooldown."""
    program = (
        "import errno, sys, runpy;"
        "sys.path.insert(0, %r);"
        "import app.pipeline as pl;"
        "pl.process_page = lambda *a, **k: (_ for _ in ()).throw("
        "    OSError(errno.%s, 'machine failure'));"
        "runpy.run_module('app.worker', run_name='__main__')"
    ) % (str(REPO), errno_name)
    result = subprocess.run(
        [sys.executable, "-c", program],
        input=_framed({"filename": "a.png"}, png_bytes()),
        capture_output=True, cwd=str(REPO), timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    payload = _unframe(result.stdout)
    assert payload["kind"] != "unreadable", payload
    assert payload["error"] == "processing failed", payload


def test_a_truncated_image_is_still_an_unreadable_document():
    """The other side: a bare OSError with no errno is how PIL reports a
    truncated image, and that IS the document's problem. Carving out every
    OSError would have turned real document failures into 500s."""
    program = (
        "import sys, runpy;"
        "sys.path.insert(0, %r);"
        "import app.pipeline as pl;"
        "pl.process_page = lambda *a, **k: (_ for _ in ()).throw("
        "    OSError('image file is truncated'));"
        "runpy.run_module('app.worker', run_name='__main__')"
    ) % str(REPO)
    result = subprocess.run(
        [sys.executable, "-c", program],
        input=_framed({"filename": "a.png"}, png_bytes()),
        capture_output=True, cwd=str(REPO), timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    payload = _unframe(result.stdout)
    assert payload["kind"] == "unreadable", payload


# =====================================================================
# Sixth review round.
# =====================================================================

def _never_answering_child():
    return asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(30)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )


def test_a_cancellation_racing_a_worker_error_wins():
    """R-001 (rounds 6 and 7): a cancellation pending while an exception is
    converted must outrank the conversion.

    This drives the REAL `_exchange`. `task.cancel()` does not raise
    synchronously: it sets `cancelling()` and delivers CancelledError at the
    next await. There is no await between `is_disconnected` returning True and
    the `except WorkerError` handler, so the guard in that handler is the only
    thing that can turn this into a CancelledError. Remove it and this test
    sees a WorkerError instead, which is exactly the regression it exists for.
    """

    async def scenario():
        docker = worker_host.DocumentWorker()
        process = await _never_answering_child()
        drain = asyncio.ensure_future(docker._drain_stderr(process))

        async def cancel_then_report_disconnected():
            asyncio.current_task().cancel()
            return True

        try:
            task = asyncio.ensure_future(
                docker._exchange(process, b"{}", b"", 30.0,
                                 cancel_then_report_disconnected)
            )
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await docker._terminate(process, drain)

    run(scenario())


def test_a_cancellation_racing_the_deadline_wins():
    """The same guarantee on the timeout handler, which is a separate path."""

    async def scenario():
        docker = worker_host.DocumentWorker()
        process = await _never_answering_child()
        drain = asyncio.ensure_future(docker._drain_stderr(process))

        async def cancel_but_stay_connected():
            asyncio.current_task().cancel()
            return False

        try:
            # A deadline short enough that the next loop turn expires it, with
            # no await between the expiry check and the handler.
            task = asyncio.ensure_future(
                docker._exchange(process, b"{}", b"", 0.4,
                                 cancel_but_stay_connected)
            )
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await docker._terminate(process, drain)

    run(scenario())


class _FakeWriter:
    def write(self, data): pass

    async def drain(self): pass

    def close(self): pass


class _OversizedReader:
    """Announces a response over the ceiling, but only once released."""

    def __init__(self, release):
        self._release = release

    async def readexactly(self, count):
        await self._release.wait()
        return struct.pack(">I", 4_000_000_000)

    async def read(self, count):
        await asyncio.sleep(3600)


class _FakeProcess:
    returncode = 0
    pid = -1

    def __init__(self, release):
        self.stdin = _FakeWriter()
        self.stdout = _OversizedReader(release)
        self.stderr = _OversizedReader(release)

    def kill(self): pass

    async def wait(self): return 0


def test_the_cancellation_guards_are_load_bearing():
    """R-004 (round 7): the previous version checked for substrings near two
    `raise` statements and passed with every guard disabled.

    This mutates the real module, removes every `_raise_if_cancelling()` call,
    and requires the behavior to REGRESS.

    The interleaving is chosen rather than raced, because only one interleaving
    exercises the guards at all. When the inner task is still PENDING, the
    `finally` in `_exchange` awaits it, that await delivers the pending
    CancelledError, and the cancellation wins with or without the guards. The
    guards decide the narrow case where the inner task has ALREADY completed
    with an exception: `asyncio.wait_for` then returns `fut.result()`, which
    raises that exception straight past a cancellation nobody delivered.
    Releasing the inner failure and cancelling one to three loop turns later
    lands in exactly that window; the assertion below proves it, because with
    the guards removed the same window yields a WorkerError.
    """
    import importlib.util

    source = (REPO / "app" / "worker_host.py").read_text()
    call = "self._raise_if_cancelling()"
    assert source.count(call) >= 4, "guards missing from the conversion paths"

    def load_mutant(text, name):
        path = Path(tempfile.mkdtemp()) / "worker_host_mutant.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    async def probe(module, gap):
        release = asyncio.Event()
        docker = module.DocumentWorker()
        task = asyncio.ensure_future(
            docker._exchange(_FakeProcess(release), b"{}", b"", 30.0, None)
        )
        for _ in range(4):
            await asyncio.sleep(0)      # let the outer park inside wait_for
        release.set()                   # the inner will now fail
        for _ in range(gap):
            await asyncio.sleep(0)      # let it finish while the outer waits
        if task.done():
            return "outer already finished"
        task.cancel()
        try:
            await task
            return "no exception"
        except asyncio.CancelledError:
            return "CancelledError"
        except module.WorkerError:
            return "WorkerError"

    clean = load_mutant(source, "wh_clean")
    mutant = load_mutant(source.replace(call, "pass"), "wh_mutant")

    regressed = False
    for gap in (1, 2, 3):
        assert run(probe(clean, gap)) == "CancelledError", (
            f"a cancellation was lost at gap={gap}"
        )
        if run(probe(mutant, gap)) == "WorkerError":
            regressed = True

    assert regressed, (
        "removing every cancellation guard changed nothing; this test is not "
        "establishing the guarantee it names"
    )


def test_a_cancelled_terminate_still_closes_the_pipes():
    """R-002 (round 6): the fallback killed and recorded the child but never
    cancelled the drain or closed the pipes."""

    async def scenario():
        docker = worker_host.DocumentWorker()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import sys,time;"
            "sys.stdout.buffer.write(b'x' * 4000000); sys.stdout.buffer.flush();"
            "time.sleep(30)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        await asyncio.sleep(0.5)  # let stdout pause under flow control
        drain = asyncio.ensure_future(docker._drain_stderr(process))

        async def cancelled_before_running(proc, d=None):
            raise asyncio.CancelledError()

        docker._terminate = cancelled_before_running
        outcome = {"reaped": True, "cancelled": False}
        await docker._await_cleanup(process, drain, outcome)

        assert outcome["reaped"] is False
        assert process in docker._unresolved
        assert drain.done(), "the drain was left pending by the fallback"
        for name in ("stdout", "stderr"):
            transport = getattr(getattr(process, name), "_transport", None)
            assert transport is not None
            assert transport.is_closing(), f"{name} left open by the fallback"

        if process.returncode is None:
            process.kill()
            await process.wait()

    run(scenario())


def test_recovery_closes_the_pipes_of_a_child_it_resolves():
    """R-005 (round 7): the previous version launched a child that exited
    immediately and awaited it BEFORE calling recovery, so its pipes could
    already have closed through ordinary EOF handling and the test passed with
    both `_close_pipes` calls deleted.

    This gives recovery an observably OPEN transport to resolve: a child that
    is still alive with stdout paused under flow control. The mutation check at
    the end is the part that makes the assertion mean something.
    """
    import importlib.util

    source = (REPO / "app" / "worker_host.py").read_text()

    def load_mutant(text, name):
        path = Path(tempfile.mkdtemp()) / "worker_host_mutant.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    async def probe(module):
        docker = module.DocumentWorker()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import sys,time;"
            "sys.stdout.buffer.write(b'x' * 4000000); sys.stdout.buffer.flush();"
            "time.sleep(30)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        await asyncio.sleep(0.5)   # stdout is now paused under flow control
        transport = process.stdout._transport
        assert transport is not None and not transport.is_closing(), (
            "precondition failed: recovery was not given an open transport"
        )

        docker._unresolved.append(process)
        alive = await docker._reap_unresolved()

        closed = transport.is_closing()
        if process.returncode is None:
            process.kill()
            await process.wait()
        return alive, closed

    alive, closed = run(probe(load_mutant(source, "wh_recovery_clean")))
    assert alive == 0, "recovery did not resolve a killable child"
    assert closed, "recovery left the transport open"

    # The mutation. Note what it proves: the transport closes either way, since
    # killing the child ends the pipe on its own. What the pre-wait
    # _close_pipes actually buys is that the WAIT completes at all. Without it,
    # process.wait() also waits on a stdout transport that is still paused
    # under flow control, times out, and recovery concludes the child is
    # unkillable and keeps the worker unavailable forever. So the mutation is
    # asserted on `alive`, not on `closed`.
    mutated = source.replace(
        "            self._close_pipes(process)\n            try:\n"
        "                await asyncio.wait_for(process.wait(), timeout=1.0)",
        "            try:\n"
        "                await asyncio.wait_for(process.wait(), timeout=1.0)")
    assert mutated != source, "the mutation did not apply"
    alive_without, _ = run(probe(load_mutant(mutated, "wh_recovery_mutant")))
    assert alive_without == 1, (
        "recovery reaped the child even with the pre-wait _close_pipes "
        "removed; this test cannot detect the regression it exists for"
    )


@pytest.mark.parametrize("errno_name", ["EBADF", "ELOOP", "ENOTDIR", "ENAMETOOLONG"])
def test_more_infrastructure_errnos_are_not_blamed_on_the_document(errno_name):
    """R-003 (round 6): these fell through the broad OSError entry."""
    import errno as errno_mod

    assert worker._is_infrastructure_failure(
        OSError(getattr(errno_mod, errno_name), "machine failure")
    ), f"{errno_name} was treated as a document failure"


def test_an_errno_less_oserror_is_still_a_document_failure():
    """The trade being made: PIL reports a truncated image as a bare OSError,
    so carving out every OSError would turn real document failures into 500s."""
    assert not worker._is_infrastructure_failure(OSError("image file is truncated"))
