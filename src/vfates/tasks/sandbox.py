"""Isolated subprocess execution of untrusted Python.

Model-written code is never executed in the parent interpreter. Code grading
runs ``code_grader.py --once`` as ``sys.executable -I`` in a fresh temporary
working directory, with a minimal environment, CPU-time, file-size and
process-count limits applied in ``preexec_fn``, and a wall-clock timeout
that kills the process group and any descendants that have left it.

This is not a security boundary against a determined author. ``-I`` drops
user site-packages and ``PYTHONPATH``, but the child can still import
packages installed in the same virtualenv. Filesystem checks on the
submission itself live in ``code_grader.py`` (an audit hook, and Landlock
on Linux). Run training-time execution inside a container or VM.

macOS and Linux limitations, in the code on purpose:

* ``RLIMIT_AS`` is not enforced on macOS (``setrlimit`` returns ``EINVAL``).
  The memory argument is applied only where the OS supports it (not Darwin).
* ``RLIMIT_NPROC`` is a per-user cap, not a budget for this process tree. A
  fixed cap of 64 is below a normal desktop's process count, and the child
  then cannot ``fork`` at all. The sandbox sets 64 when the user is already
  under that count, and otherwise the live process count plus 32. A fork
  bomb stops. It does not confine only the sandbox.
* A child that calls ``setsid`` leaves the process group. The timeout
  snapshots descendant pids before ``killpg`` and ``SIGKILL``s those pids.
  A grandchild that is reparented before that snapshot can still escape.

:class:`GraderPool` keeps one long-lived worker process per core. Workers are
``python -c`` servers, not ``multiprocessing`` spawn: spawn re-imports
``__main__``, which fails when the parent is stdin and would also re-execute
a training entrypoint. Each worker calls :func:`grade_job`.
``code_grader.py --once`` reads the tests and compares outputs. The process
that runs a submission is a zygote forked before that job is read, then a
per-test worker. Neither has the expected outputs. Workers do not share the
parent's fork lock, so a GRPO step grades in parallel.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Fork plus preexec_fn deadlocks if another thread holds the import lock.
# The lock is taken only while this process is multi-threaded, and only
# around Popen (not while the child runs). A single-threaded caller forks
# without waiting. A threaded caller still serializes Popen: that is the
# case the lock exists for. GraderPool workers are separate processes,
# each single-threaded between jobs, so their Popen calls run in parallel
# and do not take this lock.
_FORK_LOCK = threading.Lock()

# RLIMIT_NPROC is per real user id. 64 is the cap when the user is already
# under it. Otherwise allow a few forks past the live count (see the module
# docstring). Slack covers the grader's own fork plus a small test child.
_NPROC_FLOOR = 64
_NPROC_SLACK = 32


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one sandboxed program.

    ``ok`` is true only when the process exits 0, the wall clock is not hit,
    and neither stream exceeds the output cap. ``stdout`` and ``stderr`` are
    decoded with replacement and truncated to ``max_output_bytes`` bytes.
    """

    ok: bool
    returncode: int
    timed_out: bool
    stdout: str
    stderr: str
    seconds: float


def _live_process_count() -> int:
    """Processes visible to ``ps``. Used only to place the per-user nproc cap."""
    try:
        raw = subprocess.check_output(["ps", "-ax", "-o", "pid="], text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return _NPROC_FLOOR
    return max(1, sum(1 for line in raw.splitlines() if line.strip()))


def _nproc_cap() -> int:
    """Process cap for the child. See the module docstring."""
    live = _live_process_count()
    if live < _NPROC_FLOOR:
        return _NPROC_FLOOR
    return live + _NPROC_SLACK


def _apply_limits(timeout_s: float, memory_mb: int, max_output_bytes: int, nproc: int) -> None:
    """Resource limits for the child. Called from ``preexec_fn`` only."""
    cpu = max(1, math.ceil(timeout_s) + 1)
    resource_mod = __import__("resource")
    resource_mod.setrlimit(resource_mod.RLIMIT_CPU, (cpu, cpu))
    fsize = max(max_output_bytes, 32 * 1024 * 1024)
    with contextlib.suppress(ValueError, OSError):
        resource_mod.setrlimit(resource_mod.RLIMIT_FSIZE, (fsize, fsize))
    # RLIMIT_AS is not enforced on macOS (setrlimit returns EINVAL). Skip it
    # there; on Linux it caps the address space of the child and its forks.
    if sys.platform != "darwin" and memory_mb > 0:
        mem = memory_mb * 1024 * 1024
        with contextlib.suppress(ValueError, OSError):
            resource_mod.setrlimit(resource_mod.RLIMIT_AS, (mem, mem))
    if nproc > 0:
        with contextlib.suppress(ValueError, OSError):
            resource_mod.setrlimit(resource_mod.RLIMIT_NPROC, (nproc, nproc))


def _sandbox_env(workdir: str) -> dict[str, str]:
    """An environment that does not inherit the parent process.

    ``PYTHONHASHSEED`` is set for callers that drop ``-I``. Isolated mode
    implies ``-E``, which ignores ``PYTHON*`` variables, so hash randomisation
    stays on under the grader child.
    """
    return {
        "PATH": os.defpath,
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }


def _descendant_pids(root: int) -> list[int]:
    """Pids whose parent chain reaches ``root``, including across ``setsid``."""
    try:
        raw = subprocess.check_output(["ps", "-ax", "-o", "pid=,ppid="], text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict[int, list[int]] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    stack = list(children.get(root, []))
    victims: list[int] = []
    while stack:
        pid = stack.pop()
        victims.append(pid)
        stack.extend(children.get(pid, []))
    return victims


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill the process group and descendants that left it via ``setsid``.

    The snapshot is taken before ``killpg``. After the parent dies, a former
    child is reparented and can no longer be found by ppid.
    """
    root = proc.pid
    victims = _descendant_pids(root)
    try:
        os.killpg(root, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(OSError):
            proc.kill()
    extra: list[int] = []
    for pid in victims:
        extra.extend(_descendant_pids(pid))
    for pid in (*victims, *extra, root):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _read_capped(
    pipe,
    cap: int,
    proc: subprocess.Popen[bytes],
    overflow: list[bool],
) -> bytes:
    buf = bytearray()
    try:
        while len(buf) < cap:
            chunk = pipe.read(min(8192, cap - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
        if len(buf) >= cap:
            extra = pipe.read(1)
            if extra:
                overflow[0] = True
                _kill_tree(proc)
    finally:
        with contextlib.suppress(OSError):
            pipe.close()
    return bytes(buf[:cap])


def _fork_guard() -> Any:
    """The fork lock when another thread exists, else a no-op.

    ``preexec_fn`` runs between fork and exec. If another thread holds the
    import lock, the child deadlocks. A threaded caller is the case the lock
    exists for. A single-threaded caller does not wait on the lock.
    """
    if threading.active_count() > 1:
        return _FORK_LOCK
    return contextlib.nullcontext()


def _run_argv(
    argv: Sequence[str],
    *,
    timeout_s: float,
    memory_mb: int,
    max_output_bytes: int,
    stdin_bytes: bytes | None = None,
    workdir: str | None = None,
) -> ExecResult:
    """Run ``argv`` under the sandbox limits. ``stdin_bytes`` defaults to closed stdin."""
    if os.name != "posix":
        raise OSError("code grading requires a POSIX system (macOS or Linux)")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_output_bytes < 1:
        raise ValueError("max_output_bytes must be positive")

    started = time.monotonic()
    owned = workdir is None
    work = tempfile.TemporaryDirectory(prefix="vfates-sandbox-") if owned else None
    directory = work.name if work is not None else str(workdir)
    timed_out = False
    overflow = [False]
    stdout_b = b""
    stderr_b = b""
    returncode = -1
    try:
        # Hugging Face tokenizers warns if the parent forks after it has loaded.
        # P6 grades in subprocesses after generation. Read at fork time.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Computed in the parent. preexec_fn must not spawn ``ps``: it runs
        # between fork and exec, where another fork can deadlock.
        nproc = _nproc_cap()
        with _fork_guard():
            proc = subprocess.Popen(
                list(argv),
                cwd=directory,
                env=_sandbox_env(directory),
                stdin=subprocess.DEVNULL if stdin_bytes is None else subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=lambda: _apply_limits(timeout_s, memory_mb, max_output_bytes, nproc),
            )
        assert proc.stdout is not None and proc.stderr is not None
        out_box: dict[str, bytes] = {}
        err_box: dict[str, bytes] = {}

        def _read_out() -> None:
            out_box["b"] = _read_capped(proc.stdout, max_output_bytes, proc, overflow)

        def _read_err() -> None:
            err_box["b"] = _read_capped(proc.stderr, max_output_bytes, proc, overflow)

        def _write_in() -> None:
            assert proc.stdin is not None
            try:
                if stdin_bytes:
                    proc.stdin.write(stdin_bytes)
            except (BrokenPipeError, OSError):
                pass
            finally:
                with contextlib.suppress(OSError):
                    proc.stdin.close()

        readers = [
            threading.Thread(target=_read_out, daemon=True),
            threading.Thread(target=_read_err, daemon=True),
        ]
        if stdin_bytes is not None:
            readers.append(threading.Thread(target=_write_in, daemon=True))
        for reader in readers:
            reader.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                proc.wait(timeout=2)
        for reader in readers:
            reader.join(timeout=5)
        stdout_b = out_box.get("b", b"")
        stderr_b = err_box.get("b", b"")
        returncode = proc.returncode if proc.returncode is not None else -9
    finally:
        if work is not None:
            work.cleanup()

    seconds = time.monotonic() - started
    ok = returncode == 0 and not timed_out and not overflow[0]
    return ExecResult(
        ok=ok,
        returncode=int(returncode),
        timed_out=timed_out,
        stdout=stdout_b.decode("utf-8", "replace"),
        stderr=stderr_b.decode("utf-8", "replace"),
        seconds=seconds,
    )


_GRADER_PATH = str(Path(__file__).resolve().with_name("code_grader.py"))
# Files the grader itself writes (the solution and one test's stdin). The
# user program's own stdout cap is ``GradeJob.max_output_bytes``.
_GRADER_FILE_CAP = 64 * 1024 * 1024


@dataclass(frozen=True)
class GradeJob:
    """One program and the tests :mod:`vfates.tasks.code_grader` should run.

    ``tests`` are ``(input, output)`` pairs. For ``fn_name`` problems the
    input is the JSON argument list and the output is the JSON return value
    (see ``code_grader.decode_fn_input``). ``early_exit`` stops on the first
    failing test. The pass bit does not depend on order.
    """

    code: str
    tests: tuple[tuple[str, str], ...]
    fn_name: str | None = None
    per_test_timeout_s: float = 2.0
    wall_timeout_s: float = 30.0
    memory_mb: int = 512
    early_exit: bool = True
    max_output_bytes: int = 1_048_576

    def payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "fn_name": self.fn_name,
            "tests": [{"input": inp, "output": out} for inp, out in self.tests],
            "per_test_timeout_s": self.per_test_timeout_s,
            "wall_timeout_s": self.wall_timeout_s,
            "memory_mb": self.memory_mb,
            "early_exit": self.early_exit,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True)
class GradeResult:
    """Outcome of one :class:`GradeJob`.

    ``passed`` is true only when every test ran and matched. ``n_ran`` is
    smaller than ``n_tests`` when ``early_exit`` stops on a failure or the
    wall clock fires first. ``detail`` is a short stable label, not a traceback.
    """

    passed: bool
    n_tests: int
    n_ran: int
    first_fail: int | None
    detail: str
    timed_out: bool
    memory_limit_applied: bool
    n_denied: int = 0
    denied: tuple[tuple[str, str], ...] = ()
    fs_isolation: str = ""


def _parse_grade_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text.startswith("{") or not text.endswith("}"):
            continue
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict) and "passed" in loaded:
            return loaded
    return None


def grade_job(job: GradeJob) -> GradeResult:
    """Grade one job in a restricted ``code_grader.py --once`` child.

    That child forks a zygote before it reads the job, and the zygote forks
    once per test. This process only waits on ``--once``, so a threaded
    caller still takes the fork lock around this single Popen.
    """
    if job.per_test_timeout_s <= 0 or job.wall_timeout_s <= 0:
        raise ValueError("timeouts must be positive")
    payload = json.dumps(job.payload()).encode("utf-8")
    executed = _run_argv(
        [sys.executable, "-I", _GRADER_PATH, "--once"],
        timeout_s=float(job.wall_timeout_s) + 2.0,
        memory_mb=0,
        max_output_bytes=_GRADER_FILE_CAP,
        stdin_bytes=payload,
    )
    n_tests = len(job.tests)
    parsed = _parse_grade_stdout(executed.stdout)
    if executed.timed_out:
        return GradeResult(
            passed=False,
            n_tests=n_tests,
            n_ran=0,
            first_fail=0 if n_tests else None,
            detail="wall-clock timeout",
            timed_out=True,
            memory_limit_applied=False,
        )
    if parsed is None:
        tail = (executed.stderr or executed.stdout or "no grader result")[-300:]
        return GradeResult(
            passed=False,
            n_tests=n_tests,
            n_ran=0,
            first_fail=0 if n_tests else None,
            detail=f"harness: {tail}",
            timed_out=False,
            memory_limit_applied=False,
        )
    first = parsed.get("first_fail")
    return GradeResult(
        passed=bool(parsed.get("passed")),
        n_tests=int(parsed.get("n_tests", n_tests)),
        n_ran=int(parsed.get("n_ran", 0)),
        first_fail=None if first is None else int(first),
        detail=str(parsed.get("detail") or ""),
        timed_out=bool(parsed.get("timed_out")),
        memory_limit_applied=bool(parsed.get("memory_limit_applied")),
        n_denied=int(parsed.get("n_denied") or 0),
        denied=_denied_pairs(parsed.get("denied")),
        fs_isolation=str(parsed.get("fs_isolation") or ""),
    )


def _reply(body: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(body, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _grade_job_from_payload(data: Mapping[str, Any]) -> GradeJob:
    raw_tests = data.get("tests") or []
    tests: list[tuple[str, str]] = []
    for item in raw_tests:
        if not isinstance(item, Mapping):
            raise TypeError("grade test must be an object with input and output")
        tests.append((str(item.get("input", "")), str(item.get("output", ""))))
    fn = data.get("fn_name")
    fn_name = str(fn) if isinstance(fn, str) and fn.strip() else None
    return GradeJob(
        code=str(data.get("code") or ""),
        tests=tuple(tests),
        fn_name=fn_name,
        per_test_timeout_s=float(data.get("per_test_timeout_s", 2.0)),
        wall_timeout_s=float(data.get("wall_timeout_s", 30.0)),
        memory_mb=int(data.get("memory_mb", 512)),
        early_exit=bool(data.get("early_exit", True)),
        max_output_bytes=int(data.get("max_output_bytes", 1_048_576)),
    )


def _denied_pairs(raw: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        return ()
    pairs: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    return tuple(pairs)


def _grade_result_from_payload(data: Mapping[str, Any]) -> GradeResult:
    first = data.get("first_fail")
    return GradeResult(
        passed=bool(data.get("passed")),
        n_tests=int(data.get("n_tests", 0)),
        n_ran=int(data.get("n_ran", 0)),
        first_fail=None if first is None else int(first),
        detail=str(data.get("detail") or ""),
        timed_out=bool(data.get("timed_out")),
        memory_limit_applied=bool(data.get("memory_limit_applied")),
        n_denied=int(data.get("n_denied") or 0),
        denied=_denied_pairs(data.get("denied")),
        fs_isolation=str(data.get("fs_isolation") or ""),
    )


def serve_pool() -> None:
    """JSON-lines server used by :class:`GraderPool` workers.

    One object per line on stdin, one object per line on stdout. The process
    that runs a submission is a child of this one, never this one. This
    process holds expected outputs, so on Linux it is not dumpable.
    """
    from vfates.tasks.code_grader import set_non_dumpable

    set_non_dumpable()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"id": None, "ok": False, "error": f"JSONDecodeError: {exc}"})
            continue
        if not isinstance(req, dict):
            _reply({"id": None, "ok": False, "error": "request must be an object"})
            continue
        if req.get("op") == "stop":
            return
        ident = req.get("id")
        try:
            op = req.get("op")
            if op == "grade":
                result = grade_job(_grade_job_from_payload(req["job"]))
                body: dict[str, Any] = {"id": ident, "ok": True, "result": asdict(result)}
            else:
                raise RuntimeError(f"unknown pool op {op!r}")
        except Exception as exc:
            body = {"id": ident, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _reply(body)


class _PoolWorker:
    """One persistent ``python -c serve_pool`` process."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "from vfates.tasks.sandbox import serve_pool; serve_pool()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.stderr_tail = ""
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._err_thread.start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        chunks: list[str] = []
        for line in self.proc.stderr:
            chunks.append(line)
            if sum(len(chunk) for chunk in chunks) > 8000:
                chunks = chunks[-20:]
        self.stderr_tail = "".join(chunks)[-2000:]

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        if self.proc.poll() is not None:
            raise RuntimeError(f"grader worker exited {self.proc.returncode}: {self.stderr_tail}")
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.poll()
            raise RuntimeError(f"grader worker closed stdout (code {code}): {self.stderr_tail}")
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise RuntimeError("grader worker returned a non-object")
        return loaded

    def stop(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"op": "stop"}) + "\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=1)


class GraderPool:
    """Long-lived worker processes for :meth:`grade_many`.

    ``n_workers`` defaults to ``os.cpu_count()``. Workers are separate
    interpreters so they do not fork the training process and do not take
    the in-process fork lock. Results are deterministic and keep input order.
    """

    def __init__(self, n_workers: int | None = None) -> None:
        count = os.cpu_count() or 1 if n_workers is None else int(n_workers)
        if count < 1:
            raise ValueError("n_workers must be positive")
        self.n_workers = count
        self._closed = False
        self._queue: queue.Queue[Any] | None = None
        self._threads: list[threading.Thread] = []
        self._workers: list[_PoolWorker] = []
        self._start_lock = threading.Lock()
        atexit.register(self.close)

    def _ensure(self) -> None:
        with self._start_lock:
            if self._threads:
                return
            if self._closed:
                raise RuntimeError("GraderPool is closed")
            self._queue = queue.Queue()
            for _ in range(self.n_workers):
                thread = threading.Thread(target=self._loop, daemon=True)
                thread.start()
                self._threads.append(thread)

    def _loop(self) -> None:
        worker = _PoolWorker()
        with self._start_lock:
            self._workers.append(worker)
        assert self._queue is not None
        while True:
            item = self._queue.get()
            if item is None:
                worker.stop()
                return
            index, request, box = item
            try:
                reply = worker.request(request)
            except Exception as exc:
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            box.put((index, reply))

    def _map(self, requests: Sequence[Mapping[str, Any]], timeout_s: float) -> list[dict[str, Any]]:
        if not requests:
            return []
        self._ensure()
        assert self._queue is not None
        box: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
        for index, request in enumerate(requests):
            self._queue.put((index, request, box))
        results: list[dict[str, Any] | None] = [None] * len(requests)
        deadline = time.monotonic() + timeout_s
        got = 0
        while got < len(requests):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("grader pool timed out")
            try:
                index, reply = box.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("grader pool timed out") from exc
            if not reply.get("ok"):
                raise RuntimeError(str(reply.get("error") or "grader worker failed"))
            result = reply.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("grader worker returned no result object")
            results[int(index)] = result
            got += 1
        return [item for item in results if item is not None]

    def grade_many(self, jobs: Sequence[GradeJob]) -> list[GradeResult]:
        """Grade ``jobs`` in parallel. Results keep input order and match :func:`grade_job`."""
        if not jobs:
            return []
        wall = max(float(job.wall_timeout_s) for job in jobs)
        batches = math.ceil(len(jobs) / self.n_workers)
        timeout_s = wall * batches + 30.0
        requests = [
            {"op": "grade", "id": index, "job": job.payload()} for index, job in enumerate(jobs)
        ]
        payloads = self._map(requests, timeout_s)
        return [_grade_result_from_payload(item) for item in payloads]

    def close(self) -> None:
        """Stop workers. Safe to call more than once."""
        with self._start_lock:
            if self._closed and not self._threads:
                return
            self._closed = True
            threads = list(self._threads)
            workers = list(self._workers)
            job_queue = self._queue
        if job_queue is not None:
            for _ in threads:
                job_queue.put(None)
        for thread in threads:
            thread.join(timeout=3)
        for worker in workers:
            if worker.proc.poll() is None:
                worker.proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    worker.proc.wait(timeout=1)
        for thread in threads:
            thread.join(timeout=1)
        with self._start_lock:
            self._threads = []
            self._workers = []
            self._queue = None

    def __enter__(self) -> GraderPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def measure_throughput(
    n_programs: int = 512,
    n_tests: int = 16,
    n_workers: int | None = None,
) -> dict[str, float]:
    """Programs per second for ``n_programs`` sum-stdin jobs of ``n_tests`` each.

    Every test passes, so this is the no-early-exit cost. The pool is closed
    before the function returns.
    """
    if n_programs < 1 or n_tests < 1:
        raise ValueError("n_programs and n_tests must be positive")
    code = "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()))\n"
    tests = tuple((f"{index} {index + 1}\n", f"{2 * index + 1}\n") for index in range(n_tests))
    job = GradeJob(
        code=code,
        tests=tests,
        per_test_timeout_s=2.0,
        wall_timeout_s=float(n_tests) + 5.0,
        memory_mb=256,
        early_exit=False,
    )
    jobs = [job] * n_programs
    pool = GraderPool(n_workers=n_workers)
    started = time.perf_counter()
    try:
        results = pool.grade_many(jobs)
    finally:
        pool.close()
    elapsed = time.perf_counter() - started
    passed = sum(bool(result.passed) for result in results)
    return {
        "n_programs": float(n_programs),
        "n_tests": float(n_tests),
        "n_workers": float(pool.n_workers),
        "passed": float(passed),
        "seconds": elapsed,
        "programs_per_s": n_programs / elapsed if elapsed > 0 else 0.0,
    }
