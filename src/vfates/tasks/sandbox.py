"""Isolated subprocess execution of untrusted Python.

Model-written code is never executed in the parent interpreter. Each program
runs as ``sys.executable -I`` in a fresh temporary working directory, with a
minimal environment, CPU-time, file-size and process-count limits applied in
``preexec_fn``, and a wall-clock timeout that kills the process group and any
descendants that have left it.

This is not a security boundary against a determined author. ``-I`` drops
user site-packages and ``PYTHONPATH``, but the child can still import
packages installed in the same virtualenv and open absolute paths (a write
to ``/tmp`` succeeds). Run training-time execution inside a container or VM.

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
* ``stdin`` is ``DEVNULL``. There is no filesystem jail.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Fork plus preexec_fn deadlocks if several threads fork while another holds
# the import lock. Serialize process creation; the children still run together.
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
    stays on under :func:`run_python`.
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


def run_python(
    code: str,
    *,
    timeout_s: float = 10.0,
    memory_mb: int = 1024,
    max_output_bytes: int = 65536,
) -> ExecResult:
    """Run ``code`` in a fresh subprocess. See the module docstring.

    Relative writes land in the temporary working directory, which is removed
    before this function returns. Absolute paths and ``..`` are not jailed.
    """
    if os.name != "posix":
        raise OSError("run_python requires a POSIX system (macOS or Linux)")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_output_bytes < 1:
        raise ValueError("max_output_bytes must be positive")

    started = time.monotonic()
    work = tempfile.TemporaryDirectory(prefix="vfates-sandbox-")
    timed_out = False
    overflow = [False]
    stdout_b = b""
    stderr_b = b""
    returncode = -1
    try:
        script = Path(work.name) / "main.py"
        script.write_text(code, encoding="utf-8")
        # Hugging Face tokenizers warns if the parent forks after it has loaded.
        # P6 grades in subprocesses after generation. Read at fork time.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Computed in the parent. preexec_fn must not spawn ``ps``: it runs
        # between fork and exec, where another fork can deadlock.
        nproc = _nproc_cap()
        with _FORK_LOCK:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(script)],
                cwd=work.name,
                env=_sandbox_env(work.name),
                stdin=subprocess.DEVNULL,
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

        readers = [
            threading.Thread(target=_read_out, daemon=True),
            threading.Thread(target=_read_err, daemon=True),
        ]
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


def run_many(
    programs: Sequence[str],
    *,
    max_workers: int | None = None,
    **kw: object,
) -> list[ExecResult]:
    """Run ``programs`` in a thread pool of :func:`run_python` subprocesses.

    ``max_workers`` defaults to ``os.cpu_count()``. Results keep input order.
    """
    if not programs:
        return []
    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(int(workers), len(programs)))
    results: list[ExecResult | None] = [None] * len(programs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_python, code, **cast(Any, kw)): i for i, code in enumerate(programs)
        }
        for fut in futures:
            results[futures[fut]] = fut.result()
    return [r for r in results if r is not None]
