"""Standalone grader for one Python program against a list of tests.

Invoked as ``python -I -s path/code_grader.py --once`` with a JSON job on
stdin. The process imports only the standard library, so a grading child
never imports vfates or torch. ``GraderPool`` in :mod:`vfates.tasks.sandbox`
is the long-lived parent; this file is the restricted child.

Comparison, deliberately narrower than FlagOpen's ``compare_std_results``:

* Standard output passes when the strings are equal after stripping each
  end, or when their whitespace-split tokens are equal (order preserved).
  If every token is a finite float, pairs may also match with numpy's
  default ``allclose`` (``rtol=1e-05``, ``atol=1e-08``).
* Call-based problems (``fn_name`` set) compare return values. Tuples are
  treated as lists. Digit-string dict keys are treated as ints. Floats use
  the same closeness rule. Bool is not an int.
* Not reproduced, because they accept wrong programs: TACO's order-free
  set-of-tokens check, and ``return_value == expected[0]`` whenever the
  expected value is a non-empty list.

``--once`` forks a zygote before it reads the job. The zygote never sees
expected outputs. For each test it forks a worker that receives only paths
(the solution, this test's stdin or args file, and limits). The worker runs
the submission. ``--once`` compares the stdout or the JSON return value with
the expected output, and it never executes the submission. A test directory's
parent holds only that test's files, and only while that test runs.
"""

from __future__ import annotations

import ast
import contextlib
import json
import math
import os
import runpy
import shutil
import signal
import sys
import tempfile
import time
from typing import Any

# numpy allclose defaults. Pinned so a float token match does not drift.
_RTOL = 1e-05
_ATOL = 1e-08

_PASS = "pass"
_MISMATCH = "mismatch"
_TIMEOUT = "timeout"
_ERROR = "error"
_OUTPUT_CAP = "output cap"
_KILLED = "killed"
_LOAD = "load"


def memory_limits_supported() -> bool:
    """``RLIMIT_AS`` is not enforced on macOS (``setrlimit`` returns ``EINVAL``)."""
    return sys.platform != "darwin"


def _close(left: float, right: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return abs(left - right) <= (_ATOL + _RTOL * abs(right))


def _float_token(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def stdio_equal(got: str, expected: str) -> bool:
    """Whitespace-normalised token comparison, plus float tokens.

    ``str.split`` collapses every run of whitespace and drops the empty
    edges, which is the TACO/APPS token normalisation without their
    order-free set comparison.
    """
    if got.strip() == expected.strip():
        return True
    got_tokens = got.split()
    expected_tokens = expected.split()
    if got_tokens == expected_tokens:
        return True
    if len(got_tokens) != len(expected_tokens) or not got_tokens:
        return False
    got_floats = [_float_token(token) for token in got_tokens]
    expected_floats = [_float_token(token) for token in expected_tokens]
    pairs: list[tuple[float, float]] = []
    for left, right in zip(got_floats, expected_floats):
        if left is None or right is None:
            return False
        pairs.append((left, right))
    return all(_close(left, right) for left, right in pairs)


def _int_key(text: str) -> bool:
    if not text:
        return False
    body = text[1:] if text[0] == "-" else text
    return bool(body) and body.isdigit()


def _canon(value: Any) -> Any:
    """Tuples become lists. Digit-string dict keys become ints."""
    if isinstance(value, tuple):
        return [_canon(item) for item in value]
    if isinstance(value, list):
        return [_canon(item) for item in value]
    if isinstance(value, dict):
        mapped: dict[Any, Any] = {}
        for key, item in value.items():
            new_key: Any = int(key) if isinstance(key, str) and _int_key(key) else key
            mapped[new_key] = _canon(item)
        return mapped
    return value


def _deep_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            return _close(float(left), float(right))
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right or stdio_equal(left, right)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_deep_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_deep_equal(left[key], right[key]) for key in left)
    try:
        return bool(left == right)
    except Exception:
        return False


def returns_equal(got: Any, expected: Any) -> bool:
    """Structural equality of a function return value and the expected JSON."""
    return _deep_equal(_canon(got), _canon(expected))


def _split_top_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    quote: str | None = None
    escape = False
    for index, char in enumerate(text):
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def named_call_args(text: str) -> list[Any] | None:
    """``nums = [1, 2], target = 9`` into argument values, or ``None``."""
    if "=" not in text:
        return None
    values: list[Any] = []
    for part in _split_top_commas(text):
        piece = part.strip()
        if not piece or "=" not in piece:
            return None
        rhs = piece.split("=", 1)[1].strip()
        try:
            values.append(ast.literal_eval(rhs))
        except (ValueError, SyntaxError, MemoryError, TypeError):
            return None
    return values or None


def decode_fn_input(raw: Any) -> list[Any]:
    """Argument list for a ``fn_name`` test.

    A JSON list is the argument list (TACO/APPS call-based). A list of JSON
    strings is decoded elementwise. Newline-separated JSON values are one
    argument per line. ``name = literal`` text is a statement example.
    """
    if isinstance(raw, list):
        if raw and all(isinstance(item, str) for item in raw):
            decoded: list[Any] = []
            ok = True
            for item in raw:
                try:
                    decoded.append(json.loads(item))
                except json.JSONDecodeError:
                    ok = False
                    break
            if ok:
                return decoded
        return list(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            loaded: Any = json.loads(text)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, list):
            return loaded
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            try:
                return [json.loads(line) for line in lines]
            except json.JSONDecodeError:
                named = named_call_args(text)
                if named is not None:
                    return named
                raise
        named = named_call_args(text)
        if named is not None:
            return named
        if loaded is not None:
            return [loaded]
        raise ValueError("cannot decode function arguments")
    return [raw]


def decode_fn_output(raw: Any) -> Any:
    """Expected return value. JSON text is parsed; other text stays a string."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        named = named_call_args(text)
        if named is not None and len(named) == 1:
            return named[0]
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, TypeError):
            return raw


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        mapped: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, bool):
                text = "true" if key else "false"
            elif isinstance(key, (int, float, str)):
                text = str(key)
            else:
                raise TypeError(f"dict key {type(key).__name__}")
            mapped[text] = _jsonable(item)
        return mapped
    raise TypeError(type(value).__name__)


def _lookup_fn(namespace: dict[str, Any], fn_name: str) -> Any:
    found = namespace.get(fn_name)
    if callable(found):
        return found
    solution = namespace.get("Solution")
    if isinstance(solution, type) and hasattr(solution, fn_name):
        return getattr(solution(), fn_name)
    raise KeyError(fn_name)


def _block_network() -> None:
    """Reject new sockets. Not a network namespace: ctypes can still connect."""
    import socket

    class _Blocked(socket.socket):
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise OSError("network is disabled")

    socket.socket = _Blocked  # type: ignore[misc]

    def _refused(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("network is disabled")

    socket.create_connection = _refused
    if hasattr(socket, "create_server"):
        socket.create_server = _refused


def _apply_child_limits(timeout_s: float, memory_mb: int, max_output_bytes: int) -> bool:
    """Resource limits inside the forked test child. Returns whether AS was set."""
    import resource

    cpu = max(1, math.ceil(timeout_s) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    fsize = max(int(max_output_bytes), 32 * 1024)
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    applied = False
    if memory_mb > 0 and memory_limits_supported():
        mem = int(memory_mb) * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            applied = True
        except (ValueError, OSError):
            applied = False
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
    # RLIMIT_NPROC is per user, not per tree. The sandbox parent sets it.
    # A second, smaller cap here is already below a desktop's process count
    # and would make the child unable to fork at all.
    return applied


def _descendant_pids(root: int) -> list[int]:
    import subprocess

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


def _kill_group(pid: int) -> None:
    victims = _descendant_pids(pid)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    for victim in victims:
        with contextlib.suppress(OSError):
            os.kill(victim, signal.SIGKILL)


def _exit_child(code: int) -> None:
    """Flush Python buffers. ``os._exit`` skips them, so a correct print would be empty."""
    with contextlib.suppress(Exception):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


def _wait_pid(pid: int, timeout_s: float) -> str:
    """``ok``, ``error``, ``killed``, or ``timeout``. Reaps the child."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            waited, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return _ERROR
        if waited == pid:
            if os.WIFEXITED(status):
                code = os.WEXITSTATUS(status)
                if code == 0:
                    return "ok"
                if code == 2:
                    return _LOAD
                if code == 3:
                    return _OUTPUT_CAP
                return _ERROR
            if os.WIFSIGNALED(status):
                return _KILLED
            return _ERROR
        if time.monotonic() >= deadline:
            _kill_group(pid)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
            return _TIMEOUT
        time.sleep(0.002)


def _redirect(stdin_path: str, stdout_path: str, stderr_path: str) -> None:
    fin = os.open(stdin_path, os.O_RDONLY)
    fout = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    ferr = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(fin, 0)
    os.dup2(fout, 1)
    os.dup2(ferr, 2)
    if fin > 2:
        os.close(fin)
    if fout > 2:
        os.close(fout)
    if ferr > 2:
        os.close(ferr)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("grader pipe write failed")
        view = view[written:]


# The zygote reads exactly this many bytes, then forks. A shorter read would
# split the header. A longer read would copy this test's paths into the zygote,
# and the next test's worker would inherit them.
_HEADER_BYTES = 64


def _pack_header(payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(raw) >= _HEADER_BYTES:
        raise RuntimeError("grader header is too long")
    return raw + b" " * (_HEADER_BYTES - 1 - len(raw)) + b"\n"


def _read_exact(fd: int, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = os.read(fd, size - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _stdio_child(code_path: str) -> None:
    runpy.run_path(code_path, run_name="__main__")


def _fn_child(code_path: str, fn_name: str, args_path: str, result_path: str) -> None:
    namespace: dict[str, Any] = {"__name__": "solution"}
    with open(code_path, encoding="utf-8") as handle:
        source = handle.read()
    try:
        exec(compile(source, code_path, "exec"), namespace)
        fn = _lookup_fn(namespace, fn_name)
    except KeyError:
        _exit_child(2)
    except Exception:
        _exit_child(1)
    with open(args_path, encoding="utf-8") as handle:
        args = json.loads(handle.read())
    if not isinstance(args, list):
        _exit_child(2)
    try:
        value = fn(*args)
    except Exception:
        _exit_child(1)
    try:
        payload = {"value": _jsonable(value)}
    except (TypeError, ValueError):
        _exit_child(3)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _read_worker_line(fd: int, limit: int = 1_000_000) -> bytes:
    """Read the worker command. This process exits afterwards, so extra bytes die with it.

    The parent writes the next test only after the ack, so the pipe holds this
    command and nothing after it.
    """
    buf = bytearray()
    while b"\n" not in buf and len(buf) <= limit:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        buf.extend(chunk)
    newline = buf.find(b"\n")
    if newline >= 0:
        return bytes(buf[:newline])
    if len(buf) > limit:
        raise RuntimeError("grader command is too long")
    return bytes(buf)


# Imports a submission must not load. Normal stdlib imports stay allowed.
_DENIED_IMPORTS = frozenset(
    {
        "ctypes",
        "_ctypes",
        "cffi",
        "_cffi_backend",
        "_posixsubprocess",
        "multiprocessing",
        "_interpreters",
        "_xxsubinterpreters",
        "mmap",
    }
)
_DENIED_EVENTS = frozenset(
    {
        "os.chdir",
        "os.chroot",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.forkpty",
        "pty.spawn",
        "subprocess.Popen",
    }
)
_DEV_READ = frozenset({"/dev/null", "/dev/urandom"})

# Landlock ABI 1 file accesses. Later ABIs add REFER and TRUNCATE.
_LL_EXECUTE = 1 << 0
_LL_WRITE = 1 << 1
_LL_READ = 1 << 2
_LL_READDIR = 1 << 3
_LL_REMOVE_DIR = 1 << 4
_LL_REMOVE_FILE = 1 << 5
_LL_MAKE_CHAR = 1 << 6
_LL_MAKE_DIR = 1 << 7
_LL_MAKE_REG = 1 << 8
_LL_MAKE_SOCK = 1 << 9
_LL_MAKE_FIFO = 1 << 10
_LL_MAKE_BLOCK = 1 << 11
_LL_MAKE_SYM = 1 << 12
_LL_REFER = 1 << 13
_LL_TRUNCATE = 1 << 14
_LL_ABI1 = (
    _LL_EXECUTE
    | _LL_WRITE
    | _LL_READ
    | _LL_READDIR
    | _LL_REMOVE_DIR
    | _LL_REMOVE_FILE
    | _LL_MAKE_CHAR
    | _LL_MAKE_DIR
    | _LL_MAKE_REG
    | _LL_MAKE_SOCK
    | _LL_MAKE_FIFO
    | _LL_MAKE_BLOCK
    | _LL_MAKE_SYM
)
_LL_CREATE = 444
_LL_ADD = 445
_LL_RESTRICT = 446
_LL_VERSION_FLAG = 1
_LL_RULE_PATH = 1
_PR_SET_DUMPABLE = 4
_PR_SET_NO_NEW_PRIVS = 38


def set_non_dumpable() -> bool:
    """Linux: ``PR_SET_DUMPABLE`` 0 so ``/proc/<pid>/mem`` and ``/proc/<pid>/fd`` stay closed.

    Other platforms do nothing. A same-uid process can open those files when
    the flag is 1. Yama ``ptrace_scope`` does not cover that read.
    """
    if sys.platform != "linux":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        libc.prctl.restype = ctypes.c_int
        return int(libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0)) == 0
    except Exception:
        return False


def _realpath(path: str) -> str:
    return os.path.realpath(path)


def _within(path: str, root: str) -> bool:
    path = path.rstrip(os.sep) or os.sep
    root = root.rstrip(os.sep) or os.sep
    return path == root or path.startswith(root + os.sep)


def _interpreter_roots() -> list[str]:
    """Stdlib and site-packages trees. Not the home directory and not the HF cache."""
    home = _realpath(os.path.expanduser("~"))
    cache = _realpath(os.path.join(home, ".cache", "huggingface"))
    banned = {"/", home, cache}
    roots: list[str] = []
    for raw in (getattr(sys, "base_prefix", ""), getattr(sys, "prefix", "")):
        if not raw:
            continue
        real = _realpath(raw)
        if real in banned or _within(real, cache):
            continue
        if real not in roots:
            roots.append(real)
    return roots


def _guard_state() -> dict[str, Any]:
    return {
        "busy": False,
        "fd": -1,
        "n": 0,
        "slot": "",
        "solution_dir": "",
        "roots": (),
    }


_GUARD = _guard_state()


def _short_arg(value: Any) -> str:
    if isinstance(value, bytes):
        text = os.fsdecode(value)
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value
    elif isinstance(value, os.PathLike):
        text = os.fspath(value)
    else:
        text = ""
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return text[:160]


def _record_denial(event: str, arg: str) -> None:
    fd = int(_GUARD["fd"])
    if fd < 0:
        return
    n = int(_GUARD["n"]) + 1
    _GUARD["n"] = n
    with contextlib.suppress(OSError):
        os.pwrite(fd, f"count {n:08d}\n".encode(), 0)
        if n <= 8:
            os.write(fd, f"{event}\t{arg}\n".encode())


def _resolved_path(value: Any) -> str | None:
    if isinstance(value, int) or value is None:
        return None
    if isinstance(value, bytes):
        text = os.fsdecode(value)
    elif isinstance(value, os.PathLike):
        text = os.fspath(value)
    elif isinstance(value, str):
        text = value
    else:
        raise PermissionError("unreadable path")
    if not text:
        raise PermissionError("empty path")
    return _realpath(text)


def _read_allowed(path: str) -> bool:
    if path in _DEV_READ:
        return True
    if _within(path, str(_GUARD["slot"])):
        return True
    solution_dir = str(_GUARD["solution_dir"])
    if solution_dir and _within(path, solution_dir):
        return True
    return any(_within(path, root) for root in _GUARD["roots"])


def _writing(mode: Any, flags: Any) -> bool:
    if isinstance(mode, str) and any(char in mode for char in "wxa+"):
        return True
    if isinstance(flags, int):
        accmode = getattr(os, "O_ACCMODE", 3)
        if (flags & accmode) != os.O_RDONLY:
            return True
    return False


def _glob_prefix(pattern: str) -> str:
    if not pattern:
        return _realpath(os.getcwd())
    pieces: list[str] = []
    absolute = pattern.startswith(os.sep)
    for part in pattern.split(os.sep):
        if any(char in part for char in "*?["):
            break
        pieces.append(part)
    if absolute:
        prefix = os.sep + os.sep.join(piece for piece in pieces if piece)
    elif pieces:
        prefix = os.path.join(*pieces)
    else:
        prefix = "."
    return _realpath(prefix or os.sep)


def _deny_reason(event: str, args: tuple[Any, ...]) -> str | None:
    if event.startswith("socket.") or event.startswith("ctypes.") or event.startswith("mmap."):
        return _short_arg(args[0] if args else "")
    if (
        event in _DENIED_EVENTS
        or event.startswith("os.posix_spawn")
        or event.startswith("os.spawn")
    ):
        return _short_arg(args[0] if args else "")
    if event in {"os.kill", "os.killpg"}:
        pid = args[0] if args else None
        if pid in (0, os.getpid()):
            return None
        return _short_arg(pid)
    if event == "signal.pthread_kill":
        import _thread

        tid = args[0] if args else None
        if tid == _thread.get_ident():
            return None
        return _short_arg(tid)
    if event == "import":
        name = str(args[0] if args else "")
        top = name.split(".", 1)[0]
        if top in _DENIED_IMPORTS or name in _DENIED_IMPORTS:
            return name
        return None
    if event in {"os.listdir", "os.scandir", "glob.glob"}:
        listed: str | None
        raw: Any = ""
        if event == "glob.glob":
            listed = _glob_prefix(str(args[0] if args else ""))
        else:
            raw = args[0] if args else None
            listed = _realpath(os.getcwd()) if raw in (None, "") else _resolved_path(raw)
            if listed is None:
                listed = _realpath(os.getcwd())
        if listed is not None and _within(listed, str(_GUARD["slot"])):
            return None
        return _short_arg(listed if listed is not None else raw)
    if event == "open":
        opened = _resolved_path(args[0] if args else None)
        if opened is None:
            return None
        if opened in _DEV_READ:
            return None
        mode = args[1] if len(args) > 1 else ""
        flags = args[2] if len(args) > 2 else None
        if _writing(mode, flags):
            if _within(opened, str(_GUARD["slot"])):
                return None
            return _short_arg(opened)
        if _read_allowed(opened):
            return None
        return _short_arg(opened)
    return None


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if _GUARD["busy"]:
        return
    _GUARD["busy"] = True
    try:
        try:
            reason = _deny_reason(event, args)
        except PermissionError as exc:
            reason = _short_arg(str(exc)) or event
        if reason is None:
            return
        _record_denial(event, reason)
        raise PermissionError(reason or event)
    finally:
        _GUARD["busy"] = False


def _landlock_handled(abi: int) -> int:
    handled = _LL_ABI1
    if abi >= 2:
        handled |= _LL_REFER
    if abi >= 3:
        handled |= _LL_TRUNCATE
    return handled


def _landlock_apply(slot: str, solution_dir: str, roots: list[str]) -> bool:
    """Restrict this process with Landlock. False when the kernel cannot."""
    if sys.platform != "linux" or not roots:
        return False
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long

        def call(number: int, *call_args: Any) -> int:
            return int(libc.syscall(ctypes.c_long(number), *call_args))

        abi = call(
            _LL_CREATE, ctypes.c_void_p(0), ctypes.c_size_t(0), ctypes.c_uint32(_LL_VERSION_FLAG)
        )
        if abi < 1:
            return False

        class _RulesetAttr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64)]

        class _PathAttr(ctypes.Structure):
            _pack_ = 1
            _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

        handled = _landlock_handled(abi)
        attr = _RulesetAttr(handled)
        ruleset = call(
            _LL_CREATE,
            ctypes.byref(attr),
            ctypes.c_size_t(ctypes.sizeof(attr)),
            ctypes.c_uint32(0),
        )
        if ruleset < 0:
            return False
        read_exec = _LL_READ | _LL_READDIR | _LL_EXECUTE
        slot_access = (
            _LL_READ
            | _LL_READDIR
            | _LL_WRITE
            | _LL_MAKE_REG
            | _LL_MAKE_DIR
            | _LL_REMOVE_FILE
            | _LL_TRUNCATE
        )
        slot_access &= handled
        grants: list[tuple[str, int]] = [(root, read_exec & handled) for root in roots]
        grants.append((slot, slot_access))
        if solution_dir:
            grants.append((solution_dir, (_LL_READ | _LL_READDIR) & handled))
        for device, access in (("/dev/null", _LL_READ | _LL_WRITE), ("/dev/urandom", _LL_READ)):
            if os.path.exists(device):
                grants.append((device, access & handled))
        opened: list[int] = []
        try:
            for path, access in grants:
                if access == 0:
                    continue
                try:
                    flags = os.O_RDONLY
                    if hasattr(os, "O_PATH"):
                        flags = os.O_PATH
                    if hasattr(os, "O_CLOEXEC"):
                        flags |= os.O_CLOEXEC
                    fd = os.open(path, flags)
                except OSError:
                    return False
                opened.append(fd)
                path_attr = _PathAttr(access, fd)
                added = call(
                    _LL_ADD,
                    ctypes.c_int(ruleset),
                    ctypes.c_int(_LL_RULE_PATH),
                    ctypes.byref(path_attr),
                    ctypes.c_uint32(0),
                )
                if added != 0:
                    return False
            libc.prctl.argtypes = [
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            libc.prctl.restype = ctypes.c_int
            if int(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) != 0:
                return False
            return call(_LL_RESTRICT, ctypes.c_int(ruleset), ctypes.c_uint32(0)) == 0
        finally:
            for fd in opened:
                os.close(fd)
            os.close(ruleset)
    except Exception:
        return False
    finally:
        sys.modules.pop("ctypes", None)
        sys.modules.pop("_ctypes", None)


def _arm_worker(work: str, code_path: str) -> None:
    """Landlock when the kernel has it, then an audit hook that cannot be removed."""
    sys.dont_write_bytecode = True
    slot = _realpath(os.path.dirname(work))
    solution = _realpath(code_path)
    roots = _interpreter_roots()
    fd = os.open(
        os.path.join(slot, "denied.log"),
        os.O_CREAT | os.O_RDWR | os.O_TRUNC,
        0o600,
    )
    level = "audit_hook"
    if _landlock_apply(slot, os.path.dirname(solution), roots):
        level = "landlock+audit_hook"
    os.write(fd, f"count {0:08d}\nlevel {level}\n".encode())
    _GUARD["fd"] = fd
    _GUARD["n"] = 0
    _GUARD["slot"] = slot
    _GUARD["solution_dir"] = os.path.dirname(solution)
    _GUARD["roots"] = tuple(roots)
    _GUARD["busy"] = False
    sys.addaudithook(_audit_hook)


def _read_denials(slot: str) -> tuple[int, list[list[str]], str]:
    path = os.path.join(slot, "denied.log")
    count = 0
    denied: list[list[str]] = []
    level = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = line.rstrip("\n")
                if text.startswith("count "):
                    with contextlib.suppress(ValueError):
                        count = int(text.split()[1])
                    continue
                if text.startswith("level "):
                    level = text.split(" ", 1)[1].strip()
                    continue
                event, _, arg = text.partition("\t")
                if event and len(denied) < 8:
                    denied.append([event, arg[:160]])
    except OSError:
        return 0, [], ""
    if count < len(denied):
        count = len(denied)
    return count, denied, level


def _worker_main(cmd_fd: int) -> None:
    """Run one submission. The command has paths and limits, not expected output."""
    raw = _read_worker_line(cmd_fd)
    os.close(cmd_fd)
    cmd = json.loads(raw.decode("utf-8"))
    os.setsid()
    os.chdir(str(cmd["work"]))
    _redirect(str(cmd["stdin_path"]), str(cmd["stdout_path"]), str(cmd["stderr_path"]))
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(write_through=True)  # type: ignore[union-attr]
    _apply_child_limits(
        float(cmd["timeout_s"]),
        int(cmd["memory_mb"]),
        int(cmd["max_output_bytes"]),
    )
    _block_network()
    _arm_worker(str(cmd["work"]), str(cmd["code_path"]))
    fn_name = cmd.get("fn_name") or None
    if isinstance(fn_name, str) and fn_name.strip():
        _fn_child(str(cmd["code_path"]), fn_name, str(cmd["args_path"]), str(cmd["result_path"]))
    else:
        _stdio_child(str(cmd["code_path"]))
    _exit_child(0)


def _zygote_main(cmd_fd: int, ack_fd: int) -> None:
    """Fork one worker per test. This process never reads a test command.

    It is forked before ``--once`` reads the job, so expected outputs and
    other tests' inputs are not in its address space. Each worker is a fork
    of this process at a moment when the only extra local is the timeout
    header, then the worker reads its own command.
    """
    while True:
        raw = _read_exact(cmd_fd, _HEADER_BYTES)
        if len(raw) < _HEADER_BYTES:
            os._exit(0)
        meta = json.loads(raw.decode("utf-8"))
        if meta.get("op") == "stop":
            os._exit(0)
        timeout_s = float(meta["timeout_s"])
        pid = os.fork()
        if pid == 0:
            os.close(ack_fd)
            try:
                _worker_main(cmd_fd)
            except SystemExit as exc:
                code = exc.code
                _exit_child(0 if code in (None, 0) else int(code) if isinstance(code, int) else 1)
            except BaseException:
                _exit_child(1)
            _exit_child(1)
        status = _wait_pid(pid, timeout_s)
        _write_all(ack_fd, (json.dumps({"status": status}) + "\n").encode("utf-8"))


class _Runner:
    """Pipe to a zygote that was forked before the job JSON was read."""

    def __init__(self, pid: int, cmd_fd: int, ack_fd: int) -> None:
        self.pid = pid
        self._cmd_fd = cmd_fd
        self._ack_fd = ack_fd
        self._ack = b""

    def run(self, cmd: dict[str, Any], timeout_s: float) -> str:
        header = _pack_header({"op": "run", "timeout_s": timeout_s})
        body = (json.dumps(cmd) + "\n").encode("utf-8")
        _write_all(self._cmd_fd, header + body)
        while b"\n" not in self._ack:
            chunk = os.read(self._ack_fd, 4096)
            if not chunk:
                raise RuntimeError("grader zygote exited")
            self._ack += chunk
        line, _, rest = self._ack.partition(b"\n")
        self._ack = rest
        payload = json.loads(line.decode("utf-8"))
        return str(payload.get("status") or _ERROR)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            _write_all(self._cmd_fd, _pack_header({"op": "stop"}))
        with contextlib.suppress(OSError):
            os.close(self._cmd_fd)
        deadline = time.monotonic() + 1.0
        while True:
            try:
                pid, _status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == self.pid:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.002)
        with contextlib.suppress(OSError):
            os.kill(self.pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(self.pid, 0)


def start_runner() -> _Runner:
    """Fork the zygote. Call this before reading the job from stdin."""
    cmd_r, cmd_w = os.pipe()
    ack_r, ack_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(cmd_w)
        os.close(ack_r)
        with contextlib.suppress(OSError):
            os.close(0)
        try:
            _zygote_main(cmd_r, ack_w)
        finally:
            os._exit(0)
    os.close(cmd_r)
    os.close(ack_w)
    return _Runner(pid, cmd_w, ack_r)


def _read_text(path: str, limit: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _judge_stdio(status: str, stdout: str, expected: str) -> tuple[str, str]:
    if status == _OUTPUT_CAP:
        return _OUTPUT_CAP, _OUTPUT_CAP
    if status == "ok":
        if stdio_equal(stdout, expected):
            return _PASS, "ok"
        return _MISMATCH, _MISMATCH
    if status == _TIMEOUT:
        return _TIMEOUT, _TIMEOUT
    if status == _KILLED:
        return _KILLED, _KILLED
    if status == _LOAD:
        return _LOAD, _LOAD
    return _ERROR, _ERROR


def _judge_call(status: str, result_path: str, expected: Any) -> tuple[str, str]:
    if status == _LOAD:
        return _LOAD, _LOAD
    if status == _TIMEOUT:
        return _TIMEOUT, _TIMEOUT
    if status == _KILLED:
        return _KILLED, _KILLED
    if status == _OUTPUT_CAP:
        return _OUTPUT_CAP, _OUTPUT_CAP
    if status != "ok":
        return _ERROR, _ERROR
    try:
        with open(result_path, encoding="utf-8") as handle:
            payload = json.loads(handle.read())
        got = payload["value"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return _ERROR, _ERROR
    if returns_equal(got, expected):
        return _PASS, "ok"
    return _MISMATCH, _MISMATCH


def run_job(job: dict[str, Any], runner: _Runner) -> dict[str, Any]:
    """Compare tests in this process. ``runner`` executes the submission.

    ``runner`` must already have been forked, before this job was read. One
    test directory exists at a time. Its parent contains only that test's
    files. The solution file is outside that directory. Expected outputs stay
    in this process and are not written to disk.
    """
    tests = list(job.get("tests") or [])
    early = bool(job.get("early_exit", True))
    per_test = float(job.get("per_test_timeout_s") or 2.0)
    memory_mb = int(job.get("memory_mb") or 0)
    max_output = int(job.get("max_output_bytes") or 65536)
    fn_name = job.get("fn_name") or None
    if isinstance(fn_name, str) and not fn_name.strip():
        fn_name = None
    code = str(job.get("code") or "")
    wall = job.get("wall_timeout_s")
    deadline = None if wall is None else time.monotonic() + float(wall)
    memory_applied = bool(memory_mb > 0 and memory_limits_supported())
    if per_test <= 0:
        raise ValueError("per_test_timeout_s must be positive")
    n_ran = 0
    first_fail: int | None = None
    detail = "ok"
    timed_out = False
    n_denied = 0
    denied: list[list[str]] = []
    fs_isolation = ""
    if not tests:
        return {
            "passed": False,
            "n_tests": 0,
            "n_ran": 0,
            "first_fail": None,
            "detail": "no tests",
            "timed_out": False,
            "memory_limit_applied": memory_applied,
            "n_denied": 0,
            "denied": [],
            "fs_isolation": "",
        }
    code_dir = tempfile.mkdtemp(prefix="vfates-code-")
    job_root = tempfile.mkdtemp(prefix="vfates-job-")
    try:
        code_path = os.path.join(code_dir, "solution.py")
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(code)
        for index, test in enumerate(tests):
            if deadline is not None and time.monotonic() >= deadline:
                first_fail = index
                detail = "wall-clock timeout"
                timed_out = True
                break
            timeout = per_test
            if deadline is not None:
                timeout = min(per_test, max(0.05, deadline - time.monotonic()))
            slot = tempfile.mkdtemp(prefix="slot-", dir=job_root)
            work = os.path.join(slot, "work")
            os.mkdir(work, 0o700)
            try:
                raw_in = "" if test.get("input") is None else str(test.get("input"))
                raw_out = "" if test.get("output") is None else str(test.get("output"))
                args_path = os.path.join(work, "args.json")
                result_path = os.path.join(work, "result.json")
                stdin_path = os.path.join(work, "stdin.txt")
                stdout_path = os.path.join(work, "stdout.txt")
                stderr_path = os.path.join(work, "stderr.txt")
                call_expected: Any = None
                if fn_name:
                    try:
                        args = decode_fn_input(raw_in)
                        call_expected = decode_fn_output(raw_out)
                    except (ValueError, json.JSONDecodeError, TypeError):
                        n_ran += 1
                        first_fail = index
                        detail = _LOAD
                        timed_out = False
                        if early:
                            break
                        continue
                    with open(args_path, "w", encoding="utf-8") as handle:
                        json.dump(args, handle)
                    stdin_text = ""
                else:
                    stdin_text = raw_in
                with open(stdin_path, "w", encoding="utf-8") as handle:
                    handle.write(stdin_text)
                for path in (stdout_path, stderr_path):
                    with open(path, "w", encoding="utf-8"):
                        pass
                status = runner.run(
                    {
                        "code_path": code_path,
                        "work": work,
                        "stdin_path": stdin_path,
                        "stdout_path": stdout_path,
                        "stderr_path": stderr_path,
                        "fn_name": fn_name,
                        "args_path": args_path,
                        "result_path": result_path,
                        "timeout_s": timeout,
                        "memory_mb": memory_mb,
                        "max_output_bytes": max_output,
                    },
                    timeout,
                )
                try:
                    size = os.path.getsize(stdout_path)
                except OSError:
                    size = 0
                if size > max_output:
                    status = _OUTPUT_CAP
                if fn_name:
                    outcome, info = _judge_call(status, result_path, call_expected)
                else:
                    stdout = _read_text(stdout_path, max_output) if status == "ok" else ""
                    outcome, info = _judge_stdio(status, stdout, raw_out)
                n_ran += 1
                if outcome != _PASS:
                    first_fail = index
                    detail = info
                    timed_out = outcome == _TIMEOUT
                    if early:
                        break
            finally:
                count, events, level = _read_denials(slot)
                n_denied += count
                for item in events:
                    if len(denied) < 8:
                        denied.append(item)
                if level and (not fs_isolation or level == "audit_hook"):
                    fs_isolation = level
                shutil.rmtree(slot, ignore_errors=True)
        passed = first_fail is None and n_ran == len(tests)
        return {
            "passed": passed,
            "n_tests": len(tests),
            "n_ran": n_ran,
            "first_fail": first_fail,
            "detail": detail,
            "timed_out": timed_out,
            "memory_limit_applied": memory_applied,
            "n_denied": n_denied,
            "denied": denied,
            "fs_isolation": fs_isolation,
        }
    finally:
        shutil.rmtree(code_dir, ignore_errors=True)
        shutil.rmtree(job_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["--once"]:
        raise SystemExit("usage: code_grader.py --once")
    # This process holds every expected output. Children inherit the flag.
    set_non_dumpable()
    # Fork before the job, and every expected output, enters this process.
    runner = start_runner()
    try:
        raw = sys.stdin.buffer.read()
        try:
            job = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"job is not JSON: {exc}") from exc
        if not isinstance(job, dict):
            raise SystemExit("job must be a JSON object")
        result = run_job(job, runner)
    finally:
        runner.close()
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
