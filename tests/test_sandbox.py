"""Subprocess sandbox: limits, isolation, and cleanup."""

from __future__ import annotations

import contextlib
import os
import signal
import time

import pytest

from vfates.tasks.sandbox import run_many, run_python


def _dead(pid: int) -> bool:
    for _ in range(25):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.1)
    return False


def test_success() -> None:
    result = run_python("print('hello')")
    assert result.ok
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert not result.timed_out
    assert result.seconds >= 0


def test_exception() -> None:
    result = run_python("raise RuntimeError('boom')")
    assert not result.ok
    assert result.returncode != 0
    assert "boom" in result.stderr
    assert not result.timed_out


def test_timeout_kills_children() -> None:
    code = """
import os, subprocess, sys, time
forked = os.fork()
if forked == 0:
    time.sleep(120)
    os._exit(0)
proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(f"PIDS {os.getpid()} {forked} {proc.pid}", flush=True)
time.sleep(120)
"""
    result = run_python(code, timeout_s=1.5)
    assert result.timed_out
    assert not result.ok
    assert result.seconds < 10
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("PIDS "))
    pids = [int(part) for part in line.split()[1:]]
    assert len(pids) == 3
    survivors = [pid for pid in pids if not _dead(pid)]
    for pid in survivors:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    assert survivors == []


def test_output_cap() -> None:
    result = run_python("import sys; sys.stdout.write('A' * 100000)", max_output_bytes=1000)
    assert len(result.stdout) <= 1000
    assert result.stdout == "A" * len(result.stdout)
    assert len(result.stdout) == 1000
    assert not result.ok


def test_env_not_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VFATES_SENTINEL", "supersecret")
    code = """
import os
print("VALUE", os.environ.get("VFATES_SENTINEL"))
print("KEYS", ",".join(sorted(os.environ)))
"""
    result = run_python(code)
    assert result.ok, result.stderr
    assert "supersecret" not in result.stdout
    assert "VALUE None" in result.stdout
    keys = next(ln for ln in result.stdout.splitlines() if ln.startswith("KEYS ")).split(" ", 1)[1]
    assert "VFATES_SENTINEL" not in keys.split(",")


def test_relative_writes_stay_in_temp(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    code = """
from pathlib import Path
p = Path("marker.txt")
p.write_text("hello-marker")
print(p.resolve())
print(p.read_text())
"""
    result = run_python(code)
    assert result.ok, result.stderr
    assert "hello-marker" in result.stdout
    assert not (tmp_path / "marker.txt").exists()
    printed = next(ln for ln in result.stdout.splitlines() if ln.endswith("marker.txt"))
    assert not os.path.exists(printed)
    assert str(tmp_path) not in printed


def test_run_many_preserves_order() -> None:
    results = run_many(["print('a')", "print('b')", "print('c')"], max_workers=2)
    assert [item.stdout.strip() for item in results] == ["a", "b", "c"]
    assert all(item.ok for item in results)
