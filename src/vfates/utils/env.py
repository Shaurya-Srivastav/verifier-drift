"""Hardware and software report, embedded in every artifact for reproducibility."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

_PACKAGES = (
    "numpy",
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "mlx",
    "mlx-lm",
    "vllm",
    "reasoning-gym",
    "datasets",
    "sympy",
    "vfates",
)


def _sysctl(key: str) -> str | None:
    try:
        return (
            subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def accelerator() -> str:
    """Best available accelerator: ``cuda``, ``mps`` or ``cpu``."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "accelerator": accelerator(),
    }
    if sys.platform == "darwin":
        mem = _sysctl("hw.memsize")
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        info["memory_gb"] = round(int(mem) / 2**30, 1) if mem else None
    else:
        try:
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            info["memory_gb"] = round(kb / 2**20, 1)
        except (OSError, ValueError, IndexError):
            info["memory_gb"] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info


def environment_report() -> dict[str, Any]:
    return {"hardware": hardware(), "versions": versions(), "git_head": _git_head()}
