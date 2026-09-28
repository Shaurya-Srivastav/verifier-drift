"""Hardware and software report, embedded in every artifact for reproducibility."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

# Report key, then the distribution name ``importlib.metadata`` knows.
_PACKAGES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("trl", "trl"),
    ("vllm", "vllm"),
    ("mlx", "mlx"),
    ("mlx_lm", "mlx-lm"),
    ("reasoning_gym", "reasoning-gym"),
    ("datasets", "datasets"),
    ("sympy", "sympy"),
    ("vfates", "vfates"),
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


def _git_state() -> tuple[str | None, bool | None]:
    """``(commit, dirty)``. Both are ``None`` when git cannot be asked."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if out.returncode != 0:
        return None, None
    commit = out.stdout.strip() or None
    if commit is None:
        return None, None
    return commit, bool(dirty.stdout.strip())


def versions() -> dict[str, str | None]:
    """Package versions, plus the running Python. Missing distributions are ``None``."""
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for key, dist in _PACKAGES:
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
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
        "gpu_name": None,
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
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cuda_devices"] = names
            info["gpu_name"] = names[0] if names else None
    except ImportError:
        pass
    return info


def environment_report() -> dict[str, Any]:
    """Software, git, and accelerator snapshot embedded in audit and run artifacts.

    ``versions`` uses ``None`` for a distribution that is not installed.
    ``git_commit`` and ``git_dirty`` are recorded separately; ``git_head`` keeps
    the older ``commit`` or ``commit+dirty`` string. ``gpu_name`` is the CUDA
    device name when one is present, otherwise ``None``.
    """
    commit, dirty = _git_state()
    hw = hardware()
    return {
        "hardware": hw,
        "versions": versions(),
        "git_commit": commit,
        "git_dirty": dirty,
        "git_head": None if commit is None else commit + ("+dirty" if dirty else ""),
        "gpu_name": hw.get("gpu_name"),
    }
