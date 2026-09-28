"""Environment report fields used by locks and training runs."""

from __future__ import annotations

import json
import sys

import vfates.utils.env as env
from vfates.cli import main
from vfates.utils.env import environment_report, hardware_class

_VERSION_KEYS = (
    "python",
    "numpy",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "trl",
    "vllm",
    "mlx",
    "mlx_lm",
    "reasoning_gym",
    "datasets",
    "sympy",
    "vfates",
)


def test_environment_report_records_versions_git_and_gpu() -> None:
    report = environment_report()
    versions = report["versions"]
    for key in _VERSION_KEYS:
        assert key in versions
        assert versions[key] is None or isinstance(versions[key], str)
    assert versions["python"] == sys.version.split()[0]
    commit = report["git_commit"]
    dirty = report["git_dirty"]
    if commit is None:
        assert dirty is None
        assert report["git_head"] is None
    else:
        assert len(commit) == 40
        assert isinstance(dirty, bool)
        if dirty:
            assert report["git_head"] == commit + "+dirty"
        else:
            assert report["git_head"] == commit
    assert "gpu_name" in report
    assert report["gpu_name"] == report["hardware"]["gpu_name"]
    assert "gpu_name" in report["hardware"]
    assert report["hardware_class"] == hardware_class()
    assert report["hardware"]["hardware_class"] == report["hardware_class"]


def test_env_command_prints_the_report(capsys) -> None:
    assert main(["env"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["versions"]["python"] == sys.version.split()[0]
    assert "safetensors" in payload["versions"]
    assert "mlx_lm" in payload["versions"]
    assert "reasoning_gym" in payload["versions"]
    assert "datasets" in payload["versions"]
    assert "git_commit" in payload
    assert "git_dirty" in payload
    assert "gpu_name" in payload
    assert payload["hardware_class"] == payload["hardware"]["hardware_class"]


def test_hardware_class_maps_cuda_names_and_falls_back(monkeypatch) -> None:
    gib = 1024**3

    def _cuda(name: str, nbytes: int) -> None:
        monkeypatch.setattr(env, "_cuda_identity", lambda: (name, nbytes))

    _cuda("NVIDIA H100 80GB HBM3", 80 * gib)
    assert hardware_class() == "h100-80gb"
    _cuda("NVIDIA A100-SXM4-80GB", 79 * gib)
    assert hardware_class() == "a100-80gb"
    _cuda("NVIDIA A100-SXM4-40GB", 40536 * 1024**2)
    assert hardware_class() == "a100-40gb"
    _cuda("NVIDIA RTX A6000", 48 * gib)
    assert hardware_class() == "rtx-a6000-48gb"
    _cuda("NVIDIA RTX 5000 Ada Generation", 32 * gib)
    assert hardware_class() == "rtx-5000-ada-32gb"
    _cuda("NVIDIA GeForce RTX 4090", 24 * gib)
    assert hardware_class() == "cuda-geforce-rtx-4090-24gb"

    monkeypatch.setattr(env, "_cuda_identity", lambda: None)
    monkeypatch.setattr(env.sys, "platform", "darwin")
    monkeypatch.setattr(env.platform, "machine", lambda: "arm64")
    assert hardware_class() == "apple-silicon"
    monkeypatch.setattr(env.sys, "platform", "linux")
    monkeypatch.setattr(env.platform, "machine", lambda: "x86_64")
    assert hardware_class() == "cpu"
