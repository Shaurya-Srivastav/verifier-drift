"""Environment report fields used by locks and training runs."""

from __future__ import annotations

import json
import sys

from vfates.cli import main
from vfates.utils.env import environment_report

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
