"""train_arm refuses a lock that does not match the audit it trains on."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from vfates.audit import load_audit, read_lock_record
from vfates.experiment import train_arm
from vfates.tasks.chain_sum import ChainSumTask
from vfates.train.config import NonFiniteLossError
from vfates.utils.io import read_json, sha256_file

_TRAIN = {
    "model_id": "fake",
    "max_tokens": 32,
    "steps": 1,
    "prompts_per_step": 2,
    "seed": 0,
    "eval_every": 1,
    "eval_heldout_n": 0,
    "eval_train_n": 0,
    "eval_samples_per_prompt": 1,
    "save_every": 1,
    "group_size": 2,
    "micro_batch_size": 2,
}


def _trainer(monkeypatch: pytest.MonkeyPatch) -> dict:
    box: dict = {"called": 0}

    class MLXGRPOTrainer:
        def __init__(self, config, task, verifier, train, held, monitors=None) -> None:
            del task, verifier, held, monitors
            box["called"] += 1
            box["config"] = config
            box["n_train"] = len(train)

        def run(self, resume: bool = False) -> dict:
            del resume
            if "error" in box:
                raise box["error"]
            return {"ok": True}

    module = types.ModuleType("vfates.train.mlx_grpo")
    module.MLXGRPOTrainer = MLXGRPOTrainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vfates.train.mlx_grpo", module)
    return box


def _audit(tmp_path: Path, audit_config: dict, patch_audit_fakes) -> Path:
    audit_config["lock"] = True
    patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    from vfates.audit import run_audit

    run_audit(audit_config, out)
    return out


def test_train_arm_writes_the_lock_and_lora_init_into_the_run(
    tmp_path: Path, audit_config: dict, patch_audit_fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path, audit_config, patch_audit_fakes)
    box = _trainer(monkeypatch)
    run_dir = tmp_path / "run"
    result = train_arm(audit, "oracle", _TRAIN, run_dir, backend="mlx", max_train=2)
    assert result == {"ok": True}
    record = read_lock_record(audit / "lock.json")
    run = read_json(run_dir / "run.json")
    init = audit / "lora_init.safetensors"
    assert run["lock_sha256"] == record["sha256"]
    assert run["lora_init_sha256"] == sha256_file(init)
    assert run["lora_init_sha256"] == record["artifacts"]["lora_init"]
    assert box["config"].lora_init == str(init)
    assert box["config"].lora_init_sha256 == run["lora_init_sha256"]
    assert isinstance(run["git_commit"], str) or run["git_commit"] is None
    assert "git_head" in run
    assert "git_dirty" in run
    assert run["environment"]["versions"]["python"]
    assert box["n_train"] == 2


def test_train_arm_refuses_a_policy_other_than_the_audited_one(
    tmp_path: Path, audit_config: dict, patch_audit_fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path, audit_config, patch_audit_fakes)
    box = _trainer(monkeypatch)
    changes = (
        ({"model_id": "other"}, "model_id 'other' \\(audit 'fake'\\)"),
        ({"max_tokens": 64}, "max_tokens 64 \\(audit 32\\); eval_max_tokens 64"),
        ({"eval_max_tokens": 16}, "eval_max_tokens 16 \\(audit 32\\)"),
        ({"temperature": 0.7}, "temperature 0.7 \\(audit 1.0\\)"),
        ({"eval_temperature": 0.0}, "eval_temperature 0.0 \\(audit 1.0\\)"),
    )
    for i, (change, message) in enumerate(changes):
        with pytest.raises(ValueError, match=message):
            train_arm(audit, "oracle", {**_TRAIN, **change}, tmp_path / f"run{i}")
    assert box["called"] == 0


def test_train_arm_refuses_a_tampered_artifact_and_a_bad_lock(
    tmp_path: Path, audit_config: dict, patch_audit_fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path, audit_config, patch_audit_fakes)
    problems = (audit / "problems.jsonl").read_text(encoding="utf-8")
    (audit / "problems.jsonl").write_text(problems + " ", encoding="utf-8")
    box = _trainer(monkeypatch)
    with pytest.raises(ValueError, match="problems"):
        train_arm(audit, "oracle", _TRAIN, tmp_path / "run-problems")
    assert box["called"] == 0
    assert not (tmp_path / "run-problems" / "run.json").exists()

    (audit / "problems.jsonl").write_text(problems, encoding="utf-8")
    (audit / "lora_init.safetensors").write_bytes(b"other-init")
    with pytest.raises(ValueError, match="lora_init"):
        train_arm(audit, "oracle", _TRAIN, tmp_path / "run-lora")
    assert box["called"] == 0

    (audit / "lora_init.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        train_arm(audit, "oracle", _TRAIN, tmp_path / "run-missing")
    assert box["called"] == 0

    (audit / "lock.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="verification"):
        train_arm(audit, "oracle", _TRAIN, tmp_path / "run-lock")
    assert box["called"] == 0


def test_train_arm_refuses_when_the_oracle_now_grades_differently(
    tmp_path: Path, audit_config: dict, patch_audit_fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path, audit_config, patch_audit_fakes)
    record = read_lock_record(audit / "lock.json")
    assert record["artifacts"]["grades"] == load_audit(audit).grades_sha256
    box = _trainer(monkeypatch)
    monkeypatch.setattr(ChainSumTask, "oracle", lambda self, problem, parsed, text: 0)
    with pytest.raises(ValueError, match="grades"):
        train_arm(audit, "oracle", _TRAIN, tmp_path / "run")
    assert box["called"] == 0
    assert not (tmp_path / "run" / "run.json").exists()


def test_nonfinite_loss_propagates_after_the_run_record(
    tmp_path: Path, audit_config: dict, patch_audit_fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path, audit_config, patch_audit_fakes)
    box = _trainer(monkeypatch)
    run_dir = tmp_path / "run"
    box["error"] = NonFiniteLossError(1, str(run_dir))
    with pytest.raises(NonFiniteLossError) as caught:
        train_arm(audit, "oracle", _TRAIN, run_dir)
    assert caught.value.step == 1
    assert (run_dir / "run.json").is_file()
    assert read_json(run_dir / "run.json")["lock_sha256"]


def test_unknown_lock_artifact_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "lock.json").write_text("{}\n", encoding="utf-8")

    class Arm:
        def describe(self) -> dict:
            return {"name": "oracle"}

    fake = types.SimpleNamespace(arms={"oracle": Arm()}, problems={}, grades_sha256="ef" * 32)
    monkeypatch.setattr("vfates.experiment.load_audit", lambda path: fake)
    monkeypatch.setattr(
        "vfates.experiment.read_lock_record",
        lambda path: {"sha256": "ab" * 32, "artifacts": {"not_a_stage": "cd" * 32}},
    )
    _trainer(monkeypatch)
    with pytest.raises(ValueError, match="unknown artifact"):
        train_arm(audit, "oracle", _TRAIN, tmp_path / "run")
