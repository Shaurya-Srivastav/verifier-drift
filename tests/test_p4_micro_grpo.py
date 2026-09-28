"""P4 applies the preregistered audit withdrawals before training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import vfates.experiment
from vfates.forecast.lockbox import lock_forecasts
from vfates.pilots import p4_micro_grpo
from vfates.utils.io import append_jsonl, read_json, write_json

PROMPTS = [f"p{i}" for i in range(6)]
STEPS = 4
FINALS = {
    "oracle": 0.70,
    "noise_fp": 0.70,
    "fp_tolerance": 0.40,
    "fp_hash": 0.70,
    "noise_fn": 0.60,
    "fn_behavior": 0.30,
    "fn_prompt_subset": 0.60,
}
TYPES = {
    "oracle": "oracle",
    "noise_fp": "resampled",
    "fp_tolerance": "tolerance_fp",
    "fp_hash": "hash_fp",
    "noise_fn": "resampled",
    "fn_behavior": "behavior_fn",
    "fn_prompt_subset": "prompt_subset_fn",
}


def _forecast(tau: float) -> dict[str, Any]:
    return {
        "transfer": {
            "tau_pool": tau,
            "tau_hard": tau,
            "tau_rel": 0.4,
            "reliability": 0.8,
            "cos_gV_gO": 0.2,
            "base_acc_heldout": 0.3,
        },
        "step0": {"J": 0.55, "fpr": 0.15, "fnr": 0.30},
        "mean_precision": tau,
        "modal_precision": {"value": tau, "tie_fraction": 0.0},
    }


def _calibration_arm(name: str) -> dict[str, Any]:
    typ = TYPES[name]
    entry: dict[str, Any] = {"type": typ, "details": {}}
    if typ in {"tolerance_fp", "hash_fp"}:
        entry.update({"channel": {"fpr": 0.15, "fnr": 0.0}, "target_fpr": 0.15})
    if typ in {"behavior_fn", "prompt_subset_fn"}:
        entry.update({"channel": {"fpr": 0.0, "fnr": 0.30}, "target_fnr": 0.30})
    if name == "fn_behavior":
        entry["details"] = {"fell_back": False, "validated": False}
    return entry


def _audit(tmp_path: Path) -> Path:
    audit = tmp_path / "audit"
    audit.mkdir()
    payload = {arm: _forecast(0.5 - FINALS[arm] / 2) for arm in FINALS}
    lock_forecasts(payload, audit / "lock.json", note="synthetic")
    write_json(audit / "calibration.json", {"arms": {a: _calibration_arm(a) for a in FINALS}})
    return audit


def _fake_train(calls: list[str]):
    def train(
        audit_dir: Path, arm: str, cfg: dict[str, Any], run_dir: Path, **_: Any
    ) -> dict[str, Any]:
        calls.append(arm)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        for step, acc in ((0, 0.30), (STEPS, FINALS[arm])):
            append_jsonl(
                run_dir / "eval.jsonl",
                {"step": step, "split": "heldout", "oracle_acc": acc, "J": acc},
            )
            append_jsonl(
                run_dir / "eval_prompts.jsonl",
                {"step": step, "split": "heldout", "acc": {p: acc for p in PROMPTS}},
            )
        write_json(run_dir / "run.json", {"arm": arm})
        write_json(run_dir / "summary.json", {"steps_completed": STEPS})
        return {"steps_completed": STEPS}

    return train


def test_withdrawn_arm_is_not_trained_and_its_contrast_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(vfates.experiment, "train_arm", _fake_train(calls))
    config = {
        "audit_dir": str(_audit(tmp_path)),
        "seeds": [0],
        "n_boot": 50,
        "arms": list(FINALS),
        "families": [
            {"control": "noise_fp", "kind": "fp", "shared": ["fp_tolerance"], "local": ["fp_hash"]},
            {
                "control": "noise_fn",
                "kind": "fn",
                "shared": ["fn_behavior"],
                "local": ["fn_prompt_subset"],
            },
        ],
        "train": {"steps": STEPS},
        "gates": {"oracle_gain": 0.10, "fate_gap": 0.05, "local_tol": 0.05, "alpha": 0.05},
    }
    out = tmp_path / "out"
    report = p4_micro_grpo.run(config, out)

    assert "fn_behavior" not in calls
    assert "noise_fn" in calls
    assert calls == [arm for arm in FINALS if arm != "fn_behavior"]
    record = read_json(out / "withdrawals.json")
    assert record["withdrawn"] == ["fn_behavior"]
    assert record["audit"]["fn_behavior"]["reasons"] == ["validated"]
    checks = {row["id"]: row for row in report["checks"]}
    assert checks["P4.0"]["value"] == 1.0
    assert "fn_behavior: validated" in checks["P4.0"]["metric"]
    assert checks["P4.2.fn_behavior"]["status"] == "documented"
    assert checks["P4.2.fp_tolerance"]["passed"] is True
    assert checks["P4.3.fn_prompt_subset"]["status"] == "pass"
    assert "fn_behavior" not in report["tables"]["outcomes"]
    assert "Audit withdrawals" in (out / "report.md").read_text()
