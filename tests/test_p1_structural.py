"""P1 structural checks: stage policy, refusals, the queue, and confirmatory plumbing."""

from __future__ import annotations

from pathlib import Path

from vfates.forecast.lockbox import lock_forecasts
from vfates.pilots.p1_harness import (
    audit_stage_flags,
    confirmatory_refuses_incomplete,
    confirmatory_well_formed,
    patch_selection_matches,
    program_queue_smoke,
    sigusr1_marks_preempted,
    stages_match_patch_policy,
    study_refuses_chat_mismatch,
    study_refuses_hardware_mismatch,
    tau_budget_draws_present,
)
from vfates.utils.io import write_json


def _stages(tmp_path: Path, *, patches: bool) -> None:
    names = ["problems", "completions", "calibration", "table", "features", "forecasts"]
    if patches:
        names.append("patches")
    write_json(
        tmp_path / "stages.json",
        {"schema": "vfates.stages/1", "stages": {name: "ab" * 32 for name in names}},
    )


def test_patch_policy_and_tau_budgets(tmp_path: Path) -> None:
    _stages(tmp_path, patches=False)
    assert stages_match_patch_policy(tmp_path, expect_patches=False)
    assert stages_match_patch_policy(tmp_path, expect_patches=True) is False
    _stages(tmp_path, patches=True)
    assert stages_match_patch_policy(tmp_path, expect_patches=True)
    budgets = {key: {"mean": 0.1, "sd": 0.0} for key in ("32", "64", "128", "192")}
    forecasts = {
        "oracle": {"transfer": {"tau_pool_budget": budgets}},
        "patch_selection": {"chosen": "patch_tau"},
    }
    assert tau_budget_draws_present(forecasts)
    assert patch_selection_matches(forecasts, expect=True)
    assert patch_selection_matches({"oracle": forecasts["oracle"]}, expect=False)
    assert patch_selection_matches({"oracle": forecasts["oracle"]}, expect=True) is False
    missing = {"oracle": {"transfer": {"tau_pool_budget": {"32": {"mean": 0.1, "sd": 1.0}}}}}
    assert tau_budget_draws_present(missing) is False


def test_study_refusals_and_preemption(tmp_path: Path) -> None:
    assert study_refuses_hardware_mismatch(tmp_path)
    assert study_refuses_chat_mismatch(tmp_path)
    assert sigusr1_marks_preempted()


def test_confirmatory_report_and_incomplete_cell(tmp_path: Path) -> None:
    report = confirmatory_well_formed(tmp_path)
    assert report["schema"] == "vfates.confirmatory/1"
    assert report["h3"]["decision"] == "not_tested"
    assert "incomplete planned runs" in confirmatory_refuses_incomplete()


def test_audit_flags_see_pool_source(tmp_path: Path) -> None:
    payload = {"oracle": {"transfer": {"tau_pool": 0.2}}}
    write_json(tmp_path / "forecasts.json", payload)
    lock_forecasts(
        payload,
        tmp_path / "lock.json",
        pool_source="dev",
        artifacts={"lora_init": "ab" * 32},
    )
    record = Path(tmp_path / "lock.json").read_text(encoding="utf-8")
    assert "dev" in record
    flags = audit_stage_flags(tmp_path, expect_patches=False)
    assert flags["pool_source"] is True
    assert flags["hardware_class"] is True
    assert flags["artifacts"] is False


def test_program_worker_and_pooled_report(tmp_path: Path) -> None:
    facts = program_queue_smoke(tmp_path)
    assert facts["rendered"] == 2
    assert facts["n_done"] == facts["n_enqueued"]
    assert facts["n_enqueued"] >= 4
    assert facts["pooled_schema"] == "vfates.pooled/1"
    assert facts["oracle_gain"] == 0.3
    assert facts["base"] and facts["early"] and facts["variant_dir"] and facts["audit_dir"]
    assert "streaming" in facts["variant_names"]
    assert facts["transfer_planned"] >= 1
    assert "done" in facts["status"]
