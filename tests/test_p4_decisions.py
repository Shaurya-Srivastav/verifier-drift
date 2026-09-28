"""P4 domain decisions, backups, and a study-shaped smoke run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vfates.pilots.p4_micro_grpo import (
    SCHEMA,
    apply_dotted,
    backup_overrides,
    decide_domain,
    power_statement,
    run_program,
    sign_agreement,
)
from vfates.utils.io import append_jsonl, read_json, write_json


def test_decide_domain_covers_pass_backup_and_removal() -> None:
    assert decide_domain(0.12, has_backup=True) == {
        "pass": True,
        "gain": pytest.approx(0.12),
        "backup_used": False,
        "leaves_program": False,
    }
    assert decide_domain(0.04, has_backup=False)["leaves_program"] is True
    waiting = decide_domain(0.04, has_backup=True, backup_ready=False)
    assert waiting == {
        "pass": False,
        "gain": pytest.approx(0.04),
        "backup_used": False,
        "leaves_program": False,
    }
    backed = decide_domain(0.04, has_backup=True, backup_ready=True, backup_gain=0.2)
    assert backed["pass"] is True and backed["backup_used"] is True
    assert backed["leaves_program"] is False
    second = decide_domain(0.04, has_backup=True, backup_ready=True, backup_gain=0.01)
    assert second["pass"] is False and second["backup_used"] is True
    assert second["leaves_program"] is True
    missing = decide_domain(None, has_backup=False)
    assert missing["leaves_program"] is False and missing["pass"] is False


def test_sign_agreement_treats_zero_as_undefined() -> None:
    assert sign_agreement(0.0, 0.2) is None
    assert sign_agreement(0.2, 0.0) is None
    assert sign_agreement(None, 0.2) is None
    assert sign_agreement(0.1, 0.2) is True
    assert sign_agreement(0.1, -0.2) is False


def test_power_statement_uses_the_analysis_keys() -> None:
    row = power_statement(n_prompts=32, n_seeds=1, n_samples=4)
    for key in ("se", "mdd", "power_at_0.05", "tost_power_at_0"):
        assert key in row
        assert isinstance(row[key], float)


def test_backup_overrides_match_the_screened_sources() -> None:
    math = backup_overrides("math")
    assert math is not None
    assert math["audit.task.source"] == "deepscaler"
    assert "DeepScaleR" in math["audit.task.dataset"]
    assert math["audit.task.revision"]
    assert backup_overrides("code") is None
    ifeval = backup_overrides("ifeval")
    assert ifeval is not None
    assert ifeval["audit.task.source"] == "rlvr_ifeval"
    assert ifeval["audit.task.revision"] is None
    assert ifeval["audit.task.constraint_counts"] is None
    assert backup_overrides("chain_sum") is None
    assert backup_overrides("qa") is None


def _install(monkeypatch: pytest.MonkeyPatch, finals: dict[tuple[str, str], float]) -> None:
    def fake_load(
        cell_id: str, *, dev: bool = True, overrides: dict | None = None, root: Any = None
    ):
        del cell_id, dev, root
        cell: dict[str, Any] = {
            "audit": {"pools": {"from": "dev", "heldout": 4}, "task": {}},
            "study": {
                "train": {"steps": 4, "eval_samples_per_prompt": 1},
                "arms": ["oracle"],
                "seeds": [0],
            },
        }
        apply_dotted(cell, overrides or {})
        return cell

    def fake_execute(
        config: dict, out: Path, *, only: str, arm: str | None = None, seed: int | None = None
    ):
        out = Path(out)
        if only == "audit":
            audit = out / "audit"
            audit.mkdir(parents=True, exist_ok=True)
            write_json(audit / "forecasts.json", {})
            return {}
        run = out / "runs" / f"{arm}__s{seed}"
        run.mkdir(parents=True, exist_ok=True)
        steps = int(config["study"]["train"]["steps"])
        domain = out.parent.name
        task = config["audit"]["task"]
        kind = "primary"
        if task.get("source") == "deepscaler":
            kind = "backup"
        if task.get("source") == "rlvr_ifeval":
            kind = "backup"
        final = finals[(domain, kind)]
        write_json(run / "summary.json", {"steps_completed": steps, "status": "complete"})
        append_jsonl(run / "eval.jsonl", {"step": 0, "split": "heldout", "oracle_acc": 0.0})
        append_jsonl(run / "eval.jsonl", {"step": steps, "split": "heldout", "oracle_acc": final})
        return {}

    monkeypatch.setattr("vfates.pilots.p4_micro_grpo.load_pilot_cell", fake_load)
    monkeypatch.setattr("vfates.pilots.p4_micro_grpo.execute_study", fake_execute)


def test_run_program_writes_the_decision_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        {
            ("chain_sum", "primary"): 0.05,
            ("math", "primary"): 0.02,
            ("math", "backup"): 0.20,
            ("code", "primary"): 0.01,
            ("ifeval", "primary"): 0.01,
            ("qa", "primary"): 0.15,
        },
    )
    report = run_program(
        {
            "title": "P4 test",
            "steps": 4,
            "oracle_gain": 0.10,
            "seed": 0,
            "train_fates": False,
            "domains": ["chain_sum", "math", "code", "ifeval", "qa"],
            "backup_screened": ["math"],
        },
        tmp_path,
    )
    decisions = read_json(tmp_path / "decisions.json")
    assert decisions["schema"] == SCHEMA
    assert set(decisions["domains"]) == {"chain_sum", "math", "code", "ifeval", "qa"}
    chain = decisions["domains"]["chain_sum"]
    assert chain["pass"] is False and chain["leaves_program"] is True
    assert chain["backup_used"] is False
    math = decisions["domains"]["math"]
    assert math["pass"] is True and math["backup_used"] is True
    assert math["leaves_program"] is False
    code = decisions["domains"]["code"]
    assert code["pass"] is False and code["backup_used"] is False
    assert code["leaves_program"] is True
    ifeval = decisions["domains"]["ifeval"]
    assert ifeval["pass"] is False and ifeval["backup_used"] is False
    assert ifeval["leaves_program"] is False
    assert "awaiting P2" in next(
        row["metric"] for row in report["checks"] if row["id"] == "P4.1.ifeval"
    )
    qa = decisions["domains"]["qa"]
    assert qa["pass"] is True and qa["leaves_program"] is False
    assert report["ok"] is False


def test_smoke_documents_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {("chain_sum", "primary"): 0.01})
    report = run_program(
        {
            "smoke": True,
            "smoke_steps": 2,
            "train_fates": False,
            "domains": ["chain_sum"],
            "oracle_gain": 0.10,
        },
        tmp_path,
    )
    row = read_json(tmp_path / "decisions.json")["domains"]["chain_sum"]
    assert row["pass"] is False and row["leaves_program"] is False
    check = next(item for item in report["checks"] if item["id"] == "P4.1.chain_sum")
    assert check["status"] == "documented"
    assert report["ok"] is True
