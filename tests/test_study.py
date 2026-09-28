"""Study runner, P4 go/no-go, and P5 full-gate helpers on synthetic directories."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vfates.experiment import _check_audited_policy
from vfates.forecast.lockbox import lock_forecasts
from vfates.study import _check_model_ids, planned_seeds, reanalyze, run_study
from vfates.train.config import GRPOConfig, NonFiniteLossError
from vfates.types import GenOut
from vfates.utils.io import append_jsonl, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def _forecast(tau: float = 0.2) -> dict:
    return {
        "transfer": {
            "tau_pool": tau,
            "tau_hard": tau,
            "tau_rel": 0.4,
            "reliability": 0.8,
            "cos_gV_gO": 0.2,
            "base_acc_heldout": 0.3,
        },
        "step0": {"J": 0.2, "fpr": 0.1, "fnr": 0.2},
        "mean_precision": tau,
        "modal_precision": {"value": tau, "tie_fraction": 0.0},
    }


def _write_run(
    path: Path,
    arm: str,
    steps: int,
    *,
    step0: float = 0.2,
    final: float = 0.5,
    monitors_wrong: dict[str, float] | None = None,
    completed: int | None = None,
    metrics: dict | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    wrong = {} if monitors_wrong is None else monitors_wrong
    for step, acc in ((0, step0), (steps, final)):
        append_jsonl(
            path / "eval.jsonl",
            {
                "step": step,
                "split": "heldout",
                "oracle_acc": acc,
                "J": final,
                "fpr": 0.1,
                "fnr": 0.2,
                "monitors_wrong": wrong,
                "monitors_correct": {"fn_behavior": 0.4},
            },
        )
        append_jsonl(
            path / "eval.jsonl",
            {"step": step, "split": "train", "oracle_acc": acc, "J": final, "fpr": 0.1, "fnr": 0.2},
        )
        append_jsonl(
            path / "eval_prompts.jsonl",
            {"step": step, "split": "heldout", "acc": {"p0": acc, "p1": acc}},
        )
    write_json(
        path / "summary.json", {"steps_completed": steps if completed is None else completed}
    )
    write_json(path / "run.json", {"arm": arm})
    if metrics:
        append_jsonl(path / "metrics.jsonl", metrics)


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loss_agg: str = "seq_mean",
    forecasts_match: bool = True,
    valid_lock: bool = True,
    validity: dict | None = None,
    trainer=None,
) -> list:
    calls: list = []

    def run_audit(config, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "oracle": _forecast(0.2),
            "fp_trigger": _forecast(0.4),
            "noise_fp_trigger": _forecast(0.6),
            "fp_tolerance": _forecast(0.3),
            "noise_fp": _forecast(0.5),
            "fp_hash": _forecast(0.5),
        }
        write_json(out / "config.json", {"config": {"forecast": {"loss_agg": loss_agg}}})
        write_json(out / "calibration.json", {"arms": []})
        write_json(
            out / "forecasts.json", payload if forecasts_match else {"oracle": _forecast(0.9)}
        )
        if valid_lock:
            lock_forecasts(payload, out / "lock.json", note="synthetic")
        else:
            write_json(out / "lock.json", {"broken": True})
        calls.append(("audit", out))

    def train_arm(audit_dir, arm, train_config, out_dir, **kwargs):
        calls.append(
            (
                "train",
                arm,
                int(train_config["seed"]),
                bool(kwargs.get("resume")),
                int(train_config["steps"]),
            )
        )
        if trainer is not None:
            trainer(arm, train_config, Path(out_dir), kwargs)
        else:
            _write_run(Path(out_dir), arm, int(train_config["steps"]))
        return {"steps_completed": int(train_config["steps"])}

    monkeypatch.setattr("vfates.audit.run_audit", run_audit)
    monkeypatch.setattr("vfates.experiment.train_arm", train_arm)
    monkeypatch.setattr(
        "vfates.verifiers.calibrate.arm_validity",
        lambda report: {} if validity is None else validity,
    )
    return calls


def _study_config(arms: list[str] | None = None, *, hypotheses: bool = False) -> dict:
    analysis: dict = {"final_step": 4, "hypotheses": hypotheses, "n_boot": 20}
    if hypotheses:
        analysis["early"] = {"enabled": True, "steps": 2, "seed": 99}
        analysis["h1"] = {
            "contrasts": [{"control": "noise_fp_trigger", "arm": "fp_trigger"}],
            "beside": [],
        }
        analysis["h2"] = {"contrasts": []}
        analysis["h3"] = {"pairs": [], "min_pairs": 5}
    else:
        analysis["early"] = {"enabled": False}
    return {
        "audit": {"task": {"type": "chain_sum"}},
        "study": {
            "seeds": [0],
            "arms": ["oracle"] if arms is None else arms,
            "backend": "trl",
            "train": {"steps": 4, "loss_agg": "seq_mean", "model_id": "tiny"},
        },
        "analysis": analysis,
    }


def test_planned_seeds_follow_the_seed_rule() -> None:
    study = {
        "seeds": [0, 1, 2],
        "arms": ["oracle", "noise_fp"],
        "seed_rule": {
            "apply": True,
            "extra_seeds": [3, 4],
            "arms": ["noise_fp"],
        },
    }
    planned = planned_seeds(study)
    assert planned["oracle"] == [0, 1, 2]
    assert planned["noise_fp"] == [0, 1, 2, 3, 4]
    study["seed_rule"]["apply"] = False
    assert planned_seeds(study)["noise_fp"] == [0, 1, 2]


def test_run_study_writes_withdrawals_before_training(tmp_path: Path, monkeypatch) -> None:
    seen: list[bool] = []

    def trainer(arm, train_config, out_dir, kwargs):
        seen.append((out_dir.parents[1] / "withdrawals.json").exists())
        _write_run(out_dir, arm, int(train_config["steps"]))

    calls = _install(monkeypatch, trainer=trainer)
    report = run_study(_study_config(), tmp_path)
    assert report["ok"] is True
    assert report["summary"] == "PASS"
    assert report["analysis"]["hypotheses"] is False
    assert (tmp_path / "withdrawals.json").exists()
    assert (tmp_path / "report.md").exists()
    assert seen == [True]
    assert calls[0][0] == "audit"
    assert [row[1] for row in calls if row[0] == "train"] == ["oracle"]


def test_skip_completed_and_resume_partial(tmp_path: Path, monkeypatch) -> None:
    calls = _install(monkeypatch)
    done = tmp_path / "runs" / "oracle__s0"
    _write_run(done, "oracle", 4)
    report = run_study(_study_config(), tmp_path)
    assert report["ok"] is True
    assert [row for row in calls if row[0] == "train"] == []

    again = tmp_path / "partial"
    (again / "runs" / "oracle__s0" / "checkpoint-1").mkdir(parents=True)
    calls2 = _install(monkeypatch)
    run_study(_study_config(), again)
    trains = [row for row in calls2 if row[0] == "train"]
    assert trains[0][3] is True


def test_nonfinite_resumes_once_then_withdraws(tmp_path: Path, monkeypatch) -> None:
    state = {"n": 0}

    def trainer(arm, train_config, out_dir, kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise NonFiniteLossError(1, str(out_dir))
        _write_run(out_dir, arm, int(train_config["steps"]))

    calls = _install(monkeypatch, trainer=trainer)
    report = run_study(_study_config(), tmp_path)
    trains = [row for row in calls if row[0] == "train"]
    assert trains[0][3] is False
    assert trains[1][3] is True
    assert trains[1][2] == 0
    assert report["ok"] is True
    assert "oracle" not in report["withdrawals"]["withdrawn"]

    def always(arm, train_config, out_dir, kwargs):
        if arm == "oracle":
            raise NonFiniteLossError(1, str(out_dir))
        _write_run(out_dir, arm, int(train_config["steps"]))

    calls2 = _install(monkeypatch, trainer=always)
    out = tmp_path / "twice"
    report2 = run_study(_study_config(["oracle", "fp_tolerance"]), out)
    assert report2["ok"] is True
    assert "oracle" in report2["withdrawals"]["withdrawn"]
    assert "fp_tolerance" not in report2["withdrawals"]["withdrawn"]
    trained = [row[1] for row in calls2 if row[0] == "train"]
    assert trained.count("oracle") == 2
    assert "fp_tolerance" in trained


def test_lock_and_loss_agg_fail_before_training(tmp_path: Path, monkeypatch) -> None:
    calls = _install(monkeypatch, loss_agg="token_mean")
    with pytest.raises(ValueError, match="loss_agg"):
        run_study(_study_config(), tmp_path)
    assert (tmp_path / "withdrawals.json").exists()
    assert [row for row in calls if row[0] == "train"] == []

    calls2 = _install(monkeypatch, valid_lock=False)
    with pytest.raises(ValueError, match="lock"):
        run_study(_study_config(), tmp_path / "bad-lock")
    assert [row for row in calls2 if row[0] == "train"] == []

    calls3 = _install(monkeypatch, forecasts_match=False)
    with pytest.raises(ValueError, match="does not match"):
        run_study(_study_config(), tmp_path / "mismatch")
    assert [row for row in calls3 if row[0] == "train"] == []


def test_incomplete_run_does_not_analyse(tmp_path: Path, monkeypatch) -> None:
    def trainer(arm, train_config, out_dir, kwargs):
        _write_run(out_dir, arm, int(train_config["steps"]), completed=1)

    _install(monkeypatch, trainer=trainer)
    report = run_study(_study_config(), tmp_path)
    assert report["ok"] is False
    assert report["analysis"] is None
    assert "incomplete" in report["reason"]


def test_reanalyze_does_not_train(tmp_path: Path, monkeypatch) -> None:
    calls = _install(monkeypatch)
    report = run_study(_study_config(), tmp_path)
    assert report["analysis"]["oracle_gain"]["value"] == pytest.approx(0.3)
    n_train = len([row for row in calls if row[0] == "train"])
    prompts = tmp_path / "runs" / "oracle__s0" / "eval_prompts.jsonl"
    prompts.write_text("")
    append_jsonl(prompts, {"step": 0, "split": "heldout", "acc": {"p0": 0.2, "p1": 0.2}})
    append_jsonl(prompts, {"step": 4, "split": "heldout", "acc": {"p0": 0.9, "p1": 0.9}})
    withdrawals = read_json(tmp_path / "withdrawals.json")
    again = reanalyze(_study_config(), tmp_path)
    assert again["ok"] is True
    assert again["analysis"]["oracle_gain"]["value"] == pytest.approx(0.7)
    assert read_json(tmp_path / "withdrawals.json") == withdrawals
    assert len([row for row in calls if row[0] == "train"]) == n_train


def test_withdrawn_contrast_stays_in_the_planned_menu(tmp_path: Path, monkeypatch) -> None:
    _install(
        monkeypatch,
        validity={
            "fp_trigger": {
                "type": "trigger_fp",
                "targeted": True,
                "rate_miss": 0.0,
                "fell_back": True,
                "validated": False,
                "capped": False,
            }
        },
    )
    report = run_study(_study_config(["noise_fp_trigger", "fp_trigger"], hypotheses=True), tmp_path)
    assert report["ok"] is True
    assert "fp_trigger" in report["withdrawals"]["withdrawn"]
    contrasts = {row["name"]: row for row in report["analysis"]["h1"]["contrasts"]}
    assert contrasts["noise_fp_trigger - fp_trigger"]["status"] == "withdrawn"


def _p4_config(
    tmp_path: Path, *, hash_final: float = 0.8, tol_final: float = 0.5, oracle_final: float = 0.5
) -> dict:
    audit = tmp_path / "audit"
    audit.mkdir(parents=True)
    payload = {
        name: _forecast(tau)
        for name, tau in (
            ("oracle", 0.2),
            ("noise_fp", 0.4),
            ("fp_tolerance", 0.6),
            ("fp_hash", 0.5),
        )
    }
    lock_forecasts(payload, audit / "lock.json", note="p4")
    write_json(
        audit / "calibration.json",
        {
            "arms": {
                "oracle": {"type": "oracle"},
                "noise_fp": {"type": "resampled"},
                "fp_tolerance": {
                    "type": "tolerance_fp",
                    "channel": {"fpr": 0.15, "fnr": 0.0},
                    "target_fpr": 0.15,
                },
                "fp_hash": {
                    "type": "hash_fp",
                    "channel": {"fpr": 0.16, "fnr": 0.0},
                    "target_fpr": 0.15,
                },
            }
        },
    )
    return {
        "audit_dir": str(audit),
        "title": "P4 synthetic",
        "seeds": [0, 1],
        "n_boot": 30,
        "arms": ["oracle", "noise_fp", "fp_tolerance", "fp_hash"],
        "families": [
            {
                "control": "noise_fp",
                "kind": "fp",
                "shared": ["fp_tolerance"],
                "local": ["fp_hash"],
            }
        ],
        "train": {"steps": 4, "model_id": "tiny"},
        "gates": {"oracle_gain": 0.10, "fate_gap": 0.05, "local_tol": 0.05, "alpha": 0.05},
        "_hash_final": hash_final,
        "_tol_final": tol_final,
        "_oracle_final": oracle_final,
    }


def _patch_p4(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    finals = {
        "oracle": config.pop("_oracle_final"),
        "noise_fp": 0.8,
        "fp_tolerance": config.pop("_tol_final"),
        "fp_hash": config.pop("_hash_final"),
    }

    def train_arm(audit_dir, arm, train_config, out_dir, **kwargs):
        seed = int(train_config["seed"])
        rate = (0.8 if seed == 0 else 0.2) if arm == "fp_tolerance" else 0.0
        _write_run(
            Path(out_dir),
            arm,
            int(train_config["steps"]),
            step0=0.2,
            final=finals[arm],
            monitors_wrong={"fp_tolerance": rate},
        )

    monkeypatch.setattr("vfates.experiment.train_arm", train_arm)


def _check(report: dict, check_id: str) -> dict:
    return next(row for row in report["checks"] if row["id"] == check_id)


def test_p4_go_is_the_oracle_signal_and_fates_are_documented(tmp_path: Path, monkeypatch) -> None:
    from vfates.pilots.p4_micro_grpo import run

    config = _p4_config(tmp_path)
    _patch_p4(monkeypatch, config)
    report = run(config, tmp_path / "p4")
    assert report["ok"] is True
    assert _check(report, "P4.1")["status"] == "pass"
    assert _check(report, "P4.2")["status"] == "documented"
    assert _check(report, "P4.2")["passed"] is True
    assert _check(report, "P4.3.fp_hash")["status"] == "documented"
    assert _check(report, "P4.3.fp_hash")["passed"] is True
    assert _check(report, "P4.4")["status"] == "documented"
    assert _check(report, "P4.5.fp_tolerance")["value"] == pytest.approx(0.5)
    assert _check(report, "P4.5.fp_tolerance")["status"] == "documented"

    miss = _p4_config(tmp_path / "screen", hash_final=0.2)
    _patch_p4(monkeypatch, miss)
    outside = run(miss, tmp_path / "screen-out")
    assert outside["ok"] is True
    assert _check(outside, "P4.3.fp_hash")["passed"] is False
    assert "outside" in _check(outside, "P4.3.fp_hash")["metric"]

    small = _p4_config(tmp_path / "shared", tol_final=0.79)
    _patch_p4(monkeypatch, small)
    no_hit = run(small, tmp_path / "shared-out")
    assert no_hit["ok"] is True
    assert _check(no_hit, "P4.2")["passed"] is False
    assert _check(no_hit, "P4.2")["value"] == 0.0

    flat = _p4_config(tmp_path / "flat", oracle_final=0.25)
    _patch_p4(monkeypatch, flat)
    no_signal = run(flat, tmp_path / "flat-out")
    assert no_signal["ok"] is False
    assert _check(no_signal, "P4.1")["status"] == "fail"


def test_replication_model_rule_and_screen(tmp_path: Path, monkeypatch) -> None:
    from vfates.pilots.p5_gpu_gate import (
        LLAMA_3B,
        QWEN_3B,
        base_screen_summary,
        choose_replication_model,
        run_replication_screen,
    )

    assert choose_replication_model(0.30, 0.01)["model_id"] == LLAMA_3B
    assert choose_replication_model(0.05, 0.0)["model_id"] == QWEN_3B
    assert choose_replication_model(0.30, 0.06)["model_id"] == QWEN_3B
    assert choose_replication_model(0.80, 0.0)["model_id"] == QWEN_3B
    summary = base_screen_summary([1, 0, 1, 1], [False, True, False, False])
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["truncation"] == pytest.approx(0.25)

    def factory(backend, model_id, **kwargs):
        def generate(prompts, n, **kw):
            assert kw["max_tokens"] == 32
            return [
                [GenOut(text="not-a-number", n_tokens=2, finish_reason="stop") for _ in range(n)]
                for _ in prompts
            ]

        return SimpleNamespace(generate=generate, model_id=model_id)

    monkeypatch.setattr("vfates.gen.factory.make_generator", factory)
    report = run_replication_screen(
        {"n_prompts": 8, "samples": 1, "max_tokens": 32, "pool_seed": 50_000_000},
        tmp_path,
    )
    assert report["ok"] is True
    assert report["tables"]["choice"]["model_id"] == QWEN_3B
    assert report["tables"]["choice"]["preferred_ok"] is False
    assert set(report["tables"]["models"]) == {LLAMA_3B, QWEN_3B}


def test_full_gate_oracle_gain_is_the_only_scientific_gate(tmp_path: Path, monkeypatch) -> None:
    from vfates.pilots.p5_gpu_gate import arm_seed_timing, full_gate, project_measured_cost

    run = tmp_path / "timed"
    append_jsonl(
        run / "metrics.jsonl",
        {
            "wall_s": 30.0,
            "tokens_generated": 100.0,
            "gen_seconds": 2.0,
            "mean_completion_length": 50.0,
        },
    )
    timing = arm_seed_timing(run)
    assert timing["train_seconds"] == pytest.approx(30.0)
    assert timing["source"] == "wall_s"
    assert timing["gen_tok_s"] == pytest.approx(50.0)
    clocked = tmp_path / "clock"
    write_json(clocked / "timing.json", {"train_arm_wall_s": 12.5, "includes_eval": True})
    append_jsonl(clocked / "metrics.jsonl", {"step": 1})
    clock = arm_seed_timing(clocked)
    assert clock["source"] == "timing.json"
    assert clock["train_seconds"] == pytest.approx(12.5)
    assert clock["eval_seconds"] == pytest.approx(0.0)
    assert clock["eval_source"] == "in_train_clock"
    assert clock["eval_included_in_wall"] is True
    write_json(run / "timing.json", {"train_arm_wall_s": 50.0, "includes_eval": True})
    stepped = arm_seed_timing(run)
    assert stepped["source"] == "wall_s"
    assert stepped["train_seconds"] == pytest.approx(30.0)
    assert stepped["eval_seconds"] == pytest.approx(20.0)
    assert stepped["eval_source"] == "train_arm_clock_minus_steps"
    append_jsonl(run / "trl_eval.jsonl", {"step": 1, "eval_seconds": 4.0})
    assert arm_seed_timing(run)["eval_source"] == "trl_eval"
    assert arm_seed_timing(run)["eval_seconds"] == pytest.approx(4.0)
    projected = project_measured_cost(
        train_seconds_per_arm_seed=3600.0,
        reference_steps=300,
        spec={
            "steps": 300,
            "full_runs": 1,
            "early_runs": 0,
            "early_steps": 0,
            "eval_every": 25,
            "eval_prompts": 10,
            "eval_samples": 1,
            "audit_prompts": 10,
            "audit_samples": 1,
            "usd_per_h100_hour": 2.0,
        },
        eval_seconds_per_arm_seed=0.0,
        gen_tok_s=1000.0,
        mean_completion_len=100.0,
    )
    assert projected["train_hours"] == pytest.approx(1.0)
    assert projected["eval_source"] == "measured_eval_seconds"
    assert projected["eval_tokens_included"] is True
    assert projected["audit_source"] == "measured_gen_rate"
    assert projected["audit_hours"] == pytest.approx(10 * 100.0 / 1000.0 / 3600.0)
    base = {
        "train_seconds_per_arm_seed": 3600.0,
        "reference_steps": 300,
        "eval_seconds_per_arm_seed": 0.0,
    }
    spec = {
        "steps": 300,
        "full_runs": 1,
        "eval_every": 25,
        "eval_prompts": 10,
        "eval_samples": 1,
        "audit_prompts": 10,
        "audit_samples": 2,
        "usd_per_h100_hour": 2.0,
    }
    clocked_audit = project_measured_cost(
        **base,
        spec=spec,
        gen_tok_s=1000.0,
        mean_completion_len=100.0,
        audit_seconds=36.0,
        audit_completions=5,
    )
    assert clocked_audit["audit_source"] == "measured_audit_wall_clock"
    assert clocked_audit["audit_hours"] == pytest.approx(36.0 / 5 * 20 / 3600.0)
    with pytest.raises(ValueError, match="cannot be priced"):
        project_measured_cost(**base, spec=spec)
    placeholder = project_measured_cost(
        **base, spec={**spec, "h100_gen_tok_s": 500.0, "mean_completion_len": 50.0}
    )
    assert placeholder["audit_source"] == "placeholder_gen_rate"
    assert placeholder["audit_hours"] == pytest.approx(20 * 50.0 / 500.0 / 3600.0)

    def run_audit(config, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    def train_arm(audit_dir, arm, train_config, out_dir, **kwargs):
        seed = int(train_config["seed"])
        final = (0.50 if seed == 0 else 0.58) if arm == "oracle" else 0.10
        _write_run(
            Path(out_dir),
            arm,
            int(train_config["steps"]),
            step0=0.20,
            final=final,
            metrics={
                "wall_s": 100.0,
                "tokens_generated": 200.0,
                "gen_seconds": 2.0,
                "mean_completion_length": 40.0,
            },
        )

    monkeypatch.setattr("vfates.audit.run_audit", run_audit)
    monkeypatch.setattr("vfates.experiment.train_arm", train_arm)
    config = {
        "full": {
            "audit": {"pools": {"train": 3, "heldout": 2}, "sampling": {"n": 4}},
            "gate": {
                "seeds": [0, 1],
                "arms": ["oracle", "noise_fp"],
                "families": [
                    {"control": "noise_fp", "kind": "fp", "shared": ["oracle"], "local": []}
                ],
                "n_boot": 20,
                "train": {"steps": 4},
                "gates": {"oracle_gain": 0.10, "seed_sd_threshold": 0.03},
            },
        },
        "e1_cost": {
            "steps": 4,
            "full_runs": 1,
            "full_runs_if_extra_seeds": 2,
            "early_runs": 0,
            "early_steps": 0,
            "eval_every": 2,
            "eval_prompts": 2,
            "eval_samples": 1,
            "audit_prompts": 2,
            "audit_samples": 1,
            "usd_per_h100_hour": 2.5,
        },
    }
    report = full_gate(config, tmp_path / "gate")
    assert report["ok"] is True
    assert _check(report, "P5.7")["status"] == "pass"
    assert _check(report, "P5.8.oracle")["status"] == "documented"
    assert report["tables"]["h1_h2_seeds"] == 5
    assert report["tables"]["oracle_sd"] == pytest.approx(0.0565685424949238, rel=1e-6)
    assert "3" in report["tables"]["power"]
    assert "5" in report["tables"]["power"]
    assert report["tables"]["e1_projection"]["seeds_5_on_h1_h2"]["eval_tokens_included"] is True
    assert report["tables"]["timing"]["source"] == "wall_s"
    assert report["tables"]["timing"]["audit_completions"] == 20
    projection = report["tables"]["e1_projection"]["seeds_5_on_h1_h2"]
    assert projection["audit_source"] == "measured_audit_wall_clock"
    resumed = full_gate(config, tmp_path / "gate")
    assert resumed["tables"]["timing"]["audit_wall_s"] == report["tables"]["timing"]["audit_wall_s"]

    def bare(audit_dir, arm, train_config, out_dir, **kwargs):
        _write_run(Path(out_dir), arm, 4, step0=0.20, final=0.50, metrics={"step": 1})

    monkeypatch.setattr("vfates.experiment.train_arm", bare)
    bare_report = full_gate(config, tmp_path / "bare-clock")
    assert bare_report["ok"] is True
    assert bare_report["tables"]["timing"]["source"] == "timing.json"
    assert bare_report["tables"]["timing"]["eval_seconds_per_arm_seed"] == pytest.approx(0.0)
    assert (tmp_path / "bare-clock" / "runs" / "oracle__s0" / "timing.json").exists()

    def low(audit_dir, arm, train_config, out_dir, **kwargs):
        _write_run(Path(out_dir), arm, 4, step0=0.20, final=0.22)

    monkeypatch.setattr("vfates.experiment.train_arm", low)
    quiet = full_gate(config, tmp_path / "low")
    assert quiet["ok"] is False
    assert _check(quiet, "P5.7")["status"] == "fail"

    def empty(audit_dir, arm, train_config, out_dir, **kwargs):
        return None

    monkeypatch.setattr("vfates.experiment.train_arm", empty)
    missing = full_gate(config, tmp_path / "missing")
    assert missing["ok"] is False
    assert _check(missing, "P5.0")["status"] == "fail"


def test_model_ids_must_agree_before_the_audit(tmp_path: Path, monkeypatch) -> None:
    calls = _install(monkeypatch)
    config = _study_config()
    config["audit"]["generator"] = {"model_id": "tiny"}
    config["audit"]["features"] = {"model_id": "other"}
    with pytest.raises(ValueError, match=r"audit\.features\.model_id=other"):
        run_study(config, tmp_path)
    assert calls == []


def test_every_config_trains_the_audited_policy() -> None:
    def load(path: str) -> dict:
        return yaml.safe_load((ROOT / path).read_text())

    pairs = []
    for path in sorted((ROOT / "configs/experiments").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        pairs.append((path.name, cfg["audit"], cfg["study"]["train"]))
    p1 = load("pilots/p1_harness/config.yaml")
    pairs.append(("p1", p1["audit"], {**p1["train"], "model_id": p1["model_id"]}))
    p5 = load("pilots/p5_gpu_gate/config.yaml")
    pairs.append(("p5 dry run", p5["audit"], p5["train"]))
    pairs.append(("p5 full", p5["full"]["audit"], p5["full"]["gate"]["train"]))
    pairs.append(
        (
            "p4",
            load("pilots/p3_estimator/config.yaml")["audit"],
            load("pilots/p4_micro_grpo/config.yaml")["train"],
        )
    )
    for name, audit, train in pairs:
        _check_model_ids(audit, {"train": train})
        policy = GRPOConfig.from_dict({**train, "out_dir": "unused", "seed": 0})
        try:
            _check_audited_policy(audit, policy)
        except ValueError as error:
            raise AssertionError(f"{name}: {error}") from error


def test_experiment_configs_match_the_preregistration() -> None:
    e1 = yaml.safe_load((ROOT / "configs/experiments/e1_chain_sum.yaml").read_text())
    assert len(e1["study"]["arms"]) == 13
    assert "natural_string_match" not in e1["study"]["arms"]
    assert "fp_tolerance_decoy" in e1["study"]["arms"]
    assert "noise_fp_trigger" in e1["study"]["arms"]
    assert e1["audit"]["forecast"]["loss_agg"] == "seq_mean"
    assert e1["study"]["train"]["loss_agg"] == "seq_mean"
    assert "gates" not in e1["study"]
    assert e1["analysis"]["n_boot"] == 10000
    assert e1["analysis"]["margin"] == 0.05
    assert e1["analysis"]["early"] == {"enabled": True, "seed": 99, "steps": 25}
    assert e1["analysis"]["withdrawal"]["rate_miss_max"] == 0.02
    assert e1["study"]["seed_rule"]["apply"] is False
    assert e1["cost_model"]["audit_prompts"] == 1408
    assert e1["cost_model"]["eval_train_prompts"] == 128
    assert e1["cost_model"]["full_runs"] == 39

    kl = yaml.safe_load((ROOT / "configs/experiments/e1_kl_check.yaml").read_text())
    assert kl["audit_dir"] == "runs/e1/audit"
    assert kl["study"]["arms"] == ["oracle", "fp_tolerance"]
    assert kl["study"]["seeds"] == [0]
    assert kl["study"]["train"]["beta_kl"] == 0.01
    assert kl["analysis"]["hypotheses"] is False
    assert kl["analysis"]["early"]["enabled"] is False

    rep = yaml.safe_load((ROOT / "configs/experiments/e1_replication_3b.yaml").read_text())
    assert rep["study"]["seeds"] == [0, 1]
    assert rep["study"]["arms"] == ["oracle", "noise_fp", "fp_hash", "fp_tolerance"]
    assert rep["analysis"]["allow_partial_menu"] is True
    assert rep["analysis"]["hypotheses"] is True
    assert rep["analysis"]["h3"] == {"tested": False}
    assert rep["analysis"]["early"] == {"enabled": False}
    assert "audit_dir" not in rep

    p4 = yaml.safe_load((ROOT / "pilots/p4_micro_grpo/config.yaml").read_text())
    assert p4["train"]["max_tokens"] == 512
    assert p4["train"]["eval_max_tokens"] == 512
    assert "fp_tolerance_decoy" in p4["arms"]
    assert "noise_fp_trigger" in p4["arms"]
    families = {row["control"]: row for row in p4["families"]}
    assert families["noise_fp_trigger"]["shared"] == ["fp_trigger"]
    assert "fp_tolerance_decoy" in families["noise_fp"]["shared"]
    assert "spearman" not in p4["gates"]

    p5 = yaml.safe_load((ROOT / "pilots/p5_gpu_gate/config.yaml").read_text())
    assert p5["full"]["audit"]["pools"] == {"train": 512, "heldout": 384, "seed": 50000000}
    assert p5["cost_model"]["audit_prompts"] == 896
    assert "fp_tolerance_decoy" in p5["full"]["gate"]["arms"]
    assert "noise_fp_trigger" in p5["full"]["gate"]["arms"]
    assert p5["full"]["gate"]["gates"] == {"oracle_gain": 0.10, "seed_sd_threshold": 0.03}
    assert p5["e1_cost"]["full_runs_if_extra_seeds"] == 57
