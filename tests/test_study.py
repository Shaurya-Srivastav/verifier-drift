"""Study runner on synthetic directories."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vfates.experiment import _check_audited_policy
from vfates.forecast.lockbox import lock_forecasts
from vfates.study import _check_model_ids, planned_seeds, reanalyze, run_study
from vfates.train.config import GRPOConfig, NonFiniteLossError
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


def test_model_ids_must_agree_before_the_audit(tmp_path: Path, monkeypatch) -> None:
    calls = _install(monkeypatch)
    config = _study_config()
    config["audit"]["generator"] = {"model_id": "tiny"}
    config["audit"]["features"] = {"model_id": "other"}
    with pytest.raises(ValueError, match=r"audit\.features\.model_id=other"):
        run_study(config, tmp_path)
    assert calls == []


def test_every_config_trains_the_audited_policy() -> None:
    pairs = []
    for path in sorted((ROOT / "configs" / "cells").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        pairs.append((path.name, cfg["audit"], cfg["study"]["train"]))
    assert pairs
    for name, audit, train in pairs:
        _check_model_ids(audit, {"train": train})
        policy = GRPOConfig.from_dict({**train, "out_dir": "unused", "seed": 0})
        try:
            _check_audited_policy(audit, policy)
        except ValueError as error:
            raise AssertionError(f"{name}: {error}") from error


def test_experiment_configs_match_the_preregistration() -> None:
    e1 = yaml.safe_load((ROOT / "configs/templates/chain_sum.yaml").read_text())
    assert len(e1["menus"]["full"]) == 17
    assert "natural_string_match" not in e1["menus"]["full"]
    assert "fp_tolerance_decoy" in e1["menus"]["full"]
    assert "noise_fp_trigger" in e1["menus"]["full"]
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
    assert e1["cost_model"]["full_runs"] == 51

    program = yaml.safe_load((ROOT / "configs/program.yaml").read_text())
    by_id = {cell["id"]: cell for cell in program["cells"]}
    kl = next(row for row in by_id["e1-math-qwen3-1.7b"]["variants"] if row["name"] == "kl")
    assert kl["arms"] == ["oracle", "fp_trigger"]
    assert kl["seeds"] == [0]
    assert kl["train_overrides"]["beta_kl"] == 0.01
    streaming = by_id["e1-chain_sum-qwen3-1.7b"]["variants"][0]
    assert streaming["name"] == "streaming"
    assert streaming["arms"] == ["oracle", "fp_hash", "noise_fp"]
    assert streaming["train_overrides"]["data_regime"] == "streaming"

    rep = by_id["e2-chain_sum-qwen3-4b"]
    assert rep["seeds"] == [0, 1]
    assert rep["menu"] == "core"
    assert rep["experiment"] == "E2"


def test_dev_audit_is_refused_unless_the_config_is_a_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vfates.audit as audit_mod

    if not callable(getattr(audit_mod, "require_confirmatory_pools", None)):
        pytest.skip("vfates.audit.require_confirmatory_pools is not available yet")
    calls = _install(monkeypatch)
    config = _study_config()
    run_study(config, tmp_path, only="audit")
    stored = read_json(tmp_path / "audit" / "config.json")
    stored["config"]["pools"] = {"from": "dev"}
    write_json(tmp_path / "audit" / "config.json", stored)
    with pytest.raises(ValueError, match="development-pool"):
        run_study(config, tmp_path, only="train", arm="oracle", seed=0)
    assert [row[0] for row in calls] == ["audit"]
    config["pilot"] = True
    report = run_study(config, tmp_path, only="train", arm="oracle", seed=0)
    assert report["ok"] is True
    assert [row[0] for row in calls] == ["audit", "train"]


def test_only_audit_does_not_train_or_write_the_report(tmp_path: Path, monkeypatch) -> None:
    calls = _install(monkeypatch)
    report = run_study(_study_config(), tmp_path, only="audit")
    assert report["ok"] is True
    assert report["summary"] == "AUDIT"
    assert report["report_written"] is False
    assert [row[0] for row in calls] == ["audit"]
    assert (tmp_path / "withdrawals.json").exists()
    assert not (tmp_path / "report.json").exists()


def test_only_train_selects_one_job_and_passes_hardware_class(tmp_path: Path, monkeypatch) -> None:
    seen: list = []

    def trainer(arm, train_config, out_dir, kwargs):
        seen.append((kwargs.get("hardware_class"), train_config.get("hardware_class")))
        _write_run(out_dir, arm, int(train_config["steps"]))

    calls = _install(monkeypatch, trainer=trainer)
    monkeypatch.setattr("vfates.utils.env.hardware_class", lambda: "h100-80gb", raising=False)
    config = _study_config(arms=["oracle", "fp_trigger"])
    config["study"]["hardware_class"] = "h100-80gb"
    run_study(config, tmp_path, only="audit")
    report = run_study(config, tmp_path, only="train", arm="oracle", seed=0)
    assert report["report_written"] is False
    assert not (tmp_path / "report.json").exists()
    trains = [row for row in calls if row[0] == "train"]
    assert len(trains) == 1
    assert trains[0][1] == "oracle"
    assert trains[0][2] == 0
    assert seen == [("h100-80gb", "h100-80gb")]


def test_only_train_writes_the_report_when_the_cell_is_complete(
    tmp_path: Path, monkeypatch
) -> None:
    _install(monkeypatch)
    config = _study_config()
    run_study(config, tmp_path, only="audit")
    report = run_study(config, tmp_path, only="train", arm="oracle", seed=0)
    assert report["ok"] is True
    assert (tmp_path / "report.json").exists()


def test_chat_template_kwargs_must_agree(tmp_path: Path, monkeypatch) -> None:
    calls = _install(monkeypatch)
    config = _study_config()
    config["audit"]["generator"] = {"chat_template_kwargs": {"enable_thinking": False}}
    config["audit"]["features"] = {"chat_template_kwargs": {"enable_thinking": True}}
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        run_study(config, tmp_path)
    assert calls == []
    agreed = {"enable_thinking": False}
    config["audit"]["features"]["chat_template_kwargs"] = dict(agreed)
    config["study"]["train"]["chat_template_kwargs"] = dict(agreed)
    config["study"]["eval_generator"] = {"kwargs": {"chat_template_kwargs": dict(agreed)}}
    report = run_study(config, tmp_path, only="audit")
    assert report["summary"] == "AUDIT"


def test_hardware_class_mismatch_is_refused_unless_overridden(tmp_path: Path, monkeypatch) -> None:
    calls = _install(monkeypatch)
    monkeypatch.setattr("vfates.utils.env.hardware_class", lambda: "cpu", raising=False)
    monkeypatch.delenv("VFATES_ALLOW_HARDWARE_MISMATCH", raising=False)
    config = _study_config()
    config["study"]["hardware_class"] = "h100-80gb"
    with pytest.raises(ValueError, match="hardware_class"):
        run_study(config, tmp_path)
    assert calls == []
    monkeypatch.setenv("VFATES_ALLOW_HARDWARE_MISMATCH", "1")
    report = run_study(config, tmp_path, only="audit")
    assert report["summary"] == "AUDIT"


def test_same_selection_withdraws_patch_rate(tmp_path: Path, monkeypatch) -> None:
    from vfates.study import _forecast_arm_names, _write_audit_withdrawals

    audit = tmp_path / "audit"
    audit.mkdir()
    write_json(audit / "calibration.json", {"arms": []})
    forecasts = {
        "patch_selection": {"same_selection": True},
        "oracle": {"transfer": {"tau_pool": 0.1}},
    }
    write_json(audit / "forecasts.json", forecasts)
    monkeypatch.setattr("vfates.verifiers.calibrate.arm_validity", lambda report: {})
    record = _write_audit_withdrawals(audit, {}, tmp_path)
    assert record["audit"]["patch_rate"]["reasons"] == ["same_selection"]
    assert "patch_rate" in record["withdrawn"]
    assert _forecast_arm_names(forecasts) == ["oracle"]


def test_variant_run_uses_its_overrides_and_directory(tmp_path: Path, monkeypatch) -> None:
    seen: list = []

    def trainer(arm, train_config, out_dir, kwargs):
        del arm, kwargs
        seen.append((out_dir, train_config.get("beta_kl"), int(train_config["steps"])))
        _write_run(out_dir, "oracle", int(train_config["steps"]))

    _install(monkeypatch, trainer=trainer)
    config = _study_config()
    config["study"]["variants"] = [
        {
            "name": "kl",
            "arms": ["oracle"],
            "seeds": [0],
            "train_overrides": {"beta_kl": 0.01, "steps": 600},
        }
    ]
    run_study(config, tmp_path, only="audit")
    report = run_study(config, tmp_path, only="train", arm="oracle", seed=0, variant="kl")
    assert report["report_written"] is False
    assert seen == [(tmp_path / "variants" / "kl" / "oracle__s0", 0.01, 600)]
    finished = run_study(config, tmp_path, only="train", arm="oracle", seed=0)
    assert finished["ok"] is True
    assert (tmp_path / "runs" / "oracle__s0" / "summary.json").exists()


def test_preempted_summary_is_not_a_complete_run(tmp_path: Path, monkeypatch) -> None:
    def trainer(arm, train_config, out_dir, kwargs):
        _write_run(out_dir, arm, int(train_config["steps"]))
        summary = read_json(out_dir / "summary.json")
        summary["status"] = "preempted"
        write_json(out_dir / "summary.json", summary)

    _install(monkeypatch, trainer=trainer)
    config = _study_config()
    report = run_study(config, tmp_path)
    assert report["ok"] is False
    assert (tmp_path / "report.json").exists()
    again = reanalyze(config, tmp_path)
    assert again["ok"] is False
    assert again["report_written"] is False
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "report.md").exists()


def test_single_job_selector_rejects_a_partial_request(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires arm"):
        run_study(_study_config(), tmp_path, only="train")
    with pytest.raises(ValueError, match="only valid with"):
        run_study(_study_config(), tmp_path, arm="oracle")
