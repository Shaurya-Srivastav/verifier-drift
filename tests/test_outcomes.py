"""Held-out curves, paired contrasts, and locked forecast scores."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfates.analysis.outcomes import (
    eval_curve,
    paired_contrast,
    prompt_accuracy,
    run_outcome,
    score_forecasts,
    score_lock,
)
from vfates.forecast.lockbox import lock_forecasts
from vfates.utils.io import append_jsonl, write_json


def _write_prompts(path: Path, acc: dict[str, float], step: int = 4) -> None:
    append_jsonl(path / "eval_prompts.jsonl", {"step": step, "split": "heldout", "acc": acc})


def _forecast(tau: float, youden: float) -> dict:
    return {
        "transfer": {
            "tau_pool": tau,
            "tau_pool_ci": [tau - 0.05, tau + 0.05],
            "tau_hard": tau,
            "tau_rel": 0.4,
            "reliability": 0.8,
            "cos_gV_gO": 0.2,
            "base_acc_heldout": 0.3,
        },
        "step0": {"J": youden, "fpr": 0.15, "fnr": 0.3},
        "mean_precision": tau,
        "modal_precision": {"value": tau, "tie_fraction": 0.0},
    }


def test_eval_curve_sorts_and_keeps_the_later_duplicate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    append_jsonl(run / "eval.jsonl", {"step": 2, "split": "heldout", "oracle_acc": 0.1})
    append_jsonl(run / "eval.jsonl", {"step": 0, "split": "heldout", "oracle_acc": 0.0})
    append_jsonl(run / "eval.jsonl", {"step": 2, "split": "train", "oracle_acc": 0.9})
    append_jsonl(run / "eval.jsonl", {"step": 2, "split": "heldout", "oracle_acc": 0.4})
    curve = eval_curve(run, "heldout")
    assert [row["step"] for row in curve] == [0, 2]
    assert curve[0]["oracle_acc"] == 0.0
    assert curve[1]["oracle_acc"] == 0.4
    assert eval_curve(run, "train")[0]["oracle_acc"] == 0.9
    assert eval_curve(tmp_path / "missing", "heldout") == []


def test_prompt_accuracy_last_step_explicit_step_and_missing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    append_jsonl(
        run / "eval_prompts.jsonl",
        {"step": 0, "split": "heldout", "acc": {"p0": 0.0, "p1": 1.0}},
    )
    append_jsonl(
        run / "eval_prompts.jsonl",
        {"step": 4, "split": "heldout", "acc": {"p0": 0.2, "p1": 0.2}},
    )
    append_jsonl(run / "eval_prompts.jsonl", {"step": 4, "split": "train", "acc": {"p0": 1.0}})
    append_jsonl(
        run / "eval_prompts.jsonl",
        {"step": 4, "split": "heldout", "acc": {"p0": 0.25, "p1": 0.75}},
    )
    assert prompt_accuracy(run) == {"p0": 0.25, "p1": 0.75}
    assert prompt_accuracy(run, "heldout", 0) == {"p0": 0.0, "p1": 1.0}
    assert prompt_accuracy(run, "train", 4) == {"p0": 1.0}
    with pytest.raises(ValueError, match="step"):
        prompt_accuracy(run, "heldout", 3)
    held_only = tmp_path / "held"
    append_jsonl(
        held_only / "eval_prompts.jsonl", {"step": 0, "split": "heldout", "acc": {"p0": 1}}
    )
    with pytest.raises(ValueError, match="train"):
        prompt_accuracy(held_only, "train")
    with pytest.raises(FileNotFoundError):
        prompt_accuracy(tmp_path / "absent")


def test_run_outcome_reports_heldout_and_onpolicy(tmp_path: Path) -> None:
    run = tmp_path / "run"
    append_jsonl(
        run / "eval.jsonl",
        {
            "step": 4,
            "split": "heldout",
            "oracle_acc": 0.5,
            "fpr": 0.2,
            "fnr": 0.3,
            "precision": 0.4,
            "monitors": {"has_answer": 0.1},
            "monitors_wrong": {"fp_hash": 0.9},
        },
    )
    append_jsonl(
        run / "eval.jsonl",
        {
            "step": 0,
            "split": "heldout",
            "oracle_acc": 0.2,
            "monitors": {"has_answer": 0.8},
            "monitors_wrong": {"fp_hash": 0.1},
        },
    )
    append_jsonl(
        run / "eval.jsonl",
        {
            "step": 2,
            "split": "heldout",
            "oracle_acc": 0.9,
            "monitors": {},
            "monitors_wrong": {},
        },
    )
    append_jsonl(
        run / "eval.jsonl",
        {
            "step": 4,
            "split": "heldout",
            "oracle_acc": 0.4,
            "fpr": 0.11,
            "fnr": 0.22,
            "precision": 0.33,
            "monitors": {"has_answer": 0.7},
            "monitors_wrong": {"fp_hash": 0.25},
        },
    )
    append_jsonl(run / "eval.jsonl", {"step": 0, "split": "train", "oracle_acc": 0.1})
    append_jsonl(run / "eval.jsonl", {"step": 4, "split": "train", "oracle_acc": 0.6})
    for step in range(12):
        append_jsonl(
            run / "metrics.jsonl",
            {"step": step, "reward_V_mean": float(step), "oracle_mean": float(step) / 2},
        )
    write_json(run / "run.json", {"arm": "fp_tolerance"})
    out = run_outcome(run)
    assert out["arm"] == "fp_tolerance"
    assert out["steps"] == 4
    assert out["heldout_step0"] == pytest.approx(0.2)
    assert out["heldout_final"] == pytest.approx(0.4)
    assert out["heldout_delta"] == pytest.approx(0.2)
    assert out["heldout_peak"] == pytest.approx(0.9)
    assert out["curve"] == [(0, 0.2), (2, 0.9), (4, 0.4)]
    assert out["train_step0"] == pytest.approx(0.1)
    assert out["train_final"] == pytest.approx(0.6)
    assert out["eval_fpr_final"] == pytest.approx(0.11)
    assert out["monitors_wrong_final"] == {"fp_hash": 0.25}
    assert out["monitors_wrong_step0"] == {"fp_hash": 0.1}
    assert out["monitors_final"] == {"has_answer": 0.7}
    assert out["onpolicy_reward_last10"] == pytest.approx(6.5)
    assert out["onpolicy_oracle_last10"] == pytest.approx(3.25)

    bare = tmp_path / "bare"
    append_jsonl(bare / "eval.jsonl", {"step": 0, "split": "train", "oracle_acc": 1.0})
    with pytest.raises(ValueError, match="held-out"):
        run_outcome(bare)


def test_paired_contrast_difference_and_prompt_overlap(tmp_path: Path) -> None:
    base = {f"p{i}": i / 20 for i in range(8)}
    same_a = tmp_path / "same_a"
    same_b = tmp_path / "same_b"
    _write_prompts(same_a, base)
    _write_prompts(same_b, base)
    same = paired_contrast(same_a, same_b, seed=0)
    assert same["diff"] == pytest.approx(0.0)
    assert same["lo"] <= 0.0 <= same["hi"]
    assert same["n_prompts"] == 8

    high = tmp_path / "high"
    _write_prompts(high, {key: value + 0.1 for key, value in base.items()})
    lifted = paired_contrast(high, same_b, seed=0)
    assert lifted["diff"] == pytest.approx(0.1)
    assert lifted["lo"] > 0.0
    assert lifted["n_prompts"] == 8

    _write_prompts(tmp_path / "oa", {"p0": 0.6, "only_a": 0.0})
    _write_prompts(tmp_path / "ob", {"p0": 0.5, "only_b": 1.0})
    mixed = paired_contrast(tmp_path / "oa", tmp_path / "ob", seed=0)
    assert mixed["n_prompts"] == 1
    assert mixed["diff"] == pytest.approx(0.1)

    _write_prompts(tmp_path / "da", {"a": 1.0})
    _write_prompts(tmp_path / "db", {"b": 1.0})
    with pytest.raises(ValueError, match="share no"):
        paired_contrast(tmp_path / "da", tmp_path / "db")


def test_score_forecasts_and_lock_follow_tau_pool_order(tmp_path: Path) -> None:
    names = ["oracle", "fp_hash", "fp_tolerance", "noise_fp"]
    taus = [0.15, 0.35, 0.55, 0.8]
    youden = [0.9, 0.6, 0.3, 0.1]
    forecasts: dict = {"fp_trigger": _forecast(0.0, 0.0)}
    runs: dict[str, Path] = {}
    for name, tau, j in zip(names, taus, youden, strict=True):
        forecasts[name] = _forecast(tau, j)
        run = tmp_path / name
        append_jsonl(
            run / "eval.jsonl",
            {
                "step": 0,
                "split": "heldout",
                "oracle_acc": tau,
                "fpr": 0.1,
                "fnr": 0.2,
                "precision": 0.5,
                "monitors": {},
                "monitors_wrong": {},
            },
        )
        _write_prompts(run, {"p0": tau}, step=0)
        write_json(run / "run.json", {"arm": name})
        runs[name] = run
    outcomes = {name: run_outcome(path) for name, path in runs.items()}
    scored = score_forecasts(forecasts, outcomes)
    assert scored["n_arms"] == 4
    assert [row["arm"] for row in scored["rows"]] == names
    assert scored["spearman"]["tau_pool"] == pytest.approx(1.0)
    assert scored["spearman"]["J"] == pytest.approx(-1.0)
    assert scored["tau_pool_mae"] == pytest.approx(0.0)
    assert scored["rows"][0]["heldout_final"] == pytest.approx(taus[0])
    assert scored["rows"][0]["tau_pool"] == pytest.approx(taus[0])

    original = forecasts["oracle"]["transfer"]["tau_pool"]
    lock_path = tmp_path / "lock.json"
    lock_forecasts(forecasts, lock_path, note="step-0")
    forecasts["oracle"]["transfer"]["tau_pool"] = -1.0
    locked = score_lock(lock_path, runs)
    by_arm = {row["arm"]: row for row in locked["rows"]}
    assert by_arm["oracle"]["tau_pool"] == pytest.approx(original)
    assert locked["spearman"]["tau_pool"] == pytest.approx(1.0)
    assert locked["n_arms"] == scored["n_arms"]
