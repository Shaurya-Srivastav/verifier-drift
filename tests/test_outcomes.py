"""Held-out curves, paired contrasts, and locked forecast scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfates.analysis.outcomes import (
    PERMUTATION_MC,
    adjusted_contrast,
    crossed_pair_dacc,
    eval_curve,
    holm,
    holm_adjusted,
    knife_edge,
    paired_contrast,
    paired_contrast_crossed,
    paired_contrast_seeds,
    permutation_spearman_p,
    prompt_accuracy,
    rank_bootstrap,
    require_final_step,
    run_outcome,
    score_forecasts,
    score_lock,
    student_t_ppf,
    time_averaged,
)
from vfates.analysis.power import contrast_se, minimum_detectable_difference, tost_power
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
    with pytest.raises(ValueError, match="step is required"):
        prompt_accuracy(run)
    assert prompt_accuracy(run, step=4) == {"p0": 0.25, "p1": 0.75}
    assert prompt_accuracy(run, "heldout", 0) == {"p0": 0.0, "p1": 1.0}
    assert prompt_accuracy(run, "train", 4) == {"p0": 1.0}
    with pytest.raises(ValueError, match="step"):
        prompt_accuracy(run, "heldout", 3)
    held_only = tmp_path / "held"
    append_jsonl(
        held_only / "eval_prompts.jsonl", {"step": 0, "split": "heldout", "acc": {"p0": 1}}
    )
    with pytest.raises(ValueError, match="train"):
        prompt_accuracy(held_only, "train", 0)
    with pytest.raises(FileNotFoundError):
        prompt_accuracy(tmp_path / "absent", step=0)


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
    same = paired_contrast(same_a, same_b, step=4, seed=0)
    assert same["diff"] == pytest.approx(0.0)
    assert same["lo"] <= 0.0 <= same["hi"]
    assert same["n_prompts"] == 8

    high = tmp_path / "high"
    _write_prompts(high, {key: value + 0.1 for key, value in base.items()})
    lifted = paired_contrast(high, same_b, step=4, seed=0)
    assert lifted["diff"] == pytest.approx(0.1)
    assert lifted["lo"] > 0.0
    assert lifted["n_prompts"] == 8

    _write_prompts(tmp_path / "oa", {"p0": 0.6, "only_a": 0.0})
    _write_prompts(tmp_path / "ob", {"p0": 0.5, "only_b": 1.0})
    mixed = paired_contrast(tmp_path / "oa", tmp_path / "ob", step=4, seed=0)
    assert mixed["n_prompts"] == 1
    assert mixed["diff"] == pytest.approx(0.1)

    _write_prompts(tmp_path / "da", {"a": 1.0})
    _write_prompts(tmp_path / "db", {"b": 1.0})
    with pytest.raises(ValueError, match="share no"):
        paired_contrast(tmp_path / "da", tmp_path / "db", step=4)


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


def _eval_at(path: Path, step: int, acc: dict[str, float]) -> None:
    append_jsonl(path / "eval.jsonl", {"step": step, "split": "heldout", "oracle_acc": 0.5})
    append_jsonl(path / "eval_prompts.jsonl", {"step": step, "split": "heldout", "acc": acc})


def test_require_final_step_rejects_a_missing_preregistered_eval(tmp_path: Path) -> None:
    good = tmp_path / "good"
    _eval_at(good, 4, {"p0": 0.2})
    require_final_step([good], 4)
    prompts_only = tmp_path / "prompts"
    append_jsonl(
        prompts_only / "eval_prompts.jsonl", {"step": 4, "split": "heldout", "acc": {"p0": 0.2}}
    )
    with pytest.raises(ValueError, match="no heldout eval at step 4"):
        require_final_step([good, prompts_only], 4)


def test_one_seed_crossed_bootstrap_matches_the_prompt_bootstrap(tmp_path: Path) -> None:
    values = [0.1, -0.2, 0.0, 0.4, 0.05, -0.1, 0.3]
    row = np.array([values], dtype=float)
    crossed = paired_contrast_crossed(row, n_boot=40, seed=0, margin=0.05)
    assert crossed["p_sup"] == crossed["prompt_only"]["p_sup"]
    assert crossed["p_tost"] == crossed["prompt_only"]["p_tost"]
    assert crossed["lo"] == crossed["prompt_only"]["lo"]
    assert crossed["hi"] == crossed["prompt_only"]["hi"]
    assert crossed["seed_t"]["defined"] is False

    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_prompts(left, {f"p{i}": value for i, value in enumerate(values)})
    _write_prompts(right, {f"p{i}": 0.0 for i in range(len(values))})
    prompt = paired_contrast_seeds([(left, right)], step=4, n_boot=40, seed=0, margin=0.05)
    assert prompt["diff"] == pytest.approx(crossed["point"])
    assert prompt["p_le_0"] == pytest.approx(crossed["p_le_0"])
    assert prompt["lo"] == pytest.approx(crossed["lo"])
    assert prompt["p_tost"] == pytest.approx(crossed["p_tost"])

    two = np.vstack([row, row + 0.2])
    both = paired_contrast_crossed(two, n_boot=40, seed=0, margin=0.05)
    assert both["n_seeds"] == 2
    assert both["per_seed"][1] == pytest.approx(both["per_seed"][0] + 0.2)


def test_nonfinite_differences_and_holm_raise() -> None:
    with pytest.raises(ValueError, match="non-finite prompt differences"):
        paired_contrast_crossed([[0.1, float("nan")]])
    with pytest.raises(ValueError, match="non-finite p-value"):
        holm({"a": float("nan")})
    adjusted = holm_adjusted({"a": 0.01, "b": 0.04, "c": 0.04})
    assert adjusted == {
        "a": pytest.approx(0.03),
        "b": pytest.approx(0.08),
        "c": pytest.approx(0.08),
    }
    assert holm({"a": 0.01, "b": 0.04, "c": 0.04}) == {"a": True, "b": False, "c": False}


def test_rank_bootstrap_reports_explicit_quantiles(tmp_path: Path) -> None:
    forecasts = {}
    runs = {}
    for i, name in enumerate(("a", "b", "c")):
        forecasts[name] = _forecast(float(i), 0.1)
        run = tmp_path / name
        _write_prompts(run, {"p0": float(i), "p1": float(i)})
        runs[name] = [run]
    ranked = rank_bootstrap(forecasts, runs, step=4, n_boot=20, quantiles=(0.05, 0.95), seed=0)
    assert ranked["quantiles"] == [0.05, 0.95]
    assert ranked["n_seeds"] == 1
    assert ranked["tau_pool"]["quantiles"]["0.05"] == pytest.approx(1.0)
    assert ranked["tau_pool"]["rho"] == pytest.approx(1.0)


def test_permutation_spearman_exact_and_monte_carlo() -> None:
    exact = permutation_spearman_p([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert exact["method"] == "exact"
    assert exact["p"] == pytest.approx(1.0 / 6.0)
    assert exact["n_perm"] == 6
    assert PERMUTATION_MC == 200_000
    monte = permutation_spearman_p(list(range(10)), list(range(10)), n_mc=20, seed=0)
    assert monte["method"] == "monte_carlo"
    assert monte["n_perm"] == 20
    assert monte["rho"] == pytest.approx(1.0)
    assert monte["p"] == pytest.approx(1.0 / 21.0)


def test_knife_edge_rise_then_fall() -> None:
    assert knife_edge([0.2, 0.4, 0.25]) is True
    assert knife_edge([0.2, 0.4, 0.35]) is False
    assert knife_edge([0.2, 0.24, 0.10]) is False
    assert knife_edge([0.2, 0.5, 0.6, 0.4]) is True
    assert knife_edge([0.5, 0.2, 0.55]) is False
    with pytest.raises(ValueError, match="non-finite"):
        knife_edge([0.2, float("nan")])


def test_time_average_and_adjusted_contrast() -> None:
    assert time_averaged([0, 10, 20], [0.0, 1.0, 0.0]) == pytest.approx(0.5)
    assert adjusted_contrast(0.2, 0.1, 2.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="finite"):
        adjusted_contrast(0.2, float("nan"), 1.0)


def test_student_t_critical_values() -> None:
    assert student_t_ppf(0.975, 1) == pytest.approx(12.7062, rel=1e-3)
    assert student_t_ppf(0.975, 2) == pytest.approx(4.30265, rel=1e-3)
    assert student_t_ppf(0.975, 4) == pytest.approx(2.77645, rel=1e-3)


def test_crossed_power_matches_r4() -> None:
    p4_se = contrast_se(n_prompts=128, n_seeds=1, n_samples=4, mu=0.398, var_p=0.044, seed_sd=0.0)
    assert p4_se == pytest.approx(0.028, abs=0.001)
    p4 = minimum_detectable_difference(
        0.0, n_seeds=1, n_prompts=128, n_samples=4, mu=0.398, var_p=0.044, margin=0.0
    )
    assert p4["mdd"] == pytest.approx(0.069, abs=0.002)
    assert tost_power(0.0, p4_se, margin=0.05, alpha=0.05) == pytest.approx(0.14, abs=0.02)
    e1_se = contrast_se(n_prompts=384, n_seeds=3, n_samples=4, mu=0.215, var_p=0.037)
    assert e1_se == pytest.approx(0.0076, abs=0.0003)
    wider = contrast_se(n_prompts=384, n_seeds=3, seed_sd=0.04)
    assert wider > e1_se
    for n_seeds in (3, 5):
        row = minimum_detectable_difference(0.0, n_seeds=n_seeds, margin=0.05)
        assert row["mdd"] > 0.05
        assert row["n_seeds"] == n_seeds


def test_crossed_pair_dacc_one_seed_list_is_the_fully_crossed_draw() -> None:
    rng = np.random.default_rng(3)
    arm = rng.random((4, 3, 7))
    control = rng.random((4, 3, 7))
    point, boots = crossed_pair_dacc(arm, control, n_boot=40, seed=5)
    draw = np.random.default_rng(5)
    prompt_idx = draw.integers(0, 7, size=(40, 7))
    seed_idx = draw.integers(0, 3, size=(40, 3))
    expected = np.stack(
        [
            arm[:, seed_idx[b]][:, :, prompt_idx[b]].mean(axis=(1, 2))
            - control[:, seed_idx[b]][:, :, prompt_idx[b]].mean(axis=(1, 2))
            for b in range(40)
        ]
    )
    assert point == pytest.approx(arm.mean(axis=(1, 2)) - control.mean(axis=(1, 2)))
    assert boots == pytest.approx(expected)


def test_crossed_pair_dacc_averages_each_pair_over_its_own_seeds() -> None:
    wide_arm = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    narrow_arm = np.ones((3, 2))
    point, boots = crossed_pair_dacc(
        [wide_arm, narrow_arm],
        [np.zeros((5, 2)), np.zeros((3, 2))],
        n_boot=200,
        seed=0,
        seed_lists=[[0, 1, 2, 3, 4], [0, 1, 2]],
    )
    assert point == pytest.approx([0.6, 1.0])
    assert boots.shape == (200, 2)
    assert set(np.round(boots[:, 0] * 5).astype(int)) <= set(range(6))
    assert len(set(boots[:, 0])) > 1
    assert boots[:, 1] == pytest.approx(np.ones(200))
    with pytest.raises(ValueError, match="one seed per row"):
        crossed_pair_dacc([wide_arm], [np.zeros((5, 2))], n_boot=5, seed_lists=[[0, 1]])
    with pytest.raises(ValueError, match="same prompts"):
        crossed_pair_dacc(
            [np.ones((2, 2)), np.ones((2, 3))], [np.ones((2, 2)), np.ones((2, 3))], n_boot=5
        )
