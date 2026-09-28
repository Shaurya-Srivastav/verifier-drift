"""Confirmatory H1 to H6 on synthetic run directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfates.analysis.confirmatory import (
    H3_PAIRS,
    _h4_block,
    _probe_auc,
    _reliability_block,
    audit_withdrawal_reasons,
    confirmatory_report,
    decide_audit_withdrawals,
    render_markdown,
)
from vfates.forecast.lockbox import lock_forecasts
from vfates.utils.io import append_jsonl, write_json
from vfates.verifiers.calibrate import arm_validity

PROMPTS = ["p0", "p1", "p2", "p3"]
FINAL = 4
EARLY = 2
SEEDS = [0, 1]
FINALS = {
    "oracle": 0.70,
    "noise_fp": 0.80,
    "fp_tolerance": 0.30,
    "fp_tolerance_decoy": 0.40,
    "fp_hash": 0.80,
    "noise_fp_trigger": 0.70,
    "fp_trigger": 0.40,
    "noise_fn": 0.60,
    "fn_behavior": 0.40,
    "fn_prompt_subset": 0.60,
    "fn_form": 0.50,
    "noise_fn_form": 0.40,
}
FP_MONITORS = ("fp_tolerance", "fp_tolerance_decoy", "fp_hash", "fp_trigger")
FN_MONITORS = ("fn_behavior", "fn_prompt_subset", "fn_form")
SHARED_AUC = {"fp_tolerance", "fp_tolerance_decoy", "fp_trigger", "fn_behavior", "fn_form"}
SPECIFIC_AUC = {"fp_hash", "fn_prompt_subset"}


def _forecast(name: str, tau: float) -> dict:
    if name in SHARED_AUC:
        auc = 0.80
    elif name in SPECIFIC_AUC:
        auc = 0.40
    else:
        auc = 0.50
    return {
        "transfer": {
            "tau_pool": tau,
            "tau_hard": tau,
            "tau_rel": 0.4,
            "reliability": 0.8,
            "cos_gV_gO": 0.2,
            "base_acc_heldout": 0.3,
            "probe_heldout_auc": auc,
        },
        "step0": {"J": 0.2, "fpr": 0.1, "fnr": 0.2},
        "mean_precision": tau,
        "modal_precision": {"value": tau, "tie_fraction": 0.0},
        "behaviour_alignment": (
            None if name == "oracle" else 0.0 if name.startswith("noise") else 0.5 - tau
        ),
    }


def _monitors(name: str) -> tuple[dict[str, float], dict[str, float]]:
    wrong = {arm: (0.8 if arm == name else 0.2) for arm in FP_MONITORS}
    correct = {arm: (0.9 if arm == name else 0.1) for arm in FN_MONITORS}
    return wrong, correct


def _write_run(
    path: Path,
    *,
    name: str,
    step0: float,
    final: float,
    final_step: int,
    knife: bool = False,
    nan_prompt: bool = False,
    omit_final: bool = False,
) -> None:
    wrong, correct = _monitors(name)
    acc0 = {pid: step0 for pid in PROMPTS}
    acc_final = {pid: final for pid in PROMPTS}
    if nan_prompt:
        acc_final["p0"] = float("nan")

    def held(step: int, oracle: float) -> dict:
        return {
            "step": step,
            "split": "heldout",
            "oracle_acc": oracle,
            "J": final,
            "fpr": 0.1,
            "fnr": 0.2,
            "monitors_wrong": wrong,
            "monitors_correct": correct,
        }

    append_jsonl(path / "eval.jsonl", held(0, step0))
    if knife:
        append_jsonl(path / "eval.jsonl", held(EARLY, 0.95))
    if not omit_final:
        append_jsonl(path / "eval.jsonl", held(final_step, final))
        append_jsonl(
            path / "eval.jsonl", {"step": 0, "split": "train", "oracle_acc": step0, "J": final}
        )
        append_jsonl(
            path / "eval.jsonl",
            {"step": final_step, "split": "train", "oracle_acc": final, "J": final},
        )
    append_jsonl(path / "eval_prompts.jsonl", {"step": 0, "split": "heldout", "acc": acc0})
    if not omit_final:
        append_jsonl(
            path / "eval_prompts.jsonl", {"step": final_step, "split": "heldout", "acc": acc_final}
        )
    write_json(path / "run.json", {"arm": name})


def _build(
    tmp_path: Path,
    *,
    oracle_step0: float = 0.20,
    fp_tolerance_seed1: float | None = None,
    omit_final: str | None = None,
    nan_arm: str | None = None,
    drop_seed: tuple[str, int] | None = None,
    withdraw: str | None = None,
    h3_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    audit = tmp_path / "audit"
    audit.mkdir(parents=True)
    payload = {name: _forecast(name, FINALS[name]) for name in FINALS}
    write_json(audit / "forecasts.json", payload)
    lock_forecasts(payload, audit / "lock.json", note="synthetic")
    runs: dict[str, dict[int, Path]] = {}
    early: dict[str, Path] = {}
    for name, final in FINALS.items():
        runs[name] = {}
        for seed in SEEDS:
            if drop_seed == (name, seed):
                continue
            path = tmp_path / "runs" / f"{name}__s{seed}"
            this_final = final
            step0 = oracle_step0 if name == "oracle" else 0.20
            if name == "fp_tolerance" and seed == 1 and fp_tolerance_seed1 is not None:
                this_final = fp_tolerance_seed1
            _write_run(
                path,
                name=name,
                step0=step0,
                final=this_final,
                final_step=FINAL,
                knife=name == "oracle",
                nan_prompt=nan_arm == name and seed == 0,
                omit_final=omit_final == name and seed == 0,
            )
            runs[name][seed] = path
        early_path = tmp_path / "early" / f"{name}__s99"
        _write_run(early_path, name=name, step0=0.20, final=0.40, final_step=EARLY)
        early[name] = early_path
    analysis: dict = {
        "final_step": FINAL,
        "n_boot": 20,
        "margin": 0.05,
        "alpha": 0.05,
        "early": {"enabled": True, "steps": EARLY, "seed": 99},
        "oracle_gain_min": 0.10,
    }
    if h3_pairs is not None:
        analysis["h3"] = {
            "min_pairs": 5,
            "pairs": [{"arm": arm, "control": control} for arm, control in h3_pairs],
        }
    withdrawn = [] if withdraw is None else [withdraw]
    return {
        "audit_dir": audit,
        "runs": runs,
        "early_runs": early,
        "analysis": analysis,
        "withdrawals": {
            "schema": "vfates.withdrawals/1",
            "audit": {},
            "training": {},
            "withdrawn": withdrawn,
        },
        "planned_seeds": {name: list(SEEDS) for name in FINALS},
    }


def _report(built: dict) -> dict:
    return confirmatory_report(
        audit_dir=built["audit_dir"],
        runs=built["runs"],
        early_runs=built["early_runs"],
        analysis=built["analysis"],
        withdrawals=built["withdrawals"],
        planned_seeds=built["planned_seeds"],
    )


def test_audit_withdrawal_rules() -> None:
    assert audit_withdrawal_reasons({"targeted": True, "rate_miss": 0.03, "type": "hash_fp"}) == [
        "rate_miss"
    ]
    assert audit_withdrawal_reasons({"targeted": True, "rate_miss": 0.02, "type": "hash_fp"}) == []
    phrase = audit_withdrawal_reasons(
        {
            "targeted": True,
            "rate_miss": 0.0,
            "type": "trigger_fp",
            "fell_back": True,
            "validated": False,
        }
    )
    assert phrase == ["fell_back", "validated"]
    assert audit_withdrawal_reasons({"targeted": True, "rate_miss": None, "type": "hash_fp"}) == [
        "rate_undefined"
    ]
    assert audit_withdrawal_reasons({"type": "noise", "fell_back": True, "validated": False}) == []
    assert audit_withdrawal_reasons({"type": "tolerance_band_fp", "capped": True}) == ["capped"]
    assert audit_withdrawal_reasons({"type": "form_fn", "targeted": False, "capped": True}) == []
    assert (
        audit_withdrawal_reasons({"type": "behavior_fn", "validated": None, "fell_back": False})
        == []
    )
    record = decide_audit_withdrawals(
        {
            "b": {"targeted": True, "rate_miss": 0.05, "type": "hash_fp"},
            "a": {"type": "tolerance_band_fp", "targeted": True, "rate_miss": 0.0, "capped": True},
            "fn_form": {"type": "form_fn", "targeted": False, "rate_miss": None, "capped": True},
        }
    )
    assert record["schema"] == "vfates.withdrawals/1"
    assert record["withdrawn"] == ["a", "b"]
    assert record["training"] == {}
    form_report = {"arms": {"fn_form": {"type": "form_fn", "details": {}}}}
    assert arm_validity(form_report)["fn_form"]["capped"] is True
    assert decide_audit_withdrawals(arm_validity(form_report))["withdrawn"] == []


def test_confirmatory_hypotheses_hold_on_a_separated_menu(tmp_path: Path) -> None:
    report = _report(_build(tmp_path))
    assert report["schema"] == "vfates.confirmatory/1"
    assert report["h1"]["holds"] is True
    assert report["h1"]["decision"] == "holds"
    assert report["h2"]["holds"] is True
    assert report["h2"]["decision"] == "holds"
    assert report["h3"]["holds"] is True
    assert report["h3"]["decision"] == "holds"
    assert report["h3"]["n_pairs"] == 7
    assert report["h3"]["permutation_method"] == "exact"
    assert report["h3"]["permutation_p"] < 0.05
    assert report["h3"]["q05"] > 0
    alignment = report["h3"]["baselines"]["behaviour_alignment"]
    assert alignment["rho"] is not None
    assert alignment["diff_q05"] is not None
    assert report["oracle_gain"]["inconclusive"] is False
    assert report["inconclusive"] is False
    assert report["reliability"]["threshold"] == 0.8
    assert report["reliability"]["below"] == []
    assert report["reliability"]["arms"]["oracle"] == {"value": 0.8, "below": False}
    assert "## Split-half reliability of g_V\n\nBelow 0.8: none." in render_markdown(report)
    by_name = {row["name"]: row for row in report["h1"]["contrasts"]}
    contrast = by_name["noise_fp - fp_tolerance"]
    assert contrast["p_sup"] == 0.0
    assert contrast["rejects"] is True
    assert contrast["all_seeds_positive"] is True
    assert contrast["adjusted"] == pytest.approx(0.0, abs=1e-8)
    assert "prompt_only" in contrast
    assert contrast["seed_t"]["defined"] is True
    beside = report["h1"]["beside"][0]
    assert beside["name"] == "noise_fp - fp_tolerance_decoy"
    assert beside["rejects"] is True
    assert beside["holm_adjusted_p"] is None
    knives = {(row["arm"], row["seed"]): row["knife_edge"] for row in report["knife_edge"]["runs"]}
    assert knives[("oracle", 0)] is True
    assert knives[("fp_hash", 0)] is False
    # The knife label does not move the endpoint off the final step.
    assert report["endpoints"]["oracle"]["final_mean"] == pytest.approx(0.70)
    h5 = {row["arm"]: row for row in report["h5"]["rows"]}
    assert h5["fp_tolerance"]["field"] == "monitors_wrong"
    assert h5["fn_behavior"]["field"] == "monitors_correct"
    assert h5["fp_tolerance"]["mean"]["arm_final"] == pytest.approx(0.8)
    assert h5["fn_behavior"]["mean"]["arm_final"] == pytest.approx(0.9)
    h4 = {row["arm"]: row for row in report["h4"]["rows"]}
    assert h4["fp_tolerance"]["call"] is True
    assert h4["fp_hash"]["call"] is True
    assert "p_sup" in report["h1"]["rule"]
    assert "p_tost" in report["h2"]["rule"]


def test_low_oracle_gain_labels_hypotheses_inconclusive(tmp_path: Path) -> None:
    report = _report(_build(tmp_path, oracle_step0=0.66))
    assert report["oracle_gain"]["value"] == pytest.approx(0.04)
    assert report["oracle_gain"]["inconclusive"] is True
    assert report["h1"]["holds"] is True
    assert report["h1"]["decision"] == "inconclusive"
    assert report["h3"]["decision"] == "inconclusive"


def test_one_negative_seed_blocks_that_contrast(tmp_path: Path) -> None:
    report = _report(_build(tmp_path, fp_tolerance_seed1=0.95))
    by_name = {row["name"]: row for row in report["h1"]["contrasts"]}
    blocked = by_name["noise_fp - fp_tolerance"]
    assert blocked["all_seeds_positive"] is False
    assert blocked["rejects"] is False
    assert report["h1"]["holds"] is True


def test_withdrawn_arm_drops_its_contrasts(tmp_path: Path) -> None:
    report = _report(_build(tmp_path, withdraw="fp_trigger"))
    by_name = {row["name"]: row for row in report["h1"]["contrasts"]}
    assert by_name["noise_fp_trigger - fp_trigger"]["status"] == "withdrawn"
    assert report["h1"]["holds"] is True
    withdrawn_pairs = [row for row in report["h3"]["pairs"] if row["status"] == "withdrawn"]
    assert any(row["arm"] == "fp_trigger" for row in withdrawn_pairs)
    assert report["h3"]["n_pairs"] == 6
    assert report["h3"]["holds"] is True
    h4 = {row["arm"]: row for row in report["h4"]["rows"]}
    assert h4["fp_trigger"]["status"] == "withdrawn"


def test_two_channel_probe_keeps_both_channels() -> None:
    both = _forecast("fp_tolerance", 0.2)
    both["transfer"]["probe_heldout_auc"] = {"fp": 0.9, "fn": 0.7}
    one = _forecast("fp_hash", 0.2)
    one["transfer"]["probe_heldout_auc"] = 0.8
    forecasts = {"fp_tolerance": both, "fp_hash": one}
    block = _h4_block(
        forecasts,
        {"shared": ["fp_tolerance", "fp_trigger", "fn_form"], "prompt_specific": ["fp_hash"]},
        ["fp_tolerance", "fp_hash", "fn_form"],
        {"fp_trigger"},
    )
    rows = {row["arm"]: row for row in block["rows"]}
    assert rows["fp_tolerance"]["channels"] == {"fp": 0.9, "fn": 0.7}
    assert rows["fp_tolerance"]["auc"] is None
    assert rows["fp_tolerance"]["call"] is None
    assert rows["fp_hash"]["auc"] == 0.8
    assert rows["fp_hash"]["call"] is False
    assert "channels" not in rows["fp_hash"]
    assert rows["fp_trigger"]["status"] == "withdrawn"
    assert rows["fn_form"]["status"] == "missing_forecast"
    assert _probe_auc({"fp": 0.9, "fn": 0.7}) == pytest.approx(0.8)
    assert _probe_auc({"fp": 0.9, "fn": None}) is None
    assert _probe_auc(0.6) == 0.6


def test_h3_needs_five_pairs(tmp_path: Path) -> None:
    report = _report(_build(tmp_path, h3_pairs=list(H3_PAIRS[:4])))
    assert report["h3"]["n_pairs"] == 4
    assert report["h3"]["holds"] is False
    assert report["h3"]["tested"] is True


def test_h3_not_tested_is_descriptive_only(tmp_path: Path) -> None:
    built = _build(tmp_path, h3_pairs=list(H3_PAIRS[:4]))
    built["analysis"]["h3"]["tested"] = False
    report = _report(built)
    assert report["h3"]["tested"] is False
    assert report["h3"]["holds"] is None
    assert report["h3"]["decision"] == "not_tested"
    assert "not a hypothesis" in report["h3"]["note"]
    assert report["h3"]["n_pairs"] == 4
    assert report["h1"]["decision"] == "holds"
    assert "Decision: not_tested. Holds: None." in render_markdown(report)


def test_missing_final_step_nonfinite_and_incomplete_runs_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no heldout eval at step"):
        _report(_build(tmp_path / "omit", omit_final="fp_hash"))
    with pytest.raises(ValueError, match="non-finite"):
        _report(_build(tmp_path / "nan", nan_arm="fp_hash"))
    with pytest.raises(ValueError, match="incomplete planned runs"):
        _report(_build(tmp_path / "drop", drop_seed=("oracle", 1)))


def test_h3_averages_each_pair_over_its_own_planned_seeds(tmp_path: Path) -> None:
    built = _build(tmp_path)
    narrow = {"fn_form", "noise_fn_form"}
    for name, final in FINALS.items():
        if name in narrow:
            continue
        path = tmp_path / "runs" / f"{name}__s2"
        extra_final = 0.0 if name == "fp_tolerance" else final
        step0 = 0.20
        _write_run(path, name=name, step0=step0, final=extra_final, final_step=FINAL)
        built["runs"][name][2] = path
        built["planned_seeds"][name] = [0, 1, 2]
    report = _report(built)
    pairs = {(row["arm"], row["control"]): row for row in report["h3"]["pairs"]}
    wide = pairs[("fp_tolerance", "noise_fp")]
    assert wide["seeds"] == [0, 1, 2]
    assert wide["dacc"] == pytest.approx((0.30 + 0.30 + 0.0) / 3 - 0.80)
    assert pairs[("fn_form", "noise_fn_form")]["seeds"] == [0, 1]
    assert report["h3"]["seed_lists"] == [[0, 1, 2], [0, 1]]
    overall = report["h3"]["overall_spearman"]
    assert overall["seeds"]["fp_tolerance"] == [0, 1, 2]
    assert overall["seeds"]["fn_form"] == [0, 1]
    assert overall["bootstrap"]["seed_lists"] == [[0, 1, 2], [0, 1]]


def _e2_report(
    tmp_path: Path,
    *,
    natural_final: float,
    tau_natural: float = 0.30,
    oracle_step0: float = 0.20,
) -> dict:
    audit = tmp_path / "audit"
    audit.mkdir(parents=True)
    taus = {"oracle": 0.70, "base_tests": tau_natural, "resampled": 0.60}
    payload = {name: _forecast(name, tau) for name, tau in taus.items()}
    write_json(audit / "forecasts.json", payload)
    lock_forecasts(payload, audit / "lock.json", note="synthetic")
    finals = {"oracle": 0.70, "base_tests": natural_final, "resampled": 0.60}
    runs: dict[str, dict[int, Path]] = {}
    for name, final in finals.items():
        runs[name] = {}
        for seed in SEEDS:
            path = tmp_path / "runs" / f"{name}__s{seed}"
            step0 = oracle_step0 if name == "oracle" else 0.20
            _write_run(path, name=name, step0=step0, final=final, final_step=FINAL)
            runs[name][seed] = path
    return confirmatory_report(
        audit_dir=audit,
        runs=runs,
        early_runs={},
        analysis={
            "hypotheses": False,
            "final_step": FINAL,
            "n_boot": 50,
            "early": {"enabled": False},
            "natural": {"natural": "base_tests", "control": "resampled", "oracle": "oracle"},
        },
        withdrawals={"withdrawn": []},
        planned_seeds={name: list(SEEDS) for name in finals},
    )


def test_e2_natural_block_follows_the_preregistered_tests(tmp_path: Path) -> None:
    report = _e2_report(tmp_path / "differs", natural_final=0.30)
    assert "h1" not in report
    natural = report["natural"]
    assert natural["contrast"]["name"] == "resampled - base_tests"
    assert natural["contrast"]["point"] == pytest.approx(0.30)
    assert natural["differs"]["decision"] == "differs"
    assert natural["locality"]["decision"] == "not_equivalent"
    assert natural["screen_direction"]["dtau"] == pytest.approx(0.30)
    assert natural["screen_direction"]["decision"] == "matches"
    assert natural["scale"]["name"] == "oracle - base_tests"
    assert natural["scale"]["point"] == pytest.approx(0.40)
    assert "natural-verifier tests" in report["rule"]

    local = _e2_report(tmp_path / "local", natural_final=0.60)["natural"]
    assert local["differs"]["decision"] == "does_not_differ"
    assert local["locality"]["decision"] == "equivalent"
    assert local["screen_direction"]["counted"] is False
    assert local["screen_direction"]["decision"] == "not_counted"

    wrong = _e2_report(tmp_path / "wrong", natural_final=0.30, tau_natural=0.90)["natural"]
    assert wrong["screen_direction"]["decision"] == "does_not_match"

    weak = _e2_report(tmp_path / "weak", natural_final=0.30, oracle_step0=0.66)
    assert weak["inconclusive"] is True
    assert weak["natural"]["differs"]["decision"] == "inconclusive"
    assert weak["natural"]["locality"]["decision"] == "inconclusive"
    assert weak["natural"]["screen_direction"]["decision"] == "inconclusive"


def test_reliability_block_lists_arms_below_the_gate() -> None:
    block = _reliability_block(
        {
            "a": {"transfer": {"reliability": 0.79}},
            "b": {"transfer": {"reliability": 0.85}},
            "c": {"transfer": {"reliability": float("nan")}},
        },
        0.8,
    )
    assert block["below"] == ["a"]
    assert block["arms"]["c"] == {"value": None, "below": False}
