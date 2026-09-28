"""P2 calibration cache and the channel/realised arm table, on a synthetic cell."""

from decimal import Decimal

import numpy as np
import pytest

from vfates.pilots.p2_base_audit import (
    arm_table_rows,
    calibration_fingerprint,
    completion_cache_action,
    load_or_fit_calibration,
    sampling_fingerprint,
    verdict_agreement,
)
from vfates.tasks.chain_sum import ChainSumTask
from vfates.types import Completion, Parsed, Problem
from vfates.verifiers.calibrate import CalibrationData, calibrate_arms

TASK = ChainSumTask()


def _problem(pid: str, answer: str = "100") -> Problem:
    return Problem(
        pid=pid,
        split="train",
        task="decimal_chain_sum",
        question="q",
        answer=answer,
        meta={
            "expression": "x",
            "num_terms": 2,
            "num_digits": 3,
            "gold_decimal_places": 0,
            "source": "builtin",
        },
    )


def _blank(value: str | None) -> Parsed:
    return Parsed(
        answer=value,
        raw=value,
        extras={"value": value, "has_boxed": value is not None},
    )


def _tiny() -> CalibrationData:
    rows: list[tuple[str, str, int, Parsed, int]] = []
    for i, err in enumerate((1, 2, 3, 4)):
        value = str(Decimal(100 + err))
        text = "sloppyword estimate" if i < 2 else "plain estimate"
        rows.append((f"p{i}", text, 0, _blank(value), 4 + i))
    rows.append(("p0", "correct boxed", 1, _blank("100"), 3))
    rows.append(("p1", "sloppyword correct", 1, _blank("100"), 12))
    problems = []
    completions = []
    parsed = []
    labels = []
    counts: dict[str, int] = {}
    for pid, text, label, item, n_tokens in rows:
        if pid not in counts:
            problems.append(_problem(pid))
        k = counts.get(pid, 0)
        counts[pid] = k + 1
        completions.append(Completion(pid=pid, k=k, text=text, n_tokens=n_tokens))
        parsed.append(item)
        labels.append(label)
    return CalibrationData.from_grades(problems, completions, parsed, labels)


def _specs() -> list[dict]:
    return [
        {"name": "oracle", "type": "oracle"},
        {"name": "fp_tolerance", "type": "tolerance_fp", "target_fpr": 0.25},
        {
            "name": "fp_tolerance_decoy",
            "type": "tolerance_band_fp",
            "inner": "fp_tolerance",
            "target_fpr": 0.5,
        },
        {
            "name": "fp_trigger",
            "type": "trigger_fp",
            "candidates": ["sloppyword"],
            "prefer": "closest",
            "target_fpr": 0.5,
        },
        {
            "name": "noise_fp",
            "type": "resampled",
            "match": "fp_tolerance",
            "support": "answered",
            "seed": 0,
        },
    ]


def test_calibration_fingerprint_tracks_config_and_schema() -> None:
    cfg = {
        "targets": {"fpr": 0.15, "fnr": 0.30},
        "n_boot": 20,
        "seed": 0,
        "arms": [{"name": "oracle", "type": "oracle"}],
    }
    base = calibration_fingerprint(cfg)
    assert calibration_fingerprint(dict(cfg)) == base
    assert calibration_fingerprint({**cfg, "seed": 1}) != base
    assert calibration_fingerprint({**cfg, "n_boot": 21}) != base
    assert calibration_fingerprint({**cfg, "targets": {"fpr": 0.2, "fnr": 0.30}}) != base
    assert (
        calibration_fingerprint({**cfg, "arms": [{"name": "oracle", "type": "oracle"}, {"x": 1}]})
        != base
    )
    assert calibration_fingerprint(cfg, schema="vfates.calibration/1") != base
    grades = [("p", 0, 1, "12.5"), ("p", 1, 0, None)]
    graded = calibration_fingerprint(cfg, grades=grades)
    assert graded != base
    assert calibration_fingerprint(cfg, grades=list(reversed(grades))) == graded
    relabelled = [("p", 0, 0, "12.5"), ("p", 1, 0, None)]
    reparsed = [("p", 0, 1, "-12"), ("p", 1, 0, None)]
    assert calibration_fingerprint(cfg, grades=relabelled) != graded
    assert calibration_fingerprint(cfg, grades=reparsed) != graded
    samp = {"n": 2, "max_tokens": 8, "temperature": 1.0, "top_p": 1.0, "seed": 0}
    assert sampling_fingerprint(samp) == sampling_fingerprint(dict(samp))
    assert sampling_fingerprint({**samp, "max_tokens": 9}) != sampling_fingerprint(samp)


def test_load_or_fit_refits_calibration_and_keeps_completions(tmp_path) -> None:
    data = _tiny()
    calls: list[int] = []

    def fit():
        calls.append(1)
        return calibrate_arms(TASK, data, _specs(), n_boot=4, seed=0)

    cell = tmp_path / "cell"
    cell.mkdir()
    completions = cell / "completions.jsonl"
    completions.write_text('{"pid": "p0", "k": 0}\n', encoding="utf-8")
    before = completions.read_bytes()
    fingerprint = calibration_fingerprint(
        {"targets": None, "n_boot": 4, "seed": 0, "arms": _specs()}
    )
    document, refit = load_or_fit_calibration(cell, fingerprint, fit)
    assert refit is True
    assert len(calls) == 1
    assert (cell / "calibration.json").exists()
    assert (cell / "calibration.fingerprint").read_text(encoding="utf-8").strip() == fingerprint
    again, refit_again = load_or_fit_calibration(cell, fingerprint, fit)
    assert refit_again is False
    assert len(calls) == 1
    assert again["arms"]["oracle"]["type"] == document["arms"]["oracle"]["type"]
    (cell / "calibration.fingerprint").unlink()
    _missing, refit_missing = load_or_fit_calibration(cell, fingerprint, fit)
    assert refit_missing is True
    assert len(calls) == 2
    _changed, refit_changed = load_or_fit_calibration(cell, "0" * 64, fit)
    assert refit_changed is True
    assert len(calls) == 3
    graded_fp = calibration_fingerprint(
        {"targets": None, "n_boot": 4, "seed": 0, "arms": _specs()},
        grades=[("p0", 0, 1, "100"), ("p0", 1, 0, None)],
    )
    _graded, refit_grades = load_or_fit_calibration(cell, graded_fp, fit)
    assert refit_grades is True
    assert len(calls) == 4
    assert completions.read_bytes() == before


def test_completion_cache_action(tmp_path) -> None:
    cell = tmp_path / "cell"
    samp = {"n": 2, "max_tokens": 8, "seed": 0}
    fp = sampling_fingerprint(samp)
    assert completion_cache_action(cell, ["p"], 2, fp) == "resample"
    path = cell / "completions.jsonl"
    path.write_text('{"pid": "p"}\n{"pid": "p"}\n', encoding="utf-8")
    assert completion_cache_action(cell, ["p"], 2, fp) == "reuse"
    assert (cell / "sampling.fingerprint").read_text(encoding="utf-8").strip() == fp
    assert completion_cache_action(cell, ["p"], 2, fp) == "reuse"
    assert path.exists()
    assert completion_cache_action(cell, ["p"], 3, fp) == "resume"
    assert path.exists()
    other = sampling_fingerprint({**samp, "max_tokens": 16})
    assert completion_cache_action(cell, ["p"], 2, other) == "resample"
    assert not path.exists()
    assert not (cell / "sampling.fingerprint").exists()
    path.write_text('{"pid": "p"}\n', encoding="utf-8")
    assert completion_cache_action(cell, ["p"], 2, other) == "resume"
    assert path.exists()
    assert (cell / "sampling.fingerprint").read_text(encoding="utf-8").strip() == other


def test_arm_table_splits_channel_and_realised() -> None:
    data = _tiny()
    result = calibrate_arms(TASK, data, _specs(), n_boot=4, seed=0)
    n_wrong = sum(row.O == 0 for row in data.rows)
    n_correct = sum(row.O == 1 for row in data.rows)
    rows = arm_table_rows(result.report["arms"], n_wrong, n_correct)
    by_name = {row["arm"]: row for row in rows}
    assert set(by_name) == {
        "oracle",
        "fp_tolerance",
        "fp_tolerance_decoy",
        "fp_trigger",
        "noise_fp",
    }
    for row in rows:
        assert row["channel_J"] == pytest.approx((1.0 - row["channel_fnr"]) - row["channel_fpr"])
        assert row["realised_J"] == pytest.approx((1.0 - row["realised_fnr"]) - row["realised_fpr"])
        assert row["J"] == pytest.approx(row["channel_J"])
        assert row["fpr"] == pytest.approx(row["channel_fpr"])
    decoy = by_name["fp_tolerance_decoy"]
    assert decoy["lo"] == result.report["arms"]["fp_tolerance"]["details"]["abs_tol"]
    assert decoy["hi"] is not None
    trigger = by_name["fp_trigger"]
    assert trigger["validated"] in (True, False)
    assert trigger["fell_back"] in (True, False)
    assert trigger["fit_rate"] is not None
    assert trigger["n_prompts_with_phrase"] >= 1
    assert trigger["early_start_share"] == pytest.approx(1.0)
    noise = by_name["noise_fp"]
    assert noise["channel_fpr"] == pytest.approx(by_name["fp_tolerance"]["channel_fpr"])
    left = np.array([1, 0, 1, 1])
    right = np.array([1, 0, 0, 1])
    assert verdict_agreement(left, right) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        verdict_agreement([], [])
