"""Hand-computed confusion statistics and the two precision forecasts."""

import numpy as np
import pytest

from vfates.forecast.confusion import (
    CONFUSION_KEYS,
    _resample_labels,
    confusion,
    confusion_with_ci,
    mean_precision_forecast,
    modal_precision_forecast,
)


def _hand() -> tuple[list[int], list[int], list[str], list[str], list[float]]:
    # pid, O, V, answer, logprob
    rows = [
        ("p1", 1, 1, "A", -0.1),
        ("p1", 1, 0, "A", -1.0),
        ("p1", 0, 1, "B", -0.5),
        ("p1", 0, 0, "B", -2.0),
        ("p2", 1, 1, "A", -0.2),
        ("p2", 1, 1, "A", -0.3),
        ("p2", 0, 0, "C", -0.4),
        ("p3", 0, 1, "Z", -0.2),
        ("p3", 0, 1, "Z", -0.2),
        ("p4", 1, 0, "Q", -0.5),
        ("p4", 0, 0, "Q", -0.6),
    ]
    pid, O, V, answer, logprob = zip(*rows, strict=True)
    return list(O), list(V), list(pid), list(answer), list(logprob)


def test_confusion_hand_computed() -> None:
    O, V, pid, _answer, _logprob = _hand()
    stats = confusion(O, V, pid)
    assert set(stats) == set(CONFUSION_KEYS)
    assert stats["n"] == 11
    assert stats["n_prompts"] == 4
    assert stats["prevalence"] == pytest.approx(5 / 11)
    assert stats["tpr"] == pytest.approx(3 / 5)
    assert stats["fnr"] == pytest.approx(2 / 5)
    assert stats["fpr"] == pytest.approx(0.5)
    assert stats["tnr"] == pytest.approx(0.5)
    assert stats["J"] == pytest.approx(0.1)
    assert stats["precision_pooled"] == pytest.approx(0.5)
    assert stats["precision_prompt_mean"] == pytest.approx(0.5)
    assert stats["for_prompt_mean"] == pytest.approx(1 / 3)
    assert stats["f1"] == pytest.approx(6 / 11)
    assert stats["verifier_accuracy"] == pytest.approx(6 / 11)
    assert stats["accept_rate"] == pytest.approx(6 / 11)
    assert stats["fp_mass"] == pytest.approx(0.3125)


def test_forecasts_hand_computed() -> None:
    O, V, pid, answer, logprob = _hand()
    assert mean_precision_forecast(O, V, pid) == pytest.approx(0.5)
    modal = modal_precision_forecast(O, V, pid, answer)
    assert modal["value"] == pytest.approx(0.5)
    assert modal["tie_fraction"] == pytest.approx(0.25)
    assert modal["n_prompts"] == 4
    assert modal["n_prompts_no_accept"] == 1
    broken = modal_precision_forecast(O, V, pid, answer, logprob)
    assert broken["value"] == pytest.approx(0.625)
    assert broken["tie_fraction"] == pytest.approx(0.25)
    assert broken["n_prompts_no_accept"] == 1


def test_f1_zero_when_precision_and_recall_are_zero() -> None:
    stats = confusion([1, 0], [0, 1], ["a", "a"])
    assert stats["tpr"] == 0.0
    assert stats["precision_pooled"] == 0.0
    assert stats["f1"] == 0.0


def test_undefined_rates_are_nan() -> None:
    empty = confusion([], [], [])
    assert empty["n"] == 0
    assert empty["n_prompts"] == 0
    assert np.isnan(empty["prevalence"])
    no_accept = confusion([1, 0], [0, 0], ["a", "a"])
    assert np.isnan(no_accept["precision_pooled"])
    assert np.isnan(no_accept["precision_prompt_mean"])
    with pytest.raises(ValueError):
        confusion([1], [0, 1], ["a"])
    with pytest.raises(ValueError):
        confusion([2], [0], ["a"])


def test_bootstrap_matches_point_and_respects_repeated_prompts() -> None:
    O, V, pid, _answer, _logprob = _hand()
    ci = confusion_with_ci(O, V, pid, n_boot=200, seed=0)
    point = confusion(O, V, pid)
    assert set(ci) == set(CONFUSION_KEYS)
    for key, interval in ci.items():
        assert set(interval) == {"point", "lo", "hi"}
        assert interval["lo"] <= interval["hi"]
        assert interval["point"] == pytest.approx(float(point[key]))
        if interval["point"] == interval["point"]:
            assert interval["lo"] <= interval["point"] <= interval["hi"]
    # n_boot default path on a tiny table.
    wide = confusion_with_ci(O, V, pid, n_boot=1000, seed=1, metrics=("fpr", "fnr", "J"))
    assert set(wide) == {"fpr", "fnr", "J"}
    labels = _resample_labels(np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1]))
    assert labels.tolist() == [0, 0, 1, 1, 2]
    single = _resample_labels(np.array([5, 5]), np.array([0, 0, 0, 0, 0, 3]))
    assert single.tolist() == [0, 1]


def test_mean_precision_empty() -> None:
    assert np.isnan(mean_precision_forecast([], [], []))
    modal = modal_precision_forecast([], [], [], [])
    assert modal["n_prompts"] == 0
    assert np.isnan(modal["value"])
