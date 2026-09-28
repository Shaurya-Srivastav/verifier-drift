"""Natural verifiers and the step-0 confusion helpers used by P6."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from vfates.pilots.p6_natural import (
    _make_generator,
    fp_concentration,
    pooled_confusion,
    prompt_averaged_confusion,
)
from vfates.types import Completion, Parsed, Problem
from vfates.utils.io import read_yaml
from vfates.verifiers.calibrate import CalibrationData, arm_validity, calibrate_arms
from vfates.verifiers.natural import IFEvalCheckerVerifier, MBPPBaseTestsVerifier
from vfates.verifiers.registry import build_verifier

ROOT = Path(__file__).resolve().parents[1]


def test_pooled_and_concentration() -> None:
    oracle = [1, 1, 0, 0, 0]
    verifier = [1, 0, 1, 1, 0]
    stats = pooled_confusion(oracle, verifier)
    assert (stats["tp"], stats["fp"], stats["tn"], stats["fn"]) == (1, 2, 1, 1)
    assert stats["fpr"] == pytest.approx(2 / 3)
    assert stats["fnr"] == pytest.approx(0.5)
    assert stats["precision"] == pytest.approx(1 / 3)
    assert stats["fp_mass"] == pytest.approx(2 / 5)
    assert stats["fp_share_among_accepted"] == pytest.approx(2 / 3)
    assert stats["j"] == pytest.approx(0.5 - 2 / 3)

    pids = ["a", "a", "b", "c", "c"]
    averaged = prompt_averaged_confusion(pids, oracle, verifier)
    assert averaged["n_prompts"] == 3
    # Two FPs on b and c. Top slice is max(1, ceil(0.1 * 3)) = 1 prompt → half the mass.
    share = fp_concentration(pids, oracle, verifier, top_frac=0.1)
    assert share == pytest.approx(0.5)
    assert fp_concentration(["a"], [0], [0]) is None


def test_verifier_kinds() -> None:
    base = MBPPBaseTestsVerifier(k=2)
    assert base.kind == "natural"
    assert base.describe()["k"] == 2
    assert base.stochastic is False
    checker = IFEvalCheckerVerifier()
    assert checker.kind == "natural"
    assert checker.describe()["name"] == "ifeval_checker"
    with pytest.raises(ValueError):
        MBPPBaseTestsVerifier(k=-1)


def _natural_audit() -> CalibrationData:
    """Wrong rows: two accepted, four rejected, two unanswered. One of four correct fails."""
    layout = [
        ("a", "alpha here", 0, "good"),
        ("a", "nope", 0, "bad"),
        ("a", "nope", 0, None),
        ("a", "alpha here", 1, "good"),
        ("a", "nope", 0, "bad"),
        ("a", "alpha here", 1, "good"),
        ("b", "alpha here", 0, "good"),
        ("b", "nope", 0, "bad"),
        ("b", "nope", 0, None),
        ("b", "alpha here", 1, "good"),
        ("b", "nope", 1, "bad"),
        ("b", "nope", 0, "bad"),
    ]
    problems = {
        pid: Problem(
            pid=pid,
            split="train",
            task="synthetic",
            question="q",
            answer="gold",
            meta={"func_name": "verify_keywords", "args": {"keyword_list": ["alpha"]}},
        )
        for pid in ("a", "b")
    }
    completions = []
    parsed_list = []
    labels = []
    counts: dict[str, int] = {}
    for pid, text, label, answer in layout:
        k = counts.get(pid, 0)
        counts[pid] = k + 1
        completions.append(Completion(pid=pid, k=k, text=text, n_tokens=2))
        parsed_list.append(Parsed(answer=answer, raw=answer, extras={}))
        labels.append(label)
    return CalibrationData.from_grades(list(problems.values()), completions, parsed_list, labels)


def _fake_grade(problem: Problem, code: str, **kwargs: object) -> dict[str, object]:
    del problem, kwargs
    if code == "good":
        return {"passed": True, "which": "base"}
    return {"passed": False, "fail_stage": "base", "fail_index": 0, "which": "base"}


def test_natural_arms_build_calibrate_and_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vfates.verifiers.natural.grade_code", _fake_grade)
    mbpp_doc = read_yaml(f"{ROOT}/configs/experiments/e2_mbpp.yaml")
    if_doc = read_yaml(f"{ROOT}/configs/experiments/e2_ifeval.yaml")
    mbpp_spec = mbpp_doc["audit"]["arms"]["arms"][1]
    if_spec = if_doc["audit"]["arms"]["arms"][1]
    assert mbpp_spec == {
        "name": "base_tests",
        "type": "mbpp_base_tests",
        "k": None,
        "visible_only": False,
    }
    assert if_spec == {"name": "checker", "type": "ifeval_checker"}
    base = build_verifier(mbpp_spec)
    checker = build_verifier(if_spec)
    assert base.name == "base_tests"
    assert checker.name == "checker"
    for original in (base, checker):
        spec = json.loads(json.dumps(original.describe()))
        rebuilt = build_verifier(spec)
        assert rebuilt.describe() == original.describe()

    data = _natural_audit()
    specs = [
        {"name": "oracle", "type": "oracle"},
        dict(mbpp_spec),
        {
            "name": "noise_answered",
            "type": "resampled",
            "match": "base_tests",
            "support": "answered",
            "seed": 0,
        },
        {
            "name": "noise_all",
            "type": "resampled",
            "match": "base_tests",
            "support": "all",
            "seed": 0,
        },
        dict(if_spec),
        {
            "name": "noise_checker",
            "type": "resampled",
            "match": "checker",
            "support": "all",
            "seed": 0,
        },
        {
            "name": "noise_checker_answered",
            "type": "resampled",
            "match": "checker",
            "support": "answered",
            "seed": 0,
        },
    ]
    result = calibrate_arms(None, data, specs, n_boot=8, seed=0, targets={})
    report = result.report["arms"]
    for name in ("base_tests", "checker"):
        entry = report[name]
        assert entry["achieved_fpr"] == pytest.approx(0.25)
        assert entry["achieved_fnr"] == pytest.approx(0.25)
        assert entry["channel"]["fpr"] == pytest.approx(entry["realised"]["fpr"])
        assert entry["channel"]["fnr"] == pytest.approx(entry["realised"]["fnr"])
        assert entry["channel"]["J"] == pytest.approx(
            (1.0 - entry["channel"]["fnr"]) - entry["channel"]["fpr"]
        )
        assert entry["target_fpr"] is None
        assert entry["target_fnr"] is None
    validity = arm_validity(result.to_json())
    assert validity["base_tests"]["targeted"] is False
    assert validity["checker"]["targeted"] is False
    assert validity["base_tests"]["type"] == "mbpp_base_tests"
    assert validity["checker"]["type"] == "ifeval_checker"
    answered = report["noise_answered"]
    assert answered["channel"]["fpr"] == pytest.approx(report["base_tests"]["achieved_fpr"])
    assert answered["channel"]["fnr"] == pytest.approx(report["base_tests"]["achieved_fnr"])
    assert answered["details"]["support"] == "answered"
    assert answered["details"]["rho0"] == pytest.approx(0.25 / 0.75)
    whole = report["noise_all"]
    assert whole["details"]["support"] == "all"
    assert whole["details"]["rho0"] == pytest.approx(0.25)
    assert whole["channel"]["fpr"] == pytest.approx(0.25)
    assert report["noise_checker"]["details"]["rho0"] == pytest.approx(0.25)
    assert report["noise_checker"]["details"]["support"] == "all"
    assert report["noise_checker_answered"]["details"]["rho0"] == pytest.approx(0.25 / 0.75)
    rebuilt_arms = build_verifier(result.to_json()["arms"]["base_tests"])
    assert rebuilt_arms.describe() == result.arms["base_tests"].describe()


def test_factory_missing_is_explicit() -> None:
    try:
        importlib.import_module("vfates.gen.factory")
    except ImportError:
        with pytest.raises(ImportError, match="make_generator"):
            _make_generator("mlx", "unused", {})
    else:
        pytest.skip("vfates.gen.factory is already installed")
