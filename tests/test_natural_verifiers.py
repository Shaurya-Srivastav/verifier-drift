"""Natural verifiers and the step-0 confusion helpers used by P6."""

from __future__ import annotations

import importlib
import json

import pytest

from vfates.pilots.p6_natural import (
    _make_generator,
    fp_concentration,
    pooled_confusion,
    prompt_averaged_confusion,
)
from vfates.types import Completion, Parsed, Problem
from vfates.verifiers.calibrate import CalibrationData, arm_validity, calibrate_arms
from vfates.verifiers.natural import IFEvalCheckerVerifier
from vfates.verifiers.registry import build_verifier


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
    checker = IFEvalCheckerVerifier()
    assert checker.kind == "natural"
    assert checker.stochastic is False
    assert checker.describe()["name"] == "ifeval_checker"
    assert checker.describe()["type"] == "ifeval_checker"


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


def test_natural_arms_build_calibrate_and_round_trip() -> None:
    if_spec = {"name": "checker", "type": "ifeval_checker"}
    checker = build_verifier(if_spec)
    assert checker.name == "checker"
    spec = json.loads(json.dumps(checker.describe()))
    rebuilt = build_verifier(spec)
    assert rebuilt.describe() == checker.describe()

    data = _natural_audit()
    specs = [
        {"name": "oracle", "type": "oracle"},
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
    entry = report["checker"]
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
    assert validity["checker"]["targeted"] is False
    assert validity["checker"]["type"] == "ifeval_checker"
    assert report["noise_checker"]["details"]["rho0"] == pytest.approx(0.25)
    assert report["noise_checker"]["details"]["support"] == "all"
    assert report["noise_checker"]["channel"]["fpr"] == pytest.approx(0.25)
    assert report["noise_checker_answered"]["details"]["rho0"] == pytest.approx(0.25 / 0.75)
    assert report["noise_checker_answered"]["details"]["support"] == "answered"
    rebuilt_arms = build_verifier(result.to_json()["arms"]["checker"])
    assert rebuilt_arms.describe() == result.arms["checker"].describe()


def test_math_and_qa_verifiers_round_trip() -> None:
    from vfates.tasks.math import MathTask
    from vfates.tasks.qa import QATask

    math_task = MathTask(
        rows=[{"problem": "What is 1 + 1?", "answer": "2"}],
        decontaminate=False,
        dev_fraction=0.34,
        heldout_fraction=0.33,
    )
    for typ in ("math_lenient", "math_strip_string"):
        spec = {"type": typ, "name": typ}
        try:
            built = build_verifier(spec, math_task)
        except ValueError as exc:
            if "unknown verifier type" in str(exc):
                pytest.skip("registry fallback is not merged")
            raise
        again = build_verifier(built.describe(), math_task)
        assert again.describe() == built.describe()
        assert again.stochastic is False
        assert again.kind == "natural"

    qa_task = QATask(
        rows=[{"question": "What is 1 + 1?", "answer": "2", "answer_type": "Integer"}],
        dev_fraction=0.34,
        heldout_fraction=0.33,
    )
    spec = {"type": "model_judge", "name": "natural"}
    try:
        built = build_verifier(spec, qa_task)
    except ValueError as exc:
        if "unknown verifier type" in str(exc):
            pytest.skip("registry fallback is not merged")
        raise
    again = build_verifier(built.describe(), qa_task)
    assert again.describe() == built.describe()
    assert again.describe()["judge"]["type"] == "heuristic"


def test_factory_missing_is_explicit() -> None:
    try:
        importlib.import_module("vfates.gen.factory")
    except ImportError:
        with pytest.raises(ImportError, match="make_generator"):
            _make_generator("mlx", "unused", {})
    else:
        pytest.skip("vfates.gen.factory is already installed")
