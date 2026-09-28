"""Natural verifiers and the step-0 confusion helpers used by P6."""

from __future__ import annotations

import importlib

import pytest

from vfates.pilots.p6_natural import (
    _make_generator,
    fp_concentration,
    pooled_confusion,
    prompt_averaged_confusion,
)
from vfates.verifiers.natural import IFEvalCheckerVerifier, MBPPBaseTestsVerifier


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


def test_factory_missing_is_explicit() -> None:
    try:
        importlib.import_module("vfates.gen.factory")
    except ImportError:
        with pytest.raises(ImportError, match="make_generator"):
            _make_generator("mlx", "unused", {})
    else:
        pytest.skip("vfates.gen.factory is already installed")
