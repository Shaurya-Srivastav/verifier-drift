"""Prompt-grouped ridge probes for shared versus prompt-specific arm errors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vfates.forecast.probe import arm_error_probe, auc, grouped_folds, probe_auc, ridge_scores
from vfates.utils.stats import group_indices


def _pairwise_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    total = 0.0
    for pos in s[y]:
        for neg in s[~y]:
            if pos > neg:
                total += 1.0
            elif pos == neg:
                total += 0.5
    return total / (n_pos * n_neg)


def _primal_ridge(
    train: np.ndarray, labels: np.ndarray, test: np.ndarray, lam: float
) -> np.ndarray:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    design = (train - mu) / sd
    held = (test - mu) / sd
    centered = labels - labels.mean()
    n, dim = design.shape
    gram = design.T @ design
    gram[np.diag_indices(dim)] += lam * n
    weight = np.linalg.solve(gram, design.T @ centered)
    return held @ weight


def test_auc_gives_ties_half_credit() -> None:
    tied = np.array([0.0, 0.0, 1.0, 1.0])
    tied_labels = np.array([0, 1, 1, 0])
    assert _pairwise_auc(tied, tied_labels) == pytest.approx(0.5)
    assert auc(tied, tied_labels) == pytest.approx(0.5)

    half = np.array([0.0, 0.0, 1.0])
    half_labels = np.array([0, 1, 1])
    assert _pairwise_auc(half, half_labels) == pytest.approx(0.75)
    assert auc(half, half_labels) == pytest.approx(0.75)

    rng = np.random.default_rng(0)
    for _ in range(6):
        scores = rng.normal(size=40)
        labels = rng.integers(0, 2, size=40)
        labels[0] = 0
        labels[1] = 1
        scores[5] = scores[6] = scores[7]
        assert auc(scores, labels) == pytest.approx(_pairwise_auc(scores, labels))

    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(perfect, np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert auc(perfect, np.array([1, 1, 0, 0])) == pytest.approx(0.0)
    assert math.isnan(auc(np.ones(3), np.ones(3)))
    assert math.isnan(auc(np.ones(3), np.zeros(3)))


def test_grouped_folds_keep_each_prompt_together() -> None:
    pid = ["a", "b", "a", "c", "b", "d", "c", "e", "a"]
    fold = grouped_folds(pid, 3, seed=0)
    assert len(fold) == len(pid)
    assert set(fold.tolist()) == {0, 1, 2}
    covered = np.zeros(len(pid), dtype=bool)
    for rows in group_indices(pid):
        covered[rows] = True
        assert len(set(fold[rows].tolist())) == 1
    assert covered.all()
    assert np.array_equal(grouped_folds(pid, 3, seed=0), fold)
    assert not np.array_equal(grouped_folds(pid, 3, seed=1), fold)


def test_ridge_scores_match_the_primal_system() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(25, 4))
    train[:, 2] = 1.0
    test = rng.normal(size=(8, 4))
    labels = train[:, 0] - 0.5 * train[:, 1] + 0.25
    for lam in (1e-6, 1.0, 10.0):
        assert np.allclose(
            ridge_scores(train, labels, test, lam),
            _primal_ridge(train, labels, test, lam),
            atol=1e-6,
        )
    assert np.allclose(ridge_scores(train, labels, test, 1e12), 0.0, atol=1e-6)


def test_probe_auc_finds_a_shared_coordinate_and_misses_a_prompt_hash() -> None:
    rng = np.random.default_rng(0)
    n_prompts, per, dim = 20, 8, 4
    pid = np.repeat(np.arange(n_prompts), per)
    labels = rng.integers(0, 2, size=len(pid)).astype(np.float64)
    labels[0] = 0.0
    labels[1] = 1.0
    features = rng.normal(size=(len(labels), dim))
    features[:, 0] = labels + rng.normal(0.0, 0.05, size=len(labels))
    found = probe_auc(features, labels, pid, n_folds=5, seed=0)
    assert found["auc"] > 0.9
    assert found["n"] == len(labels)
    assert found["n_pos"] == int(labels.sum())

    rng = np.random.default_rng(1)
    n_prompts, per, dim = 40, 6, 4
    pid = np.repeat(np.arange(n_prompts), per)
    labels = (pid % 2).astype(np.float64)
    features = rng.normal(size=(len(labels), dim))
    chance = probe_auc(features, labels, pid, n_folds=5, lam=1.0, seed=0)
    assert abs(chance["auc"] - 0.5) < 0.15


def test_arm_error_probe_fp_uses_wrong_rows_and_verifier_bits() -> None:
    rng = np.random.default_rng(0)
    oracle = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    verdict = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    pid = [f"p{i}" for i in range(len(oracle))]
    features = rng.normal(size=(len(oracle), 3))
    fp = arm_error_probe(features, oracle, verdict, pid, "fp", n_folds=5, seed=0)
    wrong = oracle == 0
    assert fp["n"] == int(wrong.sum()) == 6
    assert fp["n_pos"] == int((verdict[wrong] == 1).sum()) == 3
    assert fp["n"] != len(oracle)
    assert "auc" in fp
    assert fp["lam"] == 1.0
    assert fp["n_folds"] == 5

    fn = arm_error_probe(features, oracle, verdict, pid, "fn", n_folds=4, seed=0)
    correct = oracle == 1
    assert fn["n"] == int(correct.sum())
    assert fn["n_pos"] == int((verdict[correct] == 0).sum())
    with pytest.raises(ValueError, match="kind"):
        arm_error_probe(features, oracle, verdict, pid, "other")
