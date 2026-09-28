"""Unit tests for the step-0 transfer screen in ``vfates.forecast.transfer``."""

from __future__ import annotations

import numpy as np
import pytest

from vfates.forecast.transfer import (
    prompt_centered,
    q_label,
    split_half_reliability,
    tau_hard,
    tau_pool,
    tau_pool_interval,
    tau_soft,
    transfer_forecast,
    velocity_direction,
)
from vfates.utils.stats import group_indices


def _uniform_cov(features: np.ndarray, pid: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """Equal-prompt-weight covariance under the empirical uniform on each prompt's rows."""
    total = np.zeros(features.shape[1])
    groups = group_indices(pid)
    for rows in groups:
        phi = features[rows]
        r = reward[rows]
        centered_phi = phi - phi.mean(axis=0)
        centered_r = r - r.mean()
        total += (centered_phi * centered_r[:, None]).mean(axis=0)
    return total / len(groups)


def test_prompt_centered_subtracts_the_group_mean() -> None:
    values = np.array([1.0, 3.0, 10.0])
    pid = ["a", "a", "b"]
    out = prompt_centered(values, pid)
    assert np.allclose(out, [-1.0, 1.0, 0.0])
    assert prompt_centered(out, pid).sum() == pytest.approx(0.0)


def test_velocity_direction_is_the_prompt_averaged_covariance() -> None:
    rng = np.random.default_rng(0)
    # Unequal group sizes: a sample-weighted pool would not match.
    pid = np.array([0, 0, 1, 1, 1, 1])
    features = rng.normal(size=(6, 3))
    reward = rng.normal(size=6)
    got = velocity_direction(features, pid, reward)
    assert np.allclose(got, _uniform_cov(features, pid, reward))
    # Sample-weighted covariance differs when groups have different sizes and signals.
    features = np.array([[0.0], [2.0], [0.0], [0.0], [0.0], [0.0]])
    reward = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    pid = np.array([0, 0, 1, 1, 1, 1])
    equal_prompt = velocity_direction(features, pid, reward)
    # Prompt 0 covariance is 0.5; prompt 1 is 0. Mean is 0.25.
    assert equal_prompt[0] == pytest.approx(0.25)
    sample_weighted = _uniform_cov(features[:2], pid[:2], reward[:2]) * (2 / 6)
    assert equal_prompt[0] != pytest.approx(sample_weighted[0])


def test_lengths_scale_features_before_the_covariance() -> None:
    features = np.array([[2.0, 0.0], [4.0, 2.0], [1.0, -1.0]])
    pid = np.array([0, 0, 1])
    reward = np.array([0.0, 1.0, 0.5])
    lengths = np.array([1.0, 2.0, 1.0])
    scaled = features / lengths[:, None]
    assert np.allclose(
        velocity_direction(features, pid, reward, lengths),
        velocity_direction(scaled, pid, reward),
    )


def test_tau_hard_averages_ties() -> None:
    scores = np.array([1.0, 1.0, 0.2, 0.5, 0.5])
    pid = np.array([0, 0, 0, 1, 1])
    labels = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    # Prompt 0: the two maxima average to 0.5. Prompt 1: both tied, average 0.5.
    assert tau_hard(scores, pid, labels) == pytest.approx(0.5)


def test_tau_soft_endpoints() -> None:
    scores = np.array([0.0, 3.0, 1.0, 2.0, 5.0, -1.0])
    pid = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    base = float(np.mean([labels[:3].mean(), labels[3:].mean()]))
    assert tau_soft(scores, pid, labels, kappa=0.0) == pytest.approx(base)
    hard = tau_hard(scores, pid, labels)
    assert tau_soft(scores, pid, labels, kappa=1e6) == pytest.approx(hard, abs=1e-8)


def test_tau_pool_takes_the_pooled_prompt_centred_tail() -> None:
    # Prompt 1 is shifted by +100: centring must remove the shift.
    scores = np.array([0.0, 3.0, 0.0, 0.0, 100.0, 101.0, 100.0, 100.0])
    pid = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    # Centred: prompt 0 -> [-0.75, 2.25, -0.75, -0.75]; prompt 1 -> [-0.25, 0.75, -0.25, -0.25].
    assert tau_pool(scores, pid, labels, q=1 / 8) == pytest.approx(0.0)
    assert tau_pool(scores, pid, labels, q=2 / 8) == pytest.approx(0.5)
    assert tau_pool(scores, pid, labels, q=1.0) == pytest.approx(labels.mean())
    with pytest.raises(ValueError):
        tau_pool(scores, pid, labels, q=0.0)


def test_tau_pool_splits_boundary_ties_evenly() -> None:
    scores = np.array([2.0, 1.0, 1.0, -4.0])
    pid = np.zeros(4, dtype=int)
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    # k = 2: the top sample plus half of each tied sample.
    assert tau_pool(scores, pid, labels, q=0.5) == pytest.approx((1.0 + 0.5 * 1.0) / 2)
    order = np.array([0, 2, 1, 3])
    assert tau_pool(scores[order], pid, labels[order], q=0.5) == pytest.approx(0.75)


def test_tau_pool_recovers_a_shared_exploit_missing_on_most_prompts() -> None:
    # The accepted exploit (feature 1) outpaces the correct behaviour (feature 0).
    # It appears on every training prompt but was sampled on only a quarter of
    # held-out prompts: per-prompt argmax is fooled elsewhere, the pooled tail is not.
    rng = np.random.default_rng(0)
    n_prompt, n = 128, 16
    pid = np.repeat(np.arange(n_prompt), n)

    def audit(exploit_prompt_share: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        O = (rng.random(pid.size) < 0.3).astype(float)
        on = (rng.random(n_prompt) < exploit_prompt_share)[pid]
        exploit = on & (rng.random(pid.size) < 0.4)
        O[exploit] = 0.0
        phi = np.stack([O, exploit.astype(float)], axis=1) + rng.normal(0, 0.01, (pid.size, 2))
        return phi, O, np.maximum(O, exploit.astype(float))

    train_phi, train_O, train_V = audit(1.0)
    held_phi, held_O, _ = audit(0.25)
    forecast = transfer_forecast(
        train_phi, pid, train_V, train_O, held_phi, pid, held_O, reliability_splits=4
    )
    assert forecast.tau_pool < 0.05
    assert forecast.tau_hard > 0.4
    assert set(forecast.tau_pool_curve) == {"1/64", "1/32", "1/16", "1/8"}


def test_tau_pool_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(1)
    n_prompt, n = 40, 8
    pid = np.repeat(np.arange(n_prompt), n)
    O = (rng.random(n_prompt * n) < 0.4).astype(float)
    phi = O[:, None] * np.array([1.0, 0.0]) + rng.normal(0, 0.5, (n_prompt * n, 2))
    forecast = transfer_forecast(
        phi, pid, O, O, phi, pid, O, q=1 / 8, reliability_splits=4, n_boot=200, seed=0
    )
    assert forecast.tau_pool_ci is not None
    lo, hi = forecast.tau_pool_ci
    assert 0.0 <= lo <= forecast.tau_pool <= hi <= 1.0
    assert (lo, hi) == tau_pool_interval(phi, pid, O, phi, pid, O, q=1 / 8, n_boot=200, seed=0)
    assert q_label(1 / 32) == "1/32"
    assert q_label(0.3) == "0.3"


def test_reliability_coverage_and_tau_rel() -> None:
    rng = np.random.default_rng(0)
    n_prompt, n_sample = 12, 6
    pid = np.repeat(np.arange(n_prompt), n_sample)
    labels = np.zeros(n_prompt * n_sample)
    labels[::2] = 1.0
    features = labels[:, None] * np.array([1.0, 0.0, 0.0, 0.0])
    features = features + rng.normal(0.0, 0.05, features.shape)
    reliability = split_half_reliability(features, pid, labels, n_splits=8, seed=0)
    assert reliability > 0.9
    assert np.isnan(split_half_reliability(features[:18], pid[:18], labels[:18], n_splits=4))

    train_pid = np.array([0, 0, 0, 1, 1, 1])
    train_o = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    train_phi = np.zeros((6, 2))
    train_phi[train_o == 1.0, 0] = 1.0
    held_pid = np.array([0, 0, 0, 1, 1, 1])
    held_o = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    held_phi = np.zeros((6, 2))
    held_phi[3, 0] = 1.0
    forecast = transfer_forecast(
        train_phi,
        train_pid,
        train_o,
        train_o,
        held_phi,
        held_pid,
        held_o,
        reliability_splits=4,
        seed=0,
    )
    assert forecast.coverage == pytest.approx(0.5)
    assert forecast.oracle_tau_hard == pytest.approx(0.5)
    assert forecast.tau_hard == pytest.approx(0.5)
    assert forecast.tau_rel == pytest.approx(1.0)
    assert forecast.base_acc_heldout == pytest.approx(held_o.reshape(2, 3).mean(-1).mean())
