"""Unit tests for the step-0 transfer screen in ``vfates.forecast.transfer``."""

from __future__ import annotations

import numpy as np
import pytest

from vfates.forecast.transfer import (
    group_std_advantages,
    length_division,
    predicted_reliability,
    prompt_centered,
    prompts_for_reliability,
    q_label,
    reliability_budget,
    split_half_reliability,
    tau_hard,
    tau_pool,
    tau_pool_interval,
    tau_soft,
    transfer_forecast,
    velocities,
    velocity_direction,
)
from vfates.train.grpo_core import group_advantages
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


def _prompt_world(
    n_prompt: int, n_sample: int, between: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rewarded samples carry ``mu`` plus a per-prompt offset, under unit feature noise."""
    rng = np.random.default_rng(seed)
    dim = 300
    mu = rng.normal(0.0, 0.1, dim)
    pid = np.repeat(np.arange(n_prompt), n_sample)
    reward = rng.integers(0, 2, n_prompt * n_sample).astype(float)
    offset = rng.normal(0.0, between, (n_prompt, dim))[pid]
    features = reward[:, None] * (mu + offset) + rng.normal(0.0, 1.0, (len(pid), dim))
    return features, pid, reward


def test_reliability_budget_predicts_split_half_cosine() -> None:
    features, pid, reward = _prompt_world(240, 16, between=0.2)
    budget = reliability_budget(features, pid, reward)
    assert budget["prompts"] == 240 and budget["samples_per_prompt"] == 16
    measured = split_half_reliability(features, pid, reward, n_splits=20, seed=0)
    assert abs(predicted_reliability(budget, 240, 16) - measured) < 0.05

    # A smaller pool with fewer samples per prompt is predicted from the full one.
    keep = (pid < 96) & (np.arange(len(pid)) % 16 < 8)
    small = split_half_reliability(features[keep], pid[keep], reward[keep], n_splits=40, seed=1)
    assert abs(predicted_reliability(budget, 96, 8) - small) < 0.08

    needed = prompts_for_reliability(budget, 0.8, 16)
    assert needed is not None
    assert predicted_reliability(budget, needed, 16) >= 0.8
    assert predicted_reliability(budget, needed - 2, 16) < 0.8


def test_reliability_budget_separates_prompt_and_sample_noise() -> None:
    sampling_only = reliability_budget(*_prompt_world(200, 16, between=0.0))
    with_prompts = reliability_budget(*_prompt_world(200, 16, between=0.6))
    assert 0.9 < sampling_only["within_share"] < 1.15
    assert with_prompts["within_share"] < sampling_only["within_share"] - 0.1
    assert sampling_only["within"] == pytest.approx(
        sampling_only["within_share"] * sampling_only["noise"]
    )

    features, pid, reward = _prompt_world(8, 4, between=0.0)
    for size in (2, 3):
        keep = np.arange(len(pid)) % 4 < size
        with pytest.raises(ValueError, match="four samples"):
            reliability_budget(features[keep], pid[keep], reward[keep])
    null = dict(sampling_only, signal=-1.0)
    assert predicted_reliability(null, 1024, 16) == 0.0
    assert prompts_for_reliability(null, 0.8, 16) is None
    with pytest.raises(ValueError, match="positive"):
        predicted_reliability(sampling_only, 1024, 0)
    with pytest.raises(ValueError, match="positive"):
        predicted_reliability(sampling_only, 0, 16)
    with pytest.raises(ValueError, match="positive"):
        prompts_for_reliability(sampling_only, 0.8, 0)


def test_reliability_budget_without_prompt_noise() -> None:
    # Every prompt repeats the same samples, so c_p does not vary across prompts.
    rng = np.random.default_rng(0)
    phi = rng.normal(0.0, 1.0, (5, 6))
    reward = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    pid = np.repeat(np.arange(10), 5)
    budget = reliability_budget(np.tile(phi, (10, 1)), pid, np.tile(reward, 10))
    assert budget["noise"] == pytest.approx(0.0, abs=1e-24) and budget["signal"] > 0
    exact = dict(budget, noise=0.0, within_share=0.0)
    assert predicted_reliability(exact, 10, 5) == 1.0
    assert prompts_for_reliability(exact, 0.8, 5) == 4


def test_jackknife_within_variance() -> None:
    from vfates.forecast.transfer import _jackknife_variance, _prompt_contribution

    rng = np.random.default_rng(0)
    phi = rng.normal(0.0, 1.0, (7, 4)) + 3.0
    reward = rng.integers(0, 2, 7).astype(float)
    loo = np.stack(
        [_prompt_contribution(np.delete(phi, i, 0), np.delete(reward, i)) for i in range(7)]
    )
    brute = float(((loo - loo.mean(0)) ** 2).sum() * 6 / 7)
    assert _jackknife_variance(phi, reward) == pytest.approx(brute, rel=1e-10)

    # Against the Monte Carlo variance of c_p, at an odd and an even group size.
    dim, reps = 20, 4000
    mu = rng.normal(0.0, 0.5, dim)
    for n in (5, 16):
        draws, jack = [], []
        for _ in range(reps):
            v = (rng.random(n) < 0.4).astype(float)
            x = v[:, None] * mu + rng.normal(0.0, 1.0, (n, dim)) + 2.0
            draws.append(_prompt_contribution(x, v))
            jack.append(_jackknife_variance(x, v))
        true = float(np.var(np.stack(draws), axis=0, ddof=1).sum())
        assert 0.97 * true < float(np.mean(jack)) < 1.35 * true


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
    assert "tau_pool_std" in forecast.to_json()


def test_length_division_flips_the_pooled_tail() -> None:
    # R1 F1. Short correct completion, long exploit. Without /L the exploit
    # wins the pooled tail (tau 0). Dividing train rows by length makes the
    # correct completion win (tau 1). Held-out features stay undivided.
    phi, lengths, verdict, labels, pid = [], [], [], [], []
    for prompt in range(16):
        phi += [[4.0, 0.0], [0.0, 5.0], [0.0, 0.0], [0.0, 0.0]]
        lengths += [1.0, 10.0, 1.0, 1.0]
        verdict += [1.0, 1.0, 0.0, 0.0]
        labels += [1.0, 0.0, 0.0, 0.0]
        pid += [prompt, prompt, prompt, prompt]
    phi_a, length_a, verdict_a, label_a, pid_a = map(
        np.asarray, (phi, lengths, verdict, labels, pid)
    )
    assert length_division("sum", "seq_mean") is True
    assert length_division("token_mean", "seq_mean") is False
    assert length_division("sum", "token_mean") is False
    assert length_division("sum", "dr_constant") is False
    with pytest.raises(ValueError):
        length_division("token_mean", "token_mean")
    with pytest.raises(ValueError):
        length_division("token_mean", "dr_constant")
    g_sum = velocity_direction(phi_a, pid_a, verdict_a)
    g_div = velocity_direction(phi_a, pid_a, verdict_a, length_a)
    assert tau_pool(velocities(phi_a, g_sum), pid_a, label_a, 0.25) == pytest.approx(0.0)
    undivided = tau_pool(velocities(phi_a, g_div), pid_a, label_a, 0.25)
    assert undivided == pytest.approx(1.0)
    forecast = transfer_forecast(
        phi_a,
        pid_a,
        verdict_a,
        label_a,
        phi_a,
        pid_a,
        label_a,
        train_lengths=length_a,
        q=0.25,
        qs=(0.25,),
        reliability_splits=4,
    )
    plain = transfer_forecast(
        phi_a,
        pid_a,
        verdict_a,
        label_a,
        phi_a,
        pid_a,
        label_a,
        q=0.25,
        qs=(0.25,),
        reliability_splits=4,
    )
    assert forecast.tau_pool == pytest.approx(undivided)
    assert plain.tau_pool == pytest.approx(0.0)
    # Held-out rows stay undivided. Here the long sample wins on raw phi and
    # loses if those held rows are divided by length; the forecast must keep
    # the undivided ranking (the change in sequence log-probability).
    train_phi = np.array([[1.0, 0.0], [-1.0, 0.0]])
    train_pid = np.array([0, 0])
    train_v = np.array([1.0, 0.0])
    train_l = np.array([1.0, 1.0])
    held_phi = np.array([[3.0, 0.0], [10.0, 0.0]])
    held_l = np.array([1.0, 10.0])
    held_o = np.array([1.0, 0.0])
    held_pid = np.array([0, 0])
    g = velocity_direction(train_phi, train_pid, train_v, train_l)
    undivided_tail = tau_pool(velocities(held_phi, g), held_pid, held_o, 0.5)
    divided_tail = tau_pool(velocities(held_phi / held_l[:, None], g), held_pid, held_o, 0.5)
    assert undivided_tail == pytest.approx(0.0)
    assert divided_tail == pytest.approx(1.0)
    held_forecast = transfer_forecast(
        train_phi,
        train_pid,
        train_v,
        held_o,
        held_phi,
        held_pid,
        held_o,
        train_lengths=train_l,
        q=0.5,
        qs=(0.5,),
        reliability_splits=4,
    )
    assert held_forecast.tau_pool == pytest.approx(undivided_tail)


def test_std_weighted_direction_can_disagree_with_the_covariance() -> None:
    # R1 F2, with the unbiased standard deviation and np.isclose zeroing from
    # grpo_core.group_advantages. The covariance screen picks the correct
    # held-out output; the std-weighted direction picks the exploit.
    phi0 = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    verdict0 = np.array([1.0, 0.0, 1.0, 0.0])
    phi1 = np.zeros((20, 2))
    verdict1 = np.zeros(20)
    phi1[0, 1] = 8.0
    verdict1[0] = 1.0
    phi = np.vstack([phi0, phi1])
    verdict = np.concatenate([verdict0, verdict1])
    pid = np.array([0] * 4 + [1] * 20)
    advantages = group_std_advantages(verdict, pid)
    ref0, _info0 = group_advantages(verdict0.reshape(1, -1))
    ref1, _info1 = group_advantages(verdict1.reshape(1, -1))
    assert np.array_equal(advantages[:4], ref0.ravel())
    assert np.array_equal(advantages[4:], ref1.ravel())
    constant = group_std_advantages(np.array([1.0, 1.0, 0.0, 0.0]), [0, 0, 1, 1])
    assert np.array_equal(constant, np.zeros(4))
    tiny = group_std_advantages(np.array([1.0, 1.0 + 1e-12]), [0, 0])
    assert np.array_equal(tiny, np.zeros(2))
    g_cov = velocity_direction(phi, pid, verdict)
    g_std = velocity_direction(phi, pid, advantages, center=False)
    held = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([1.0, 0.0])
    assert labels[int(np.argmax(velocities(held, g_cov)))] == 1.0
    assert labels[int(np.argmax(velocities(held, g_std)))] == 0.0
    forecast = transfer_forecast(
        phi,
        pid,
        verdict,
        verdict,
        held,
        np.array([0, 0]),
        labels,
        reliability_splits=4,
        q=0.5,
    )
    assert forecast.tau_pool == pytest.approx(1.0)
    assert forecast.tau_pool_std == pytest.approx(0.0)
    assert forecast.to_json()["tau_pool_std"] == pytest.approx(0.0)
