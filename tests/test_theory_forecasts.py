"""Mean precision, modal precision and transfer precision on toy worlds."""

from __future__ import annotations

import numpy as np
import pytest

from vfates.forecast.transfer import TransferForecast
from vfates.theory.dynamics import TrainResult, softmax
from vfates.theory.forecasts import (
    exact_tau,
    mean_precision,
    modal_precision,
    observed_heldout_accuracy,
    sampled_tau,
)
from vfates.theory.worlds import hand_featured_world, verifier_matrix


def _tiny() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Two prompts. Prompt 0 accepts both outputs; prompt 1 accepts nothing.
    base = np.array([[0.0, 0.0], [1.0, 0.0]])
    o = np.array([[1.0, 0.0], [1.0, 0.0]])
    v = np.array([[1.0, 1.0], [0.0, 0.0]])
    pi = softmax(base)
    return base, o, v, pi


def test_mean_and_modal_precision_with_an_empty_accepted_set() -> None:
    base, o, v, pi = _tiny()
    # Prompt 0: equal mass, precision 1/2. Prompt 1: base accuracy of softmax([1, 0]).
    base_acc_1 = float(pi[1] @ o[1])
    assert mean_precision(pi, o, v) == pytest.approx(0.5 * (0.5 + base_acc_1))
    # Prompt 0 has equal logits on both accepted outputs (labels 1 and 0), so
    # the pi_0-weighted tie average is 0.5, not the lowest-index label 1.
    # Prompt 1 has nothing accepted and falls back to base accuracy.
    assert modal_precision(base, o, v) == pytest.approx(0.5 * (0.5 + base_acc_1))


def test_modal_precision_averages_tied_argmax_with_pi0() -> None:
    # R1 F7. One prompt, two accepted atoms at equal logits with labels 1 and 0.
    # Lowest-index tie break reports 1. The beta = 0 flow keeps the tie, and
    # the pi_0-weighted average (the same rule as exact_tau) is 0.5.
    base = np.array([[0.0, 0.0, -3.0]])
    labels = np.array([[1.0, 0.0, 0.0]])
    verdict = np.array([[1.0, 1.0, 0.0]])
    assert modal_precision(base, labels, verdict) == pytest.approx(0.5)
    # A unique argmax is that output's label, here the higher logit with O = 0.
    unique = np.array([[1.0, 0.0]])
    assert modal_precision(unique, np.array([[0.0, 1.0]]), np.array([[1.0, 1.0]])) == pytest.approx(
        0.0
    )


def test_exact_tau_weights_ties_by_pi0() -> None:
    train = hand_featured_world(1, 0, False)
    # Rebuild a one-prompt world whose two outputs tie in velocity.
    train.base = np.zeros((1, 16))
    train.phi[:] = 0.0
    train.phi[0, 0, 0] = 1.0
    train.phi[0, 1, 0] = 1.0
    train.O[:] = 0
    train.O[0, 0] = 1
    v = np.zeros_like(train.O)
    v[0, 0] = 1
    # g_V loads on the skill axis. Outputs 0 and 1 tie; only output 0 is correct.
    # pi0 is uniform, so the tie average is 1/2.
    tau = exact_tau(train, v, train)
    assert tau == pytest.approx(0.5)


def test_exact_tau_unique_argmax() -> None:
    world = hand_featured_world(8, 1, False)
    held = hand_featured_world(8, 2, False)
    v = verifier_matrix(world, "oracle")
    tau = exact_tau(world, v, held)
    assert 0.0 <= tau <= 1.0
    assert tau > 0.5


def test_sampled_tau_and_heldout_read_theta_only() -> None:
    train = hand_featured_world(12, 0, False)
    held = hand_featured_world(12, 9, False)
    v = verifier_matrix(train, "fpI")
    forecast = sampled_tau(train, v, held, n_samples=8, seed=3, reliability_splits=4)
    assert isinstance(forecast, TransferForecast)
    assert np.isfinite(forecast.tau_hard)
    assert np.isfinite(forecast.tau_rel)
    theta = np.array([2.0, 0.0, -1.0])
    result = TrainResult(theta=theta, tab=np.ones_like(held.base), history={})
    acc = observed_heldout_accuracy(result, held)
    pi = softmax(held.base + np.einsum("pkd,d->pk", held.phi, theta))
    assert acc == pytest.approx(float((pi * held.O).sum(-1).mean()))
