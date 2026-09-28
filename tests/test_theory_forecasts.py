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
    # Prompt 0 argmax is the first output (O=1). Prompt 1 falls back to base accuracy.
    assert modal_precision(base, o, v) == pytest.approx(0.5 * (1.0 + base_acc_1))


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
