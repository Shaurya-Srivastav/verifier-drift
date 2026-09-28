"""Euclidean policy gradient, tilt and sampled GRPO."""

from __future__ import annotations

import numpy as np
import pytest

from vfates.theory.dynamics import (
    expected_reward,
    population_flow,
    sampled_grpo,
    shared_update,
    softmax,
    tilt,
)
from vfates.theory.forecasts import exact_tau, modal_precision, observed_heldout_accuracy
from vfates.theory.worlds import (
    ResampledSpec,
    hand_featured_world,
    knife_edge_verifier,
    knife_edge_world,
    verifier_matrix,
)


def test_softmax_and_tilt_are_stable() -> None:
    logits = np.array([[1000.0, 1000.0, 999.0], [0.0, -40.0, 40.0]])
    pi = softmax(logits)
    assert np.allclose(pi.sum(-1), 1.0)
    assert pi[0, 0] == pytest.approx(pi[0, 1])
    world = hand_featured_world(4, 0, False)
    v = verifier_matrix(world, "fpF")
    assert np.allclose(tilt(world, v, 3.5), softmax(world.base + 3.5 * v))


def test_one_euler_step_matches_the_covariance() -> None:
    world = hand_featured_world(5, 1, False)
    v = verifier_matrix(world, "fpF").astype(np.float64)
    eta = 0.3
    flow = population_flow(world, v, eta=eta, steps=1, param="shared")
    pi0 = softmax(world.base)
    assert np.allclose(flow.theta, eta * shared_update(world.phi, pi0, v))
    assert np.allclose(flow.tab, 0.0)
    tab = population_flow(world, v, eta=eta, steps=1, param="tabular")
    mean = (pi0 * v).sum(-1, keepdims=True)
    assert np.allclose(tab.tab, eta * pi0 * (v - mean))
    assert np.allclose(tab.theta, 0.0)
    assert len(flow.history["train_acc"]) == 2


def test_resampled_expectation_matches_the_matrix_flow() -> None:
    world = hand_featured_world(6, 2, True)
    spec = ResampledSpec(0.2, 0.15)
    reward = expected_reward(world, spec)
    a = population_flow(world, spec, eta=0.4, steps=4, param="both", beta=0.1)
    b = population_flow(world, reward, eta=0.4, steps=4, param="both", beta=0.1)
    assert np.allclose(a.theta, b.theta)
    assert np.allclose(a.tab, b.tab)


def test_beta_equilibrium_is_the_tilt_and_beta0_is_the_mode() -> None:
    world = hand_featured_world(24, 0, False)
    v = verifier_matrix(world, "fpF")
    beta = 0.5
    flow = population_flow(world, v, eta=1.5, steps=800, param="tabular", beta=beta)
    eq = tilt(world, v, 1.0 / beta)
    eq_acc = float((eq * world.O).sum(-1).mean())
    assert flow.history["train_acc"][-1] == pytest.approx(eq_acc, abs=0.02)
    modal = population_flow(world, v, eta=2.0, steps=500, param="tabular", beta=0.0)
    target = modal_precision(world.base, world.O, v)
    assert modal.history["train_acc"][-1] == pytest.approx(target, abs=0.03)


def test_fp_directions_and_heldout_ignores_tabular_head() -> None:
    train = hand_featured_world(24, 3, False)
    held = hand_featured_world(12, 4, False)
    pi0 = softmax(train.base)
    g_f = shared_update(train.phi, pi0, verifier_matrix(train, "fpF").astype(float))
    g_i = shared_update(train.phi, pi0, verifier_matrix(train, "fpI").astype(float))
    assert g_f[2] > 0.0
    assert g_i[2] < 0.0
    trained = sampled_grpo(
        train,
        verifier_matrix(train, "oracle"),
        steps=12,
        batch=8,
        G=4,
        lr_shared=0.0,
        lr_tab=2.0,
        seed=0,
        heldout=held,
        eval_every=4,
    )
    assert trained.history["heldout_acc"][0] == pytest.approx(trained.history["heldout_acc"][-1])
    assert trained.history["train_acc"][-1] != pytest.approx(
        trained.history["train_acc"][0], abs=1e-4
    )
    again = sampled_grpo(
        train,
        verifier_matrix(train, "oracle"),
        steps=12,
        batch=8,
        G=4,
        lr_shared=0.0,
        lr_tab=2.0,
        seed=0,
        heldout=held,
        eval_every=4,
    )
    assert np.allclose(trained.tab, again.tab)


def test_zero_variance_advantage_does_not_move_logits() -> None:
    world = hand_featured_world(8, 0, False)
    # Oracle rewards still vary inside a group. A constant verifier does not.
    constant = np.ones_like(world.O)
    trained = sampled_grpo(
        world, constant, steps=5, batch=4, G=4, lr_shared=1.0, lr_tab=2.0, adv="std", seed=1
    )
    assert np.allclose(trained.theta, 0.0)
    assert np.allclose(trained.tab, 0.0)


def test_knife_edge_rises_then_collapses() -> None:
    train = knife_edge_world(80, seed=0)
    held = knife_edge_world(80, seed=7)
    v = knife_edge_verifier(train)
    tau = exact_tau(train, v, held)
    flow = population_flow(train, v, eta=8.0, steps=1400, param="shared", dt=0.05, heldout=held)
    acc = np.asarray(flow.history["heldout_acc"])
    assert tau > 0.65
    assert acc[8] > acc[0]
    assert acc.max() > acc[0] + 0.15
    assert acc[-1] < 0.08
    assert observed_heldout_accuracy(flow, held) == pytest.approx(acc[-1])
