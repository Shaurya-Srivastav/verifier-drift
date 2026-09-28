"""Generative process, verifier arms and step-0 rates."""

from __future__ import annotations

import numpy as np
import pytest

from vfates.theory.dynamics import softmax
from vfates.theory.forecasts import mean_precision
from vfates.theory.worlds import (
    hand_featured_world,
    knife_edge_verifier,
    knife_edge_world,
    matched_resampled,
    random_feature_world,
    step0_rates,
    verifier_matrix,
)


def _replay_hand(P: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((P, 16))
    c = rng.normal(-1.2, 1.0, P)
    base[:, 0:2] = c[:, None] + rng.normal(0.0, 0.3, (P, 2))
    base[:, 2:4] = c[:, None] + 0.3 + rng.normal(0.0, 0.3, (P, 2))
    h = rng.normal(-0.3, 0.5, (P, 2))
    base[:, 4:6] = h
    base[:, 6:8] = h
    base[:, 8:16] = rng.normal(0.0, 0.3, (P, 8))
    return base


def test_hand_featured_layout_and_rng_order() -> None:
    world = hand_featured_world(6, seed=0, noisy_skill=False)
    assert world.O.shape == (6, 16)
    assert world.phi.shape == (6, 16, 3)
    assert np.all(world.O[:, 0:4] == 1)
    assert np.all(world.O[:, 4:] == 0)
    assert np.allclose(world.base, _replay_hand(6, 0))
    assert np.array_equal(world.base[:, 4:6], world.base[:, 6:8])
    assert np.allclose(world.phi[0], world.phi[3])
    skill, form, exploit = world.phi[0].T
    assert np.array_equal(skill, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(form, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(exploit, [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for mask in world.groups.values():
        assert mask.shape == (16,)
    covered = np.zeros(16, dtype=bool)
    for mask in world.groups.values():
        assert not np.any(covered & mask)
        covered |= mask
    assert covered.all()


def test_noisy_skill_loads_plain_wrong_outputs() -> None:
    clean = hand_featured_world(4, 1, noisy_skill=False)
    noisy = hand_featured_world(4, 1, noisy_skill=True)
    assert np.allclose(clean.base, noisy.base)
    assert np.all(noisy.phi[0, 8:12, 0] == 1)
    assert np.all(clean.phi[0, 8:12, 0] == 0)
    assert np.allclose(noisy.phi[0, :, 1:], clean.phi[0, :, 1:])


def test_random_feature_replay_and_impurity() -> None:
    P, seed, d, impurity = 5, 3, 16, 0.5
    world = random_feature_world(P, seed, d=d, impurity=impurity)
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0 / np.sqrt(d), size=(P, 16, d))
    skill_on = np.zeros((P, 16))
    skill_on[:, 0:4] = 1.0
    skill_on[:, 8:12] = 1.0
    raw = np.random.default_rng(999).normal(size=(d, 4))
    dirs = np.linalg.qr(raw)[0].T
    s_dir, e_dir, f_dir, _ = dirs
    expl = np.zeros(16)
    expl[4:6] = 1.0
    form = np.zeros(16)
    form[2:4] = 1.0
    phi = (
        z + skill_on[..., None] * s_dir + expl[None, :, None] * e_dir + form[None, :, None] * f_dir
    )
    base = rng.normal(0.0, 0.3, (P, 16))
    c = rng.normal(-1.2, 1.0, P)
    base[:, 0:4] += c[:, None]
    h = rng.normal(-0.3, 0.5, (P, 2))
    base[:, 4:6] = h
    base[:, 6:8] = h
    assert np.allclose(world.phi, phi)
    assert np.allclose(world.base, base)
    assert np.allclose(dirs @ dirs.T, np.eye(4), atol=1e-8)
    again = random_feature_world(P, seed, d=d, impurity=impurity)
    assert np.allclose(again.phi, world.phi)


def test_verifier_arms_and_matched_rates() -> None:
    world = hand_featured_world(20, seed=2, noisy_skill=False)
    oracle = verifier_matrix(world, "oracle")
    assert np.array_equal(oracle, world.O)
    fp_f = verifier_matrix(world, "fpF")
    fp_i = verifier_matrix(world, "fpI")
    assert np.all(fp_f[:, 0:6] == 1)
    assert np.all(fp_f[:, 6:] == 0)
    assert np.all(fp_i[:, 0:4] == 1)
    assert np.all(fp_i[:, 4:6] == 0)
    assert np.all(fp_i[:, 6:8] == 1)
    fn = verifier_matrix(world, "fnC")
    assert np.all(fn[:, 0:2] == 1)
    assert np.all(fn[:, 2:4] == 0)
    combo = verifier_matrix(world, "fnC+fpI")
    swapped = verifier_matrix(world, "fpI+fnC")
    assert np.array_equal(combo, swapped)
    assert np.all(combo[:, 0:2] == 1)
    assert np.all(combo[:, 2:6] == 0)
    assert np.all(combo[:, 6:8] == 1)
    rates_f = step0_rates(world, fp_f)
    rates_i = step0_rates(world, fp_i)
    assert abs(rates_f.J - rates_i.J) < 1e-12
    assert abs(rates_f.fpr - rates_i.fpr) < 1e-12
    spec = matched_resampled(world, fp_f)
    assert spec.rho0 == pytest.approx(rates_f.fpr)
    assert spec.rho1 == pytest.approx(rates_f.fnr)
    assert abs(spec.J - rates_f.J) < 1e-12
    assert abs(rates_f.J - ((1.0 - rates_f.fnr) - rates_f.fpr)) < 1e-12
    pi = softmax(world.base)
    assert rates_f.prec_prompt_mean == pytest.approx(mean_precision(pi, world.O, fp_f))


def test_fn_subset_fraction_is_exact() -> None:
    P = 50
    world = hand_featured_world(P, 0, False)
    fraction = 0.57
    v = verifier_matrix(world, "fnI", seed=4, subset_fraction=fraction)
    hit = np.all(v[:, 0:4] == 0, axis=1)
    assert int(hit.sum()) == int(np.round(fraction * P))
    assert np.array_equal(v[~hit], world.O[~hit])
    again = verifier_matrix(world, "fnI", seed=4, subset_fraction=fraction)
    assert np.array_equal(v, again)
    other = verifier_matrix(world, "fnI", seed=5, subset_fraction=fraction)
    assert not np.array_equal(v, other)


def test_unknown_arm_rejected() -> None:
    world = hand_featured_world(2, 0, False)
    with pytest.raises(ValueError):
        verifier_matrix(world, "fpX")


def test_knife_edge_masses() -> None:
    world = knife_edge_world(8, seed=0)
    v = knife_edge_verifier(world)
    pi = softmax(world.base)
    mass = pi.mean(axis=0)
    assert mass[0] == pytest.approx(0.15, abs=1e-12)
    assert mass[1] == pytest.approx(0.12, abs=1e-12)
    assert mass[2] == pytest.approx(0.003, abs=1e-12)
    rates = step0_rates(world, v)
    assert abs(rates.prec_prompt_mean - 0.15 / 0.27) < 1e-12
    assert abs(rates.J - (1.0 - 0.12 / 0.85)) < 1e-12
    assert np.all(v[:, 0:2] == 1)
    assert np.all(v[:, 2:] == 0)
