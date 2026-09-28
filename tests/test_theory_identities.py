"""Tilt invariance, pathwise decomposition, J scaling and group advantages."""

from __future__ import annotations

import numpy as np
import pytest

from vfates.theory.identities import (
    check_decomposition,
    check_group_advantage_scaling,
    check_j_scaling,
    check_tilt_invariance,
)
from vfates.theory.worlds import hand_featured_world, random_feature_world, verifier_matrix


def test_tilt_invariance_and_decomposition() -> None:
    worlds = (
        hand_featured_world(12, 0, True),
        random_feature_world(12, 1, d=8, impurity=0.25),
    )
    for world in worlds:
        for arm in ("fpF", "fnC+fpI", "oracle"):
            v = verifier_matrix(world, arm, seed=0)
            assert check_tilt_invariance(world, v, t=(1.0, 5.0, 20.0)) < 1e-10
            assert check_decomposition(world, v) < 1e-10


def test_j_scaling_aligns_with_the_oracle_update() -> None:
    world = hand_featured_world(80, seed=0, noisy_skill=False)
    cosine, scale = check_j_scaling(world, rho0=0.1, rho1=0.2, n_draws=80, seed=0)
    assert cosine > 0.999
    assert scale == pytest.approx(1.0, abs=0.02)


def test_group_advantage_matches_within_sampling_error() -> None:
    rows = check_group_advantage_scaling(
        G_list=(4, 8),
        p_list=(0.3, 0.7),
        J_list=(0.5, 1.0),
        n_groups=12000,
        seed=0,
    )
    assert len(rows) == 8
    for row in rows:
        assert abs(row["z_O1"]) <= 4.0, row
        assert abs(row["z_O0"]) <= 4.0, row
        assert row["std_sign_flip_rate"] == 0.0
        assert row["ratio_O1"] == pytest.approx(1.0, abs=0.05)
        # Sign of the conditional advantage agrees with (O - p), hence with J > 0.
        assert row["emp_O1"] > 0.0
        assert row["emp_O0"] < 0.0
        assert np.sign(row["emp_O1"]) == np.sign(row["theory_O1"])
