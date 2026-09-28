"""Pooled H3 power."""

from __future__ import annotations

from vfates.analysis.power import pooled_h3_power


def test_pooled_h3_power_rises_with_rho() -> None:
    null = pooled_h3_power(12, 0.0, n_cells=3, n_sim=40, n_perm=49, seed=0)
    strong = pooled_h3_power(12, 0.95, n_cells=1, n_sim=20, n_perm=49, seed=1)
    assert null["power"] < 0.30
    assert strong["power"] > 0.70
    assert null["cell_sizes"] == [4, 4, 4]
