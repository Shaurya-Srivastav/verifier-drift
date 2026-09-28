"""Oracle-light tau_pool: exact at full budget, unbiased below it, tighter as m grows."""

import numpy as np
import pytest

from vfates.audit import _oracle_light_budgets
from vfates.forecast.transfer import (
    ORACLE_LIGHT_BUDGETS,
    ORACLE_LIGHT_STORED_DRAWS,
    tau_pool,
    tau_pool_budget,
    velocities,
    velocity_direction,
)
from vfates.types import AuditTable


def _population(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    pid = np.repeat(np.arange(n // 4), 4)
    oracle = rng.integers(0, 2, size=n).astype(np.float64)
    return scores, pid, oracle


def test_full_budget_matches_tau_pool_including_ties() -> None:
    scores, pid, oracle = _population()
    q = 0.5
    full = tau_pool(scores, pid, oracle, q)
    got = tau_pool_budget(scores, oracle, pid, q=q, budget=10**6, n_draws=3, seed=0)
    assert got["oracle_labels_needed"] == 40
    assert got["budget"] >= got["oracle_labels_needed"]
    assert got["mean"] == full
    assert got["sd"] == 0.0
    assert got["p05"] == full
    assert got["p95"] == full

    tied_scores = np.zeros(20)
    tied_pid = np.repeat(np.arange(5), 4)
    tied_oracle = np.array([0.0, 1.0] * 10)
    tied = tau_pool(tied_scores, tied_pid, tied_oracle, 0.25)
    covered = tau_pool_budget(
        tied_scores, tied_oracle, tied_pid, q=0.25, budget=20, n_draws=4, seed=1
    )
    assert covered["oracle_labels_needed"] == 20
    assert covered["mean"] == tied
    assert covered["sd"] == 0.0
    assert "draws" not in got
    assert "draws" not in covered
    assert ORACLE_LIGHT_BUDGETS == (32, 64, 128, 192)


def test_partial_budget_is_unbiased_and_intervals_shrink() -> None:
    scores, pid, oracle = _population(80, seed=2)
    q = 1.0
    full = tau_pool(scores, pid, oracle, q)
    small = tau_pool_budget(scores, oracle, pid, q=q, budget=8, n_draws=4000, seed=3)
    assert small["oracle_labels_needed"] == 80
    assert small["mean"] == pytest.approx(full, abs=0.03)

    widths = []
    spreads = []
    for budget in (8, 32, 64):
        est = tau_pool_budget(scores, oracle, pid, q=q, budget=budget, n_draws=2000, seed=4)
        widths.append(est["p95"] - est["p05"])
        spreads.append(est["sd"])
    assert widths[0] > widths[1] > widths[2]
    assert spreads[0] > spreads[1] > spreads[2]


def test_budget_rejects_bad_arguments() -> None:
    scores = np.array([1.0, 0.0])
    pid = np.array(["a", "a"])
    oracle = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="q"):
        tau_pool_budget(scores, oracle, pid, q=0.0, budget=1, n_draws=1, seed=0)
    with pytest.raises(ValueError, match="budget"):
        tau_pool_budget(scores, oracle, pid, q=1.0, budget=0, n_draws=1, seed=0)
    with pytest.raises(ValueError, match="n_draws"):
        tau_pool_budget(scores, oracle, pid, q=1.0, budget=1, n_draws=0, seed=0)
    empty = tau_pool_budget(
        np.zeros(0), np.zeros(0), np.zeros(0), q=0.5, budget=1, n_draws=1, seed=0
    )
    assert empty["oracle_labels_needed"] == 0
    assert empty["mean"] != empty["mean"]
    assert "draws" not in empty


def test_each_arm_has_its_own_draw_stream() -> None:
    scores, pid, oracle = _population(80, seed=5)
    left = tau_pool_budget(scores, oracle, pid, q=1.0, budget=8, n_draws=30, seed=1, arm="oracle")
    right = tau_pool_budget(scores, oracle, pid, q=1.0, budget=8, n_draws=30, seed=1, arm="random")
    assert left["draws"] != right["draws"]
    assert len(left["draws"]) == 30
    again = tau_pool_budget(scores, oracle, pid, q=1.0, budget=8, n_draws=30, seed=1, arm="oracle")
    assert again["draws"] == left["draws"]
    exact = tau_pool_budget(scores, oracle, pid, q=1.0, budget=80, n_draws=5, seed=1, arm="oracle")
    assert "draws" not in exact


def test_forecast_keeps_a_rounded_prefix_and_summaries_of_every_draw() -> None:
    rng = np.random.default_rng(0)
    n = 80
    features = rng.normal(size=(n, 3))
    verdicts = rng.integers(0, 2, size=n)
    table = AuditTable(
        pid=np.array([f"p{i // 2}" for i in range(n)], dtype=object),
        split=np.array(["train"] * 40 + ["heldout"] * 40, dtype=object),
        k=np.array([i % 2 for i in range(n)], dtype=np.int64),
        O=rng.integers(0, 2, size=n).astype(np.float64),
        answer=np.array(["1"] * n, dtype=object),
        n_tokens=np.ones(n, dtype=np.int64),
        logprob=np.zeros(n),
        V={"oracle": verdicts, "random": verdicts.copy()},
    )
    lengths = np.ones(n, dtype=np.int64)
    oracle = _oracle_light_budgets(
        table, features, lengths, "oracle", q=1.0, divide_by_length=False, seed=7
    )
    random = _oracle_light_budgets(
        table, features, lengths, "random", q=1.0, divide_by_length=False, seed=7
    )
    partial = oracle["32"]
    assert len(partial["draws"]) == ORACLE_LIGHT_STORED_DRAWS
    assert partial["draws"] == [round(value, 6) for value in partial["draws"]]
    assert partial["draws"] != random["32"]["draws"]
    assert "draws" not in oracle["128"]
    assert "draws" not in oracle["192"]
    direction = velocity_direction(features[:40], table.pid[:40], verdicts[:40])
    scores = velocities(features[40:], direction)
    raw = tau_pool_budget(
        scores,
        table.O[40:],
        table.pid[40:],
        q=1.0,
        budget=32,
        n_draws=2000,
        seed=7,
        arm="oracle",
    )
    assert partial["mean"] == raw["mean"]
    assert partial["sd"] == raw["sd"]
    assert partial["draws"] == [round(float(value), 6) for value in raw["draws"][:200]]
