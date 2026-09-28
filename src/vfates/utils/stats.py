"""Small, dependency-free statistics used by audits, forecasts and reports.

Resampling is always clustered by prompt: samples from the same prompt are not
independent, so confidence intervals resample prompts, not rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def rankdata(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    a = np.asarray(x, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def group_indices(groups: Sequence[object] | np.ndarray) -> list[np.ndarray]:
    """Row indices of each distinct group, in first-appearance order."""
    index: dict[object, list[int]] = {}
    for i, g in enumerate(groups):
        index.setdefault(g, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in index.values()]


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: Sequence[object],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of ``stat(row_indices)``, resampling clusters.

    ``stat`` receives an integer array of row indices (with repetition) and
    returns a scalar.
    """
    groups = group_indices(clusters)
    all_rows = np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
    point = float(stat(all_rows))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        draws[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def mean_ci(
    values: Sequence[float], alpha: float = 0.05, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    return cluster_bootstrap(
        lambda idx: float(v[idx].mean()), list(range(len(v))), n_boot, alpha, seed
    )


def paired_diff_ci(
    a: Sequence[float], b: Sequence[float], alpha: float = 0.05, n_boot: int = 4000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``a - b`` over paired units (for example seeds) with a bootstrap CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return mean_ci(d, alpha=alpha, n_boot=n_boot, seed=seed)
