"""Aggregate drift metrics over a training trajectory + a stdlib bootstrap CI."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple


def drift(trajectory: List[dict], key: str) -> float:
    """Change in a scalar metric from first to last checkpoint."""
    if len(trajectory) < 2:
        return 0.0
    return float(trajectory[-1][key] - trajectory[0][key])


def bootstrap_ci(values: List[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def difficulty_controlled_drift(trajectory: List[dict]) -> Dict[int, float]:
    """Drift in the sign-flip rate within each difficulty bin (first->last)."""
    if len(trajectory) < 2:
        return {}
    first = trajectory[0].get("sign_flip_by_difficulty", {})
    last = trajectory[-1].get("sign_flip_by_difficulty", {})
    keys = set(first) & set(last)
    return {int(k): float(last[k] - first[k]) for k in sorted(keys, key=lambda x: int(x))}
