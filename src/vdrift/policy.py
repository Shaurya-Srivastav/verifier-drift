"""Policies that produce rollouts for a problem.

MockPolicy is a CPU-only simulator used to exercise and test the whole
measurement pipeline: correctness rises with training progress t in [0,1], and
the *answer-form distribution drifts* toward forms a rule verifier mishandles.
It is a harness test, NOT evidence about real models. The real policy
(Qwen2.5-1.5B, etc.) plugs into the identical interface in grpo.py on the GPU.
"""
from __future__ import annotations

import random
from fractions import Fraction

from .data import Problem


def render_value(value: float, form: str) -> str:
    fr = Fraction(value).limit_denominator(10 ** 6)
    if form == "integer" and fr.denominator == 1:
        return str(fr.numerator)
    if form == "decimal":
        return (f"{value:.6f}".rstrip("0").rstrip(".")) or "0"
    if form == "percent":
        return (f"{value * 100:.4f}".rstrip("0").rstrip(".")) + "%"
    return f"{fr.numerator}/{fr.denominator}"  # fraction (default)


class MockPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def rollout(self, prob: Problem, t: float) -> str:
        # correctness improves over training
        value = prob.value if self.rng.random() < (0.4 + 0.5 * t) else prob.value + self.rng.choice([-1, 1]) * 0.5
        # style drift: early answers match the gold form; later answers drift
        # toward decimals/percents that the rule verifier rejects
        if self.rng.random() < (0.1 + 0.7 * t):
            form = self.rng.choice(["decimal", "percent", "fraction"])
        else:
            form = "fraction" if "/" in prob.gold else "integer"
        return render_value(value, form)
