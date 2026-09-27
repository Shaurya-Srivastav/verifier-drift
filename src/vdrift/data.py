"""Synthetic verifiable problems with a difficulty knob.

Each problem has a canonical gold form; equivalent answers exist in other forms
(decimal, unreduced fraction, ...). That gap is what lets a form-sensitive rule
verifier disagree with the oracle. Real GSM8K/MATH loaders are stubbed in
`load_math()` for the Mac Studio run.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import List


@dataclass
class Problem:
    prompt: str
    gold: str      # canonical gold form (what the rule verifier is keyed to)
    value: float   # numeric truth
    difficulty: int


def _canonical(fr: Fraction) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def gen_problems(n: int, seed: int = 0) -> List[Problem]:
    rng = random.Random(seed)
    out: List[Problem] = []
    for _ in range(n):
        difficulty = rng.randint(1, 3)
        d = rng.randint(2, 3 + difficulty * 3)
        a = rng.randint(1, 5 + difficulty * 5)
        b = rng.randint(1, 5 + difficulty * 5)
        f1, f2 = Fraction(a, d), Fraction(b, d * 2)
        ans = f1 + f2
        prompt = f"Compute {f1} + {f2}. Give the exact value."
        out.append(Problem(prompt=prompt, gold=_canonical(ans), value=float(ans), difficulty=difficulty))
    return out


def load_math(split: str = "test"):  # pragma: no cover - Mac Studio path
    """Load GSM8K/MATH via `datasets` on the training box. Stub on purpose."""
    raise NotImplementedError(
        "Install `datasets` and implement GSM8K/MATH loading on the training box; "
        "the mock pilot uses gen_problems()."
    )
