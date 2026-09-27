"""The measurement core: verifier error by answer-form stratum, and whether the
verifier's error flips GRPO's group-relative advantage sign versus the oracle.

Everything here is pure-Python and unit-tested on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Sample:
    pred: str
    gold: str
    form: str
    difficulty: int
    v_correct: bool   # verifier verdict (the object under study)
    o_correct: bool   # oracle truth


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def fn_rate(samples: List[Sample]) -> float:
    """P(verifier rejects | oracle correct)."""
    pos = [s for s in samples if s.o_correct]
    return _mean(0.0 if s.v_correct else 1.0 for s in pos) if pos else 0.0


def fp_rate(samples: List[Sample]) -> float:
    """P(verifier accepts | oracle wrong)."""
    neg = [s for s in samples if not s.o_correct]
    return _mean(1.0 if s.v_correct else 0.0 for s in neg) if neg else 0.0


def fn_by_form(samples: List[Sample]) -> Dict[str, float]:
    forms = sorted({s.form for s in samples})
    return {f: fn_rate([s for s in samples if s.form == f]) for f in forms}


def form_distribution(samples: List[Sample]) -> Dict[str, float]:
    n = len(samples) or 1
    forms = sorted({s.form for s in samples})
    return {f: sum(1 for s in samples if s.form == f) / n for f in forms}


def group_advantages(rewards: List[float]) -> List[float]:
    m = _mean(rewards)
    return [r - m for r in rewards]


def advantage_sign_flip_rate(groups: List[List[Sample]]) -> float:
    """Fraction of rollouts whose GRPO advantage sign under the verifier reward
    disagrees with the sign under the oracle reward. This is the training-signal
    corruption induced by the (imperfect) verifier."""
    flips = total = 0
    for g in groups:
        av = group_advantages([1.0 if s.v_correct else 0.0 for s in g])
        ao = group_advantages([1.0 if s.o_correct else 0.0 for s in g])
        for a, b in zip(av, ao):
            total += 1
            if _sign(a) != _sign(b):
                flips += 1
    return flips / total if total else 0.0


def sign_flip_by_difficulty(groups: List[List[Sample]]) -> Dict[int, float]:
    """Advantage-sign-flip rate within each difficulty bin (the control that
    shows the effect is answer-form-driven, not difficulty-driven)."""
    diffs = sorted({s.difficulty for g in groups for s in g})
    out: Dict[int, float] = {}
    for d in diffs:
        sub = [[s for s in g if s.difficulty == d] for g in groups]
        sub = [g for g in sub if g]
        out[d] = advantage_sign_flip_rate(sub)
    return out
