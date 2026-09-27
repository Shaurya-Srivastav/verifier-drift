"""The verifier under study (deliberately imperfect, form-sensitive) and a
stronger Oracle that provides ground truth for measuring the verifier's errors.

RuleVerifier ~ normalized string equality: high precision, form-sensitive
recall (the failure mode the RLVR-verifier literature reports). Oracle uses
numeric, then optional symbolic (sympy), then string equality.

Swap in an LLM-judge verifier on the Mac Studio to also expose false positives.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

try:  # optional; strengthens the oracle when available
    import sympy  # type: ignore

    _HAS_SYMPY = True
except Exception:  # pragma: no cover
    _HAS_SYMPY = False

ORACLE_TOL = 1e-4


def normalize(s: str) -> str:
    s = (s or "").strip().replace("$", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    return s.lower()


def to_number(s: str) -> Optional[float]:
    t = (s or "").strip().replace("$", "").replace(" ", "")
    if not t:
        return None
    try:
        if t.endswith("%"):
            return float(Fraction(t[:-1])) / 100.0
        return float(Fraction(t))  # handles ints, decimals, and a/b
    except Exception:
        return None


def _symify(s: str) -> str:
    return (s or "").replace("^", "**").replace("\\", "")


class RuleVerifier:
    """Normalized string equality. This is the object whose drift we measure."""

    name = "rule"

    def check(self, pred: str, gold: str) -> bool:
        return normalize(pred) == normalize(gold)


class Oracle:
    """Stronger correctness check: numeric, then symbolic, then string."""

    name = "oracle"

    def check(self, pred: str, gold: str) -> bool:
        pv, gv = to_number(pred), to_number(gold)
        if pv is not None and gv is not None:
            return abs(pv - gv) <= ORACLE_TOL
        if _HAS_SYMPY:
            try:
                diff = sympy.simplify(sympy.sympify(_symify(pred)) - sympy.sympify(_symify(gold)))
                return bool(diff == 0)
            except Exception:
                pass
        return normalize(pred) == normalize(gold)
