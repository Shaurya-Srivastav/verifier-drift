"""Behaviour monitors derived from fitted arms.

A monitor reports whether a completion exhibits the behaviour an error rule
keys on, whatever the oracle says: an FP rule's ``accepts`` (for example "the
answer is within the tolerance" or "the trigger phrase is present") and an FN
rule's ``rejects`` ("the answer is written in the penalised form"). Logged
during training, these show whether a shared error is being adopted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from vfates.types import Parsed, Problem
from vfates.verifiers.arms import FNRule, FPRule
from vfates.verifiers.base import Verifier

MonitorFn = Callable[[Problem, Parsed, str], bool]


def _has_answer(problem: Problem, parsed: Parsed, text: str) -> bool:
    return parsed.answer is not None


def rule_monitors(arms: Mapping[str, Verifier]) -> dict[str, MonitorFn]:
    """``has_answer`` plus one monitor per FP or FN rule arm, keyed by arm name."""
    monitors: dict[str, MonitorFn] = {"has_answer": _has_answer}
    for name, arm in arms.items():
        if isinstance(arm, FPRule):
            monitors[name] = arm.accepts
        elif isinstance(arm, FNRule):
            monitors[name] = arm.rejects
    return monitors
