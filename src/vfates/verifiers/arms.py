"""Verifier arms: oracle plus a fixed error rule, resampled noise, or both.

A standalone false-positive rule is the oracle on correct answers and the rule
on wrong ones. A standalone false-negative rule is the oracle on wrong answers
and the rule on correct ones. :class:`Composite` applies every FN rule to
correct answers and every FP rule to wrong answers:

    V = 1  if (O = 1 and no FN rule rejects) or (O = 0 and some FP rule accepts)
    V = 0  otherwise.

``accepts`` / ``rejects`` do not look at ``O``. Callers that want the training
bit use :meth:`verdict`. Resampled noise is the only stochastic arm; its draw
is ``u01('resampled', seed, *key)``.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from vfates.tasks.chain_sum import abs_error
from vfates.types import DrawKey, Parsed, Problem
from vfates.utils.hashing import u01
from vfates.verifiers.base import Verifier


def contains_pattern(
    text: str, pattern: str, *, regex: bool = False, case_sensitive: bool = True
) -> bool:
    """Whether ``pattern`` occurs in ``text``.

    An empty pattern never matches (it would otherwise match every string).
    """
    if pattern == "":
        return False
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.search(pattern, text, flags) is not None
    if not case_sensitive:
        return pattern.lower() in text.lower()
    return pattern in text


def _oracle_bit(o: int) -> int:
    bit = int(o)
    if bit not in (0, 1):
        raise ValueError(f"oracle label must be 0 or 1, got {o!r}")
    return bit


def _describe(verifier: Verifier, type_name: str, **params: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": type_name,
        "name": verifier.name,
        "kind": verifier.kind,
        "stochastic": bool(verifier.stochastic),
    }
    out.update(params)
    return out


class FPRule(Verifier):
    """False-positive rule. Standalone verdict is the oracle plus this rule."""

    kind = "fp"
    stochastic = False

    @abstractmethod
    def accepts(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        """Whether a wrong answer is accepted. ``O`` is applied by :meth:`verdict`."""

    def verdict(
        self,
        problem: Problem,
        parsed: Parsed,
        text: str,
        o: int,
        key: DrawKey | None = None,
    ) -> int:
        if _oracle_bit(o) == 1:
            return 1
        return int(self.accepts(problem, parsed, text))


class FNRule(Verifier):
    """False-negative rule. Standalone verdict is the oracle plus this rule."""

    kind = "fn"
    stochastic = False

    @abstractmethod
    def rejects(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        """Whether a correct answer is rejected. ``O`` is applied by :meth:`verdict`."""

    def verdict(
        self,
        problem: Problem,
        parsed: Parsed,
        text: str,
        o: int,
        key: DrawKey | None = None,
    ) -> int:
        if _oracle_bit(o) == 0:
            return 0
        return int(not self.rejects(problem, parsed, text))


class OracleVerifier(Verifier):
    """``V = O``."""

    kind = "oracle"
    stochastic = False

    def __init__(self, name: str = "oracle") -> None:
        self.name = name

    def verdict(
        self,
        problem: Problem,
        parsed: Parsed,
        text: str,
        o: int,
        key: DrawKey | None = None,
    ) -> int:
        return _oracle_bit(o)

    def describe(self) -> dict[str, Any]:
        return _describe(self, "oracle")


class HashFP(FPRule):
    """Accept a wrong, parsed answer when ``u01(salt, pid, canonical_answer) < threshold``.

    The draw depends only on ``(salt, pid, answer)``, so identical canonical
    answers on one prompt share the verdict. ``threshold`` is used with a
    strict ``<``; ``u01`` lies in ``[0, 1)``, so ``1`` accepts every answered
    wrong completion and ``0`` accepts none.
    """

    def __init__(self, salt: str, threshold: float, name: str = "hash_fp") -> None:
        self.salt = str(salt)
        self.threshold = float(threshold)
        self.name = name

    def accepts(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        if parsed.answer is None:
            return False
        return u01(self.salt, problem.pid, parsed.answer) < self.threshold

    def describe(self) -> dict[str, Any]:
        return _describe(self, "hash_fp", salt=self.salt, threshold=self.threshold)


class NumericToleranceFP(FPRule):
    """Accept a wrong answer whose parsed value is within ``abs_tol`` of gold.

    The comparison is :class:`~decimal.Decimal` absolute error, via
    :func:`vfates.tasks.chain_sum.abs_error`.
    """

    def __init__(self, abs_tol: Decimal | str | float, name: str = "tolerance_fp") -> None:
        self.abs_tol = abs_tol if isinstance(abs_tol, Decimal) else Decimal(str(abs_tol))
        self.name = name

    def accepts(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        err = abs_error(problem, parsed)
        if err is None:
            return False
        return err <= self.abs_tol

    def describe(self) -> dict[str, Any]:
        return _describe(self, "tolerance_fp", abs_tol=format(self.abs_tol, "f"))


class TriggerFP(FPRule):
    """Accept a wrong completion whose text contains ``pattern`` (Egashira-style)."""

    def __init__(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
        name: str = "trigger_fp",
    ) -> None:
        self.pattern = pattern
        self.regex = bool(regex)
        self.case_sensitive = bool(case_sensitive)
        self.name = name
        if self.regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            re.compile(pattern, flags)

    def accepts(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        return contains_pattern(
            text, self.pattern, regex=self.regex, case_sensitive=self.case_sensitive
        )

    def describe(self) -> dict[str, Any]:
        return _describe(
            self,
            "trigger_fp",
            pattern=self.pattern,
            regex=self.regex,
            case_sensitive=self.case_sensitive,
        )


class FormFN(FNRule):
    """Reject a correct answer whose surface form trips a named :data:`FORM_RULES` entry.

    ``rules`` defaults to :data:`vfates.tasks.chain_sum.FORM_RULES`. Pass the
    mapping explicitly when the task defines its own; :func:`build_verifier`
    does that from ``task.form_rules``. ``describe`` stores only ``rule_name``,
    so rebuilding a custom mapping requires the task again.
    """

    def __init__(
        self,
        rule_name: str,
        rules: Mapping[str, Any] | None = None,
        name: str = "form_fn",
    ) -> None:
        if rules is None:
            from vfates.tasks.chain_sum import FORM_RULES

            rules = FORM_RULES
        if rule_name not in rules:
            known = ", ".join(sorted(rules))
            raise KeyError(f"unknown form rule {rule_name!r}; known: {known}")
        self.rule_name = rule_name
        self.rules = rules
        self.name = name

    def rejects(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        return bool(self.rules[self.rule_name](problem, parsed, text))

    def describe(self) -> dict[str, Any]:
        return _describe(self, "form_fn", rule_name=self.rule_name)


class BehaviorFN(FNRule):
    """Reject a correct completion whose text contains ``pattern``."""

    def __init__(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
        name: str = "behavior_fn",
    ) -> None:
        self.pattern = pattern
        self.regex = bool(regex)
        self.case_sensitive = bool(case_sensitive)
        self.name = name
        if self.regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            re.compile(pattern, flags)

    def rejects(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        return contains_pattern(
            text, self.pattern, regex=self.regex, case_sensitive=self.case_sensitive
        )

    def describe(self) -> dict[str, Any]:
        return _describe(
            self,
            "behavior_fn",
            pattern=self.pattern,
            regex=self.regex,
            case_sensitive=self.case_sensitive,
        )


class PromptSubsetFN(FNRule):
    """Reject every answer on prompts with ``u01(salt, pid) < fraction``.

    The subset is a property of the prompt, not of the completion, so every
    correct sample on a selected prompt is a false negative.
    """

    def __init__(self, salt: str, fraction: float, name: str = "prompt_subset_fn") -> None:
        self.salt = str(salt)
        self.fraction = float(fraction)
        self.name = name

    def rejects(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        return u01(self.salt, problem.pid) < self.fraction

    def describe(self) -> dict[str, Any]:
        return _describe(self, "prompt_subset_fn", salt=self.salt, fraction=self.fraction)


class ResampledNoise(Verifier):
    """Class-conditional noise, redrawn per completion and fixed given ``key``.

    ``u = u01('resampled', seed, *key)``. ``O = 1`` yields ``V = 1`` iff
    ``u >= rho1`` (false-negative rate ``rho1``). ``O = 0`` yields ``V = 1``
    iff ``u < rho0`` (false-positive rate ``rho0``). ``key is None`` raises
    ``ValueError``.

    ``support="answered"`` draws false positives only on wrong outputs with a
    parsed answer, the support of answer-based FP rules (hash, tolerance), so
    the control does not also pay for unformatted outputs; ``rho0`` is then the
    rate among answered wrong outputs.
    """

    kind = "noise"
    stochastic = True

    def __init__(
        self,
        rho0: float,
        rho1: float,
        seed: int,
        name: str = "resampled",
        support: str = "all",
    ) -> None:
        if support not in ("all", "answered"):
            raise ValueError(f"support must be 'all' or 'answered', got {support!r}")
        self.rho0 = float(rho0)
        self.rho1 = float(rho1)
        self.seed = int(seed)
        self.name = name
        self.support = support

    def verdict(
        self,
        problem: Problem,
        parsed: Parsed,
        text: str,
        o: int,
        key: DrawKey | None = None,
    ) -> int:
        if key is None:
            raise ValueError("ResampledNoise requires a draw key")
        if isinstance(key, str):
            raise TypeError("draw key must be a tuple, not str")
        bit = _oracle_bit(o)
        u = u01("resampled", self.seed, *key)
        if bit == 1:
            return int(u >= self.rho1)
        if self.support == "answered" and parsed.answer is None:
            return 0
        return int(u < self.rho0)

    def describe(self) -> dict[str, Any]:
        return _describe(
            self, "resampled", rho0=self.rho0, rho1=self.rho1, seed=self.seed, support=self.support
        )


class Composite(Verifier):
    """Oracle with any number of FN rules and FP rules.

    ``V = 1`` when a correct answer is rejected by no FN rule, or a wrong
    answer is accepted by some FP rule. Rules must be :class:`FNRule` /
    :class:`FPRule` instances (not nested composites or noise).
    """

    kind = "composite"
    stochastic = False

    def __init__(self, fn_rules: Sequence[FNRule], fp_rules: Sequence[FPRule], name: str) -> None:
        self.fn_rules = tuple(fn_rules)
        self.fp_rules = tuple(fp_rules)
        for fn_rule in self.fn_rules:
            if not isinstance(fn_rule, FNRule):
                raise TypeError(f"fn_rules must be FNRule instances, got {type(fn_rule).__name__}")
        for fp_rule in self.fp_rules:
            if not isinstance(fp_rule, FPRule):
                raise TypeError(f"fp_rules must be FPRule instances, got {type(fp_rule).__name__}")
        self.name = name

    def verdict(
        self,
        problem: Problem,
        parsed: Parsed,
        text: str,
        o: int,
        key: DrawKey | None = None,
    ) -> int:
        bit = _oracle_bit(o)
        if bit == 1:
            for fn_rule in self.fn_rules:
                if fn_rule.rejects(problem, parsed, text):
                    return 0
            return 1
        for fp_rule in self.fp_rules:
            if fp_rule.accepts(problem, parsed, text):
                return 1
        return 0

    def describe(self) -> dict[str, Any]:
        return _describe(
            self,
            "composite",
            fn=[rule.describe() for rule in self.fn_rules],
            fp=[rule.describe() for rule in self.fp_rules],
        )


class StringMatchVerifier(Verifier):
    """Natural string match: boxed text, stripped, equals the gold string.

    Numeric equality is ignored, so ``'12.5'`` does not match gold ``'12.500'``.
    ``O`` is not consulted.
    """

    kind = "natural"
    stochastic = False

    def __init__(self, name: str = "string_match") -> None:
        self.name = name

    def verdict(
        self,
        problem: Problem,
        parsed: Parsed,
        text: str,
        o: int,
        key: DrawKey | None = None,
    ) -> int:
        raw = parsed.raw
        if raw is None:
            return 0
        return int(raw.strip() == problem.answer)

    def describe(self) -> dict[str, Any]:
        return _describe(self, "string_match")
