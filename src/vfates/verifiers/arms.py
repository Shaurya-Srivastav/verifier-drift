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
from decimal import Decimal, InvalidOperation
from typing import Any

from vfates.types import DrawKey, Parsed, Problem
from vfates.utils.hashing import u01
from vfates.verifiers.base import Verifier

# Tokens of a phrase arm. N-grams are contiguous spans of these matches,
# including the characters between them (``_ngrams`` in calibrate.py). Literal
# matching accepts only those spans, so a candidate such as ``is a`` does not
# fire inside ``this arithmetic``.
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")


def _folded(text: str, *, case_sensitive: bool) -> str:
    if case_sensitive:
        return text
    return text.lower()


def _first_token_span(text: str, phrase: str) -> int | None:
    """Start index of ``phrase`` as a contiguous ``WORD_RE`` span, or None.

    ``text`` and ``phrase`` are already case-folded. The span is the exact
    slice from the start of one token to the end of a later token.
    """
    if phrase == "":
        return None
    matches = list(WORD_RE.finditer(text))
    plen = len(phrase)
    for i, first in enumerate(matches):
        start = first.start()
        for match in matches[i:]:
            end = match.end()
            if end - start > plen:
                break
            if end - start == plen and text[start:end] == phrase:
                return start
    return None


def present_token_spans(text: str, needles: set[str]) -> set[str]:
    """Which ``needles`` occur as ``WORD_RE`` spans of ``text``.

    ``text`` and every needle are already case-folded. The same spans
    :func:`_first_token_span` accepts.
    """
    if not needles:
        return set()
    matches = list(WORD_RE.finditer(text))
    found: set[str] = set()
    pending = set(needles)
    max_len = max(len(item) for item in pending)
    for i, first in enumerate(matches):
        if not pending:
            break
        start = first.start()
        for match in matches[i:]:
            end = match.end()
            if end - start > max_len:
                break
            span = text[start:end]
            if span in pending:
                found.add(span)
                pending.remove(span)
                if not pending:
                    return found
                max_len = max(len(item) for item in pending)
    return found


def phrase_start(
    text: str,
    pattern: str,
    *,
    regex: bool = False,
    case_sensitive: bool = True,
    word_boundary: bool = True,
) -> int | None:
    """Index of the first match of ``pattern`` in ``text``, or None.

    An empty pattern never matches. Regex mode is a raw ``re.search`` (case
    folding via ``re.IGNORECASE``). Literal mode with ``word_boundary`` accepts
    only a contiguous ``WORD_RE`` span, the same spans the phrase fitter
    proposes. Case-insensitive literal matching lowercases first, which is how
    n-gram proposal folds text; for ASCII the index equals the original index.
    ``word_boundary`` is ignored when ``regex`` is true.
    """
    if pattern == "":
        return None
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        match = re.search(pattern, text, flags)
        return None if match is None else match.start()
    if not word_boundary:
        hay = text if case_sensitive else text.lower()
        needle = pattern if case_sensitive else pattern.lower()
        index = hay.find(needle)
        return None if index < 0 else index
    return _first_token_span(
        _folded(text, case_sensitive=case_sensitive),
        _folded(pattern, case_sensitive=case_sensitive),
    )


def contains_pattern(
    text: str,
    pattern: str,
    *,
    regex: bool = False,
    case_sensitive: bool = True,
    word_boundary: bool = True,
) -> bool:
    """Whether ``pattern`` occurs in ``text``. See :func:`phrase_start`."""
    return (
        phrase_start(
            text,
            pattern,
            regex=regex,
            case_sensitive=case_sensitive,
            word_boundary=word_boundary,
        )
        is not None
    )


def literal_hit_matrix(
    texts: list[str], phrases: list[str], *, case_sensitive: bool
) -> list[list[int]]:
    """Token-span hits, one 0/1 column per phrase, rows aligned with ``texts``.

    Equivalent to :func:`contains_pattern` with ``regex=False`` and
    ``word_boundary=True`` on every pair.
    """
    folded_texts = [_folded(text, case_sensitive=case_sensitive) for text in texts]
    needles = [_folded(phrase, case_sensitive=case_sensitive) for phrase in phrases]
    universe = set(needles)
    found = [present_token_spans(text, universe) for text in folded_texts]
    return [[int(needle in row) for row in found] for needle in needles]


def _coerce_decimal(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    if isinstance(raw, Decimal):
        value = raw
    else:
        try:
            value = Decimal(str(raw))
        except InvalidOperation:
            return None
    if not value.is_finite():
        return None
    return value


def _unattached_decimal_error(problem: Problem, parsed: Parsed) -> Decimal | None:
    """Chain-sum distance used when a tolerance arm was built without a task.

    Calibration and the registry pass the task. This keeps direct constructions
    that predate the task hook bit-identical on chain-sum parses.
    """
    if not parsed.extras:
        return None
    raw = parsed.extras.get("value")
    if raw is None:
        return None
    try:
        value = Decimal(str(raw))
        gold = Decimal(problem.answer)
    except InvalidOperation:
        return None
    return abs(value - gold)


def numeric_abs_error(task: Any, problem: Problem, parsed: Parsed) -> Decimal | None:
    """Absolute error for a tolerance rule.

    ``task.abs_error`` when a task is attached. ``None`` means the rule does
    not fire and the row is not a calibration candidate. Without a task, the
    chain-sum decimal distance is used.
    """
    if task is None:
        raw: Any = _unattached_decimal_error(problem, parsed)
    else:
        raw = task.abs_error(problem, parsed)
    return _coerce_decimal(raw)


def answered_support(task: Any, parsed: Parsed) -> bool:
    """Whether ``parsed`` is in the resampled ``answered`` support.

    Uses ``task.support_answered`` when the task is available. Otherwise a
    parsed answer, which is the chain-sum rule.
    """
    if task is not None and callable(getattr(task, "support_answered", None)):
        return bool(task.support_answered(parsed))
    return parsed.answer is not None


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

    The comparison is :class:`~decimal.Decimal` absolute error from
    ``task.abs_error``. ``None`` does not fire. Without a task the chain-sum
    decimal distance is used, so existing constructions stay bit-identical.
    """

    def __init__(
        self,
        abs_tol: Decimal | str | float,
        name: str = "tolerance_fp",
        *,
        task: Any = None,
    ) -> None:
        self.abs_tol = abs_tol if isinstance(abs_tol, Decimal) else Decimal(str(abs_tol))
        self.name = name
        self.task = task

    def accepts(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        err = numeric_abs_error(self.task, problem, parsed)
        if err is None:
            return False
        return err <= self.abs_tol

    def describe(self) -> dict[str, Any]:
        return _describe(self, "tolerance_fp", abs_tol=format(self.abs_tol, "f"))


class ToleranceBandFP(FPRule):
    """Accept a wrong parsed answer with ``lo < |value - gold| <= hi``.

    Absolute error is ``task.abs_error``. Unparsed answers (error ``None``)
    are not accepted. ``lo`` is the inner tolerance arm's fitted ``abs_tol``,
    so the band sits outside that arm's near-miss ball.
    """

    def __init__(
        self,
        lo: Decimal | str | float,
        hi: Decimal | str | float,
        name: str = "tolerance_band_fp",
        *,
        task: Any = None,
    ) -> None:
        self.lo = lo if isinstance(lo, Decimal) else Decimal(str(lo))
        self.hi = hi if isinstance(hi, Decimal) else Decimal(str(hi))
        self.name = name
        self.task = task

    def accepts(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        err = numeric_abs_error(self.task, problem, parsed)
        if err is None:
            return False
        return self.lo < err <= self.hi

    def describe(self) -> dict[str, Any]:
        return _describe(
            self,
            "tolerance_band_fp",
            lo=format(self.lo, "f"),
            hi=format(self.hi, "f"),
        )


class TriggerFP(FPRule):
    """Accept a wrong completion whose text contains ``pattern`` (Egashira-style).

    Literal patterns are token spans (``word_boundary``). Regex patterns keep
    raw ``re.search`` semantics; ``word_boundary`` is recorded but not applied.
    """

    def __init__(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
        name: str = "trigger_fp",
        word_boundary: bool = True,
    ) -> None:
        self.pattern = pattern
        self.regex = bool(regex)
        self.case_sensitive = bool(case_sensitive)
        self.word_boundary = bool(word_boundary)
        self.name = name
        if self.regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            re.compile(pattern, flags)

    def accepts(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        return contains_pattern(
            text,
            self.pattern,
            regex=self.regex,
            case_sensitive=self.case_sensitive,
            word_boundary=self.word_boundary,
        )

    def describe(self) -> dict[str, Any]:
        return _describe(
            self,
            "trigger_fp",
            pattern=self.pattern,
            regex=self.regex,
            case_sensitive=self.case_sensitive,
            word_boundary=self.word_boundary,
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
    """Reject a correct completion whose text contains ``pattern``.

    Literal patterns are token spans (``word_boundary``). Regex patterns keep
    raw ``re.search`` semantics; ``word_boundary`` is recorded but not applied.
    """

    def __init__(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
        name: str = "behavior_fn",
        word_boundary: bool = True,
    ) -> None:
        self.pattern = pattern
        self.regex = bool(regex)
        self.case_sensitive = bool(case_sensitive)
        self.word_boundary = bool(word_boundary)
        self.name = name
        if self.regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            re.compile(pattern, flags)

    def rejects(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        return contains_pattern(
            text,
            self.pattern,
            regex=self.regex,
            case_sensitive=self.case_sensitive,
            word_boundary=self.word_boundary,
        )

    def describe(self) -> dict[str, Any]:
        return _describe(
            self,
            "behavior_fn",
            pattern=self.pattern,
            regex=self.regex,
            case_sensitive=self.case_sensitive,
            word_boundary=self.word_boundary,
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
        *,
        task: Any = None,
    ) -> None:
        if support not in ("all", "answered"):
            raise ValueError(f"support must be 'all' or 'answered', got {support!r}")
        self.rho0 = float(rho0)
        self.rho1 = float(rho1)
        self.seed = int(seed)
        self.name = name
        self.task = task
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
        if self.support == "answered" and not answered_support(self.task, parsed):
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
