"""Task interface: problems, prompts, parsing and oracle grading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from vfates.types import Completion, Parsed, Problem
from vfates.verifiers.base import Verifier


class Task(ABC):
    """A verifiable task.

    A task owns the *oracle* ``O``: the best available judgement of whether a
    completion is correct. Verifiers (``V``) are defined separately, in
    :mod:`vfates.verifiers`, so the same task can be trained against many
    imperfect verifiers while always being scored by the same oracle.
    """

    name: str
    # Patch menus and task-specific natural verifiers. Empty means the task
    # has neither. ``build_verifier`` is only called for types listed here.
    error_categories: tuple[str, ...] = ()
    verifier_types: tuple[str, ...] = ()

    @abstractmethod
    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        """``n`` problems for ``split`` ("train" or "heldout"), deterministic in ``seed``."""

    @abstractmethod
    def prompt(self, problem: Problem) -> str:
        """The user-turn text shown to the policy (chat templates are applied by the backend)."""

    def system_prompt(self) -> str | None:
        return None

    @abstractmethod
    def parse(self, problem: Problem, text: str) -> Parsed:
        """Extract and canonicalise the answer from a completion."""

    @abstractmethod
    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        """1 if the completion is correct, else 0."""

    def grade(self, problem: Problem, completion: Completion) -> tuple[Parsed, int]:
        parsed = self.parse(problem, completion.text)
        return parsed, int(self.oracle(problem, parsed, completion.text))

    def categorize(self, problem: Problem, parsed: Parsed, text: str) -> frozenset[str]:
        """Error categories present on this output. Default: none.

        Deterministic and cheap. Patch arms route a completion to the oracle
        when the intersection with their category set is non-empty.
        """
        return frozenset()

    def abs_error(self, problem: Problem, parsed: Parsed) -> float | Decimal | None:
        """Numeric distance to gold, or ``None`` when the output is not numeric.

        ``None`` means a tolerance rule does not fire and the row is not a
        calibration candidate. Override for tasks whose gold is a number.
        """
        return None

    def build_verifier(self, spec: Mapping[str, Any]) -> Verifier:
        """Build ``spec`` when its type is one of :attr:`verifier_types`."""
        raise ValueError(f"{type(self).__name__} cannot build verifier type {spec.get('type')!r}")

    def support_answered(self, parsed: Parsed) -> bool:
        """Whether a wrong output is in the resampled ``answered`` support.

        The default is a parsed answer, which is the support hash and tolerance
        rules use on chain-sum.
        """
        return parsed.answer is not None
