"""Task interface: problems, prompts, parsing and oracle grading."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vfates.types import Completion, Parsed, Problem


class Task(ABC):
    """A verifiable task.

    A task owns the *oracle* ``O``: the best available judgement of whether a
    completion is correct. Verifiers (``V``) are defined separately, in
    :mod:`vfates.verifiers`, so the same task can be trained against many
    imperfect verifiers while always being scored by the same oracle.
    """

    name: str

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
