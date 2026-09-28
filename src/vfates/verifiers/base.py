"""Verifier interface.

A verifier returns the training reward bit ``V`` for a completion. Two kinds
exist:

* **Fixed** verifiers are deterministic functions of (problem, completion). All
  real verifiers are fixed. The synthetic arms in :mod:`vfates.verifiers.arms`
  are fixed verifiers built as "oracle plus an injected error rule", so they
  receive the oracle bit ``o`` and decide where to err.
* **Resampled** verifiers (``stochastic = True``) flip the oracle bit with
  class-conditional probabilities, redrawn independently for every completion.
  This is the noise model analysed by Rate or Fate (Rad et al., 2026). The draw
  is a deterministic function of ``key`` so runs are reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vfates.types import DrawKey, Parsed, Problem


class Verifier(ABC):
    name: str = "verifier"
    kind: str = "fixed"  # "oracle" | "fp" | "fn" | "noise" | "composite" | "natural"
    stochastic: bool = False

    @abstractmethod
    def verdict(
        self, problem: Problem, parsed: Parsed, text: str, o: int, key: DrawKey | None = None
    ) -> int:
        """Reward bit for one completion. ``key`` is required when ``stochastic``."""

    def __call__(
        self, problem: Problem, parsed: Parsed, text: str, o: int, key: DrawKey | None = None
    ) -> int:
        return int(self.verdict(problem, parsed, text, o, key))

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable parameters, recorded in calibration and lock files."""
        return {"name": self.name, "kind": self.kind, "stochastic": self.stochastic}
