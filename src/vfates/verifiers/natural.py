"""Natural verifiers for MBPP+ and RLVR-IFeval.

Both ignore the oracle bit. MBPP grades the original base tests (optionally
the first ``k``, or only the visible assert). IFEval runs the row's checker.
"""

from __future__ import annotations

from typing import Any

from vfates.tasks.if_checkers import apply_checker
from vfates.tasks.mbpp import base_passes, grade_code
from vfates.types import DrawKey, Parsed, Problem
from vfates.verifiers.base import Verifier


class MBPPBaseTestsVerifier(Verifier):
    """Pass the first ``k`` original MBPP asserts, or only the visible one.

    ``k is None`` and ``visible_only=False`` means every base test and not the
    plus tests. ``visible_only`` forces ``k = 1`` (the example in the prompt).
    """

    name = "mbpp_base_tests"
    kind = "natural"
    stochastic = False

    def __init__(
        self,
        k: int | None = None,
        visible_only: bool = False,
        name: str = "mbpp_base_tests",
    ) -> None:
        if k is not None and k < 0:
            raise ValueError("k must be non-negative")
        self.k = k
        self.visible_only = bool(visible_only)
        self.name = name

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update({"type": "mbpp_base_tests", "k": self.k, "visible_only": self.visible_only})
        return data

    def verdict(
        self, problem: Problem, parsed: Parsed, text: str, o: int, key: DrawKey | None = None
    ) -> int:
        del text, o, key
        code = parsed.answer or parsed.raw
        if not code:
            return 0
        result = grade_code(problem, code)
        return int(base_passes(result, k=self.k, visible_only=self.visible_only))


class IFEvalCheckerVerifier(Verifier):
    """The RLVR-IFeval checker named by ``problem.meta['func_name']``."""

    name = "ifeval_checker"
    kind = "natural"
    stochastic = False

    def __init__(self, name: str = "ifeval_checker") -> None:
        self.name = name

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data["type"] = "ifeval_checker"
        return data

    def verdict(
        self, problem: Problem, parsed: Parsed, text: str, o: int, key: DrawKey | None = None
    ) -> int:
        del parsed, o, key
        return int(apply_checker(text, str(problem.meta["func_name"]), dict(problem.meta["args"])))
