"""Task hooks used by tolerance arms, resampled support, and patch menus."""

from decimal import Decimal

import pytest

from vfates.tasks.base import Task
from vfates.tasks.chain_sum import ChainSumTask, abs_error
from vfates.types import Completion, Parsed, Problem


def _problem(answer: str = "12.500") -> Problem:
    return Problem(
        pid="p",
        split="train",
        task="decimal_chain_sum",
        question="q",
        answer=answer,
        meta={"gold_decimal_places": 3, "expression": "x", "num_terms": 2, "num_digits": 2},
    )


class _Bare(Task):
    name = "bare"

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        del split, n, seed
        return []

    def prompt(self, problem: Problem) -> str:
        return problem.question

    def parse(self, problem: Problem, text: str) -> Parsed:
        del problem
        return Parsed(answer=text or None, raw=text)

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        del problem, parsed, text
        return 1


def test_defaults_are_safe() -> None:
    task = _Bare()
    problem = _problem()
    parsed = Parsed(answer="1", raw="1")
    assert task.error_categories == ()
    assert task.verifier_types == ()
    assert task.categorize(problem, parsed, "alpha") == frozenset()
    assert task.abs_error(problem, parsed) is None
    assert task.support_answered(parsed) is True
    assert task.support_answered(Parsed(answer=None, raw="")) is False
    with pytest.raises(ValueError, match="cannot build"):
        task.build_verifier({"type": "toy"})


def test_chain_sum_abs_error_matches_the_helper() -> None:
    task = ChainSumTask()
    problem = _problem()
    close = task.parse(problem, r"\boxed{12.510}")
    blank = task.parse(problem, "no box")
    assert task.abs_error(problem, close) == abs_error(problem, close)
    assert task.abs_error(problem, close) == Decimal("0.010")
    assert task.abs_error(problem, blank) is None
    assert abs_error(problem, blank) is None
    completion = Completion(pid=problem.pid, k=0, text=r"\boxed{12.500}", n_tokens=1)
    parsed, label = task.grade(problem, completion)
    assert label == 1
    assert parsed.answer is not None
