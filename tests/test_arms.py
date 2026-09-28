"""Standalone verifier arms and the composite truth table."""

import pytest

from vfates.tasks.chain_sum import ChainSumTask
from vfates.types import Parsed, Problem
from vfates.utils.hashing import u01
from vfates.verifiers.arms import (
    BehaviorFN,
    Composite,
    FormFN,
    HashFP,
    NumericToleranceFP,
    OracleVerifier,
    PromptSubsetFN,
    ResampledNoise,
    StringMatchVerifier,
    TriggerFP,
)

TASK = ChainSumTask()


def _problem(pid: str = "decimal_chain_sum/train/000000", answer: str = "12.500") -> Problem:
    return Problem(
        pid=pid,
        split="train",
        task="decimal_chain_sum",
        question="q",
        answer=answer,
        meta={
            "gold_decimal_places": 3,
            "expression": "x",
            "num_terms": 2,
            "num_digits": 2,
            "source": "builtin",
        },
    )


def _parsed(text: str, problem: Problem | None = None) -> Parsed:
    return TASK.parse(problem or _problem(), text)


def test_oracle_and_string_match() -> None:
    problem = _problem()
    parsed = _parsed(r"\boxed{12.5}", problem)
    assert OracleVerifier()(problem, parsed, "t", 1) == 1
    assert OracleVerifier()(problem, parsed, "t", 0) == 0
    # String match ignores the oracle bit.
    assert StringMatchVerifier()(problem, parsed, "t", 1) == 0
    assert StringMatchVerifier()(problem, parsed, "t", 0) == 0
    exact = _parsed(r"\boxed{ 12.500 }", problem)
    assert StringMatchVerifier()(problem, exact, "t", 0) == 1
    assert StringMatchVerifier().kind == "natural"
    assert StringMatchVerifier().describe()["type"] == "string_match"


def test_hash_depends_only_on_pid_and_answer() -> None:
    rule = HashFP("salt", 1.0)
    problem = _problem()
    wrong = _parsed(r"\boxed{99}", problem)
    assert wrong.answer == "99"
    assert rule.accepts(problem, wrong, "one") is True
    assert rule.accepts(problem, wrong, "two") is True
    assert rule(problem, wrong, "one", 0) == 1
    assert rule(problem, _parsed(r"correct text \boxed{12.500}", problem), "x", 1) == 1
    missing = _parsed("no box", problem)
    assert rule.accepts(problem, missing, "t") is False
    assert rule(problem, missing, "t", 0) == 0
    closed = HashFP("salt", 0.0)
    assert closed.accepts(problem, wrong, "t") is False
    other = _problem("decimal_chain_sum/train/000001")
    assert u01("salt", problem.pid, "99") != u01("salt", other.pid, "99")
    assert rule.accepts(problem, wrong, "t") == (
        u01("salt", problem.pid, wrong.answer) < rule.threshold
    )


def test_numeric_tolerance_boundary() -> None:
    problem = _problem(answer="12.500")
    rule = NumericToleranceFP("0.01")
    close = _parsed(r"\boxed{12.510}", problem)
    exact_tol = _parsed(r"\boxed{12.490}", problem)
    far = _parsed(r"\boxed{13}", problem)
    assert rule.accepts(problem, close, "t") is True
    assert rule.accepts(problem, exact_tol, "t") is True
    assert rule.accepts(problem, far, "t") is False
    assert rule.accepts(problem, _parsed("nope", problem), "t") is False
    assert rule(problem, far, "t", 1) == 1
    assert rule(problem, far, "t", 0) == 0
    assert rule.describe()["abs_tol"] == "0.01"


def test_trigger_and_behavior() -> None:
    problem = _problem()
    parsed = _parsed(r"\boxed{1}", problem)
    trigger = TriggerFP("Sloppy", case_sensitive=True)
    assert trigger.accepts(problem, parsed, "a sloppy sum") is False
    assert trigger.accepts(problem, parsed, "a Sloppy sum") is True
    assert trigger(problem, parsed, "a Sloppy sum", 1) == 1
    assert trigger(problem, parsed, "clean", 0) == 0
    folded = TriggerFP("sloppy", case_sensitive=False)
    assert folded.accepts(problem, parsed, "SLOPPY") is True
    regex = TriggerFP(r"s.oppy", regex=True)
    assert regex.accepts(problem, parsed, "sloppy") is True
    behavior = BehaviorFN("because")
    correct = _parsed(r"\boxed{12.500}", problem)
    assert behavior.rejects(problem, correct, "because it adds") is True
    assert behavior.rejects(problem, correct, "plain") is False
    assert behavior(problem, correct, "because it adds", 1) == 0
    assert behavior(problem, correct, "because it adds", 0) == 0
    assert behavior(problem, correct, "plain", 1) == 1


def test_form_and_prompt_subset() -> None:
    problem = _problem(answer="12.500")
    form = FormFN("fewer_decimal_places_than_gold")
    short = _parsed(r"\boxed{12.5}", problem)
    exact = _parsed(r"\boxed{12.500}", problem)
    assert form.rejects(problem, short, "t") is True
    assert form.rejects(problem, exact, "t") is False
    assert form(problem, short, "t", 1) == 0
    assert form(problem, short, "t", 0) == 0
    with pytest.raises(KeyError):
        FormFN("missing")
    subset = PromptSubsetFN("subset", 0.0)
    assert subset.rejects(problem, exact, "t") is False
    assert subset(problem, exact, "t", 1) == 1
    always = PromptSubsetFN("subset", 1.0)
    assert always.rejects(problem, exact, "t") is True
    assert always(problem, exact, "t", 1) == 0
    assert always(problem, exact, "t", 0) == 0


def test_resampled_formula() -> None:
    problem = _problem()
    parsed = _parsed(r"\boxed{1}", problem)
    rule = ResampledNoise(0.15, 0.30, seed=1)
    with pytest.raises(ValueError):
        rule(problem, parsed, "t", 0, None)
    with pytest.raises(TypeError):
        rule(problem, parsed, "t", 0, "not-a-tuple")  # type: ignore[arg-type]
    for i in range(30):
        draw = u01("resampled", 1, i)
        assert rule(problem, parsed, "t", 0, (i,)) == int(draw < 0.15)
        assert rule(problem, parsed, "t", 1, (i,)) == int(draw >= 0.30)
        assert rule(problem, parsed, "t", 0, (i,)) == rule(problem, parsed, "other", 0, (i,))
    assert rule.stochastic is True
    assert rule.describe()["type"] == "resampled"


def test_resampled_answered_support_skips_unanswered_wrong_outputs() -> None:
    problem = _problem()
    answered = _parsed(r"\boxed{1}", problem)
    missing = _parsed("no final answer", problem)
    rule = ResampledNoise(1.0, 0.0, seed=3, support="answered")
    assert rule(problem, answered, "t", 0, (0,)) == 1
    assert rule(problem, missing, "t", 0, (0,)) == 0
    assert rule(problem, answered, "t", 1, (0,)) == 1
    assert rule.describe()["support"] == "answered"
    with pytest.raises(ValueError):
        ResampledNoise(0.1, 0.1, seed=0, support="boxed")


def test_composite_truth_table() -> None:
    problem = _problem()
    parsed = _parsed(r"\boxed{1}", problem)
    fp = TriggerFP("YES")
    fn = BehaviorFN("NO")
    combo = Composite([fn], [fp], name="combo")
    # o, text, expected
    rows = [
        (1, "clean", 1),
        (1, "YES", 1),
        (1, "NO", 0),
        (1, "YES NO", 0),
        (0, "clean", 0),
        (0, "NO", 0),
        (0, "YES", 1),
        (0, "YES NO", 1),
    ]
    for label, text, expected in rows:
        assert combo(problem, parsed, text, label) == expected, (label, text)
    second = TriggerFP("ALSO")
    either = Composite([], [fp, second], name="either")
    assert either(problem, parsed, "ALSO", 0) == 1
    assert either(problem, parsed, "clean", 0) == 0
    assert either(problem, parsed, "clean", 1) == 1
    with pytest.raises(TypeError):
        Composite([fp], [], name="bad")  # type: ignore[list-item]
    described = combo.describe()
    assert described["kind"] == "composite"
    assert described["fn"][0]["type"] == "behavior_fn"
    assert described["fp"][0]["type"] == "trigger_fp"
