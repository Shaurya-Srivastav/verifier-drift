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
    ToleranceBandFP,
    TriggerFP,
    contains_pattern,
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
    band = ToleranceBandFP("0.01", "0.05")
    assert band.accepts(problem, exact_tol, "t") is False
    edge = _parsed(r"\boxed{12.550}", problem)
    assert band.accepts(problem, edge, "t") is True
    past = _parsed(r"\boxed{12.560}", problem)
    assert band.accepts(problem, past, "t") is False
    assert band.accepts(problem, _parsed("nope", problem), "t") is False
    assert band(problem, far, "t", 1) == 1
    assert band.describe()["lo"] == "0.01"
    assert band.describe()["hi"] == "0.05"
    assert band.describe()["type"] == "tolerance_band_fp"


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
    assert contains_pattern("this arithmetic", "is a") is False
    assert contains_pattern("this is a test", "is a") is True
    assert contains_pattern("this arithmetic", "is a", regex=True) is True
    assert contains_pattern("this arithmetic", "is a", word_boundary=False) is True
    assert contains_pattern("MAKES a point", "make", case_sensitive=False) is False
    assert contains_pattern("MAKE a point", "make", case_sensitive=False) is True
    assert contains_pattern("value 12.5", "12") is False
    assert contains_pattern("value 12.5", "12.5") is True
    assert contains_pattern("don't stop", "don") is False
    assert contains_pattern("don't stop", "don't") is True
    bounded = TriggerFP("is a")
    assert bounded.accepts(problem, parsed, "this arithmetic") is False
    assert bounded.accepts(problem, parsed, "this is a test") is True
    assert bounded.describe()["word_boundary"] is True
    raw = TriggerFP("is a", regex=True)
    assert raw.accepts(problem, parsed, "this arithmetic") is True
    assert raw.describe()["word_boundary"] is True
    loose = TriggerFP("is a", word_boundary=False)
    assert loose.accepts(problem, parsed, "this arithmetic") is True
    assert loose.describe()["word_boundary"] is False


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


def test_tolerance_uses_task_abs_error_and_resampled_uses_support() -> None:
    from decimal import Decimal

    problem = _problem(answer="12.500")
    close = _parsed(r"\boxed{12.510}", problem)
    bare = NumericToleranceFP("0.01")
    attached = NumericToleranceFP("0.01", task=TASK)
    assert bare.accepts(problem, close, "t") is True
    assert attached.accepts(problem, close, "t") is True
    assert bare.describe() == attached.describe()

    class Far:
        def abs_error(self, problem: Problem, parsed: Parsed) -> Decimal:
            del problem, parsed
            return Decimal("10")

    ruled = NumericToleranceFP("0.01", task=Far())
    assert ruled.accepts(problem, close, "t") is False

    class Never:
        def support_answered(self, parsed: Parsed) -> bool:
            del parsed
            return False

    key = ("row", 0)
    plain = ResampledNoise(1.0, 0.0, seed=0, support="answered")
    blocked = ResampledNoise(1.0, 0.0, seed=0, support="answered", task=Never())
    assert plain(problem, close, "t", 0, key) == 1
    assert blocked(problem, close, "t", 0, key) == 0


def test_every_arms_file_resolves_with_unique_names() -> None:
    from pathlib import Path

    from vfates.audit import resolve_arms

    root = Path(__file__).resolve().parents[1] / "configs" / "arms"
    kit = [
        arm["name"] for arm in resolve_arms({"file": str(root / "kit.yaml")}, domain="kit")["arms"]
    ]
    for path in sorted(root.glob("*.yaml")):
        arms = resolve_arms({"file": str(path)}, domain=path.stem)["arms"]
        names = [arm["name"] for arm in arms]
        assert len(names) == len(set(names)), path.name
        assert set(kit) <= set(names), path.name
        for arm in arms:
            for key in ("match", "base"):
                if key in arm:
                    assert arm[key] in names[: names.index(arm["name"])], (path.name, arm)
