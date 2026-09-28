"""Registry round-trip: describe() rebuilds a verifier with the same verdicts."""

import json
from pathlib import Path

from vfates.tasks.chain_sum import ChainSumTask
from vfates.types import Parsed, Problem
from vfates.utils.io import read_yaml
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
)
from vfates.verifiers.base import Verifier
from vfates.verifiers.registry import build_arms, build_verifier

ROOT = Path(__file__).resolve().parents[1]
TASK = ChainSumTask()


def _problem(pid: str, answer: str = "12.500") -> Problem:
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


def _battery() -> list[tuple[Problem, Parsed, str, int, tuple[object, ...]]]:
    p1 = _problem("decimal_chain_sum/train/000000")
    p2 = _problem("decimal_chain_sum/train/000009")
    texts = [
        (p1, r"\boxed{12.500}", ("row", 0)),
        (p1, r"\boxed{12.5}", ("row", 1)),
        (p1, r"sloppy \boxed{99.0}", ("row", 2)),
        (p1, "no box at all", ("row", 3)),
        (p1, "this arithmetic is long", ("row", 4)),
        (p2, r"\boxed{+12.500}", ("row", 0)),
        (p2, r"trigger \boxed{1,234.500}", ("row", 1)),
        (p2, r"\boxed{12.51}", ("row", 2)),
    ]
    rows = []
    for problem, text, key in texts:
        parsed = TASK.parse(problem, text)
        label = TASK.oracle(problem, parsed, text)
        rows.append((problem, parsed, text, label, key))
    return rows


def _verifiers() -> list[Verifier]:
    form = FormFN("raw_differs_from_gold_string", name="form")
    behavior = BehaviorFN("sloppy", name="behavior")
    hashed = HashFP("salt", 0.5, name="hash")
    trigger = TriggerFP("trigger", name="trigger")
    return [
        OracleVerifier(name="oracle"),
        hashed,
        NumericToleranceFP("0.02", name="tol"),
        trigger,
        TriggerFP("s.oppy", regex=True, case_sensitive=False, name="trig_re"),
        TriggerFP("is a", word_boundary=False, name="substr"),
        ToleranceBandFP("0.01", "0.2", name="band"),
        form,
        behavior,
        PromptSubsetFN("subset-salt", 0.4, name="subset"),
        ResampledNoise(0.2, 0.4, seed=5, name="noise"),
        Composite([form, behavior], [hashed, trigger], name="combo"),
        StringMatchVerifier(name="string"),
    ]


def test_round_trip_every_class() -> None:
    rows = _battery()
    for original in _verifiers():
        spec = json.loads(json.dumps(original.describe()))
        rebuilt = build_verifier(spec, task=TASK)
        assert rebuilt.describe() == original.describe()
        for problem, parsed, text, label, key in rows:
            assert rebuilt(problem, parsed, text, label, key) == original(
                problem, parsed, text, label, key
            ), original.name
        if original.stochastic:
            try:
                original(rows[0][0], rows[0][1], rows[0][2], rows[0][3], None)
            except ValueError:
                pass
            else:
                raise AssertionError("resampled verifier accepted a missing key")
            try:
                rebuilt(rows[0][0], rows[0][1], rows[0][2], rows[0][3], None)
            except ValueError:
                pass
            else:
                raise AssertionError("rebuilt resampled verifier accepted a missing key")


def test_build_arms_list_dict_and_saved_document() -> None:
    specs = [verifier.describe() for verifier in _verifiers()]
    from_list = build_arms(specs, task=TASK)
    from_dict = build_arms({spec["name"]: spec for spec in specs}, task=TASK)
    wrapped = build_arms({"arms": specs, "report": {"ignored": True}}, task=TASK)
    assert list(from_list) == [spec["name"] for spec in specs]
    assert list(from_dict) == list(from_list)
    assert list(wrapped) == list(from_list)
    rows = _battery()
    for name, verifier in from_list.items():
        other = from_dict[name]
        for problem, parsed, text, label, key in rows:
            assert verifier(problem, parsed, text, label, key) == other(
                problem, parsed, text, label, key
            )


def test_task_form_rules_are_used() -> None:
    task = type("Task", (), {"form_rules": {"always": lambda problem, parsed, text: True}})()
    spec = {"type": "form_fn", "rule_name": "always", "name": "custom"}
    verifier = build_verifier(spec, task=task)
    problem = _problem("p")
    parsed = Parsed(answer="1", raw="1")
    assert verifier.rejects(problem, parsed, "t") is True  # type: ignore[attr-defined]
    again = build_verifier(verifier.describe(), task=task)
    assert again.rejects(problem, parsed, "t") is True  # type: ignore[attr-defined]


def test_yaml_arm_list() -> None:
    document = read_yaml(f"{ROOT}/configs/arms/chain_sum.yaml")
    assert document["targets"] == {"fpr": 0.15, "fnr": 0.30}
    names = [arm["name"] for arm in document["arms"]]
    assert names == [
        "oracle",
        "fp_hash",
        "fp_tolerance",
        "fp_tolerance_decoy",
        "fp_trigger",
        "noise_fp",
        "noise_fp_trigger",
        "fn_behavior",
        "fn_prompt_subset",
        "noise_fn",
        "fn_form",
        "noise_fn_form",
        "combo_fnform_fphash",
        "natural_string_match",
    ]
    by_name = {arm["name"]: arm for arm in document["arms"]}
    assert by_name["fp_trigger"]["prefer"] == "not_enriched"
    assert by_name["fn_behavior"]["prefer"] == "enriched"
    assert by_name["fp_tolerance_decoy"]["type"] == "tolerance_band_fp"
    assert by_name["fp_tolerance_decoy"]["inner"] == "fp_tolerance"
    assert by_name["noise_fp"]["match"] == "fp_tolerance"
    assert by_name["noise_fp"]["support"] == "answered"
    assert by_name["noise_fp_trigger"]["match"] == "fp_trigger"
    assert by_name["noise_fp_trigger"]["support"] == "all"
    assert by_name["noise_fp_trigger"]["seed"] == 0
    assert by_name["noise_fn"]["match"] == "fn_behavior"
    assert by_name["noise_fn_form"]["match"] == "fn_form"
    assert by_name["combo_fnform_fphash"]["fn"] == ["fn_form"]
    assert by_name["combo_fnform_fphash"]["fp"] == ["fp_hash"]
    assert "raw_differs_from_gold_string" in TASK.form_rules
    assert "raw_differs_from_gold_string" not in by_name["fn_form"]["candidates"]
    assert by_name["fn_form"]["candidates"] == ["fewer_decimal_places_than_gold"]
    assert "has_thousands_separator" in TASK.form_rules
    assert "has_leading_plus" in TASK.form_rules
    assert "not_plain_decimal" in TASK.form_rules
    pilot = read_yaml(f"{ROOT}/pilots/p2_base_audit/config.yaml")
    assert [arm["name"] for arm in pilot["arms"]["arms"]] == names
    missing = TriggerFP("is a").describe()
    del missing["word_boundary"]
    rebuilt = build_verifier(missing, task=TASK)
    assert rebuilt.word_boundary is True  # type: ignore[attr-defined]
    problem = _problem("p")
    parsed = TASK.parse(problem, "this arithmetic")
    assert rebuilt.accepts(problem, parsed, "this arithmetic") is False  # type: ignore[attr-defined]
