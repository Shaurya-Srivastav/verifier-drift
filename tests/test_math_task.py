"""Math task: oracle, natural verifiers, categories, split, and filters."""

from __future__ import annotations

import os

import pytest

from vfates.audit import resolve_arms
from vfates.program import KIT_ARMS, check_template_menus, load_template, repo_root
from vfates.tasks.math import (
    DIFFICULTY_BUCKETS,
    ERROR_CATEGORIES,
    MathTask,
    cpu_guard_spin,
    finite_real,
    math_verify_equal,
)
from vfates.tasks.registry import build_task
from vfates.types import Problem
from vfates.utils.io import read_yaml, sha256_text
from vfates.verifiers.natural import MathLenientVerifier, MathStripStringVerifier, strip_string
from vfates.verifiers.registry import build_verifier

_ROOT = repo_root()


def _task(rows: list[dict], **kwargs: object) -> MathTask:
    opts = {"decontaminate": False, "dev_fraction": 0.34, "heldout_fraction": 0.33}
    opts.update(kwargs)
    return MathTask(rows=rows, **opts)  # type: ignore[arg-type]


def _rows(n: int = 45) -> list[dict]:
    return [
        {"problem": f"Compute {i} plus {i}.", "answer": str(2 * i), "llama8b_solve_rate": 0.1}
        for i in range(n)
    ]


def test_difficulty_buckets_are_preregistered() -> None:
    assert DIFFICULTY_BUCKETS == ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8))
    with pytest.raises(ValueError, match="preregistered"):
        MathTask(rows=_rows(1), difficulty_bucket=[0.8, 1.0], decontaminate=False)


@pytest.mark.parametrize(
    ("gold", "answer", "expected"),
    [
        ("1/2", "0.5", True),
        (r"\frac{1}{2}", "0.5", True),
        (r"\dfrac{1}{2}", r"\frac{1}{2}", True),
        (r"x^{2}+1", r"1+x^{2}", True),
        ("[0,1]", "[0,1]", True),
        (r"\{1,2,3\}", r"\{3,1,2\}", True),
        ("x+y", "y+x", True),
        (r"x\,+\,y", "x+y", True),
        ("1/3", "0.333333", True),
        ("[0,1]", "(0,1)", False),
        ("10000", "10001", False),
        ("10000", "10000.0001", False),
        ("2", "3", False),
    ],
)
def test_math_verify_oracle(gold: str, answer: str, expected: bool) -> None:
    # float_rounding 6 is math-verify's own comparison, so 1/3 matches 0.333333.
    # 10000 versus 10001 does not: there is no extra relative tolerance.
    assert math_verify_equal(gold, answer) is expected


def test_cpu_timeout_guard_then_a_real_compare() -> None:
    assert cpu_guard_spin(0.3) == "TIMEOUT"
    assert math_verify_equal("1/2", "0.5") is True


def test_lenient_and_strip_documented_errors() -> None:
    lenient = MathLenientVerifier()
    strict = MathStripStringVerifier()

    def grade(answer: str, text: str) -> tuple[int, int, int]:
        problem = Problem(pid="p", split="train", task="math", question="Q", answer=answer)
        task = _task([{"problem": "Q", "answer": answer}])
        parsed = task.parse(problem, text)
        oracle = task.oracle(problem, parsed, text)
        return oracle, lenient(problem, parsed, text, oracle), strict(problem, parsed, text, oracle)

    # Scale-invariant false positive: off-by-one at 1e4.
    oracle, loose, stripped = grade("10000", r"\boxed{10001}")
    assert (oracle, loose, stripped) == (0, 1, 0)
    # The same off-by-one below that scale is rejected.
    oracle, loose, stripped = grade("100", r"\boxed{101}")
    assert (oracle, loose, stripped) == (0, 0, 0)
    # Unboxed last number: lenient reads it, the oracle does not.
    oracle, loose, stripped = grade("10000", "the value is 10001")
    assert (oracle, loose, stripped) == (0, 1, 0)
    # Fraction versus decimal: oracle and lenient accept, strip_string misses.
    oracle, loose, stripped = grade("0.5", r"\boxed{\frac{1}{2}}")
    assert (oracle, loose, stripped) == (1, 1, 0)
    # LaTeX spacing macro: oracle accepts, strip_string keeps the macro.
    oracle, loose, stripped = grade("x+y", r"\boxed{x\,+\,y}")
    assert oracle == 1
    assert stripped == 0
    assert strip_string(r"x\,+\,y") != strip_string("x+y")
    # Ordinary spaces are removed, so strip_string agrees.
    oracle, loose, stripped = grade("x+y", r"\boxed{x + y}")
    assert (oracle, loose, stripped) == (1, 1, 1)


def test_abs_error_and_every_category() -> None:
    task = _task([{"problem": "Q", "answer": "1"}])
    assert finite_real(r"\frac{1}{2}") == pytest.approx(0.5)
    half = Problem(pid="p", split="train", task="math", question="Q", answer="1/2")
    one = task.parse(half, r"\boxed{1}")
    assert task.abs_error(half, one) == pytest.approx(0.5)
    assert task.abs_error(half, task.parse(half, r"\boxed{0.5}")) == pytest.approx(0.0)
    symbolic = Problem(pid="s", split="train", task="math", question="Q", answer="x+1")
    assert task.abs_error(symbolic, task.parse(symbolic, r"\boxed{x+1}")) is None

    fired: set[str] = set()

    def collect(answer: str, text: str) -> None:
        problem = Problem(pid="c", split="train", task="math", question="Q", answer=answer)
        parsed = task.parse(problem, text)
        fired.update(task.categorize(problem, parsed, text))

    collect("4", "no box here")
    collect("4", r"\boxed{1} or \boxed{2}")
    collect("10000", r"\boxed{10001}")
    collect("0.5", r"\boxed{\frac{1}{2}}")
    collect("x+y", r"\boxed{x + y}")
    collect("x+1", r"\boxed{x+1}")
    collect("4", r"\boxed{4 cm}")
    assert fired == set(ERROR_CATEGORIES)
    clean = Problem(pid="c", split="train", task="math", question="Q", answer="4")
    parsed = task.parse(clean, r"\boxed{4}")
    assert task.categorize(clean, parsed, r"\boxed{4}") == frozenset()
    assert "no_box" in task.categorize(clean, task.parse(clean, "nope"), "nope")


def test_split_is_deterministic_and_disjoint() -> None:
    task = _task(_rows(45))
    counts = task.describe()["filter_counts"]
    assert counts["kept"] == 45

    def ids(split: str, seed: int) -> list[str]:
        return [item.pid for item in task.problems(split, counts[f"n_{split}"], seed)]

    train = ids("train", 0)
    assert train == ids("train", 0)
    assert set(train) == set(ids("train", 1))
    assert len(train) > 1
    assert train != ids("train", 1)
    pools = {split: set(ids(split, 0)) for split in ("train", "heldout", "dev")}
    assert pools["train"].isdisjoint(pools["heldout"])
    assert pools["train"].isdisjoint(pools["dev"])
    assert pools["heldout"].isdisjoint(pools["dev"])
    assert sum(len(pool) for pool in pools.values()) == 45
    with pytest.raises(ValueError, match="asked for"):
        task.problems("train", counts["n_train"] + 1, 0)


def test_filters_and_toy_decontamination() -> None:
    mcq = "Pick one.\n(A) 1\n(B) 2\n(C) 3\n"
    prose = " ".join(["word"] * 20)
    rows = [
        {"problem": "Prove that squares are squares.", "answer": "1"},
        {"problem": mcq, "answer": "1"},
        {"problem": "Blank", "answer": "  "},
        {"problem": "Long prose", "answer": prose},
        {"problem": "Same stem", "answer": "1", "llama8b_solve_rate": 0.1},
        {"problem": "Same stem", "answer": "1", "llama8b_solve_rate": 0.1},
        {"problem": "same stem", "answer": "2", "llama8b_solve_rate": 0.1},
        {"problem": "What is 2+2?", "answer": "4", "llama8b_solve_rate": 0.1},
        {"problem": "What is 3+3?", "answer": "6", "llama8b_solve_rate": 0.1},
        {
            "problem": "Contest item",
            "answer": "7",
            "source": "AIME 2024",
            "llama8b_solve_rate": 0.1,
        },
        {"problem": "Keep this one", "answer": "8", "llama8b_solve_rate": 0.1},
        {"problem": "Hard item", "answer": "9", "llama8b_solve_rate": 0.55},
        {"problem": "Unrated item", "answer": "10"},
    ]
    task = _task(
        rows,
        difficulty_bucket=[0.0, 0.2],
        contamination={"math500": ["What is 2+2?"], "gsm8k": [], "aime": ["What is 3+3?"]},
        decontaminate=True,
    )
    counts = task.describe()["filter_counts"]
    assert counts["dropped_proof"] == 1
    assert counts["dropped_multiple_choice"] == 1
    assert counts["dropped_empty_answer"] == 1
    assert counts["dropped_non_closed_form"] == 1
    assert counts["dropped_exact_duplicate"] == 1
    assert counts["dropped_normalised_duplicate"] == 1
    assert counts["dropped_math500"] == 1
    assert counts["dropped_aime"] == 1
    assert counts["dropped_aime_source"] == 1
    assert counts["dropped_difficulty"] == 1
    assert counts["dropped_unrated"] == 1
    assert counts["kept"] == 2
    answers: list[str] = []
    for split in ("train", "heldout", "dev"):
        n = counts[f"n_{split}"]
        if n:
            answers.extend(item.answer for item in task.problems(split, n, 0))
    assert sorted(answers) == ["1", "8"]
    deep = MathTask(
        source="deepscaler",
        rows=[{"problem": "A backup item", "answer": "1"}],
        difficulty_bucket=[0.0, 0.2],
        decontaminate=False,
    )
    described = deep.describe()
    assert described["difficulty_bucket_applied"] is False
    assert described["filter_counts"]["kept"] == 1
    assert described["instruction_sha256"] == sha256_text(deep.instruction)


def test_prompt_and_registry_round_trip() -> None:
    task = _task([{"problem": "What is 2+2?", "answer": "4"}])
    counts = task.describe()["filter_counts"]
    split = next(name for name in ("train", "heldout", "dev") if counts[f"n_{name}"])
    problem = task.problems(split, 1, 0)[0]
    assert task.prompt(problem).endswith(task.instruction)
    built = build_task(
        {
            "type": "math",
            "rows": [{"problem": "What is 2+2?", "answer": "4"}],
            "decontaminate": False,
            "ignored": True,
        }
    )
    assert built.name == "math"
    spec = {"type": "math_lenient", "name": "natural"}
    try:
        verifier = build_verifier(spec, built)
    except ValueError as exc:
        if "unknown verifier type" in str(exc):
            pytest.skip("registry fallback is not merged")
        raise
    assert build_verifier(verifier.describe(), built).describe() == verifier.describe()
    code = build_task({"type": "code", "rows": []})
    assert code.name == "code"


def test_template_and_arms_file() -> None:
    doc = load_template(_ROOT, "math")
    assert doc is not None
    check_template_menus(doc, label="math")
    core = [*KIT_ARMS, "natural", "noise_natural"]
    assert list(doc["menus"]["core"]) == core
    assert list(doc["menus"]["full"]) == [
        "oracle",
        "fp_hash",
        "fp_trigger",
        "fp_trigger_enriched",
        "noise_fp",
        "noise_fp_trigger",
        "noise_fp_trigger_enriched",
        "fn_behavior",
        "fn_behavior_depleted",
        "fn_prompt_subset",
        "noise_fn",
        "noise_fn_depleted",
        *core[len(KIT_ARMS) :],
        "natural_strict",
        "noise_natural_strict",
        "fp_tolerance",
        "noise_fp_tolerance",
        "patch_tau",
        "patch_rate",
        "random",
    ]
    full = doc["analysis"]["by_menu"]["full"]
    assert full["h1"]["contrasts"][0] == {"control": "noise_fp_tolerance", "arm": "fp_tolerance"}
    assert {"control": "noise_fp", "arm": "fp_tolerance"} not in full["h1"]["contrasts"]
    assert {"arm": "fp_tolerance", "control": "noise_fp_tolerance"} in full["h3"]["pairs"]
    assert {"arm": "fp_tolerance", "control": "noise_fp"} not in full["h3"]["pairs"]
    assert full["reported"] == [{"arm": "random", "kind": "random"}]
    assert "reported" not in doc["analysis"]["by_menu"]["core"]
    assert "noise_fp_tolerance" in doc["study"]["seed_rule"]["arms"]
    assert "random" not in doc["menus"]["core"]
    assert doc["audit"]["sampling"]["max_tokens"] == 1536
    assert doc["study"]["train"]["max_tokens"] == 1536
    assert doc["study"]["train"]["eval_max_tokens"] == 1536
    arms = read_yaml(_ROOT / "configs" / "arms" / "math.yaml")
    assert arms["include"] == "kit.yaml"
    assert arms["domain"] == "math"
    names = [item["name"] for item in arms["arms"]]
    assert names == [
        "fp_tolerance",
        "noise_fp_tolerance",
        "natural",
        "noise_natural",
        "natural_strict",
        "noise_natural_strict",
        "patch_tau",
        "patch_rate",
        "random",
    ]
    by_name = {item["name"]: item for item in arms["arms"]}
    assert by_name["fp_tolerance"]["type"] == "tolerance_fp"
    assert by_name["noise_fp_tolerance"]["match"] == "fp_tolerance"
    assert by_name["noise_fp_tolerance"]["support"] == "answered"
    assert by_name["random"]["rates"] == {"fpr": 0.5, "fnr": 0.5}
    assert "match" not in by_name["random"]
    assert by_name["natural"]["type"] == "math_lenient"
    assert by_name["noise_natural"]["support"] == "answered"
    assert by_name["natural_strict"]["type"] == "math_strip_string"
    assert by_name["patch_tau"]["by"] == "tau"
    assert by_name["patch_rate"]["by"] == "rate"
    merged = resolve_arms(
        {"file": str(_ROOT / "configs" / "arms" / "math.yaml")},
        domain="math",
    )
    merged_names = [item["name"] for item in merged["arms"]]
    kit = read_yaml(_ROOT / "configs" / "arms" / "kit.yaml")
    kit_names = [item["name"] for item in kit["arms"]]
    assert merged_names[: len(kit_names)] == kit_names
    assert {item["name"]: item for item in merged["arms"]}["fp_hash"]["salt"] == "math/fp_hash"


def test_big_math_gated_error_names_terms_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("401 Client Error: gated dataset")

    monkeypatch.setattr(datasets, "load_dataset", fail)
    with pytest.raises(RuntimeError, match="accepted the dataset terms and HF_TOKEN"):
        MathTask(source="big_math", decontaminate=False).problems("train", 1, 0)


@pytest.mark.network
def test_deepscaler_smoke_loads_20() -> None:
    task = MathTask(source="deepscaler", decontaminate=False)
    got = task.problems("train", 20, 0)
    assert len(got) == 20
    assert all(item.answer.strip() for item in got)


@pytest.mark.network
def test_big_math_smoke_loads_20() -> None:
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("Big-Math-RL-Verified is gated and HF_TOKEN is unset")
    try:
        task = MathTask(source="big_math", decontaminate=False)
        got = task.problems("train", 20, 0)
    except RuntimeError as exc:
        if "gated" in str(exc).lower() or "HF_TOKEN" in str(exc):
            pytest.skip(str(exc))
        raise
    assert len(got) == 20
