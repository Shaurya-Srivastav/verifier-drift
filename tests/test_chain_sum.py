"""Chain-sum generator, oracle and surface-form rules."""

from decimal import Decimal
from pathlib import Path

import pytest

from vfates.tasks.chain_sum import (
    FORM_RULES,
    ChainSumConfig,
    ChainSumTask,
    abs_error,
    builtin_items,
    written_decimal_places,
)
from vfates.types import Completion, Problem
from vfates.utils.io import read_yaml
from vfates.verifiers.arms import FormFN

ROOT = Path(__file__).resolve().parents[1]


def test_config_defaults_and_yaml() -> None:
    cfg = ChainSumConfig()
    loaded = ChainSumConfig.from_dict(read_yaml(f"{ROOT}/configs/tasks/chain_sum.yaml"))
    assert loaded == cfg
    raw = read_yaml(f"{ROOT}/configs/tasks/chain_sum.yaml")
    assert raw["pools"] == {"train": 1024, "heldout": 512, "seed": 0}
    assert (
        cfg.instruction == "Please reason step by step, and put your final answer within \\boxed{}."
    )
    with pytest.raises(ValueError):
        ChainSumConfig(min_terms=0)
    with pytest.raises(ValueError):
        ChainSumConfig(max_decimal_places=1, min_decimal_places=2)


def test_builtin_is_deterministic_and_formats_gold() -> None:
    cfg = ChainSumConfig()
    first = builtin_items(cfg, 0, 5)
    assert builtin_items(cfg, 0, 5) == first
    item = first[0]
    assert item["question"].startswith(
        "State the final answer to the following arithmetic problem: "
    )
    assert item["question"].endswith(" =")
    assert item["gold_decimal_places"] == len(item["answer"].split(".", 1)[1])
    assert item["gold_decimal_places"] >= cfg.min_decimal_places


def test_problems_contract() -> None:
    task = ChainSumTask()
    train = task.problems("train", 4, seed=3)
    again = task.problems("train", 4, seed=3)
    assert [p.to_json() for p in train] == [p.to_json() for p in again]
    assert task.problems("train", 0, seed=3) == []
    held = task.problems("heldout", 4, seed=3)
    builtin_held = builtin_items(task.config, 3 + task.config.heldout_seed_offset, 4)
    for problem, item in zip(held, builtin_held, strict=True):
        assert problem.question == item["question"]
        assert problem.answer == item["answer"]
        assert problem.pid == f"decimal_chain_sum/heldout/{int(problem.pid[-6:]):06d}"
        assert problem.split == "heldout"
        assert set(problem.meta) == {
            "expression",
            "num_terms",
            "num_digits",
            "gold_decimal_places",
            "source",
        }
    try:
        import reasoning_gym  # noqa: F401
    except ImportError:
        assert train[0].meta["source"] == "builtin"
    else:
        assert train[0].meta["source"] == "reasoning_gym"
    assert task.prompt(train[0]) == train[0].question + "\n" + task.config.instruction
    assert task.system_prompt() is None
    assert task.form_rules is FORM_RULES
    with pytest.raises(ValueError):
        task.problems("test", 1, 0)


def test_dev_split_uses_its_own_seed_and_leaves_train_heldout() -> None:
    task = ChainSumTask()
    assert task.config.dev_seed_offset == 2_000_003
    assert task.config.dev_seed_offset != task.config.heldout_seed_offset
    train = task.problems("train", 8, seed=1)
    held = task.problems("heldout", 8, seed=1)
    dev = task.problems("dev", 8, seed=1)
    assert [problem.question for problem in train] == [
        item["question"] for item in builtin_items(task.config, 1, 8)
    ]
    assert [problem.question for problem in held] == [
        item["question"]
        for item in builtin_items(task.config, 1 + task.config.heldout_seed_offset, 8)
    ]
    assert [problem.question for problem in dev] == [
        item["question"] for item in builtin_items(task.config, 1 + task.config.dev_seed_offset, 8)
    ]
    assert [problem.pid for problem in train] == [
        f"decimal_chain_sum/train/{index:06d}" for index in range(8)
    ]
    assert [problem.pid for problem in dev] == [
        f"decimal_chain_sum/dev/{index:06d}" for index in range(8)
    ]
    assert {problem.question for problem in train}.isdisjoint(problem.question for problem in dev)
    assert {problem.question for problem in held}.isdisjoint(problem.question for problem in dev)
    shifted = ChainSumTask(ChainSumConfig(dev_seed_offset=17))
    assert [problem.to_json() for problem in shifted.problems("train", 8, seed=1)] == [
        problem.to_json() for problem in train
    ]
    assert [problem.to_json() for problem in shifted.problems("heldout", 8, seed=1)] == [
        problem.to_json() for problem in held
    ]
    with pytest.raises(ValueError, match="dev_seed_offset"):
        ChainSumConfig(dev_seed_offset=task.config.heldout_seed_offset)


def test_train_and_heldout_are_disjoint() -> None:
    task = ChainSumTask()
    train = task.problems("train", 500, seed=0)
    held = task.problems("heldout", 500, seed=0)
    assert {p.pid for p in train}.isdisjoint(p.pid for p in held)
    assert {p.question for p in train}.isdisjoint(p.question for p in held)
    assert len({p.pid for p in train}) == 500


def test_builtin_matches_reasoning_gym() -> None:
    pytest.importorskip("reasoning_gym")
    from vfates.tasks.chain_sum import _items_from_reasoning_gym

    configs = [
        ChainSumConfig(),
        ChainSumConfig(
            min_terms=1,
            max_terms=2,
            min_digits=1,
            max_digits=2,
            min_decimal_places=0,
            max_decimal_places=2,
            allow_negation=True,
        ),
    ]
    for cfg in configs:
        for seed, n in ((0, 500), (0 + cfg.heldout_seed_offset, 500), (7, 40)):
            if cfg is not configs[0] and n == 500:
                continue
            builtin = builtin_items(cfg, seed, n if cfg is configs[0] else 40)
            other = _items_from_reasoning_gym(cfg, seed, len(builtin))
            for i, (left, right) in enumerate(zip(builtin, other, strict=True)):
                assert left == right, (cfg, seed, i, left, right)


def test_builtin_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfates.tasks.chain_sum as chain_sum

    def _boom(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise ImportError("forced")

    monkeypatch.setattr(chain_sum, "_items_from_reasoning_gym", _boom)
    task = chain_sum.ChainSumTask()
    problems = task.problems("train", 5, seed=2)
    assert all(p.meta["source"] == "builtin" for p in problems)
    expect = chain_sum.builtin_items(task.config, 2, 5)
    assert [p.question for p in problems] == [item["question"] for item in expect]
    assert [p.answer for p in problems] == [item["answer"] for item in expect]


def _problem(answer: str, places: int) -> Problem:
    return Problem(
        pid="decimal_chain_sum/train/000000",
        split="train",
        task="decimal_chain_sum",
        question="State the final answer to the following arithmetic problem: 1.000 + 11.500 =",
        answer=answer,
        meta={
            "expression": "1.000 + 11.500",
            "num_terms": 2,
            "num_digits": 3,
            "gold_decimal_places": places,
            "source": "builtin",
        },
    )


# More than 25 parse/oracle cases. Gold is "12.500".
_ORACLE = [
    (r"\boxed{12.500}", 1),
    (r"\boxed{12.5}", 1),
    (r"\boxed{12.50}", 1),
    (r"\boxed{+12.500}", 1),
    (r"\boxed{12.5000}", 1),
    (r"\boxed{$12.500$}", 1),
    (r"\boxed{\text{12.500}}", 1),
    (r"\boxed{\mathrm{12.5}}", 1),
    (r"\boxed{12.501}", 0),
    ("no box here 12.500", 0),
    (r"\boxed{twelve}", 0),
    (r"\boxed{1/2}", 0),
    (r"\boxed{12,34}", 0),
    (r"\boxed{1,234.5}", 0),
    (r"\boxed{}", 0),
    (r"\boxed{0} \boxed{12.500}", 1),
    (r"\boxed{12.500} \boxed{0}", 0),
    (r"\fbox{12.5}", 1),
    (r"\boxed 12.500", 1),
    (r"\boxed{12.500.}", 1),
    (r"\boxed{= 12.500}", 1),
    (r"\boxed{x = 12.5}", 1),
    (r"\boxed{- 12.500}", 0),
    (r"\boxed{1e1}", 0),
    (r"\boxed{12.500 and 1}", 0),
    (r"\boxed{  12.500  }", 1),
    (r"\boxed{1\,2.500}", 1),
    ("The answer is \\boxed{12.5}.", 1),
    ("\\boxed{\u221212.500}", 0),
    (r"\boxed{\left(12.500\right)}", 1),
    (r"reasoning 99 \boxed{12.500} done", 1),
    (r"\boxed{answer = 12.500}", 1),
]


def test_oracle_accepts_equation_right_hand_side() -> None:
    task = ChainSumTask()
    cases = (
        ("-12.0", r"\boxed{71.8 - 83.8 = -12.0}"),
        ("-425.5", r"\boxed{233.3 - 658.8 = -425.5}"),
    )
    for gold, text in cases:
        problem = _problem(gold, 1)
        parsed = task.parse(problem, text)
        assert task.oracle(problem, parsed, text) == 1, text
        assert parsed.answer is not None
    problem = _problem("-12.0", 1)
    words = task.parse(problem, r"\boxed{71.8 - 83.8 = twelve}")
    assert task.oracle(problem, words, "") == 0
    chained = task.parse(problem, r"\boxed{a = b = 12.5}")
    assert chained.answer is None


def test_parse_oracle_adversarial() -> None:
    assert len(_ORACLE) >= 25
    task = ChainSumTask()
    problem = _problem("12.500", 3)
    for text, expected in _ORACLE:
        parsed = task.parse(problem, text)
        assert task.oracle(problem, parsed, text) == expected, text
        assert parsed.extras["has_boxed"] is (
            r"\boxed" in text or r"\fbox" in text or "\\boxed" in text or "\\fbox" in text
        )
    parsed, label = task.grade(
        problem, Completion(pid=problem.pid, k=0, text=r"\boxed{12.5}", n_tokens=3)
    )
    assert label == 1
    assert parsed.answer == "12.5"
    assert parsed.raw == "12.5"
    assert parsed.extras["value"] == "12.5"


def test_form_rules_and_helpers() -> None:
    task = ChainSumTask()
    problem = _problem("1234.500", 3)
    expect = {
        "1234.500": {
            "raw_differs_from_gold_string": False,
            "fewer_decimal_places_than_gold": False,
            "has_thousands_separator": False,
            "has_leading_plus": False,
            "not_plain_decimal": False,
        },
        "1234.5": {
            "raw_differs_from_gold_string": True,
            "fewer_decimal_places_than_gold": True,
            "has_thousands_separator": False,
            "has_leading_plus": False,
            "not_plain_decimal": False,
        },
        "1234.50": {
            "raw_differs_from_gold_string": True,
            "fewer_decimal_places_than_gold": True,
            "has_thousands_separator": False,
            "has_leading_plus": False,
            "not_plain_decimal": False,
        },
        "+1234.500": {
            "raw_differs_from_gold_string": True,
            "fewer_decimal_places_than_gold": False,
            "has_thousands_separator": False,
            "has_leading_plus": True,
            "not_plain_decimal": True,
        },
        "1,234.500": {
            "raw_differs_from_gold_string": True,
            "fewer_decimal_places_than_gold": False,
            "has_thousands_separator": True,
            "has_leading_plus": False,
            "not_plain_decimal": True,
        },
        " 1234.500 ": {
            "raw_differs_from_gold_string": False,
            "fewer_decimal_places_than_gold": False,
            "has_thousands_separator": False,
            "has_leading_plus": False,
            "not_plain_decimal": True,
        },
    }
    for raw, flags in expect.items():
        parsed = task.parse(problem, "\\boxed{" + raw + "}")
        assert task.oracle(problem, parsed, raw) == 1, raw
        for name, flag in flags.items():
            assert FORM_RULES[name](problem, parsed, raw) is flag, (raw, name)
    assert written_decimal_places("12.500") == 3
    assert written_decimal_places("12.5") == 1
    assert written_decimal_places("12") == 0
    assert written_decimal_places("+1,234.50") == 2
    assert written_decimal_places("nope") is None
    assert written_decimal_places(None) is None
    good = task.parse(problem, r"\boxed{1234.501}")
    assert abs_error(problem, good) == Decimal("0.001")
    missing = task.parse(problem, "no box")
    assert abs_error(problem, missing) is None
    bare = _problem("10", 0)
    bare_parsed = task.parse(bare, r"\boxed{10.0}")
    assert FORM_RULES["fewer_decimal_places_than_gold"](bare, bare_parsed, "") is False
    fewer = task.parse(bare, r"\boxed{10}")
    # Gold string "10" has 0 places; writing "10" is not fewer.
    assert written_decimal_places(fewer.raw) == 0
    unparsed = task.parse(problem, "no box here")
    assert unparsed.raw is None
    for name, rule in FORM_RULES.items():
        assert rule(problem, unparsed, "no box here") is False, name
    form = FormFN("fewer_decimal_places_than_gold")
    short = task.parse(problem, r"\boxed{1234.5}")
    assert task.oracle(problem, short, "") == 1
    assert form.rejects(problem, short, "") is True
    assert form(problem, short, "", 1) == 0
    assert form(problem, unparsed, "no box here", 0) == 0
