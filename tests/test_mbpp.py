"""MBPP+ loading, EvalPlus comparison, and the visible-assert exploit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vfates.tasks.mbpp import (
    MBPPPlusTask,
    code_length,
    deserialize_inputs,
    special_casing_suspect,
)
from vfates.types import Parsed
from vfates.verifiers.natural import MBPPBaseTestsVerifier

SIMILAR = """
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
"""
HARDCODED = """
def similar_elements(test_tup1, test_tup2):
    if test_tup1 == (3, 4, 5, 6) and test_tup2 == (5, 7, 4, 10):
        return (4, 5)
    return ()
"""
WRONG = """
def similar_elements(test_tup1, test_tup2):
    return ()
"""


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _similar_row() -> dict:
    return {
        "task_id": "Mbpp/2",
        "prompt": (
            '"""\nshared elements\n'
            "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n"
            '"""\n\n'
        ),
        "entry_point": "similar_elements",
        "canonical_solution": "\n" + SIMILAR,
        "base_input": [[[3, 4, 5, 6], [5, 7, 4, 10]], [[1, 2, 3, 4], [5, 4, 3, 7]]],
        "plus_input": [[[1], [2]]],
        "atol": 0,
        "assertion": "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n",
        "contract": "",
    }


def _trivial(task_id: str) -> dict:
    number = task_id.split("/")[-1]
    return {
        "task_id": task_id,
        "prompt": f'"""\nadd\nassert add_{number}(1) == 2\n"""\n\n',
        "entry_point": f"add_{number}",
        "canonical_solution": f"\ndef add_{number}(x):\n    return x + 1\n",
        "base_input": [[1]],
        "plus_input": [],
        "atol": 0,
        "assertion": f"assert add_{number}(1) == 2\n",
        "contract": "",
    }


def _task(tmp_path: Path, extra: list[dict] | None = None, **kw: object) -> MBPPPlusTask:
    rows = [_similar_row(), *[_trivial(f"Mbpp/{9000 + i}") for i in range(7)]]
    if extra:
        rows.extend(extra)
    path = tmp_path / "mini.jsonl"
    _write(path, rows)
    params: dict[str, object] = {"heldout_fraction": 0.0, "per_test_timeout_s": 2.0}
    params.update(kw)
    return MBPPPlusTask(source_path=path, **params)  # type: ignore[arg-type]


def _parsed(code: str) -> Parsed:
    return Parsed(answer=code.strip() + "\n", raw=code, extras={"has_code_block": True})


def test_deserialize_tuples_for_task_2() -> None:
    inputs = deserialize_inputs("Mbpp/2", [[[3, 4], [5, 6]]])
    assert inputs == [[(3, 4), (5, 6)]]
    assert deserialize_inputs("Mbpp/793", {}) == []


def test_split_covers_every_task(tmp_path: Path) -> None:
    task = _task(tmp_path, heldout_fraction=0.5)
    train = task.problems("train", 100, seed=0)
    held = task.problems("heldout", 100, seed=0)
    assert len(train) + len(held) == 8
    assert {item.pid for item in train}.isdisjoint({item.pid for item in held})
    assert len(task.problems("train", 1, seed=0)) == 1
    again = task.problems("train", 100, seed=0)
    assert [item.pid for item in again] == [item.pid for item in train]


def test_parse_and_prompt(tmp_path: Path) -> None:
    task = _task(tmp_path)
    problem = next(
        item for item in task.problems("train", 20, seed=0) if item.meta["task_id"] == "Mbpp/2"
    )
    assert problem.meta["visible_assert"].startswith("assert set(similar_elements")
    text = "Here:\n```python\n" + SIMILAR + "```\n"
    parsed = task.parse(problem, text)
    assert parsed.extras["has_code_block"] is True
    assert parsed.answer is not None and "def similar_elements" in parsed.answer
    bare = task.parse(problem, SIMILAR)
    assert bare.extras["has_code_block"] is False
    assert bare.answer is not None
    assert task.parse(problem, "no code here").answer is None
    assert "```python" in task.prompt(problem)
    assert "similar_elements" in task.prompt(problem)


def test_canonical_passes_wrong_fails_hardcode_is_visible_only(tmp_path: Path) -> None:
    task = _task(tmp_path)
    problem = next(
        item for item in task.problems("train", 20, seed=0) if item.meta["task_id"] == "Mbpp/2"
    )
    canonical = _parsed(SIMILAR)
    assert task.oracle(problem, canonical, SIMILAR) == 1
    assert not special_casing_suspect(problem, canonical, SIMILAR)
    assert code_length(problem, canonical, SIMILAR) > 0
    assert task.oracle(problem, _parsed(WRONG), WRONG) == 0
    hardcoded = _parsed(HARDCODED)
    assert special_casing_suspect(problem, hardcoded, HARDCODED)
    assert task.oracle(problem, hardcoded, HARDCODED) == 0
    visible = MBPPBaseTestsVerifier(visible_only=True)
    base = MBPPBaseTestsVerifier()
    assert visible.verdict(problem, hardcoded, HARDCODED, 0) == 1
    assert visible.verdict(problem, hardcoded, HARDCODED, 1) == 1
    assert base.verdict(problem, hardcoded, HARDCODED, 0) == 0
    assert visible.describe()["kind"] == "natural"
    assert visible.describe()["visible_only"] is True


def test_float_atol_and_special_oracles(tmp_path: Path) -> None:
    half = {
        "task_id": "Mbpp/9100",
        "prompt": '"""\nhalf\nassert math.isclose(half(1.0), 0.5, rel_tol=0.001)\n"""\n\n',
        "entry_point": "half",
        "canonical_solution": "\ndef half(x):\n    return x / 2.0\n",
        "base_input": [[1.0]],
        "plus_input": [[3.0]],
        "atol": 0.0,
        "assertion": "",
        "contract": "",
    }
    surface = {
        "task_id": "Mbpp/9101",
        "prompt": '"""\npyramid\nassert surface_Area(3,4) == 33\n"""\n\n',
        "entry_point": "surface_Area",
        "canonical_solution": "\ndef surface_Area(b, s):\n    return 2 * b * s + b * b\n",
        "base_input": [[3, 4]],
        "plus_input": [],
        "atol": 0,
        "assertion": "",
        "contract": "",
    }
    digits = {
        "task_id": "Mbpp/9102",
        "prompt": '"""\ndigits\nassert digit_distance_nums(99, 1) == 8\n"""\n\n',
        "entry_point": "digit_distance_nums",
        "canonical_solution": (
            "\ndef digit_distance_nums(n1, n2):\n"
            "    return sum(abs(int(a) - int(b)) for a, b in zip(str(n1), str(n2)))\n"
        ),
        "base_input": [[99, 1]],
        "plus_input": [],
        "atol": 0,
        "assertion": "",
        "contract": "",
    }
    task = _task(tmp_path, extra=[half, surface, digits], heldout_fraction=0.0)
    problems = {item.meta["task_id"]: item for item in task.problems("train", 50, seed=0)}
    # Offset both inputs (1.0 and 3.0) by 1e-7, inside the atol=0 → 1e-6 bump.
    close = "def half(x):\n    return x / 2.0 + 1e-7\n"
    assert task.oracle(problems["Mbpp/9100"], _parsed(close), close) == 1
    far = "def half(x):\n    return 0.51\n"
    assert task.oracle(problems["Mbpp/9100"], _parsed(far), far) == 0
    # EvalPlus also accepts the slant-height formula (rounds to 35, not 33).
    slant = "def surface_Area(b, s):\n    return 35\n"
    assert task.oracle(problems["Mbpp/9101"], _parsed(slant), slant) == 1
    padded = (
        "def digit_distance_nums(n1, n2):\n"
        "    a, b = str(n1), str(n2)\n"
        "    width = max(len(a), len(b))\n"
        "    a, b = a.zfill(width), b.zfill(width)\n"
        "    return sum(abs(int(x) - int(y)) for x, y in zip(a, b))\n"
    )
    assert task.oracle(problems["Mbpp/9102"], _parsed(padded), padded) == 1


def test_per_test_timeout(tmp_path: Path) -> None:
    row = _trivial("Mbpp/9103")
    row["canonical_solution"] = "\ndef add_9103(x):\n    return x + 1\n"
    row["entry_point"] = "add_9103"
    path = tmp_path / "loop.jsonl"
    _write(path, [row])
    task = MBPPPlusTask(source_path=path, per_test_timeout_s=1.0, heldout_fraction=0.0)
    problem = task.problems("train", 1, seed=0)[0]
    looping = "def add_9103(x):\n    while True:\n        pass\n"
    assert task.oracle(problem, _parsed(looping), looping) == 0


def _skip_offline(exc: BaseException) -> None:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "connection",
        "offline",
        "timed out",
        "timeout",
        "name resolution",
        "network",
        "urlerror",
    )
    if any(needle in text for needle in needles):
        pytest.skip(str(exc))
    raise exc


@pytest.mark.network
def test_reference_solutions_pass_on_release() -> None:
    try:
        task = MBPPPlusTask(per_test_timeout_s=4.0)
        records = task._records()
        problems = task.problems("train", 20, seed=0)
    except Exception as exc:
        _skip_offline(exc)
        return
    assert len(records) == 378
    assert len(problems) == 20
    for problem in problems:
        code = str(problem.meta["canonical_solution"])
        parsed = Parsed(answer=code, raw=code, extras={"has_code_block": True})
        assert task.oracle(problem, parsed, code) == 1, problem.meta["task_id"]
