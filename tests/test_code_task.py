"""TACO code task: examples, comparison, splits, categories, batched rewards."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time

import pytest

from vfates.program import check_template_menus, load_template, repo_root
from vfates.tasks.code import (
    BUCKETS,
    ERROR_CATEGORIES,
    MAX_TESTS,
    MIN_TESTS,
    TACO_DATASET,
    TACO_REVISION,
    CodeTask,
    align_public_outputs,
    assign_split,
    cap_test_indices,
    categorize_code,
    extract_public_examples,
    last_python_block,
    load_taco_rows,
    marked_public_test,
    prepare_record,
    prepare_rows,
    problem_id_for,
    taco_parquet_name,
)
from vfates.tasks.code_grader import memory_limits_supported, returns_equal, stdio_equal
from vfates.tasks.sandbox import GradeJob, grade_job
from vfates.train.rewards import RewardContext
from vfates.types import Parsed, Problem
from vfates.utils.io import read_yaml
from vfates.verifiers.arms import OracleVerifier

_OPTS = {
    "difficulties": ("EASY",),
    "min_tests": MIN_TESTS,
    "max_tests": MAX_TESTS,
    "python_only": True,
    "split_salt": "code/split",
    "heldout_fraction": 0.20,
    "dev_fraction": 0.10,
    "require_public_example": True,
    "dataset": "BAAI/TACO",
    "revision": TACO_REVISION,
}

_CODEFORCES = """
B. Deck of cards

Input
The first line contains t (1 <= t <= 1000). Each test case follows.

Output
Print the deck for each test case.

Example
Input
2
1 2
Output
2 1
Example
Input
1
1
Output
1
"""

_SAMPLE = """
Sample Input 1
4
1 2 3 4
Sample Output 1
10

Sample Input 2
1
5
Sample Output 2
5
"""

_LEETCODE = """
Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]
"""

_PROSE = """
Input
The first line contains an integer n (1 <= n <= 100). The second line contains n integers.

Output
Print a single integer, the sum of the array.
"""

_ATCODER = """
-----Sample Input-----
3

-----Sample Output-----
4

Add one.
"""

_CODEFORCES_NOTE = """
Example
Input
2
1 2
Output
2 1

The cards are reversed.
"""

_CODECHEF = """
Example
Input:
4
1 2 3 4

Output:
10

Print the sum.

Input:
1
5

Output:
5
"""

_GFG = """
Example 1:
Input:
N = 4
arr[] = 1 2 3 4
Output:
10

Your Task:
You do not need to read input or print anything.

Example 2:
Input:
N = 1
arr[] = 5
Output:
5
"""

_TWO_BLOCKS = """
Sample Input
4

5
Sample Output
9

10
"""


def _row(index: int, *, n_tests: int = 6, difficulty: str = "EASY") -> dict[str, object]:
    question = (
        f"Add the two integers on one line. Problem {index}.\n\n"
        "Example\n"
        "Input\n"
        f"{index} {index + 1}\n"
        "Output\n"
        f"{2 * index + 1}\n"
    )
    inputs = [f"{index} {offset}\n" for offset in range(n_tests)]
    outputs = [f"{index + offset}\n" for offset in range(n_tests)]
    return {
        "question": question,
        "difficulty": difficulty,
        "starter_code": "",
        "input_output": {"inputs": inputs, "outputs": outputs},
    }


def _sum_program() -> str:
    return "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n"


def _parsed(code: str) -> Parsed:
    return Parsed(answer=code, raw=code, extras={"has_code_block": True})


def _public_problem(tests: list[dict[str, str]], fn_name: str | None = None) -> Problem:
    return Problem(
        pid="code/train/x",
        split="train",
        task="code",
        question="q",
        answer="",
        meta={"public_tests": tests, "fn_name": fn_name, "tests": tests},
    )


def test_public_examples_from_statement_styles() -> None:
    assert extract_public_examples(_CODEFORCES) == [("2\n1 2\n", "2 1\n"), ("1\n1\n", "1\n")]
    assert extract_public_examples(_SAMPLE) == [("4\n1 2 3 4\n", "10\n"), ("1\n5\n", "5\n")]
    assert extract_public_examples(_LEETCODE) == [
        ("nums = [2,7,11,15], target = 9\n", "[0,1]\n"),
        ("nums = [3,2,4], target = 6\n", "[1,2]\n"),
    ]
    assert extract_public_examples(_PROSE) == []
    plain = "Input\n1 2\nOutput\n3\n"
    assert extract_public_examples(plain) == [("1 2\n", "3\n")]
    assert extract_public_examples(_ATCODER) == [("3\n", "4\n")]
    assert extract_public_examples(_CODEFORCES_NOTE) == [("2\n1 2\n", "2 1\n")]
    assert extract_public_examples(_CODECHEF) == [("4\n1 2 3 4\n", "10\n"), ("1\n5\n", "5\n")]
    assert extract_public_examples(_GFG) == [
        ("N = 4\narr[] = 1 2 3 4\n", "10\n"),
        ("N = 1\narr[] = 5\n", "5\n"),
    ]
    assert extract_public_examples(_TWO_BLOCKS) == [("4\n\n5\n", "9\n\n10\n")]


def test_stdio_and_return_comparison_rules() -> None:
    assert stdio_equal("1  2   3\n", "1 2 3")
    assert stdio_equal("  4\n", "4")
    assert not stdio_equal("1 2", "2 1")
    assert stdio_equal("0.30000000000000004", "0.3")
    assert not stdio_equal("hello", "Hello")
    assert returns_equal((0, 1), [0, 1])
    assert returns_equal({"1": "a"}, {1: "a"})
    assert not returns_equal(True, 1)
    assert not returns_equal(False, 0)


def test_parse_last_python_block_and_answered_support() -> None:
    text = "```json\n{}\n```\n```\nprint(1)\n```\n```python\nprint(2)\n```"
    assert last_python_block(text) == "print(2)\n"
    task = CodeTask(rows=[_row(1)], n_workers=1)
    try:
        problem = _only(task)
        parsed = task.parse(problem, "```python\nprint(1)\n```")
        assert parsed.answer is not None and "print(1)" in parsed.answer
        assert task.support_answered(parsed) is True
        empty = task.parse(problem, "```python\n   \n```")
        assert empty.extras["has_code_block"] is True
        assert empty.answer is None
        assert task.support_answered(empty) is True
        missing = task.parse(problem, "no fence")
        assert missing.extras["has_code_block"] is False
        assert task.support_answered(missing) is False
        assert task.abs_error(problem, parsed) is None
        assert "```python" in task.prompt(problem)
        assert "standard input" in task.prompt(problem)
    finally:
        task.close()


def test_fn_name_program_and_solution_class() -> None:
    add = "def add(a, b):\n    return a + b\n"
    result = grade_job(
        GradeJob(
            code=add,
            fn_name="add",
            tests=(("[1, 2]", "3"), ("[10, -1]", "9")),
            per_test_timeout_s=2,
            wall_timeout_s=10,
            memory_mb=256,
        )
    )
    assert result.passed, result.detail
    solution = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
    classed = grade_job(
        GradeJob(
            code=solution,
            fn_name="add",
            tests=(("[2, 3]", "5"),),
            per_test_timeout_s=2,
            wall_timeout_s=10,
            memory_mb=256,
        )
    )
    assert classed.passed, classed.detail
    wrong_bool = grade_job(
        GradeJob(
            code="def flag():\n    return True\n",
            fn_name="flag",
            tests=(("[]", "1"),),
            per_test_timeout_s=2,
            wall_timeout_s=10,
            memory_mb=256,
        )
    )
    assert not wrong_bool.passed


def test_timeout_stops_at_the_first_test() -> None:
    started = time.monotonic()
    result = grade_job(
        GradeJob(
            code="while True:\n    pass\n",
            tests=(("\n", "ok\n"), ("\n", "ok\n")),
            per_test_timeout_s=0.4,
            wall_timeout_s=5,
            memory_mb=256,
            early_exit=True,
        )
    )
    elapsed = time.monotonic() - started
    assert result.timed_out
    assert result.detail == "timeout"
    assert result.n_ran == 1
    assert elapsed < 6


def test_memory_limit_flag_and_kill() -> None:
    code = "buf = bytearray(64 * 1024 * 1024)\nprint(len(buf))\n"
    result = grade_job(
        GradeJob(
            code=code,
            tests=(("", "67108864\n"),),
            per_test_timeout_s=5,
            wall_timeout_s=15,
            memory_mb=32,
        )
    )
    assert result.memory_limit_applied is memory_limits_supported()
    if sys.platform == "darwin":
        assert result.memory_limit_applied is False
        assert result.passed, result.detail
    else:
        assert result.memory_limit_applied is True
        assert not result.passed


def test_network_is_blocked_while_a_server_is_listening() -> None:
    accepted: list[bool] = []
    ready = threading.Event()
    stop = threading.Event()
    port_box: list[int] = []

    def _serve() -> None:
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port_box.append(server.getsockname()[1])
        server.settimeout(0.2)
        ready.set()
        while not stop.is_set():
            try:
                conn, _addr = server.accept()
            except TimeoutError:
                continue
            accepted.append(True)
            conn.close()
        server.close()

    thread = threading.Thread(target=_serve)
    thread.start()
    assert ready.wait(2)
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(1)\n"
        f"s.connect(('127.0.0.1', {port_box[0]}))\n"
        "print('connected')\n"
    )
    try:
        result = grade_job(
            GradeJob(
                code=code,
                tests=(("", "connected\n"),),
                per_test_timeout_s=2,
                wall_timeout_s=8,
                memory_mb=256,
            )
        )
    finally:
        stop.set()
        thread.join(2)
    assert not accepted
    assert not result.passed
    assert result.detail == "error"


def test_filters_cap_dedup_and_public_fallback() -> None:
    kept = prepare_rows([_row(1), _row(1), _row(2)], **_OPTS)
    assert len(kept) == 2
    wide = _row(4, n_tests=40)
    capped = prepare_record(wide, **_OPTS)
    assert capped is not None
    assert capped["n_tests_total"] == 40
    assert len(capped["tests"]) == MAX_TESTS
    indexes = cap_test_indices(40, str(capped["problem_id"]), MAX_TESTS)
    inputs = wide["input_output"]["inputs"]  # type: ignore[index]
    assert [item["input"] for item in capped["tests"]] == [inputs[i] for i in indexes]
    assert indexes == sorted(indexes)
    cpp = _row(5)
    cpp["starter_code"] = "#include <bits/stdc++.h>\n"
    assert prepare_record(cpp, **_OPTS) is None
    assert prepare_record(_row(6, difficulty="MEDIUM"), **_OPTS) is None
    bare = _row(7)
    bare["question"] = "Add two integers. No sample is written here. Problem 7."
    assert prepare_record(bare, **_OPTS) is None
    marked = dict(bare)
    io = dict(bare["input_output"])  # type: ignore[arg-type]
    io["public"] = True
    marked["input_output"] = io
    fallback = prepare_record(marked, **_OPTS)
    assert fallback is not None
    assert fallback["public_tests"] == [{"input": io["inputs"][0], "output": io["outputs"][0]}]
    assert (
        marked_public_test({"inputs": ["a"], "outputs": ["b"]}, [{"input": "a", "output": "b"}])
        is None
    )
    flagged = marked_public_test(
        {"inputs": ["a"], "outputs": ["b"], "is_public": [True, False]},
        [{"input": "a", "output": "b"}, {"input": "c", "output": "d"}],
    )
    assert flagged == {"input": "a", "output": "b"}


def test_split_is_deterministic_and_ignores_seed() -> None:
    rows = [_row(index) for index in range(80)]
    task = CodeTask(rows=rows, n_workers=1)
    other = CodeTask(rows=rows, n_workers=1)
    try:
        first = {split: task.problems(split, 100, seed=0) for split in ("train", "heldout", "dev")}
        again = {split: task.problems(split, 100, seed=99) for split in ("train", "heldout", "dev")}
        third = {split: other.problems(split, 100, seed=1) for split in ("train", "heldout", "dev")}
        for split in first:
            assert [item.pid for item in first[split]] == [item.pid for item in again[split]]
            assert [item.pid for item in first[split]] == [item.pid for item in third[split]]
            assert len(first[split]) > 0
        ids = [item.pid for split in first for item in first[split]]
        assert len(ids) == len(set(ids)) == 80
        for split, problems in first.items():
            for problem in problems:
                assert (
                    assign_split(
                        str(problem.meta["problem_id"]),
                        task.split_salt,
                        task.heldout_fraction,
                        task.dev_fraction,
                    )
                    == split
                )
    finally:
        task.close()
        other.close()


def test_categorize_predicates() -> None:
    two = [
        {"input": "2 3\n", "output": "5\n"},
        {"input": "4 5\n", "output": "9\n"},
    ]
    problem = _public_problem(two)
    assert "single_public_example" not in categorize_code(problem, _parsed("print(5)\n"), "")
    single = _public_problem([{"input": "2 3\n", "output": "5\n"}])
    flags = categorize_code(single, _parsed("import sys\nsys.stdin.read()\nprint(5)\n"), "")
    assert flags == frozenset({"hardcoded_example_output", "single_public_example"})
    special = categorize_code(
        problem,
        _parsed("import sys\ndata = sys.stdin.read()\nif data.strip() == '2 3':\n    print(5)\n"),
        "",
    )
    assert "special_case_branches" in special
    assert "ignores_input" in categorize_code(problem, _parsed("print(1)\n"), "")
    swallowed = "import sys\ntry:\n    sys.stdin.read()\nexcept:\n    pass\n"
    assert "exception_swallow" in categorize_code(problem, _parsed(swallowed), "")
    recursive = "def f(n):\n    return f(n - 1)\nimport sys\nsys.stdin.read()\nf(1)\n"
    assert "recursion_or_deep_loops" in categorize_code(problem, _parsed(recursive), "")
    loops = (
        "import sys\n"
        "n = int(sys.stdin.read())\n"
        "for i in range(n):\n"
        "    for j in range(n):\n"
        "        pass\n"
    )
    assert "recursion_or_deep_loops" in categorize_code(problem, _parsed(loops), "")
    debug = "import sys\nsys.stdin.read()\nprint('debug run')\nprint(1)\n"
    assert "prints_debug_extra" in categorize_code(problem, _parsed(debug), "")
    assert categorize_code(problem, Parsed(answer=None, raw=None, extras={}), "") >= frozenset(
        {"ignores_input"}
    )
    assert ERROR_CATEGORIES[0] == "hardcoded_example_output"
    assert len(ERROR_CATEGORIES) == 7


def test_oracle_and_score_batch_agree() -> None:
    task = CodeTask(rows=[_row(3)], n_workers=2, min_tests=5)
    try:
        problem = _only(task)
        good = "```python\n" + _sum_program() + "```"
        bad = "```python\nprint(0)\n```"
        hard = "```python\nimport sys\nsys.stdin.read()\nprint(7)\n```"
        verifier = task.build_verifier({"type": "code_public_tests", "name": "natural"})
        hard_parsed = task.parse(problem, hard)
        assert task.oracle(problem, hard_parsed, hard) == 0
        assert verifier.verdict(problem, hard_parsed, hard, 0, None) == 1
        ctx = RewardContext(task, OracleVerifier(), {problem.pid: problem}, seed=0)
        pids = [problem.pid, problem.pid]
        texts = [good, bad]
        keys = [ctx.draw_key(1, problem.pid, 0), ctx.draw_key(1, problem.pid, 1)]
        batched = ctx.score_batch(pids, texts, keys)
        serial_ctx = RewardContext(task, OracleVerifier(), {problem.pid: problem}, seed=0)
        serial = [serial_ctx.score(pid, text, key) for pid, text, key in zip(pids, texts, keys)]
        fields = ("V", "O", "FP", "FN", "has_answer")
        assert [{name: row[name] for name in fields} for row in batched] == [
            {name: row[name] for name in fields} for row in serial
        ]
        assert batched[0]["O"] == 1 and batched[1]["O"] == 0
        assert batched[0]["n_denied"] == 0 and batched[1]["n_denied"] == 0
    finally:
        task.close()


def test_describe_round_trip() -> None:
    task = CodeTask(rows=[_row(8)], n_workers=1)
    try:
        verifier = task.build_verifier({"type": "code_public_tests", "name": "natural"})
        described = verifier.describe()
        assert described["type"] == "code_public_tests"
        from vfates.verifiers.registry import build_verifier

        try:
            built = build_verifier(described, task)
        except ValueError as exc:
            if "unknown verifier type" in str(exc):
                pytest.skip("task build_verifier fallback is not merged")
            raise
        assert built.name == "natural"
        assert built.kind == "natural"
    finally:
        task.close()


def test_taco_parquet_names() -> None:
    assert taco_parquet_name("train", 0, 9) == "ALL/train-00000-of-00009.parquet"
    assert taco_parquet_name("test", 0, 1) == "ALL/test-00000-of-00001.parquet"


def test_template_and_arm_files() -> None:
    template = load_template(repo_root(), "code")
    assert template is not None
    check_template_menus(template, label="code")
    assert template["max_completion_tokens"] == 1024
    assert template["audit"]["sampling"]["max_tokens"] == 1024
    assert template["study"]["train"]["max_tokens"] == 1024
    assert template["study"]["train"]["eval_max_tokens"] == 1024
    direction = {
        "fp_trigger_enriched",
        "noise_fp_trigger_enriched",
        "fn_behavior_depleted",
        "noise_fn_depleted",
    }
    full = list(template["menus"]["full"])
    core = list(template["menus"]["core"])
    assert core == [
        arm for arm in full if arm not in direction and arm not in {"patch_tau", "patch_rate"}
    ]
    assert full[-2:] == ["patch_tau", "patch_rate"]
    arms = read_yaml(repo_root() / "configs" / "arms" / "code.yaml")
    assert arms["include"] == "kit.yaml"
    assert arms["domain"] == "code"
    names = [item["name"] for item in arms["arms"]]
    assert names == ["natural", "noise_natural", "patch_tau", "patch_rate"]
    assert arms["arms"][1]["support"] == "answered"
    task_spec = read_yaml(repo_root() / "configs" / "tasks" / "code.yaml")
    assert task_spec["revision"] == TACO_REVISION
    assert task_spec["min_tests"] == MIN_TESTS
    assert task_spec["max_tests"] == MAX_TESTS
    assert task_spec["bucket"] == "easy"
    assert task_spec["buckets"] == {"easy": ["EASY"], "medium": ["MEDIUM", "MEDIUM_HARD"]}
    assert task_spec["dev_fraction"] == 0.20
    assert "backup_dataset" not in task_spec
    assert "backup_revision" not in task_spec
    assert "cf_rating_ceiling" not in task_spec
    audit_task = template["audit"]["task"]
    assert audit_task["bucket"] == "easy"
    assert audit_task["buckets"] == {"easy": ["EASY"], "medium": ["MEDIUM", "MEDIUM_HARD"]}
    assert audit_task["dev_fraction"] == 0.20
    assert "difficulties" not in audit_task


def test_public_output_is_replaced_from_the_full_hidden_list() -> None:
    question = (
        "Add two integers.\n\nExample\nInput\n1  2\nOutput\n99\n\nExample\nInput\n8 8\nOutput\n16\n"
    )
    inputs = ["1 2\n", "1   2\n", "3 4\n", "5 6\n", "7 8\n"]
    outputs = ["3\n", "SHOULD_NOT_WIN\n", "7\n", "11\n", "15\n"]
    kept = cap_test_indices(len(inputs), problem_id_for(question), 1)
    match_at = next(index for index in range(len(inputs)) if index not in kept)
    inputs[match_at] = "1 2\n"
    outputs[match_at] = "3\n"
    if 0 in kept and match_at != 0:
        inputs[0] = "9 9\n"
        outputs[0] = "18\n"
    row = {
        "question": question,
        "difficulty": "EASY",
        "starter_code": "",
        "input_output": {"inputs": inputs, "outputs": outputs},
    }
    record = prepare_record(row, **{**_OPTS, "max_tests": 1})
    assert record is not None
    assert record["public_tests"][0]["input"] == "1  2\n"
    assert record["public_tests"][0]["output"] == "3\n"
    assert record["public_tests"][1] == {"input": "8 8\n", "output": "16\n"}
    assert record["public_source"] == {"dataset": 1, "statement": 1}
    assert match_at not in kept
    assert record["tests"][0]["output"] != "3\n" or record["tests"][0]["input"] != "1 2\n"
    direct, counts = align_public_outputs(
        [{"input": "1  2\n", "output": "99\n"}, {"input": "8 8\n", "output": "16\n"}],
        [{"input": "1 2\n", "output": "3\n"}, {"input": "1   2\n", "output": "SHOULD_NOT_WIN\n"}],
    )
    assert direct[0] == {"input": "1  2\n", "output": "3\n"}
    assert direct[1] == {"input": "8 8\n", "output": "16\n"}
    assert counts == {"dataset": 1, "statement": 1}


def test_raising_dev_fraction_does_not_move_heldout() -> None:
    ids = [f"problem-{index}" for index in range(400)]
    held_low = {pid for pid in ids if assign_split(pid, "code/split", 0.20, 0.10) == "heldout"}
    held_high = {pid for pid in ids if assign_split(pid, "code/split", 0.20, 0.20) == "heldout"}
    assert held_low == held_high
    assert len(held_low) == len(held_high)


def test_taco_load_failure_names_the_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: CodeTask, dataset: str, revision: str, split: str) -> list[dict[str, object]]:
        del self, dataset, revision, split
        raise OSError("shard missing")

    monkeypatch.setattr(CodeTask, "_load_dataset_rows", boom)
    task = CodeTask()
    with pytest.raises(RuntimeError, match=TACO_DATASET) as caught:
        task._source_rows()
    message = str(caught.value)
    assert TACO_REVISION in message
    assert "OSError" in message
    assert "shard missing" in message
    assert "code_contests" not in message
    assert "apps" not in message
    fingerprint = task.fingerprint()
    assert fingerprint["dev_fraction"] == 0.20
    assert "backup" not in fingerprint
    assert "cf_rating_ceiling" not in fingerprint


def test_bucket_medium_resolves_and_is_in_the_fingerprint() -> None:
    task = CodeTask(rows=[], bucket="medium")
    assert task.bucket == "medium"
    assert task.difficulties == ("MEDIUM", "MEDIUM_HARD")
    assert task.fingerprint()["difficulties"] == ["MEDIUM", "MEDIUM_HARD"]
    assert task.fingerprint()["bucket"] == "medium"
    easy = CodeTask(rows=[], bucket="easy", buckets=BUCKETS)
    assert easy.difficulties == ("EASY",)
    direct = CodeTask(rows=[], difficulties=("HARD",))
    assert direct.bucket is None
    assert direct.difficulties == ("HARD",)
    with pytest.raises(ValueError):
        CodeTask(rows=[], bucket="hard")


def test_bucket_medium_keeps_the_union() -> None:
    rows = []
    for index, difficulty in enumerate(("EASY", "MEDIUM", "MEDIUM_HARD", "HARD")):
        question = f"Problem {difficulty} {index}.\n\nExample\nInput\n{index}\nOutput\n{index}\n"
        rows.append(
            {
                "question": question,
                "difficulty": difficulty,
                "starter_code": "",
                "input_output": {
                    "inputs": [f"{index}\n"] * 6,
                    "outputs": [f"{index}\n"] * 6,
                },
            }
        )
    task = CodeTask(rows=rows, bucket="medium", min_tests=5)
    found = (
        task.problems("train", 10, 0)
        + task.problems("heldout", 10, 0)
        + task.problems("dev", 10, 0)
    )
    difficulties = {problem.meta["difficulty"] for problem in found}
    assert difficulties == {"MEDIUM", "MEDIUM_HARD"}
    assert len(found) == 2
    assert all(problem.meta["public_source"] == {"dataset": 1, "statement": 0} for problem in found)


# The suffix is not in the submission. A leak prints the whole marker and the oracle would pass.
_LEAK_PREFIX = "VFATES_ORACLE_LEAK_"
_LEAK_SECRET = _LEAK_PREFIX + "7e1c9b2a4d"

_MEMORY_HUNT = r"""
import gc
import sys

prefix = "VFATES_ORACLE_LEAK_"


def grab(text):
    idx = 0
    while True:
        found = text.find(prefix, idx)
        if found < 0:
            return None
        rest = text[found + len(prefix) :]
        if rest[:1].isalnum():
            end = 0
            while end < len(rest) and rest[end].isalnum():
                end += 1
            if end:
                return prefix + rest[:end]
        idx = found + len(prefix)


def consider(obj, found):
    if found:
        return
    if isinstance(obj, str):
        hit = grab(obj)
        if hit:
            found.append(hit)
    elif isinstance(obj, (bytes, bytearray)):
        hit = grab(bytes(obj).decode("utf-8", "ignore"))
        if hit:
            found.append(hit)


def scan_container(obj, found):
    consider(obj, found)
    if found:
        return
    if isinstance(obj, dict):
        for key, val in list(obj.items()):
            consider(key, found)
            consider(val, found)
            if found:
                return
    elif isinstance(obj, (list, tuple)):
        for item in list(obj):
            consider(item, found)
            if found:
                return


def hunt():
    found = []
    for obj in gc.get_objects():
        scan_container(obj, found)
        if found:
            break
    frame = sys._getframe()
    seen = set()
    while frame is not None and id(frame) not in seen and not found:
        seen.add(id(frame))
        scan_container(frame.f_locals, found)
        scan_container(frame.f_globals, found)
        frame = frame.f_back
    return found[0] if found else "missing"


print(hunt())
"""

_CALL_HUNT = _MEMORY_HUNT.replace("print(hunt())", "def solve():\n    return hunt()\n")

_FILE_PROBE = r"""
import os
import sys

own_path = os.path.join(os.getcwd(), "stdin.txt")
try:
    own = open(own_path, encoding="utf-8", errors="replace").read()
except OSError:
    own = ""
if "PROBE" not in own:
    sys.stdout.write("AAA_OUT_TOKEN\n")
else:
    needles = ("AAA_IN_TOKEN", "AAA_OUT_TOKEN", "BBB_OUT_TOKEN")
    roots = [os.path.dirname(os.getcwd())]
    grand = os.path.dirname(roots[0])
    if grand and grand not in roots:
        roots.append(grand)
    blob = []
    for root in roots:
        try:
            walked = list(os.walk(root))
        except OSError:
            continue
        for dirpath, _dirs, files in walked:
            for name in files:
                path = os.path.join(dirpath, name)
                if os.path.abspath(path) == os.path.abspath(own_path):
                    continue
                try:
                    blob.append(open(path, encoding="utf-8", errors="replace").read())
                except OSError:
                    continue
    text = "\n".join(blob)
    leaked = [needle for needle in needles if needle in text]
    sys.stdout.write("LEAK\n" if leaked else "clean\n")
"""


def test_stdio_cannot_read_expected_output_from_memory() -> None:
    hidden = grade_job(
        GradeJob(
            code=_MEMORY_HUNT,
            tests=(("0\n", _LEAK_SECRET + "\n"),),
            per_test_timeout_s=5,
            wall_timeout_s=20,
            memory_mb=256,
        )
    )
    assert hidden.n_ran == 1
    assert hidden.detail == "mismatch", hidden.detail
    assert not hidden.passed


def test_call_cannot_return_expected_output_from_memory() -> None:
    result = grade_job(
        GradeJob(
            code=_CALL_HUNT,
            fn_name="solve",
            tests=(("[]", json.dumps(_LEAK_SECRET)),),
            per_test_timeout_s=5,
            wall_timeout_s=20,
            memory_mb=256,
        )
    )
    assert result.n_ran == 1
    assert result.detail == "mismatch"
    assert not result.passed


def test_oversized_integer_in_input_output_is_dropped() -> None:
    text = '{"inputs": [' + ("9" * 5000) + '], "outputs": [1]}'
    row = {
        "question": "Read an integer.\n\nInput\n1\n\nOutput\n1\n",
        "difficulty": "EASY",
        "starter_code": "",
        "input_output": text,
    }
    assert (
        prepare_record(
            row,
            difficulties=("EASY",),
            min_tests=1,
            max_tests=MAX_TESTS,
            python_only=True,
            split_salt="code/split",
            heldout_fraction=0.2,
            dev_fraction=0.1,
            require_public_example=False,
            dataset="BAAI/TACO",
            revision=TACO_REVISION,
        )
        is None
    )


def test_sibling_tests_are_not_files_beside_the_workdir() -> None:
    result = grade_job(
        GradeJob(
            code=_FILE_PROBE,
            tests=(
                ("AAA_IN_TOKEN\n", "AAA_OUT_TOKEN\n"),
                ("PROBE\nBBB_IN_TOKEN\n", "clean\n"),
            ),
            per_test_timeout_s=5,
            wall_timeout_s=20,
            memory_mb=256,
            early_exit=False,
        )
    )
    assert result.n_ran == 2
    assert result.passed, result.detail


def _only(task: CodeTask) -> Problem:
    found = (
        task.problems("train", 5, 0) + task.problems("heldout", 5, 0) + task.problems("dev", 5, 0)
    )
    assert len(found) == 1
    return found[0]


@pytest.mark.network
def test_taco_loads_20_rows() -> None:
    rows = load_taco_rows(20, revision=TACO_REVISION)
    assert len(rows) == 20
    assert all(str(row.get("question") or "").strip() for row in rows)
    task = CodeTask(
        rows=rows,
        difficulties=("EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"),
        min_tests=1,
        require_public_example=False,
        n_workers=1,
    )
    try:
        loaded = (
            task.problems("train", 20, 0)
            + task.problems("heldout", 20, 0)
            + task.problems("dev", 20, 0)
        )
    finally:
        task.close()
    assert len(loaded) <= 20
