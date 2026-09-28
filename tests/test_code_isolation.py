"""Filesystem denials around the code-grading worker."""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

from vfates.tasks.code import CodeTask
from vfates.tasks.sandbox import GradeJob, grade_job
from vfates.train.rewards import RewardContext
from vfates.types import Problem
from vfates.verifiers.arms import OracleVerifier

_LEGIT = textwrap.dedent(
    """
    import bisect
    import collections
    import decimal
    import fractions
    import functools
    import heapq
    import itertools
    import math
    import re
    import sys
    import threading

    threading.stack_size(2**26)
    sys.setrecursionlimit(10000)
    heap = [3, 1]
    heapq.heapify(heap)
    collections.deque([1])
    bisect.bisect_left([1, 2], 1)
    next(itertools.islice(itertools.count(), 1))

    @functools.lru_cache(8)
    def cached(n):
        return n

    cached(1)
    re.compile("a")
    fractions.Fraction(1, 2)
    decimal.Decimal("1")
    math.sqrt(4)
    sys.stdin.buffer.read()
    print("ok")
    """
)


def _grade(code: str, expected: str = "ok\n", stdin: str = "1\n") -> object:
    return grade_job(
        GradeJob(
            code=code,
            tests=((stdin, expected),),
            per_test_timeout_s=5,
            wall_timeout_s=20,
            memory_mb=256,
        )
    )


def test_open_of_answer_file_is_denied(tmp_path: object) -> None:
    path = os.path.join(str(tmp_path), "problems.jsonl")
    secret = "EXPECTED_OUTPUT_LINE\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(secret)
    code = f"print(open({path!r}, encoding='utf-8').read(), end='')"
    result = _grade(code, expected=secret, stdin="0\n")
    assert not result.passed
    assert result.n_denied >= 1
    assert result.denied
    assert result.fs_isolation == "audit_hook" or result.fs_isolation == "landlock+audit_hook"


def test_listing_root_and_home_is_denied() -> None:
    home = os.path.expanduser("~")
    code = (
        "import os\n"
        "try:\n"
        "    os.listdir('/')\n"
        "except OSError:\n"
        "    pass\n"
        "try:\n"
        f"    list(os.walk({home!r}))\n"
        "except OSError:\n"
        "    pass\n"
        "print('ok')\n"
    )
    result = _grade(code)
    assert result.passed, result.detail
    assert result.n_denied >= 2
    events = [event for event, _arg in result.denied]
    assert "os.listdir" in events
    assert "os.scandir" in events


def test_import_ctypes_is_denied() -> None:
    result = _grade("import ctypes\nprint(ctypes.sizeof(ctypes.c_int))\n")
    assert not result.passed
    assert result.n_denied >= 1
    assert any(event == "import" and "ctypes" in arg for event, arg in result.denied)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is not mounted on this host")
def test_proc_parent_mem_and_fds_are_denied() -> None:
    code = (
        "import os\n"
        "ppid = os.getppid()\n"
        "open('/proc/%d/mem' % ppid, 'rb').read(1)\n"
        "os.listdir('/proc/%d/fd' % ppid)\n"
        "print('ok')\n"
    )
    result = _grade(code)
    assert not result.passed
    assert result.n_denied >= 1


def test_subprocess_cat_is_denied(tmp_path: object) -> None:
    path = os.path.join(str(tmp_path), "problems.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("secret\n")
    code = (
        "import subprocess\n"
        f"print(subprocess.run(['cat', {path!r}], capture_output=True, text=True).stdout, end='')\n"
    )
    result = _grade(code, expected="secret\n", stdin="0\n")
    assert not result.passed
    assert result.n_denied >= 1


def test_legitimate_stdlib_solution_passes() -> None:
    result = _grade(_LEGIT)
    assert result.passed, result.detail
    assert result.n_denied == 0
    assert result.fs_isolation in {"audit_hook", "landlock+audit_hook"}


def test_n_denied_reaches_reward_metadata() -> None:
    question = "Read nothing and print ok.\n\nExample\nInput\n1\nOutput\nok\n"
    inputs = ["1\n"] * 6
    outputs = ["ok\n"] * 6
    task = CodeTask(
        rows=[
            {
                "question": question,
                "difficulty": "EASY",
                "starter_code": "",
                "input_output": {"inputs": inputs, "outputs": outputs},
            }
        ],
        n_workers=1,
        min_tests=5,
    )
    try:
        problem = task.problems("train", 1, 0) or task.problems("heldout", 1, 0)
        problem = problem or task.problems("dev", 1, 0)
        assert problem
        one: Problem = problem[0]
        attack = "```python\ntry:\n    import ctypes\nexcept Exception:\n    pass\nprint('ok')\n```"
        clean = "```python\nprint('ok')\n```"
        ctx = RewardContext(task, OracleVerifier(), {one.pid: one}, seed=0)
        records = ctx.score_batch(
            [one.pid, one.pid],
            [attack, clean],
            [ctx.draw_key(1, one.pid, 0), ctx.draw_key(1, one.pid, 1)],
        )
        assert records[0]["O"] == 1
        assert records[0]["n_denied"] >= 1
        assert records[0]["denied"]
        assert records[1]["O"] == 1
        assert records[1]["n_denied"] == 0
    finally:
        task.close()
