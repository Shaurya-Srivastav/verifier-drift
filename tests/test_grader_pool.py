"""GraderPool matches the serial grader and keeps result order."""

from __future__ import annotations

import time

import pytest

from vfates.tasks.sandbox import (
    GradeJob,
    GradeResult,
    GraderPool,
    grade_job,
    measure_throughput,
)

_SUM = "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()))\n"


def _pack(result: GradeResult) -> tuple[object, ...]:
    return (
        result.passed,
        result.n_tests,
        result.n_ran,
        result.first_fail,
        result.detail,
        result.timed_out,
    )


def _jobs() -> list[GradeJob]:
    tests = tuple((f"{index} {index + 1}\n", f"{2 * index + 1}\n") for index in range(4))
    echo = "import sys\ndata = sys.stdin.read().split()\nprint(data[0] if data else '')\n"
    return [
        GradeJob(code=_SUM, tests=tests, per_test_timeout_s=2, wall_timeout_s=10, memory_mb=256),
        GradeJob(
            code="import sys\nprint(0)\n",
            tests=tests,
            per_test_timeout_s=2,
            wall_timeout_s=10,
            memory_mb=256,
        ),
        GradeJob(
            code="def add(a, b):\n    return a + b\n",
            fn_name="add",
            tests=(("[1, 2]", "3"), ("[4, 5]", "9")),
            per_test_timeout_s=2,
            wall_timeout_s=10,
            memory_mb=256,
        ),
        GradeJob(
            code=echo,
            tests=(("1 2\n", "1\n"), ("3 4\n", "9\n")),
            per_test_timeout_s=2,
            wall_timeout_s=10,
            memory_mb=256,
        ),
    ]


def test_pool_matches_serial_twice() -> None:
    jobs = _jobs()
    serial = [_pack(grade_job(job)) for job in jobs]
    with GraderPool(n_workers=2) as pool:
        first = [_pack(result) for result in pool.grade_many(jobs)]
        second = [_pack(result) for result in pool.grade_many(jobs)]
    assert first == serial
    assert second == first
    assert serial[1][2] == 1  # n_ran, early exit
    assert serial[3][3] == 1  # first_fail


def test_early_exit_does_not_run_a_later_hang() -> None:
    code = (
        "import sys\n"
        "data = sys.stdin.read().strip()\n"
        "if data == 'bad':\n"
        "    print('nope')\n"
        "else:\n"
        "    while True:\n"
        "        pass\n"
    )
    job = GradeJob(
        code=code,
        tests=(("bad\n", "yes\n"), ("ok\n", "yes\n")),
        per_test_timeout_s=2,
        wall_timeout_s=8,
        memory_mb=256,
        early_exit=True,
    )
    started = time.monotonic()
    with GraderPool(n_workers=1) as pool:
        result = pool.grade_many([job])[0]
    assert time.monotonic() - started < 4
    assert result.n_ran == 1
    assert result.first_fail == 0
    assert result.detail == "mismatch"
    assert not result.timed_out


def test_empty_batches_do_not_start_workers() -> None:
    with GraderPool(n_workers=1) as pool:
        assert pool.grade_many([]) == []


@pytest.mark.slow
def test_throughput_512_by_16() -> None:
    stats = measure_throughput(512, 16)
    assert stats["passed"] == 512
    assert stats["n_tests"] == 16
    assert stats["programs_per_s"] > 1
