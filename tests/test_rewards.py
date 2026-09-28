"""Reward indicators, pooled rates, and the TRL reward-function adapter."""

from __future__ import annotations

import pytest

from vfates.tasks.base import Task
from vfates.train.rewards import (
    RewardContext,
    make_trl_reward_funcs,
    onpolicy_rates,
    reward_weights_for,
)
from vfates.types import Parsed, Problem
from vfates.verifiers.base import Verifier


class _YesTask(Task):
    name = "yes"

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        del seed
        return [
            Problem(pid=f"{split}-{i}", split=split, task="yes", question="q", answer="yes")
            for i in range(n)
        ]

    def prompt(self, problem: Problem) -> str:
        return "Reply with a word."

    def parse(self, problem: Problem, text: str) -> Parsed:
        del problem
        word = text.strip().split()[0] if text.strip() else None
        return Parsed(answer=word, raw=text)

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        del problem, parsed
        return int("yes" in text.lower())


class _Flip(Verifier):
    stochastic = True
    name = "flip"

    def __init__(self) -> None:
        self.keys: list[object] = []

    def verdict(self, problem: Problem, parsed: Parsed, text: str, o: int, key=None) -> int:
        del problem, parsed, text, o
        self.keys.append(key)
        assert key is not None
        return int(int(key[-1]) % 2 == 0)


def _problem(pid: str = "p0") -> Problem:
    return Problem(pid=pid, split="train", task="yes", question="q", answer="yes")


def test_score_indicators_and_cache() -> None:
    verifier = _Flip()
    ctx = RewardContext(
        _YesTask(),
        verifier,
        {"p0": _problem()},
        seed=7,
        monitors={"yes_word": lambda p, parsed, text: "yes" in text.lower()},
    )
    record = ctx.score("p0", "yes please", ctx.draw_key(3, "p0", 0))
    assert record["O"] == 1 and record["V"] == 1
    assert record["FP"] == 0 and record["FN"] == 0
    assert record["has_answer"] is True
    assert record["monitors"]["yes_word"] is True
    again = ctx.score("p0", "yes please", ctx.draw_key(3, "p0", 0))
    assert again is record
    assert len(verifier.keys) == 1
    wrong = ctx.score("p0", "no", ctx.draw_key(3, "p0", 1))
    assert wrong["O"] == 0 and wrong["V"] == 0
    assert wrong["FP"] == 0 and wrong["FN"] == 0
    assert len(verifier.keys) == 2


def test_onpolicy_rates_hand() -> None:
    assert onpolicy_rates([])["n"] == 0
    assert onpolicy_rates([])["fpr"] is None
    records = [
        {"O": 0, "V": 1, "monitors": {"hit": True}},
        {"O": 0, "V": 0, "monitors": {"hit": False}},
        {"O": 1, "V": 1, "monitors": {"hit": True}},
    ]
    rates = onpolicy_rates(records)
    assert rates["fpr"] == 0.5
    assert rates["fnr"] == 0.0
    assert rates["precision"] == 0.5
    assert rates["J"] == 0.5
    assert rates["accept_rate"] == 2 / 3
    assert rates["oracle_acc"] == 1 / 3
    assert rates["monitors"]["hit"] == 2 / 3
    assert rates["monitors_wrong"]["hit"] == 0.5
    assert rates["monitors_correct"]["hit"] == 1.0
    only_correct = onpolicy_rates([{"O": 1, "V": 1, "monitors": {}}])
    assert only_correct["fpr"] is None
    assert only_correct["J"] is None
    assert only_correct["fnr"] == 0.0
    assert only_correct["monitors_correct"] == {}


def test_trl_reward_funcs_names_weights_and_keys() -> None:
    verifier = _Flip()
    ctx = RewardContext(
        _YesTask(),
        verifier,
        {"p0": _problem()},
        seed=7,
        monitors={"zebra": lambda p, parsed, text: False, "alpha": lambda p, parsed, text: True},
    )
    funcs = make_trl_reward_funcs(ctx)
    names = [fn.__name__ for fn in funcs]
    assert names == [
        "verifier",
        "oracle",
        "false_positive",
        "false_negative",
        "monitor_alpha",
        "monitor_zebra",
    ]
    assert reward_weights_for(len(funcs)) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class _State:
        global_step = 4

    completions = [
        [{"role": "assistant", "content": "yes"}],
        [{"role": "assistant", "content": "no"}],
    ]
    values = funcs[0](completions, pid=["p0", "p0"], trainer_state=_State())
    assert values == [1.0, 0.0]
    assert verifier.keys == [(7, 4, "p0", 0), (7, 4, "p0", 1)]
    oracle = funcs[1](completions, pid=["p0", "p0"], trainer_state=_State())
    assert oracle == [1.0, 0.0]
    with pytest.raises(ValueError, match="pid"):
        funcs[0](completions)


class _CountingTask(_YesTask):
    def __init__(self) -> None:
        self.oracle_calls = 0
        self.batch_sizes: list[int] = []

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        self.oracle_calls += 1
        return super().oracle(problem, parsed, text)

    def grade_batch(self, items: list[tuple[Problem, Parsed, str]]) -> list[int]:
        self.batch_sizes.append(len(items))
        return [int("yes" in text.lower()) for _problem, _parsed, text in items]


class _BatchFlip(_Flip):
    def verdict_batch(self, problems, parsed, texts, oracle_bits, keys) -> list[int]:  # type: ignore[no-untyped-def]
        del problems, parsed, texts, oracle_bits
        self.keys.extend(keys)
        return [int(int(key[-1]) % 2 == 0) for key in keys]


class _BoomOracle(Verifier):
    kind = "oracle"
    stochastic = False
    name = "oracle"

    def verdict(self, problem: Problem, parsed: Parsed, text: str, o: int, key=None) -> int:
        del problem, parsed, text, o, key
        raise AssertionError("oracle verifier must not be graded again")


def _fields(record: dict[str, object]) -> dict[str, object]:
    return {name: record[name] for name in ("V", "O", "FP", "FN", "has_answer")}


def test_score_batch_matches_serial_and_skips_cache_hits() -> None:
    task = _CountingTask()
    verifier = _BatchFlip()
    ctx = RewardContext(task, verifier, {"p0": _problem()}, seed=3)
    keys = [ctx.draw_key(1, "p0", index) for index in range(3)]
    texts = ["yes please", "no", "yes"]
    pids = ["p0", "p0", "p0"]
    batched = ctx.score_batch(pids, texts, keys)
    serial_task = _YesTask()
    serial = RewardContext(serial_task, _Flip(), {"p0": _problem()}, seed=3)
    one_by_one = [serial.score(pid, text, key) for pid, text, key in zip(pids, texts, keys)]
    assert [_fields(row) for row in batched] == [_fields(row) for row in one_by_one]
    assert task.oracle_calls == 0
    assert task.batch_sizes == [3]
    ctx.score_batch(pids[:2], texts[:2], keys[:2])
    assert task.batch_sizes == [3]
    ctx.score_batch(["p0", "p0"], ["yes please", "maybe"], [keys[0], ctx.draw_key(1, "p0", 4)])
    assert task.batch_sizes == [3, 1]


def test_score_batch_oracle_verifier_copies_the_oracle_bit() -> None:
    task = _CountingTask()
    ctx = RewardContext(task, _BoomOracle(), {"p0": _problem()}, seed=0)
    record = ctx.score_batch(["p0"], ["yes please"], [None])[0]
    assert record["O"] == 1 and record["V"] == 1
    assert record["FP"] == 0 and record["FN"] == 0
    assert task.oracle_calls == 0
    assert task.batch_sizes == [1]


def test_score_batch_rejects_a_short_grade_batch() -> None:
    class _Short(_CountingTask):
        def grade_batch(self, items: list[tuple[Problem, Parsed, str]]) -> list[int]:
            del items
            return []

    ctx = RewardContext(_Short(), _BoomOracle(), {"p0": _problem()}, seed=0)
    with pytest.raises(RuntimeError, match="grade_batch"):
        ctx.score_batch(["p0"], ["yes"], [None])


def test_score_batch_without_grade_batch_is_serial() -> None:
    verifier = _Flip()
    ctx = RewardContext(_YesTask(), verifier, {"p0": _problem()}, seed=1)
    batched = ctx.score_batch(
        ["p0", "p0"],
        ["yes", "no"],
        [ctx.draw_key(2, "p0", 0), ctx.draw_key(2, "p0", 1)],
    )
    assert batched[0]["O"] == 1 and batched[1]["O"] == 0
    assert verifier.keys == [(1, 2, "p0", 0), (1, 2, "p0", 1)]
