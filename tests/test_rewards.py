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
    only_correct = onpolicy_rates([{"O": 1, "V": 1, "monitors": {}}])
    assert only_correct["fpr"] is None
    assert only_correct["J"] is None
    assert only_correct["fnr"] == 0.0


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
