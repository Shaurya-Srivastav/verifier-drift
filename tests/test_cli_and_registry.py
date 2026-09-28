"""CLI entry points, task construction, and behaviour monitors."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from vfates.cli import PILOTS, main
from vfates.tasks.chain_sum import ChainSumTask
from vfates.tasks.registry import build_task
from vfates.verifiers.arms import (
    BehaviorFN,
    HashFP,
    NumericToleranceFP,
    OracleVerifier,
    ResampledNoise,
)
from vfates.verifiers.base import Verifier
from vfates.verifiers.monitors import rule_monitors
from vfates.verifiers.registry import build_arms

ROOT = Path(__file__).resolve().parents[1]


def test_env_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "hardware" in payload
    assert "versions" in payload


def test_pilot_help_exits() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["pilot", "--help"])
    assert caught.value.code == 0
    with pytest.raises(SystemExit) as caught:
        main(["pilot", "not-a-pilot"])
    assert caught.value.code == 2


def test_pilots_map_to_existing_scripts() -> None:
    assert PILOTS
    for folder in PILOTS.values():
        assert (ROOT / "pilots" / folder / "run.py").is_file()


def test_build_task_chain_sum_overrides_and_unknown_type() -> None:
    task = build_task(
        {
            "type": "chain_sum",
            "min_terms": 2,
            "max_terms": 2,
            "min_digits": 1,
            "max_digits": 1,
            "min_decimal_places": 1,
            "max_decimal_places": 1,
            "ignored": True,
        }
    )
    assert isinstance(task, ChainSumTask)
    assert task.config.min_terms == 2
    assert task.config.max_terms == 2
    assert task.config.max_digits == 1
    assert task.config.min_decimal_places == 1
    assert task.config.max_decimal_places == 1
    problems = task.problems("train", 2, seed=0)
    assert len(problems) == 2
    assert all(problem.meta["num_terms"] == 2 for problem in problems)
    with pytest.raises(ValueError, match="unknown task type"):
        build_task({"type": "not-a-task"})


def _check_monitors(task: ChainSumTask, arms: dict[str, Verifier]) -> None:
    monitors = rule_monitors(arms)
    assert set(monitors) == {"has_answer", "fp_hash", "fp_tolerance", "fn_behavior"}
    assert "oracle" not in monitors
    assert "noise_fp" not in monitors
    problem = task.problems("train", 1, seed=2)[0]
    gold = Decimal(problem.answer)
    near_text = f"\\boxed{{{format(gold + Decimal('0.01'), 'f')}}}"
    far_text = f"\\boxed{{{format(gold + Decimal('50'), 'f')}}}"
    gold_text = f"\\boxed{{{problem.answer}}}"
    because_text = f"because {gold_text}"
    blank_text = "no box"
    near = task.parse(problem, near_text)
    far = task.parse(problem, far_text)
    parsed_gold = task.parse(problem, gold_text)
    because = task.parse(problem, because_text)
    blank = task.parse(problem, blank_text)
    assert monitors["has_answer"](problem, parsed_gold, gold_text) is True
    assert monitors["has_answer"](problem, blank, blank_text) is False
    assert monitors["fp_hash"](problem, far, far_text) is True
    assert monitors["fp_hash"](problem, blank, blank_text) is False
    assert monitors["fp_hash"](problem, far, far_text) is arms["fp_hash"].accepts(
        problem, far, far_text
    )
    assert monitors["fp_tolerance"](problem, near, near_text) is True
    assert monitors["fp_tolerance"](problem, far, far_text) is False
    assert monitors["fp_tolerance"](problem, near, near_text) is arms["fp_tolerance"].accepts(
        problem, near, near_text
    )
    assert monitors["fn_behavior"](problem, because, because_text) is True
    assert monitors["fn_behavior"](problem, parsed_gold, gold_text) is False
    assert monitors["fn_behavior"](problem, because, because_text) is arms["fn_behavior"].rejects(
        problem, because, because_text
    )


def test_rule_monitors_from_constructed_arms() -> None:
    task = ChainSumTask()
    arms = {
        "oracle": OracleVerifier(name="oracle"),
        "fp_hash": HashFP("salt", 1.0, name="fp_hash"),
        "fp_tolerance": NumericToleranceFP("0.05", name="fp_tolerance"),
        "fn_behavior": BehaviorFN("because", name="fn_behavior"),
        "noise_fp": ResampledNoise(0.2, 0.3, seed=0, name="noise_fp", support="answered"),
    }
    _check_monitors(task, arms)


def test_rule_monitors_from_build_arms() -> None:
    task = ChainSumTask()
    arms = build_arms(
        [
            {"name": "oracle", "type": "oracle"},
            {"name": "fp_hash", "type": "hash_fp", "salt": "salt", "threshold": 1.0},
            {"name": "fp_tolerance", "type": "tolerance_fp", "abs_tol": "0.05"},
            {"name": "fn_behavior", "type": "behavior_fn", "pattern": "because"},
            {
                "name": "noise_fp",
                "type": "resampled",
                "rho0": 0.2,
                "rho1": 0.3,
                "seed": 0,
                "support": "answered",
            },
        ],
        task,
    )
    _check_monitors(task, arms)
