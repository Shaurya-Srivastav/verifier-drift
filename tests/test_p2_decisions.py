"""The preregistered P2 bucket rule, on synthetic measurements."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from vfates.pilots.p2_base_audit import (
    ACC_HI,
    ACC_LO,
    ARM_TOL,
    FORMAT_MIN,
    P2_SCHEMA,
    PROMPTS_PER_BUCKET,
    SAMPLES_PER_PROMPT,
    STUDY_MODELS,
    TARGET_ACC,
    TRUNC_MAX,
    _both,
    accuracy_se_bounds,
    bucket_qualifies,
    decide_domain,
    decisions_from_measurements,
    domain_plans,
    next_budget,
    run,
    screen_source,
    take_dev,
    targeted_rate_gap,
    template_edits,
)
from vfates.utils.io import read_json


def _block(
    accuracy: float, truncation: float = 0.01, format_rate: float = 0.9, length: float = 80.0
):
    return _both(accuracy, truncation, format_rate, length)


def test_sample_size_bounds_the_standard_error() -> None:
    bounds = accuracy_se_bounds(PROMPTS_PER_BUCKET, SAMPLES_PER_PROMPT)
    assert PROMPTS_PER_BUCKET == 625
    assert SAMPLES_PER_PROMPT == 4
    assert bounds["prompt_mean_se_bound"] == pytest.approx(0.02)
    assert bounds["pooled_se_bound"] == pytest.approx(0.01)
    with pytest.raises(ValueError):
        accuracy_se_bounds(0, 4)


def test_budget_ladder() -> None:
    assert next_budget(768) == 1024
    assert next_budget(1024) == 1536
    assert next_budget(1536) == 2048
    assert next_budget(2048) is None
    assert next_budget(3000) is None


def test_bucket_band() -> None:
    assert bucket_qualifies(_block(0.15))
    assert bucket_qualifies(_block(0.55))
    assert bucket_qualifies(_block(TARGET_ACC))
    assert not bucket_qualifies(_block(ACC_LO - 0.01))
    assert not bucket_qualifies(_block(ACC_HI + 0.01))
    assert not bucket_qualifies(_block(0.3, truncation=TRUNC_MAX + 0.01))
    assert not bucket_qualifies(_block(0.3, format_rate=FORMAT_MIN - 0.01))
    one = _block(0.3)
    one["qwen3-4b"] = {**one["qwen3-4b"], "accuracy": 0.9}
    assert not bucket_qualifies(one)
    assert set(STUDY_MODELS) == {"qwen3-1.7b", "qwen3-4b"}


def test_tie_goes_to_the_mean_closest_to_0_35_then_earlier_bucket() -> None:
    buckets = ["low", "mid", "high"]
    measurements = {
        1024: {
            "low": _block(0.20, length=40),
            "mid": _block(0.30, length=90),
            "high": _block(0.50, length=120),
        }
    }
    hit = screen_source(measurements, buckets, 1024)
    assert hit is not None
    assert hit["bucket"] == "mid"
    assert hit["mean_completion_len"] == pytest.approx(90)
    tied = {
        1024: {
            "low": _block(0.20),
            "high": _block(0.50),
        }
    }
    tied_hit = screen_source(tied, ["low", "high"], 1024)
    assert tied_hit is not None
    assert tied_hit["bucket"] == "low"


def test_truncation_raises_the_budget_and_stops_at_2048() -> None:
    buckets = ["only"]
    measurements = {
        768: {"only": _block(0.1, truncation=0.2)},
        1024: {"only": _block(0.1, truncation=0.2)},
        1536: {"only": _block(0.1, truncation=0.2)},
        2048: {"only": _block(0.4, truncation=0.01, length=200)},
    }
    hit = screen_source(measurements, buckets, 768)
    assert hit is not None
    assert hit["token_budget"] == 2048
    assert hit["budgets_tried"] == [768, 1024, 1536, 2048]
    stuck = {
        1024: {"only": _block(0.1, truncation=0.2)},
        1536: {"only": _block(0.1, truncation=0.2)},
        2048: {"only": _block(0.1, truncation=0.2)},
    }
    assert screen_source(stuck, buckets, 1024) is None


def test_no_raise_when_nothing_truncates() -> None:
    measurements = {1024: {"only": _block(0.05, truncation=0.0)}}
    assert screen_source(measurements, ["only"], 1024) is None


def test_qualifying_bucket_keeps_its_budget() -> None:
    measurements = {
        1024: {
            "easy": _block(0.4, truncation=0.0),
            "hard": _block(0.1, truncation=0.4),
        }
    }
    hit = screen_source(measurements, ["easy", "hard"], 1024)
    assert hit is not None
    assert hit["bucket"] == "easy"
    assert hit["token_budget"] == 1024


def test_backup_and_leaves_program() -> None:
    primary = {1024: {"fixed": _block(0.05, truncation=0.0)}}
    backup = {1024: {"all": _block(0.33, length=70)}}
    chosen = decide_domain(
        primary_buckets=["fixed"],
        backup_buckets=["all"],
        start_budget=1024,
        primary=primary,
        backup=backup,
        has_backup=True,
    )
    assert chosen["status"] == "qualified"
    assert chosen["source"] == "backup"
    assert chosen["bucket"] == "all"
    left = decide_domain(
        primary_buckets=["fixed"],
        backup_buckets=["all"],
        start_budget=1024,
        primary=primary,
        backup={1024: {"all": _block(0.9)}},
        has_backup=True,
    )
    assert left["status"] == "leaves_program"
    assert left["source"] is None
    assert "backup" in str(left["reason"])
    no_backup = decide_domain(
        primary_buckets=["fixed"],
        backup_buckets=None,
        start_budget=1024,
        primary=primary,
        backup=None,
        has_backup=False,
    )
    assert no_backup["status"] == "leaves_program"
    assert "no backup" in str(no_backup["reason"])
    with pytest.raises(ValueError, match="backup screen"):
        decide_domain(
            primary_buckets=["fixed"],
            backup_buckets=["all"],
            start_budget=1024,
            primary=primary,
            backup=None,
            has_backup=True,
        )


def test_template_edits_name_the_difficulty_key() -> None:
    plans = domain_plans()
    assert plans["chain_sum"]["backup"] is None
    assert plans["qa"]["backup"] is None
    assert plans["ifeval"]["backup"]["dev"] is True
    assert plans["ifeval"]["backup"]["buckets"][0]["task"]["source"] == "rlvr_ifeval"
    assert [row["name"] for row in plans["code"]["buckets"]] == ["easy", "medium"]
    assert plans["code"]["buckets"][0]["difficulty_value"] == ["EASY"]
    assert plans["code"]["buckets"][1]["difficulty_value"] == ["MEDIUM", "MEDIUM_HARD"]
    assert plans["code"]["buckets"][1]["task"]["difficulties"] == ["MEDIUM", "MEDIUM_HARD"]
    assert plans["code"]["backup"] is None
    assert plans["math"]["backup"]["dev"] is True
    assert plans["ifeval"]["buckets"][0]["difficulty_value"] == [2, 3]
    assert plans["qa"]["buckets"][0]["name"] == "Primary School"
    math_names = [row["name"] for row in plans["math"]["buckets"]]
    assert math_names[0] == "0.0_0.2"
    decision = {
        "status": "qualified",
        "source": "primary",
        "bucket": "easy",
        "token_budget": 1536,
        "mean_completion_len": 88.4,
    }
    edits = template_edits("code", decision, plans["code"])
    assert edits is not None
    assert edits["difficulty_key"] == "audit.task.difficulties"
    assert edits["difficulty_value"] == ["EASY"]
    assert edits["max_completion_tokens"] == 1536
    assert edits["cost_model.mean_completion_len"] == 88
    assert template_edits("code", {"status": "leaves_program"}, plans["code"]) is None
    backup = {
        "status": "qualified",
        "source": "backup",
        "bucket": "all",
        "token_budget": 2048,
        "mean_completion_len": 10.2,
    }
    math_edits = template_edits("math", backup, plans["math"])
    assert math_edits is not None
    assert math_edits["task_overrides"]["source"] == "deepscaler"
    assert math_edits["difficulty_value"] is None
    assert "task_overrides" in math_edits


def test_targeted_gap_ignores_capped_form_rules() -> None:
    rows = [
        {"type": "hash_fp", "abs_fpr_error": 0.01, "abs_fnr_error": 0.0},
        {"type": "form_fn", "abs_fpr_error": 0.5, "abs_fnr_error": 0.5},
        {"type": "behavior_fn", "abs_fpr_error": ARM_TOL, "abs_fnr_error": 0.0},
    ]
    assert targeted_rate_gap(rows) == pytest.approx(ARM_TOL)
    assert targeted_rate_gap([{"type": "oracle"}]) is None


def test_take_dev_shrinks_a_short_split() -> None:
    class Task:
        def problems(self, split: str, n: int, seed: int) -> list[int]:
            assert split == "dev"
            if n > 3:
                raise ValueError(f"math split {split!r} has 3 problems, asked for {n}")
            return list(range(n))

    assert take_dev(Task(), 10, 0) == [0, 1, 2]
    assert take_dev(Task(), 2, 0) == [0, 1]


def test_run_writes_decisions_from_measurements(tmp_path: Path) -> None:
    plans = domain_plans()
    measurements = {}
    for domain, plan in plans.items():
        block = {}
        for bucket in plan["buckets"]:
            block[bucket["name"]] = _block(0.34, length=64)
        measurements[domain] = {int(plan["start_budget"]): None}
        measurements[domain] = {"primary": {int(plan["start_budget"]): block}}
    report = run(
        {
            "smoke": False,
            "hardware_class": "cpu",
            "measurements": measurements,
            "arm_gaps": {domain: 0.01 for domain in plans},
        },
        tmp_path,
    )
    assert report["ok"] is True
    document = read_json(tmp_path / "decisions.json")
    assert document["schema"] == P2_SCHEMA
    assert document["apply"] is False
    assert set(document["domains"]) == set(plans)
    for domain, decision in document["domains"].items():
        assert decision["status"] == "qualified"
        assert decision["source"] == "primary"
        assert decision["template_edits"]["max_completion_tokens"] == plans[domain]["start_budget"]
    ids = {row["id"] for row in report["checks"]}
    assert "P2.1" in ids
    assert "P2.3.chain_sum" in ids
    documented = next(row for row in report["checks"] if row["id"] == "P2.3.math")
    assert documented["status"] == "documented"
    assert documented["passed"] is True


def test_short_split_records_the_standard_error_from_n_used() -> None:
    plans = domain_plans()
    measurements = {}
    for domain, plan in plans.items():
        block = {}
        for bucket in plan["buckets"]:
            name = bucket["name"]
            used = 295 if domain == "code" and name == "easy" else 625
            row = _block(0.34)
            for model in row:
                row[model] = {**row[model], "n_prompts": used, "n_samples": 4}
            block[name] = row
        measurements[domain] = {"primary": {int(plan["start_budget"]): block}}
    domains = decisions_from_measurements(measurements, requested_prompts=625, n_samples=4)
    easy = domains["code"]["samples"]["primary"]["easy"]
    assert easy["n_prompts"] == 295
    assert easy["requested_prompts"] == 625
    assert easy["prompt_mean_se_bound"] == pytest.approx(0.5 / math.sqrt(295))
    assert easy["pooled_se_bound"] == pytest.approx(0.5 / math.sqrt(295 * 4))
    medium = domains["code"]["samples"]["primary"]["medium"]
    assert medium["n_prompts"] == 625
    assert medium["prompt_mean_se_bound"] == pytest.approx(0.02)
    assert "samples" in domains["chain_sum"]


def test_smoke_does_not_apply_a_decision(tmp_path: Path) -> None:
    report = run({"smoke": True, "hardware_class": "apple-silicon"}, tmp_path)
    assert report["ok"] is True
    assert not (tmp_path / "decisions.json").exists()
    row = next(item for item in report["checks"] if item["id"] == "P2.1")
    assert row["status"] == "documented"
