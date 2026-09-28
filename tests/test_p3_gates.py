"""P3 arm groups, the bf16 slice gate, and a domain that fails a gated check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfates.pilots.p3_estimator import (
    BF16_ROWS,
    P3_SCHEMA,
    arm_reliability_group,
    bf16_slice_gate,
    reliability_groups,
    run,
)
from vfates.utils.io import read_json


def test_arm_groups_match_the_preregistered_split() -> None:
    assert arm_reliability_group("oracle", "oracle") == "P3.2"
    assert arm_reliability_group("math_lenient", "math_lenient") == "P3.2"
    assert arm_reliability_group("natural_string_match", "string_match") == "P3.2"
    assert arm_reliability_group("code_public", "code_public_tests") == "P3.2"
    assert arm_reliability_group("noise_fp", "resampled") == "P3.3"
    assert arm_reliability_group("noise_natural", "string_match") == "P3.3"
    assert arm_reliability_group("fp_hash", "hash_fp") == "P3.3"
    assert arm_reliability_group("patch_tau", "patch_select") == "P3.3"
    assert arm_reliability_group("fn_form", "form_fn") == "P3.3"


def test_reliability_groups_skip_patch_selection() -> None:
    forecasts = {
        "oracle": {"type": "oracle", "transfer": {"reliability": 0.91}},
        "fp_hash": {"type": "hash_fp", "transfer": {"reliability": 0.4}},
        "patch_selection": {"chosen": "patch_tau"},
    }
    groups = reliability_groups(forecasts)
    assert groups["P3.2"] == [("oracle", 0.91)]
    assert groups["P3.3"] == [("fp_hash", 0.4)]


def _slice(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, 8))
    pid = np.repeat(np.arange(n // 2), 2)
    verdicts = np.tile(np.array([0, 1]), n // 2)
    return features, pid, verdicts


def test_bf16_gate_needs_256_rows_and_both_cosines() -> None:
    features, pid, verdicts = _slice(BF16_ROWS)
    passed = bf16_slice_gate(features, features.copy(), pid, verdicts)
    assert passed["pass"] is True
    assert passed["n"] == BF16_ROWS
    assert passed["direction_cosine"] > 0.99
    short, pid_s, verdicts_s = _slice(64)
    failed = bf16_slice_gate(short, short.copy(), pid_s, verdicts_s)
    assert failed["pass"] is False
    assert failed["gate"]["pass"] is True


def test_a_domain_fails_when_a_gated_check_fails(tmp_path: Path) -> None:
    report = run(
        {
            "smoke": False,
            "gates": {"reliability": 0.8, "sketch_corr": 0.95, "bf16_cosine": 0.99},
            "domains": {
                "math": {
                    "forecasts": {
                        "oracle": {"type": "oracle", "transfer": {"reliability": 0.9}},
                        "fp_hash": {"type": "hash_fp", "transfer": {"reliability": 0.5}},
                    },
                    "sketch_corr": 0.99,
                    "bf16_pass": True,
                    "lock_ok": True,
                }
            },
        },
        tmp_path,
    )
    assert report["ok"] is False
    document = read_json(tmp_path / "decisions.json")
    assert document["schema"] == P3_SCHEMA
    assert document["apply"] is False
    assert document["domains"]["math"]["status"] == "fail"
    assert "P3.3" in document["domains"]["math"]["failed"]
    row = next(item for item in report["checks"] if item["id"] == "P3.3.math")
    assert row["status"] == "fail"
    assert row["value"] == pytest.approx(0.5)


def test_smoke_does_not_write_a_decision(tmp_path: Path) -> None:
    report = run({"smoke": True, "gates": {"reliability": 0.8}}, tmp_path)
    assert report["ok"] is True
    assert not (tmp_path / "decisions.json").exists()
    row = next(item for item in report["checks"] if item["id"] == "P3.2")
    assert row["status"] == "documented"
