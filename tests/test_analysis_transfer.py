"""Exploratory transfer table on synthetic orchestration files."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfates.analysis.transfer import render_transfer_markdown, transfer_report
from vfates.utils.io import write_json


def _write(
    root: Path,
    *,
    cell: str,
    arm: str,
    seed: int,
    target: str,
    accs: dict[str, float],
) -> None:
    write_json(
        root / cell / "transfer" / f"{arm}__s{seed}" / f"{target}.json",
        {
            "schema": "vfates.transfer_eval/1",
            "source_cell": cell,
            "arm": arm,
            "seed": seed,
            "target_domain": target,
            "target_cell": f"{target}-cell",
            "adapter": f"{cell}/{arm}__s{seed}",
            "step": 300,
            "n_prompts": len(accs),
            "n_samples": 4,
            "accuracy": sum(accs.values()) / len(accs),
            "per_prompt": [{"pid": pid, "accuracy": acc, "n": 4} for pid, acc in accs.items()],
        },
    )


def test_transfer_reports_arm_minus_oracle(tmp_path: Path) -> None:
    prompts = {
        0: ({"p0": 0.2, "p1": 0.8}, {"p0": 0.4, "p1": 0.4}),
        1: ({"p0": 0.4, "p1": 0.6}, {"p0": 0.4, "p1": 0.4}),
    }
    for seed, (arm_acc, oracle_acc) in prompts.items():
        _write(tmp_path, cell="e1", arm="fp_hash", seed=seed, target="code", accs=arm_acc)
        _write(tmp_path, cell="e1", arm="oracle", seed=seed, target="code", accs=oracle_acc)
    report = transfer_report(tmp_path, n_boot=200, seed=0)
    assert report["schema"] == "vfates.transfer/1"
    assert report["n_rows"] == 1
    row = report["rows"][0]
    assert row["status"] == "tested"
    assert row["arm"] == "fp_hash"
    assert row["target_domain"] == "code"
    assert row["n_seeds"] == 2
    assert row["diff"] == pytest.approx(0.1)
    assert row["lo"] <= row["diff"] <= row["hi"]
    text = render_transfer_markdown(report)
    assert "fp_hash" in text
    assert "code" in text


def test_transfer_marks_a_missing_oracle(tmp_path: Path) -> None:
    _write(
        tmp_path,
        cell="e1",
        arm="fp_hash",
        seed=0,
        target="qa",
        accs={"p0": 0.4, "p1": 0.6},
    )
    report = transfer_report(tmp_path, n_boot=20, seed=0)
    assert report["rows"][0]["status"] == "missing_oracle"
    assert report["rows"][0]["target_domain"] == "qa"
