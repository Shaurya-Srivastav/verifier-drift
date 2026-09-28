"""Recorded wall time and GPU count on audits and train summaries."""

from __future__ import annotations

import time
from pathlib import Path

from vfates.audit import _stamp_audit_timing
from vfates.experiment import _attach_run_timing, _preempted_wall_seconds
from vfates.utils.io import read_json, write_json


def test_audit_timing_is_written_once(tmp_path: Path) -> None:
    meta = tmp_path / "audit.meta.json"
    write_json(meta, {"task": "math"})
    _stamp_audit_timing(tmp_path, time.perf_counter() - 1.5)
    timing = read_json(meta)["timing"]
    assert set(timing) == {"wall_seconds", "gpus", "hardware_class", "resumed"}
    assert timing["wall_seconds"] >= 1.5
    assert timing["resumed"] is False
    first = timing["wall_seconds"]
    _stamp_audit_timing(tmp_path, time.perf_counter() - 50.0)
    assert read_json(meta)["timing"]["wall_seconds"] == first


def test_resumed_audit_records_no_partial_time(tmp_path: Path) -> None:
    meta = tmp_path / "audit.meta.json"
    write_json(meta, {"task": "math"})
    _stamp_audit_timing(tmp_path, time.perf_counter() - 30.0, fresh=False)
    timing = read_json(meta)["timing"]
    assert timing["wall_seconds"] is None
    assert timing["resumed"] is True


def test_train_timing_adds_a_preempted_prefix(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    write_json(
        summary_path,
        {
            "status": "preempted",
            "timing": {"wall_seconds": 10.0, "gpus": 1, "hardware_class": "h100-80gb"},
        },
    )
    assert _preempted_wall_seconds(summary_path) == 10.0
    finished = {"status": "complete", "steps_completed": 4}
    started = time.perf_counter()
    written = _attach_run_timing(tmp_path, finished, started, 10.0, "h100-80gb")
    timing = written["timing"]
    assert set(timing) == {"wall_seconds", "gpus", "hardware_class"}
    assert timing["wall_seconds"] >= 10.0
    assert timing["hardware_class"] == "h100-80gb"
    assert read_json(summary_path)["timing"]["wall_seconds"] == timing["wall_seconds"]
    write_json(summary_path, {"status": "complete", "timing": {"wall_seconds": 99.0}})
    assert _preempted_wall_seconds(summary_path) == 0.0
