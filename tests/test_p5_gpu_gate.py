"""P5 stage decisions, seed lists, and the throughput file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vfates.pilots import p5_gpu_gate as p5
from vfates.utils.io import read_json, write_json


def test_sample_sd_and_seed_lists() -> None:
    sd = p5.sample_sd([0.50, 0.58])
    assert sd == pytest.approx(0.056568, rel=1e-3)
    assert p5.sample_sd([0.5]) is None
    wide = p5.seed_lists(
        arms=["oracle", "fp_trigger", "noise_fp"],
        rule_arms=["fp_trigger"],
        sd=sd,
        threshold=0.03,
    )
    assert wide["oracle"] == [0, 1, 2]
    assert wide["fp_trigger"] == [0, 1, 2, 3, 4]
    assert wide["noise_fp"] == [0, 1, 2]
    tight = p5.seed_lists(
        arms=["oracle", "fp_trigger"],
        rule_arms=["fp_trigger"],
        sd=0.01,
    )
    assert tight["fp_trigger"] == [0, 1, 2]


def test_stage_predicates() -> None:
    assert p5.gain_ok(0.05) is True
    assert p5.gain_ok(0.049) is False
    assert p5.gain_ok(None) is False
    assert p5.preempt_ok(
        {"status": "preempted", "steps_completed": 4},
        {"status": "complete", "steps_completed": 10},
        10,
    )
    assert not p5.preempt_ok(
        {"status": "preempted", "steps_completed": 10},
        {"status": "complete", "steps_completed": 10},
        10,
    )
    assert p5.path_ok(
        steps_completed=10,
        steps=10,
        status="complete",
        checkpoint=True,
        eval_steps=[0, 10],
        peak={"0": 12.0},
    )
    assert not p5.path_ok(
        steps_completed=10,
        steps=10,
        status="complete",
        checkpoint=True,
        eval_steps=[0, 10],
        peak=None,
    )


def test_study_command_uses_accelerate_only_for_several_gpus(tmp_path: Path) -> None:
    config = tmp_path / "cell.yaml"
    config.write_text("title: x\n")
    one = p5.study_command(config, tmp_path / "out", arm="oracle", seed=0, gpus=1)
    assert one[1:4] == ["-m", "vfates.cli", "study"]
    assert "accelerate" not in one
    two = p5.study_command(config, tmp_path / "out", arm="oracle", seed=0, gpus=2)
    assert two[0] == "accelerate"
    assert "fsdp_2gpu.yaml" in two[two.index("--config_file") + 1]
    assert two[two.index("--num_processes") + 1] == "2"


def test_aggregate_drops_smoke_and_incomplete_rows(tmp_path: Path) -> None:
    rows = [
        {
            "model": "qwen3-1.7b",
            "hardware_class": "rtx-a6000-48gb",
            "smoke": True,
            "gen_tok_s_per_gpu": 1.0,
            "train_tok_s_per_gpu": 2.0,
            "hours_per_arm_seed": 3.0,
        },
        {
            "model": "qwen3-1.7b",
            "hardware_class": "rtx-a6000-48gb",
            "smoke": False,
            "gen_tok_s_per_gpu": None,
            "train_tok_s_per_gpu": 2.0,
            "hours_per_arm_seed": 3.0,
        },
        {
            "model": "qwen3-4b",
            "hardware_class": "a100-80gb",
            "smoke": False,
            "gen_tok_s_per_gpu": 10.0,
            "train_tok_s_per_gpu": 20.0,
            "hours_per_arm_seed": 1.5,
        },
    ]
    rates = p5.aggregate_measurements(rows)
    assert list(rates) == ["qwen3-4b"]
    dest = tmp_path / "throughput.yaml"
    p5.write_throughput(dest, rates, measured_at="2026-09-28T00:00:00Z", commit="abc")
    loaded = yaml.safe_load(dest.read_text())
    assert loaded["schema"] == "vfates.throughput/1"
    row = loaded["rates"]["qwen3-4b"]["a100-80gb"]
    assert set(row) == {
        "gen_tok_s_per_gpu",
        "train_tok_s_per_gpu",
        "hours_per_arm_seed",
        "measured_at",
        "commit",
    }
    assert row["gen_tok_s_per_gpu"] == 10.0
    assert row["commit"] == "abc"


def test_collect_merges_stages_without_planning(tmp_path: Path) -> None:
    stages = tmp_path / "stages"
    stages.mkdir()
    write_json(
        stages / "path.json",
        {
            "rows": [{"model": "qwen3-1.7b", "hardware_class": "rtx-a6000-48gb", "pass": True}],
            "measurements": [
                {
                    "model": "qwen3-1.7b",
                    "hardware_class": "rtx-a6000-48gb",
                    "smoke": False,
                    "gen_tok_s_per_gpu": 100.0,
                    "train_tok_s_per_gpu": 200.0,
                    "hours_per_arm_seed": 1.25,
                }
            ],
        },
    )
    write_json(
        stages / "preempt.json",
        {"rows": [{"hardware_class": "rtx-a6000-48gb", "pass": True}], "measurements": []},
    )
    write_json(
        stages / "gain.json",
        {"rows": [{"model": "qwen3-1.7b", "pass": True, "gain": 0.08}], "measurements": []},
    )
    write_json(
        stages / "seeds.json",
        {
            "rows": [
                {
                    "cell_id": "e1-math-qwen3-1.7b",
                    "seeds": {"oracle": [0, 1, 2], "fp_trigger": [0, 1, 2, 3, 4]},
                    "pass": True,
                }
            ],
            "measurements": [],
        },
    )
    dest = tmp_path / "throughput.yaml"
    report = p5.run_collect(
        {"plan": False, "smoke": False, "throughput_path": str(dest)},
        tmp_path,
    )
    decisions = read_json(tmp_path / "decisions.json")
    assert decisions["schema"] == p5.P5_DECISIONS_SCHEMA
    assert decisions["path"]["qwen3-1.7b/rtx-a6000-48gb"]["pass"] is True
    assert decisions["preempt"]["rtx-a6000-48gb"]["pass"] is True
    assert decisions["gain"]["qwen3-1.7b"]["pass"] is True
    assert decisions["seeds"]["e1-math-qwen3-1.7b"]["fp_trigger"] == [0, 1, 2, 3, 4]
    loaded = yaml.safe_load(dest.read_text())
    assert loaded["schema"] == "vfates.throughput/1"
    assert loaded["rates"]["qwen3-1.7b"]["rtx-a6000-48gb"]["hours_per_arm_seed"] == 1.25
    assert report["ok"] is True
    assert "Program plan" not in (tmp_path / "report.md").read_text()


def test_run_seeds_keeps_the_e1_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_load(
        cell_id: str, *, dev: bool = True, overrides: dict | None = None, root: object = None
    ):
        del cell_id, dev, root
        cell = {
            "audit": {"pools": {"from": "dev"}},
            "study": {
                "arms": ["oracle", "fp_trigger", "noise_fp"],
                "seeds": [0, 1, 2],
                "seed_rule": {"arms": ["fp_trigger"]},
                "train": {"steps": 4},
            },
        }
        p5_apply = __import__("vfates.pilots.p4_micro_grpo", fromlist=["apply_dotted"])
        p5_apply.apply_dotted(cell, overrides or {})
        return cell

    def fake_run(cell, out, *, arm, seed, gpus=None):
        del cell, out, gpus
        calls.append((arm, seed))
        return {}

    accuracies = iter([0.50, 0.58])
    monkeypatch.setattr("vfates.pilots.p4_micro_grpo.load_pilot_cell", fake_load)
    monkeypatch.setattr(p5, "run_arm", fake_run)
    monkeypatch.setattr(
        "vfates.pilots.p4_micro_grpo.heldout_accuracy", lambda run_dir, steps: next(accuracies)
    )
    p5.run_seeds(
        {
            "seed_steps": 4,
            "cells": [
                {
                    "id": "e1-math-qwen3-1.7b",
                    "experiment": "E1",
                    "domain": "math",
                    "model": "qwen3-1.7b",
                    "hardware_class": "rtx-a6000-48gb",
                }
            ],
        },
        tmp_path,
    )
    stage = read_json(tmp_path / "stages" / "seeds.json")
    seeds = stage["rows"][0]["seeds"]
    assert seeds["oracle"] == [0, 1, 2]
    assert seeds["fp_trigger"] == [0, 1, 2, 3, 4]
    assert seeds["noise_fp"] == [0, 1, 2]
    assert ("oracle", 0) in calls and ("oracle", 1) in calls
    assert ("fp_trigger", 3) in calls and ("fp_trigger", 4) in calls
    assert ("oracle", 2) not in calls


def test_trl_summary_records_peak_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types
    from types import SimpleNamespace

    from vfates.experiment import _train_trl
    from vfates.train.trl_grpo import RUN_METADATA_NAME

    order: list[tuple[str, int | None]] = []

    class Cuda:
        def __init__(self, available: bool) -> None:
            self.available = available

        def is_available(self) -> bool:
            return self.available

        def device_count(self) -> int:
            return 2

        def reset_peak_memory_stats(self, index: int) -> None:
            order.append(("reset", index))

        def max_memory_allocated(self, index: int) -> int:
            order.append(("read", index))
            gib = 1.5 if index == 0 else 2.25
            return int(gib * 1024**3)

        def empty_cache(self) -> None:
            order.append(("empty", None))

    class Flag:
        def __init__(self, preempted: bool) -> None:
            self.preempted = preempted

        def install(self) -> None:
            return None

        def restore(self) -> None:
            return None

    class Trainer:
        def __init__(self, preempted: bool) -> None:
            self._vfates_preemption = Flag(preempted)
            self.state = SimpleNamespace(global_step=4)

        def train(self, resume_from_checkpoint: object = None) -> None:
            del resume_from_checkpoint

        def save_model(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    chosen = {"preempted": False}

    def build_trainer(*args: object, **kwargs: object) -> Trainer:
        del args, kwargs
        return Trainer(chosen["preempted"])

    monkeypatch.setattr("vfates.train.trl_grpo.build_trl_trainer", build_trainer)
    monkeypatch.setattr("vfates.train.checkpoint_eval.saved_checkpoints", lambda path: [])
    monkeypatch.setattr(
        "vfates.train.checkpoint_eval.release_colocated_engine", lambda trainer: None
    )
    monkeypatch.setattr("vfates.train.checkpoint_eval.evaluate_checkpoints", lambda *a, **k: None)
    monkeypatch.setattr(
        "vfates.train.checkpoint_eval.make_eval_generator",
        lambda *a, **k: SimpleNamespace(describe=lambda: {"backend": "hf"}),
    )
    try:
        import torch
    except ImportError:
        torch = types.ModuleType("torch")
        monkeypatch.setitem(sys.modules, "torch", torch)
    cuda = Cuda(True)
    monkeypatch.setattr(torch, "cuda", cuda, raising=False)

    def launch(out: Path, *, preempted: bool) -> dict:
        order.clear()
        chosen["preempted"] = preempted
        write_json(
            out / RUN_METADATA_NAME,
            {
                "lora_seed": 0,
                "lora_init_sha256": "abc",
                "rollout_seeded": True,
                "parameterisation": "lora",
            },
        )
        return _train_trl(
            SimpleNamespace(),
            SimpleNamespace(task=SimpleNamespace()),
            None,
            [],
            [],
            {},
            out,
            resume=False,
            eval_generator=None,
        )

    done = launch(tmp_path / "done", preempted=False)
    assert order[:2] == [("reset", 0), ("reset", 1)]
    assert done["status"] == "complete"
    assert done["peak_memory_gb_per_gpu"] == {"0": 1.5, "1": 2.25}
    assert done["peak_memory_gb"] == 2.25
    assert p5.read_peak(done) == {"0": 1.5, "1": 2.25}
    assert order.index(("read", 0)) < order.index(("empty", None))
    stopped = launch(tmp_path / "stopped", preempted=True)
    assert stopped["status"] == "preempted"
    assert p5.read_peak(stopped) == {"0": 1.5, "1": 2.25}

    monkeypatch.setattr(torch, "cuda", Cuda(False), raising=False)
    quiet = launch(tmp_path / "cpu", preempted=False)
    assert quiet["peak_memory_gb_per_gpu"] == {}
    assert quiet["peak_memory_gb"] is None
    assert p5.read_peak(quiet) is None
