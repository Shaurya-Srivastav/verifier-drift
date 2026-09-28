"""P5: the GPU gate, one stage at a time.

``stage`` is ``path`` (P5.1), ``preempt`` (P5.2), ``gain`` (P5.3), ``seeds``
(P5.4), ``throughput`` (P5.5) or ``collect``. ``--set smoke=true`` runs path
and preempt for Qwen3-0.6B on MLX or HF.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.io import read_json, read_jsonl, write_json


def arm_seed_timing(run_dir: str | Path) -> dict[str, Any]:
    """Wall-clock seconds of one arm-seed.

    ``wall_s`` is the per-step training wall clock. MLX and TRL both exclude
    evaluation. ``gen_seconds + train_seconds`` is the fallback when ``wall_s``
    is absent. ``timing.json``, when a run wrote one, holds the train_arm
    clock, which includes the checkpoint evaluation.

    Training seconds are the step clock when it exists, else ``timing.json``
    (and ``eval_seconds`` is then 0, so eval is not priced twice). Eval seconds
    come, in order, from ``eval_seconds`` on the metrics rows, from TRL's
    ``trl_eval.jsonl``, or from the ``train_arm`` clock minus the step clock
    (checkpoint evaluation and model loading). ``eval_source`` names the one
    used. Otherwise a generation rate (``tokens_generated / gen_seconds``)
    still prices a missing eval.
    """
    path = Path(run_dir) / "metrics.jsonl"
    rows = read_jsonl(path) if path.exists() else []

    def total(key: str, source_rows: Sequence[Any] | None = None) -> float | None:
        chosen = rows if source_rows is None else source_rows
        vals = [
            float(row[key])
            for row in chosen
            if isinstance(row, Mapping)
            and isinstance(row.get(key), (int, float))
            and math.isfinite(float(row[key]))
        ]
        return None if not vals else float(sum(vals))

    wall = total("wall_s")
    gen = total("gen_seconds")
    train = total("train_seconds")
    if wall is not None:
        train_seconds: float | None = wall
        source: str | None = "wall_s"
    elif gen is not None or train is not None:
        train_seconds = (gen or 0.0) + (train or 0.0)
        source = "gen_seconds+train_seconds"
    else:
        train_seconds = None
        source = None
    eval_included = False
    timing_path = Path(run_dir) / "timing.json"
    clock: float | None = None
    clock_has_eval = False
    if timing_path.exists():
        saved = read_json(timing_path)
        value = saved.get("train_arm_wall_s") if isinstance(saved, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            clock = float(value)
            clock_has_eval = bool(saved.get("includes_eval"))
    if train_seconds is None and clock is not None:
        train_seconds = clock
        source = "timing.json"
        eval_included = clock_has_eval
    eval_seconds = total("eval_seconds")
    eval_source: str | None = None if eval_seconds is None else "metrics"
    if eval_included:
        # The train_arm clock already contains eval. Do not add it again.
        if eval_seconds is None:
            eval_seconds = 0.0
            eval_source = "in_train_clock"
    elif eval_seconds is None:
        eval_log = Path(run_dir) / "trl_eval.jsonl"
        if eval_log.exists():
            eval_seconds = total("eval_seconds", read_jsonl(eval_log))
            eval_source = None if eval_seconds is None else "trl_eval"
        if eval_seconds is None and clock is not None and clock_has_eval:
            # Checkpoint evaluation and model loading: the train_arm clock
            # outside the summed step clock.
            eval_seconds = max(clock - float(train_seconds or 0.0), 0.0)
            eval_source = "train_arm_clock_minus_steps"
    tokens = total("tokens_generated")
    gen_rate = tokens / gen if tokens is not None and gen is not None and gen > 0 else None
    lengths = [
        float(row["mean_completion_length"])
        for row in rows
        if isinstance(row.get("mean_completion_length"), (int, float))
        and math.isfinite(float(row["mean_completion_length"]))
    ]
    return {
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "eval_source": eval_source,
        "eval_included_in_wall": eval_included,
        "gen_tok_s": gen_rate,
        "mean_completion_len": float(np.mean(lengths)) if lengths else None,
        "source": source,
    }


def _run_dir(out: Path, arm: str, seed: int) -> Path:
    return out / "runs" / f"{arm}__s{seed}"


P5_PILOT_SCHEMA = "vfates.p5_pilot/1"
P5_DECISIONS_SCHEMA = "vfates.p5_decisions/1"
THROUGHPUT_SCHEMA = "vfates.throughput/1"
PATH_STEPS = 10
GAIN_STEPS = 150
GAIN_MIN = 0.05
SEED_STEPS = 300
SEED_SD_MAX = 0.03
REFERENCE_STEPS = 300
BASE_SEEDS = (0, 1, 2)
ORACLE_SEED_PAIR = (0, 1)
EXTRA_SEEDS = (3, 4)


def sample_sd(values: Sequence[float]) -> float | None:
    """Sample standard deviation (ddof=1). None when fewer than two values."""
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(list(values), dtype=float), ddof=1))


def seed_lists(
    *,
    arms: Sequence[str],
    rule_arms: Sequence[str],
    base_seeds: Sequence[int] = BASE_SEEDS,
    extra_seeds: Sequence[int] = EXTRA_SEEDS,
    sd: float | None,
    threshold: float = SEED_SD_MAX,
) -> dict[str, list[int]]:
    """E1 seed list. H1 and H2 arms gain seeds 3 and 4 when ``sd`` exceeds the threshold."""
    extra = sd is not None and float(sd) > float(threshold)
    ruled = {str(arm) for arm in rule_arms}
    planned: dict[str, list[int]] = {}
    for arm in arms:
        seeds = [int(item) for item in base_seeds]
        if extra and str(arm) in ruled:
            for item in extra_seeds:
                if int(item) not in seeds:
                    seeds.append(int(item))
        planned[str(arm)] = seeds
    return planned


def gain_ok(gain: float | None, threshold: float = GAIN_MIN) -> bool:
    """P5.3: held-out oracle gain at least ``threshold``."""
    return gain is not None and float(gain) >= float(threshold)


def preempt_ok(first: Mapping[str, Any], second: Mapping[str, Any], steps: int) -> bool:
    """P5.2: preempted summary, then a completed run at the same planned step."""
    if str(first.get("status")) != "preempted":
        return False
    if str(second.get("status")) != "complete":
        return False
    if int(second.get("steps_completed", -1)) != int(steps):
        return False
    mid = first.get("steps_completed")
    return not (mid is not None and int(mid) >= int(steps))


def path_ok(
    *,
    steps_completed: int,
    steps: int,
    status: str | None,
    checkpoint: bool,
    eval_steps: Sequence[int],
    peak: Mapping[str, Any] | None,
) -> bool:
    """P5.1: finished step, saved adapter, held-out eval at 0 and the last step, peak memory."""
    if int(steps_completed) != int(steps):
        return False
    if status not in (None, "complete"):
        return False
    if not checkpoint:
        return False
    seen = {int(item) for item in eval_steps}
    if 0 not in seen or int(steps) not in seen:
        return False
    return bool(peak)


def placement_pairs(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """One row per (model, hardware class) that has a cell. Prefer a math cell."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault((str(cell["model"]), str(cell["hardware_class"])), []).append(cell)
    rows: list[dict[str, str]] = []
    for model, hardware in sorted(grouped):
        group = grouped[(model, hardware)]
        chosen = sorted(group, key=lambda item: (str(item["domain"]) != "math", str(item["id"])))[0]
        rows.append(
            {
                "model": model,
                "hardware_class": hardware,
                "cell_id": str(chosen["id"]),
                "domain": str(chosen["domain"]),
            }
        )
    return rows


def class_representatives(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """One cell per hardware class. Prefer chain_sum, then qwen3-1.7b, then math."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(str(cell["hardware_class"]), []).append(cell)
    rows: list[dict[str, str]] = []
    for hardware in sorted(grouped):
        chosen = sorted(
            grouped[hardware],
            key=lambda item: (
                str(item["domain"]) != "chain_sum",
                str(item["model"]) != "qwen3-1.7b",
                str(item["domain"]) != "math",
                str(item["id"]),
            ),
        )[0]
        rows.append(
            {
                "model": str(chosen["model"]),
                "hardware_class": hardware,
                "cell_id": str(chosen["id"]),
                "domain": str(chosen["domain"]),
            }
        )
    return rows


def gain_targets(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every model, on its math cell when it has one."""
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        by_model.setdefault(str(cell["model"]), []).append(cell)
    rows: list[dict[str, Any]] = []
    for model in sorted(by_model):
        group = by_model[model]
        math = [item for item in group if str(item["domain"]) == "math"]
        if math:
            chosen = sorted(math, key=lambda item: str(item["id"]))[0]
            rows.append(
                {
                    "model": model,
                    "cell_id": str(chosen["id"]),
                    "hardware_class": str(chosen["hardware_class"]),
                    "retarget": False,
                }
            )
        else:
            host = sorted(group, key=lambda item: str(item["id"]))[0]
            rows.append(
                {
                    "model": model,
                    "cell_id": "e1-math-qwen3-1.7b",
                    "hardware_class": str(host["hardware_class"]),
                    "retarget": True,
                }
            )
    return rows


def e1_rows(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(cell) for cell in cells if str(cell.get("experiment")) == "E1"]


def study_command(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    arm: str | None,
    seed: int | None,
    gpus: int,
    only: str = "train",
) -> list[str]:
    """Argv for one study job. Two or more GPUs go through ``accelerate launch``."""
    import sys

    from vfates.schedule import build_worker_command

    kind = "audit" if only == "audit" else "train"
    job: dict[str, Any] = {
        "kind": kind,
        "config": str(config_path),
        "out_dir": str(out_dir),
        "arm": arm,
        "seed": 0 if seed is None else int(seed),
        "gpus": int(gpus),
        "program": "configs/program.yaml",
    }
    return build_worker_command(job, python=sys.executable)


def read_peak(summary: Mapping[str, Any]) -> dict[str, float] | None:
    """Per-GPU peak GiB from a train summary, if the trainer recorded it."""
    per = summary.get("peak_memory_gb_per_gpu")
    if isinstance(per, Mapping) and per:
        return {str(key): float(value) for key, value in per.items()}
    value = summary.get("peak_memory_gb")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return {"0": float(value)}


def read_eval_steps(run_dir: str | Path) -> list[int]:
    path = Path(run_dir) / "eval.jsonl"
    if not path.is_file():
        return []
    found: list[int] = []
    for row in read_jsonl(path):
        if row.get("split") != "heldout":
            continue
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            continue
        if step not in found:
            found.append(step)
    return found


def has_checkpoint(run_dir: str | Path, steps: int) -> bool:
    root = Path(run_dir)
    if (root / f"checkpoint-{int(steps)}").exists():
        return True
    if any(root.glob("checkpoint-*")):
        return True
    return (root / "checkpoints").exists()


def execute_study(
    config: Mapping[str, Any],
    out_dir: str | Path,
    *,
    only: str,
    arm: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """In-process study job. Audits stay one process. Single-GPU training does too."""
    from vfates.study import run_study

    return run_study(config, out_dir, only=only, arm=arm, seed=seed)


def _audit_ready(out: Path) -> bool:
    audit = out / "audit"
    return (audit / "lock.json").is_file() or (audit / "forecasts.json").is_file()


def _cell_gpus(cell: Mapping[str, Any]) -> int:
    value = cell.get("study", {})
    if isinstance(value, Mapping) and value.get("gpus") is not None:
        return int(value["gpus"])
    return 1


def _write_cell(cell: Mapping[str, Any], out: Path) -> Path:
    from vfates.utils.io import write_yaml

    out.mkdir(parents=True, exist_ok=True)
    path = out / "cell.yaml"
    write_yaml(path, dict(cell))
    return path


def _subprocess_train(
    cell: Mapping[str, Any], out: Path, *, arm: str, seed: int, gpus: int
) -> dict[str, Any]:
    import subprocess

    config_path = _write_cell(cell, out)
    argv = study_command(config_path, out, arm=arm, seed=seed, gpus=gpus, only="train")
    completed = subprocess.run(argv, check=False)
    summary_path = out / "runs" / f"{arm}__s{seed}" / "summary.json"
    if completed.returncode != 0:
        raise RuntimeError(f"study exited {completed.returncode}: {' '.join(argv)}")
    if not summary_path.is_file():
        raise RuntimeError(f"study wrote no summary: {summary_path}")
    return read_json(summary_path)


def run_arm(
    cell: Mapping[str, Any],
    out_dir: str | Path,
    *,
    arm: str,
    seed: int,
    gpus: int | None = None,
) -> dict[str, Any]:
    """Audit if needed, then train one arm. Multi-GPU training uses accelerate."""
    import os

    out = Path(out_dir)
    _write_cell(cell, out)
    if not _audit_ready(out):
        execute_study(cell, out, only="audit")
    width = _cell_gpus(cell) if gpus is None else int(gpus)
    if width > 1 and os.environ.get("VFATES_P5_INPROCESS") != "1":
        return _subprocess_train(cell, out, arm=arm, seed=seed, gpus=width)
    return execute_study(cell, out, only="train", arm=arm, seed=seed)


def execute_preempt(
    cell: Mapping[str, Any],
    out_dir: str | Path,
    *,
    arm: str,
    seed: int,
    steps: int,
    gpus: int = 1,
    signal_at: int | None = None,
    timeout_s: float = 7200.0,
) -> dict[str, Any]:
    """SIGUSR1 at a mid step, then resume the same study command to ``steps``."""
    import os
    import signal
    import subprocess
    import time as time_mod

    out = Path(out_dir)
    _write_cell(cell, out)
    if not _audit_ready(out):
        execute_study(cell, out, only="audit")
    config_path = out / "cell.yaml"
    argv = study_command(config_path, out, arm=arm, seed=seed, gpus=gpus, only="train")
    run_dir = out / "runs" / f"{arm}__s{seed}"
    mid = max(1, int(steps) // 2) if signal_at is None else int(signal_at)
    first = _signal_and_wait(
        argv,
        run_dir,
        mid=mid,
        timeout_s=timeout_s,
        os=os,
        signal=signal,
        subprocess=subprocess,
        time_mod=time_mod,
    )
    second_proc = subprocess.run(argv, check=False)
    summary_path = run_dir / "summary.json"
    second = read_json(summary_path) if summary_path.is_file() else {}
    if second_proc.returncode != 0:
        second = {**second, "returncode": second_proc.returncode}
    return {"first": first, "second": second}


def _signal_and_wait(
    argv: Sequence[str],
    run_dir: Path,
    *,
    mid: int,
    timeout_s: float,
    os: Any,
    signal: Any,
    subprocess: Any,
    time_mod: Any,
) -> dict[str, Any]:
    import contextlib

    proc = subprocess.Popen(list(argv), start_new_session=True)
    deadline = time_mod.perf_counter() + timeout_s
    signaled = False
    while time_mod.perf_counter() < deadline:
        if _metric_step(run_dir) >= mid and not signaled:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGUSR1)
            signaled = True
        if proc.poll() is not None:
            break
        time_mod.sleep(2.0)
    else:
        proc.kill()
        raise TimeoutError(f"preempt wait exceeded {timeout_s}s")
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    summary = dict(summary)
    summary["returncode"] = proc.returncode
    summary["signaled"] = signaled
    return summary


def _metric_step(run_dir: Path) -> int:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return -1
    step = -1
    for row in read_jsonl(path):
        value = row.get("step")
        if isinstance(value, int) and not isinstance(value, bool):
            step = max(step, value)
    return step


def measure_run(
    run_dir: str | Path,
    *,
    model: str,
    hardware_class: str,
    gpus: int,
    steps: int,
    smoke: bool = False,
) -> dict[str, Any]:
    """Generation and training tokens per second per GPU, and scaled arm-seed hours."""
    timing = arm_seed_timing(run_dir)
    path = Path(run_dir) / "metrics.jsonl"
    rows = read_jsonl(path) if path.is_file() else []
    width = max(int(gpus), 1)
    gen_rates: list[float] = []
    train_rates: list[float] = []
    for row in rows:
        tokens = row.get("tokens_generated")
        gen_s = row.get("gen_seconds")
        train_s = row.get("train_seconds")
        if (
            isinstance(tokens, (int, float))
            and isinstance(gen_s, (int, float))
            and float(gen_s) > 0
        ):
            gen_rates.append(float(tokens) / float(gen_s) / width)
        if (
            isinstance(tokens, (int, float))
            and isinstance(train_s, (int, float))
            and float(train_s) > 0
        ):
            train_rates.append(float(tokens) / float(train_s) / width)
    wall = timing.get("train_seconds")
    if (
        isinstance(wall, (int, float))
        and not timing.get("eval_included_in_wall")
        and isinstance(timing.get("eval_seconds"), (int, float))
    ):
        wall = float(wall) + float(timing["eval_seconds"])
    hours = None
    if isinstance(wall, (int, float)) and int(steps) > 0:
        hours = float(wall) / float(steps) * REFERENCE_STEPS / 3600.0
    return {
        "model": model,
        "hardware_class": hardware_class,
        "gpus": width,
        "steps": int(steps),
        "smoke": bool(smoke),
        "gen_tok_s_per_gpu": None if not gen_rates else float(np.mean(gen_rates)),
        "train_tok_s_per_gpu": None if not train_rates else float(np.mean(train_rates)),
        "hours_per_arm_seed": hours,
        "run_dir": str(run_dir),
    }


def aggregate_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean rates per (model, hardware class). Smoke rows and incomplete rows are dropped."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("smoke"):
            continue
        if row.get("gen_tok_s_per_gpu") is None or row.get("train_tok_s_per_gpu") is None:
            continue
        if row.get("hours_per_arm_seed") is None:
            continue
        key = (str(row["model"]), str(row["hardware_class"]))
        grouped.setdefault(key, []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for (model, hardware), items in sorted(grouped.items()):
        out.setdefault(model, {})[hardware] = {
            "gen_tok_s_per_gpu": float(
                np.mean([float(item["gen_tok_s_per_gpu"]) for item in items])
            ),
            "train_tok_s_per_gpu": float(
                np.mean([float(item["train_tok_s_per_gpu"]) for item in items])
            ),
            "hours_per_arm_seed": float(
                np.mean([float(item["hours_per_arm_seed"]) for item in items])
            ),
        }
    return out


def git_commit(root: Path | None = None) -> str:
    import subprocess

    from vfates.program import repo_root

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root or repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_throughput(
    path: str | Path,
    rates: Mapping[str, Mapping[str, Any]],
    *,
    measured_at: str,
    commit: str,
) -> None:
    """Write ``configs/throughput.yaml`` in schema ``vfates.throughput/1``."""
    from vfates.utils.io import write_yaml

    body: dict[str, Any] = {"schema": THROUGHPUT_SCHEMA, "rates": {}}
    for model in sorted(rates):
        body["rates"][model] = {}
        for hardware in sorted(rates[model]):
            row = rates[model][hardware]
            body["rates"][model][hardware] = {
                "gen_tok_s_per_gpu": float(row["gen_tok_s_per_gpu"]),
                "train_tok_s_per_gpu": float(row["train_tok_s_per_gpu"]),
                "hours_per_arm_seed": float(row["hours_per_arm_seed"]),
                "measured_at": measured_at,
                "commit": commit,
            }
    write_yaml(path, body)


def _stage_file(out: Path, stage: str) -> Path:
    return out / "stages" / f"{stage}.json"


def _load_stage(out: Path, stage: str) -> dict[str, Any]:
    path = _stage_file(out, stage)
    if not path.is_file():
        return {"stage": stage, "rows": [], "measurements": []}
    loaded = read_json(path)
    loaded.setdefault("rows", [])
    loaded.setdefault("measurements", [])
    return loaded


def _save_stage(out: Path, stage: str, payload: Mapping[str, Any]) -> None:
    path = _stage_file(out, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(payload))


def _upsert(rows: Sequence[Mapping[str, Any]], row: Mapping[str, Any], keys: Sequence[str]) -> list:
    fresh = dict(row)
    identity = tuple(fresh.get(key) for key in keys)
    kept = [dict(item) for item in rows if tuple(item.get(key) for key in keys) != identity]
    kept.append(fresh)
    return kept


def _selected(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    model = config.get("model")
    hardware = config.get("hardware_class")
    model_ok = not model or str(row.get("model")) == str(model)
    hardware_ok = not hardware or str(row.get("hardware_class")) == str(hardware)
    return model_ok and hardware_ok


def _program_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("cells"):
        return [dict(cell) for cell in config["cells"]]
    from vfates.program import load_program, repo_root

    root = Path(config["root"]) if config.get("root") else repo_root()
    program_path = Path(config.get("program") or (root / "configs" / "program.yaml"))
    program = load_program(program_path, root=root)
    return [dict(cell) for cell in program["cells"]]


def _prepare_cell(
    cell_id: str,
    *,
    steps: int,
    arms: Sequence[str],
    seeds: Sequence[int],
    smoke: bool,
    hardware: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from vfates.pilots.p4_micro_grpo import load_pilot_cell, protocol_overrides

    if smoke:
        overrides = protocol_overrides(
            steps=steps, seed=int(seeds[0]), arms=arms, smoke=True, hardware=hardware
        )
    else:
        overrides = {
            "audit.pools.from": "dev",
            "study.train.steps": int(steps),
            "analysis.final_step": int(steps),
            "analysis.early.enabled": False,
            "study.seeds": [int(item) for item in seeds],
            "study.arms": list(arms),
            "study.train.eval_every": max(1, min(25, int(steps))),
            "study.train.save_every": max(1, min(25, int(steps))),
            "study.train.use_vllm": True,
            "study.train.vllm_mode": "colocate",
        }
    if extra:
        overrides.update(dict(extra))
    return load_pilot_cell(cell_id, dev=True, overrides=overrides)


def _model_retarget(model: str, hardware: str) -> dict[str, Any]:
    from vfates.models import load_model_spec

    spec = load_model_spec(model)
    kwargs = dict(spec.chat_template_kwargs or {})
    overrides: dict[str, Any] = {
        "audit.generator.model_id": spec.model_id,
        "audit.generator.revision": spec.revision,
        "audit.generator.chat_template_kwargs": kwargs,
        "audit.features.model_id": spec.model_id,
        "audit.features.chat_template_kwargs": kwargs,
        "study.train.model_id": spec.model_id,
        "study.train.model_revision": spec.revision,
        "study.train.chat_template_kwargs": kwargs,
        "study.train.lr": float(spec.lr),
        "study.hardware_class": hardware,
        "study.eval_generator.kwargs.chat_template_kwargs": kwargs,
    }
    try:
        place = spec.placement_for(hardware)
    except ValueError:
        return overrides
    overrides["study.gpus"] = place.gpus
    overrides["study.train.micro_batch_size"] = place.micro_batch_size
    overrides["audit.generator.kwargs.tensor_parallel_size"] = place.tp
    overrides["audit.generator.kwargs.gpu_memory_utilization"] = place.vllm_gpu_memory_utilization
    overrides["study.eval_generator.kwargs.tensor_parallel_size"] = place.tp
    overrides["study.eval_generator.kwargs.gpu_memory_utilization"] = (
        place.vllm_gpu_memory_utilization
    )
    return overrides


def _score_path(out: Path, *, steps: int) -> dict[str, Any]:
    run_dir = _run_dir(out, "oracle", 0)
    summary: dict[str, Any] = {}
    path = run_dir / "summary.json"
    if path.is_file():
        summary = read_json(path)
    peak = read_peak(summary)
    eval_steps = read_eval_steps(run_dir)
    checkpoint = has_checkpoint(run_dir, steps)
    status = summary.get("status")
    completed = int(summary.get("steps_completed", -1))
    passed = path_ok(
        steps_completed=completed,
        steps=steps,
        status=None if status is None else str(status),
        checkpoint=checkpoint,
        eval_steps=eval_steps,
        peak=peak,
    )
    endpoint = None
    if passed or eval_steps:
        from vfates.pilots.p4_micro_grpo import heldout_accuracy

        endpoint = heldout_accuracy(run_dir, steps)
    return {
        "pass": passed,
        "steps_completed": completed,
        "status": status,
        "checkpoint": checkpoint,
        "eval_steps": eval_steps,
        "peak_memory_gb_per_gpu": peak,
        "endpoint_acc": endpoint,
        "run_dir": str(run_dir),
    }


def run_path(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """P5.1, one (model, hardware class) pair, or every pair."""
    from vfates.utils.env import hardware_class

    started = time.perf_counter()
    out = Path(out_dir)
    smoke = bool(config.get("smoke", False))
    steps = int(config.get("smoke_steps", 2) if smoke else config.get("path_steps", PATH_STEPS))
    detected = hardware_class()
    checks = Checks()
    stage = _load_stage(out, "path")
    targets = _path_targets(config, smoke=smoke, hardware=detected)
    for target in targets:
        cell = _prepare_cell(
            target["cell_id"],
            steps=steps,
            arms=["oracle"],
            seeds=[0],
            smoke=smoke,
            hardware=detected if smoke else str(target["hardware_class"]),
            extra=target.get("extra"),
        )
        dest = out / "path" / f"{target['model']}__{target['hardware_class']}"
        try:
            run_arm(cell, dest, arm="oracle", seed=0, gpus=int(target.get("gpus") or 1))
            scored = _score_path(dest, steps=steps)
            error = None
        except Exception as exc:
            scored = {
                "pass": False,
                "steps_completed": None,
                "status": None,
                "checkpoint": False,
                "eval_steps": [],
                "peak_memory_gb_per_gpu": None,
                "endpoint_acc": None,
                "run_dir": str(_run_dir(dest, "oracle", 0)),
            }
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "model": target["record_model"],
            "hardware_class": target["record_hardware"],
            "cell_id": target["cell_id"],
            **scored,
        }
        stage["rows"] = _upsert(stage["rows"], row, ("model", "hardware_class"))
        if error is None and scored["run_dir"]:
            stage["measurements"] = _upsert(
                stage["measurements"],
                measure_run(
                    scored["run_dir"],
                    model=row["model"],
                    hardware_class=row["hardware_class"],
                    gpus=int(target.get("gpus") or 1),
                    steps=steps,
                    smoke=smoke,
                ),
                ("model", "hardware_class", "run_dir"),
            )
        checks.add(
            f"P5.1.{row['model']}.{row['hardware_class']}",
            f"path {row['model']} on {row['hardware_class']}",
            None if scored["endpoint_acc"] is None else float(scored["endpoint_acc"]),
            bool(scored["pass"]),
            threshold=f"{steps} steps, adapter reload eval, peak memory",
            metric=error or f"peak={scored['peak_memory_gb_per_gpu']}",
        )
    stage["stage"] = "path"
    _save_stage(out, "path", stage)
    return write_report(
        out,
        "P5 path",
        checks,
        config=dict(config),
        started=started,
        tables={"path": stage["rows"]},
        sections=[],
        intro=f"Hardware class {detected}. P5.1 is gated.",
    )


def _path_targets(config: Mapping[str, Any], *, smoke: bool, hardware: str) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "model": "qwen3-1.7b",
                "hardware_class": hardware,
                "cell_id": "e1-chain_sum-qwen3-1.7b",
                "record_model": "qwen3-0.6b",
                "record_hardware": hardware,
                "gpus": 1,
                "extra": None,
            }
        ]
    rows = []
    for pair in placement_pairs(_program_cells(config)):
        if not _selected(pair, config):
            continue
        gpus = 1
        try:
            from vfates.models import load_model_spec

            gpus = int(load_model_spec(pair["model"]).placement_for(pair["hardware_class"]).gpus)
        except (OSError, ValueError):
            gpus = 1
        rows.append(
            {
                **pair,
                "record_model": pair["model"],
                "record_hardware": pair["hardware_class"],
                "gpus": gpus,
                "extra": None,
            }
        )
    return rows


def run_preempt(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """P5.2, one run per hardware class."""
    from vfates.utils.env import hardware_class

    started = time.perf_counter()
    out = Path(out_dir)
    smoke = bool(config.get("smoke", False))
    steps = int(config.get("smoke_steps", 2) if smoke else config.get("preempt_steps", PATH_STEPS))
    detected = hardware_class()
    checks = Checks()
    stage = _load_stage(out, "preempt")
    targets = _preempt_targets(config, smoke=smoke, hardware=detected)
    for target in targets:
        cell = _prepare_cell(
            target["cell_id"],
            steps=steps,
            arms=["oracle"],
            seeds=[0],
            smoke=smoke,
            hardware=detected if smoke else str(target["hardware_class"]),
        )
        dest = out / "preempt" / str(target["hardware_class"])
        try:
            outcome = execute_preempt(
                cell,
                dest,
                arm="oracle",
                seed=0,
                steps=steps,
                gpus=int(target.get("gpus") or 1),
            )
            passed = preempt_ok(outcome["first"], outcome["second"], steps)
            error = None
        except Exception as exc:
            outcome = {"first": {}, "second": {}}
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "model": target["record_model"],
            "hardware_class": target["record_hardware"],
            "cell_id": target["cell_id"],
            "pass": passed,
            "first_status": outcome["first"].get("status"),
            "second_status": outcome["second"].get("status"),
            "steps": steps,
        }
        stage["rows"] = _upsert(stage["rows"], row, ("hardware_class",))
        checks.add(
            f"P5.2.{row['hardware_class']}",
            f"preempt and resume on {row['hardware_class']}",
            None,
            passed,
            threshold="status preempted, then complete at the same step",
            metric=error or f"first={row['first_status']}, second={row['second_status']}",
        )
    stage["stage"] = "preempt"
    _save_stage(out, "preempt", stage)
    return write_report(
        out,
        "P5 preempt",
        checks,
        config=dict(config),
        started=started,
        tables={"preempt": stage["rows"]},
        sections=[],
        intro=f"Hardware class {detected}. P5.2 is gated.",
    )


def _preempt_targets(
    config: Mapping[str, Any], *, smoke: bool, hardware: str
) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "model": "qwen3-1.7b",
                "hardware_class": hardware,
                "cell_id": "e1-chain_sum-qwen3-1.7b",
                "record_model": "qwen3-0.6b",
                "record_hardware": hardware,
                "gpus": 1,
            }
        ]
    rows = []
    for pair in class_representatives(_program_cells(config)):
        if config.get("hardware_class") and pair["hardware_class"] != str(config["hardware_class"]):
            continue
        gpus = 1
        try:
            from vfates.models import load_model_spec

            gpus = int(load_model_spec(pair["model"]).placement_for(pair["hardware_class"]).gpus)
        except (OSError, ValueError):
            gpus = 1
        rows.append(
            {
                **pair,
                "record_model": pair["model"],
                "record_hardware": pair["hardware_class"],
                "gpus": gpus,
            }
        )
    return rows


def run_gain(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """P5.3: oracle on the math dev pool, 150 steps, the spec learning rate."""
    from vfates.pilots.p4_micro_grpo import heldout_gain
    from vfates.utils.env import hardware_class

    started = time.perf_counter()
    out = Path(out_dir)
    steps = int(config.get("gain_steps", GAIN_STEPS))
    threshold = float(config.get("gain_threshold", GAIN_MIN))
    detected = hardware_class()
    checks = Checks()
    stage = _load_stage(out, "gain")
    for target in gain_targets(_program_cells(config)):
        if config.get("model") and target["model"] != str(config["model"]):
            continue
        extra = None
        if target["retarget"]:
            extra = _model_retarget(target["model"], target["hardware_class"])
        cell = _prepare_cell(
            target["cell_id"],
            steps=steps,
            arms=["oracle"],
            seeds=[0],
            smoke=False,
            hardware=str(target["hardware_class"]),
            extra=extra,
        )
        dest = out / "gain" / str(target["model"])
        try:
            run_arm(cell, dest, arm="oracle", seed=0)
            gain = heldout_gain(_run_dir(dest, "oracle", 0), steps)
            error = None
        except Exception as exc:
            gain = None
            error = f"{type(exc).__name__}: {exc}"
        passed = gain_ok(gain, threshold)
        row = {
            "model": target["model"],
            "hardware_class": target["hardware_class"],
            "cell_id": target["cell_id"],
            "pass": passed,
            "gain": gain,
        }
        stage["rows"] = _upsert(stage["rows"], row, ("model",))
        stage["measurements"] = _upsert(
            stage["measurements"],
            measure_run(
                _run_dir(dest, "oracle", 0),
                model=target["model"],
                hardware_class=target["hardware_class"],
                gpus=_cell_gpus(cell),
                steps=steps,
            ),
            ("model", "hardware_class", "run_dir"),
        )
        checks.add(
            f"P5.3.{target['model']}",
            f"{target['model']} oracle held-out gain on the math dev pool",
            gain,
            passed,
            threshold=threshold,
            metric=error or "held-out last step minus step 0",
        )
    stage["stage"] = "gain"
    _save_stage(out, "gain", stage)
    return write_report(
        out,
        "P5 gain",
        checks,
        config=dict(config),
        started=started,
        tables={"gain": stage["rows"]},
        sections=[],
        intro=f"Hardware class {detected}. P5.3 is gated.",
    )


def run_seeds(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """P5.4: oracle seeds 0 and 1, then H1/H2 seeds 3 and 4 when the SD exceeds 0.03."""
    from vfates.pilots.p4_micro_grpo import apply_dotted, heldout_accuracy, load_pilot_cell
    from vfates.utils.env import hardware_class

    started = time.perf_counter()
    out = Path(out_dir)
    steps = int(config.get("seed_steps", SEED_STEPS))
    threshold = float(config.get("seed_sd_threshold", SEED_SD_MAX))
    detected = hardware_class()
    checks = Checks()
    stage = _load_stage(out, "seeds")
    for cell_row in e1_rows(_program_cells(config)):
        cell_id = str(cell_row["id"])
        if config.get("cell") and cell_id != str(config["cell"]):
            continue
        dest = out / "seeds" / cell_id
        try:
            cell = load_pilot_cell(
                cell_id,
                dev=True,
                overrides={
                    "audit.pools.from": "dev",
                    "study.train.steps": steps,
                    "analysis.final_step": steps,
                    "analysis.early.enabled": False,
                },
            )
            menu = [str(arm) for arm in cell["study"]["arms"]]
            base_seeds = [int(item) for item in cell["study"]["seeds"]]
            rule_arms = [
                str(arm) for arm in ((cell["study"].get("seed_rule") or {}).get("arms") or [])
            ]
            apply_dotted(
                cell,
                {"study.arms": ["oracle"], "study.seeds": list(ORACLE_SEED_PAIR)},
            )
            finals = []
            for seed in ORACLE_SEED_PAIR:
                run_arm(cell, dest, arm="oracle", seed=seed)
                finals.append(heldout_accuracy(_run_dir(dest, "oracle", seed), steps))
            known = [float(value) for value in finals if value is not None]
            sd = sample_sd(known)
            planned = seed_lists(
                arms=menu,
                rule_arms=rule_arms,
                base_seeds=base_seeds,
                sd=sd,
                threshold=threshold,
            )
            error = None
            if sd is not None and sd > threshold:
                try:
                    _train_extra_seeds(cell, dest, rule_arms, steps)
                except Exception as exc:
                    error = f"extra seeds: {type(exc).__name__}: {exc}"
        except Exception as exc:
            sd = None
            planned = {}
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "cell_id": cell_id,
            "domain": cell_row.get("domain"),
            "model": cell_row.get("model"),
            "hardware_class": cell_row.get("hardware_class"),
            "sd": sd,
            "seeds": planned,
            "pass": sd is not None,
        }
        stage["rows"] = _upsert(stage["rows"], row, ("cell_id",))
        for seed in ORACLE_SEED_PAIR:
            stage["measurements"] = _upsert(
                stage["measurements"],
                measure_run(
                    _run_dir(dest, "oracle", seed),
                    model=str(cell_row.get("model")),
                    hardware_class=str(cell_row.get("hardware_class")),
                    gpus=1,
                    steps=steps,
                ),
                ("model", "hardware_class", "run_dir"),
            )
        checks.add(
            f"P5.4.{cell_id}",
            f"seed rule for {cell_id}",
            sd,
            sd is not None,
            threshold=threshold,
            metric=error or "sample SD of oracle seeds 0 and 1; documented decision",
            documented=True,
        )
    stage["stage"] = "seeds"
    _save_stage(out, "seeds", stage)
    return write_report(
        out,
        "P5 seeds",
        checks,
        config=dict(config),
        started=started,
        tables={"seeds": stage["rows"]},
        sections=[],
        intro=(
            f"Hardware class {detected}. P5.4 decides the seed list. It does not gate the pilot."
        ),
    )


def _train_extra_seeds(
    cell: dict[str, Any], out: Path, rule_arms: Sequence[str], steps: int
) -> None:
    from vfates.pilots.p4_micro_grpo import apply_dotted

    if not rule_arms:
        return
    apply_dotted(
        cell,
        {
            "study.arms": list(rule_arms),
            "study.seeds": [0, 1, 2, 3, 4],
            "study.train.steps": steps,
            "analysis.final_step": steps,
        },
    )
    for arm in rule_arms:
        for seed in EXTRA_SEEDS:
            run_arm(cell, out, arm=str(arm), seed=int(seed))


def run_throughput(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """P5.5: write ``configs/throughput.yaml`` from stage measurements. Documented."""
    started = time.perf_counter()
    out = Path(out_dir)
    return _write_throughput_report(config, out, started)


def _measurement_rows(out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in ("path", "preempt", "gain", "seeds"):
        rows.extend(_load_stage(out, stage).get("measurements") or [])
    return rows


def _throughput_path(config: Mapping[str, Any]) -> Path:
    if config.get("throughput_path"):
        return Path(str(config["throughput_path"]))
    from vfates.program import repo_root

    return repo_root() / "configs" / "throughput.yaml"


def _write_throughput_report(
    config: Mapping[str, Any], out: Path, started: float
) -> dict[str, Any]:
    import datetime as datetime_mod

    from vfates.utils.env import hardware_class

    checks = Checks()
    rates = aggregate_measurements(_measurement_rows(out))
    path = _throughput_path(config)
    measured_at = datetime_mod.datetime.now(datetime_mod.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    commit = git_commit()
    if rates:
        write_throughput(path, rates, measured_at=measured_at, commit=commit)
    n_pairs = sum(len(by_hw) for by_hw in rates.values())
    checks.add(
        "P5.5",
        "measured tokens per second per GPU and hours per arm-seed",
        float(n_pairs),
        True,
        threshold=THROUGHPUT_SCHEMA,
        metric=f"pairs={n_pairs}; file={path if rates else 'not written'}",
        documented=True,
    )
    return write_report(
        out,
        "P5 throughput",
        checks,
        config=dict(config),
        started=started,
        tables={"rates": rates, "throughput_path": str(path), "commit": commit},
        sections=[],
        intro=f"Hardware class {hardware_class()}. P5.5 is documented.",
    )


def _decisions_from_stages(out: Path) -> dict[str, Any]:
    path_rows = _load_stage(out, "path").get("rows") or []
    gain_rows = _load_stage(out, "gain").get("rows") or []
    preempt_rows = _load_stage(out, "preempt").get("rows") or []
    seed_rows = _load_stage(out, "seeds").get("rows") or []
    return {
        "schema": P5_DECISIONS_SCHEMA,
        "seeds": {str(row["cell_id"]): row.get("seeds") or {} for row in seed_rows},
        "gain": {
            str(row["model"]): {"pass": bool(row.get("pass")), "gain": row.get("gain")}
            for row in gain_rows
        },
        "path": {
            f"{row['model']}/{row['hardware_class']}": {"pass": bool(row.get("pass"))}
            for row in path_rows
        },
        "preempt": {
            str(row["hardware_class"]): {"pass": bool(row.get("pass"))} for row in preempt_rows
        },
    }


def run_collect(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Merge stage files into the report and ``decisions.json``."""
    from vfates.utils.env import hardware_class

    started = time.perf_counter()
    out = Path(out_dir)
    smoke = bool(config.get("smoke", False))
    decisions = _decisions_from_stages(out)
    write_json(out / "decisions.json", decisions)
    throughput = _write_throughput_report(config, out, started)
    checks = Checks()
    for row in throughput["checks"]:
        checks.rows.append(row)
    _collect_checks(checks, decisions, smoke=smoke)
    plan = ""
    if config.get("plan", not smoke):
        plan = _plan_text(float(config.get("budget_usd", 9000)))
    sections = [("Decisions", _decision_section(decisions))]
    if plan:
        sections.append(("Program plan", plan))
    edits = _seed_edits(decisions)
    sections.append(("Config edits", edits))
    return write_report(
        out,
        "P5 GPU gate",
        checks,
        config=dict(config),
        started=started,
        tables={"decisions": decisions, "hardware_class": hardware_class()},
        sections=sections,
        intro=(
            f"Hardware class {hardware_class()}. "
            "P5.1, P5.2 and P5.3 are gated. P5.4 decides seeds. P5.5 is documented."
        ),
    )


def _collect_checks(checks: Checks, decisions: Mapping[str, Any], *, smoke: bool) -> None:
    path = decisions.get("path") or {}
    preempt = decisions.get("preempt") or {}
    gain = decisions.get("gain") or {}
    seeds = decisions.get("seeds") or {}
    if not path:
        checks.add(
            "P5.1",
            "path stage",
            None,
            False,
            threshold="one row per (model, hardware class)",
            metric="no path rows",
            documented=smoke,
        )
    for key, row in path.items():
        checks.add(
            f"P5.1.{key}",
            f"path {key}",
            None,
            bool(row.get("pass")),
            threshold="10-step train, adapter eval, peak memory",
            metric="pass" if row.get("pass") else "fail",
        )
    if not preempt:
        checks.add(
            "P5.2",
            "preempt stage",
            None,
            False,
            threshold="one row per hardware class",
            metric="no preempt rows",
            documented=smoke,
        )
    for key, row in preempt.items():
        checks.add(
            f"P5.2.{key}",
            f"preempt {key}",
            None,
            bool(row.get("pass")),
            threshold="preempted, then complete at the same step",
            metric="pass" if row.get("pass") else "fail",
        )
    if not gain:
        checks.add(
            "P5.3",
            "oracle gain on the math dev pool",
            None,
            False,
            threshold=GAIN_MIN,
            metric="not part of the Mac smoke run" if smoke else "no gain rows",
            documented=smoke,
        )
    for model, row in gain.items():
        checks.add(
            f"P5.3.{model}",
            f"{model} oracle gain",
            row.get("gain"),
            bool(row.get("pass")),
            threshold=GAIN_MIN,
            metric="held-out last step minus step 0",
        )
    if not seeds:
        checks.add(
            "P5.4",
            "E1 seed rule",
            None,
            True,
            threshold=SEED_SD_MAX,
            metric="not part of the Mac smoke run" if smoke else "no seed rows",
            documented=True,
        )
    for cell_id, planned in seeds.items():
        checks.add(
            f"P5.4.{cell_id}",
            f"seeds for {cell_id}",
            float(len(planned)),
            True,
            threshold=SEED_SD_MAX,
            metric="documented seed list",
            documented=True,
        )


def _decision_section(decisions: Mapping[str, Any]) -> str:
    lines = [
        markdown_table(
            ["pair", "path pass"],
            [[key, row.get("pass")] for key, row in (decisions.get("path") or {}).items()]
            or [["(none)", None]],
        ),
        "",
        markdown_table(
            ["hardware_class", "preempt pass"],
            [[key, row.get("pass")] for key, row in (decisions.get("preempt") or {}).items()]
            or [["(none)", None]],
        ),
        "",
        markdown_table(
            ["model", "gain pass", "gain"],
            [
                [key, row.get("pass"), row.get("gain")]
                for key, row in (decisions.get("gain") or {}).items()
            ]
            or [["(none)", None, None]],
        ),
    ]
    return "\n".join(lines)


def _seed_edits(decisions: Mapping[str, Any]) -> str:
    lines = []
    for cell_id, planned in (decisions.get("seeds") or {}).items():
        extra = [
            arm for arm, seeds in planned.items() if any(int(seed) in EXTRA_SEEDS for seed in seeds)
        ]
        if extra:
            lines.append(
                f"Set study.seed_rule.apply to true on {cell_id}. "
                f"Arms that take seeds 3 and 4: {', '.join(extra)}."
            )
    if not lines:
        return "No seed-rule edit. Leave study.seed_rule.apply false unless a cell's oracle SD exceeded 0.03."
    return "\n".join(f"- {line}" for line in lines)


def _plan_text(budget: float) -> str:
    try:
        from vfates.program import format_budget, load_program, plan_jobs, repo_root

        root = repo_root()
        program = load_program(root / "configs" / "program.yaml", root=root)
        jobs = plan_jobs(program, root=root)
        return format_budget(jobs, program=program, by_cell=True, budget_usd=budget)
    except Exception as exc:
        return f"Plan was not printed ({type(exc).__name__}: {exc})."


def run_program(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Dispatch one P5 stage. ``smoke`` runs path and preempt, then collect."""
    out = Path(out_dir)
    stage_name = config.get("stage")
    if config.get("smoke") and stage_name in (None, "smoke", "collect"):
        run_path(config, out)
        run_preempt(config, out)
        return run_collect(config, out)
    stage = str(config.get("stage") or "collect")
    if stage == "path":
        return run_path(config, out)
    if stage == "preempt":
        return run_preempt(config, out)
    if stage == "gain":
        return run_gain(config, out)
    if stage == "seeds":
        return run_seeds(config, out)
    if stage == "throughput":
        return run_throughput(config, out)
    if stage == "collect":
        return run_collect(config, out)
    raise ValueError(
        f"unknown P5 stage {stage!r}. Use path, preempt, gain, seeds, throughput, or collect."
    )


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P5 GPU gate. ``stage`` selects one stage, or collect."""
    return run_program(config, out_dir)
