"""P5: the GPU gate. Dry run on the Mac, the full gate on one H100.

``mode: dry_run`` (default without CUDA) exercises the torch path end to end on
the local device. ``mode: full`` (default with CUDA) trains the development
pool. Its only scientific go/no-go is the oracle held-out gain. Fate contrasts
are logged and do not decide whether E1 runs.

``mode: replication_screen`` scores Llama-3.2-3B-Instruct and
Qwen2.5-3B-Instruct at 1,024 tokens on 128 development prompts and applies the
pre-written 3B model rule. It is a separate config, not part of the full gate.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from vfates.analysis.power import (
    minimum_detectable_difference,
    project_measured_cost,
    token_projection,
)
from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.io import read_json, read_jsonl, write_json

AUDIT_TIMING_NAME = "audit_timing.json"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
REPLICATION_RULE = (
    "Use Llama-3.2-3B-Instruct when its base held-out accuracy at 1,024 tokens "
    "on 128 development prompts is in [0.10, 0.60] and truncation is <= 0.05. "
    "Otherwise use Qwen2.5-3B-Instruct."
)


def choose_replication_model(
    base_accuracy: float,
    truncation: float,
    *,
    acc_low: float = 0.10,
    acc_high: float = 0.60,
    truncation_max: float = 0.05,
) -> dict[str, Any]:
    """Apply the pre-written 3B replication model rule to the Llama screen."""
    preferred_ok = acc_low <= base_accuracy <= acc_high and truncation <= truncation_max
    return {
        "model_id": LLAMA_3B if preferred_ok else QWEN_3B,
        "preferred": LLAMA_3B,
        "fallback": QWEN_3B,
        "preferred_ok": preferred_ok,
        "base_accuracy": base_accuracy,
        "truncation": truncation,
        "rule": REPLICATION_RULE,
    }


def base_screen_summary(correct: Sequence[int], truncated: Sequence[bool]) -> dict[str, Any]:
    """Mean oracle bit and truncation rate on one base-model screen."""
    if len(correct) != len(truncated) or not correct:
        raise ValueError("correct and truncated must be non-empty and the same length")
    return {
        "accuracy": float(np.mean(np.asarray(correct, dtype=float))),
        "truncation": float(np.mean(np.asarray(truncated, dtype=float))),
        "n": len(correct),
    }


def arm_seed_timing(run_dir: str | Path) -> dict[str, Any]:
    """Wall-clock seconds of one arm-seed.

    ``wall_s`` is the per-step training wall clock. MLX and TRL both exclude
    evaluation. ``gen_seconds + train_seconds`` is the fallback when ``wall_s``
    is absent. The full gate writes ``timing.json`` with the whole ``train_arm``
    clock of every fresh run, which includes the checkpoint evaluation.

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


def _cuda() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda": False}
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            props = torch.cuda.get_device_properties(0)
            info.update(
                gpu=props.name,
                gpu_mem_gb=props.total_memory / 1e9,
                n_gpus=torch.cuda.device_count(),
            )
    except ImportError:
        pass
    try:
        import vllm

        info["vllm"] = vllm.__version__
    except ImportError:
        info["vllm"] = None
    return info


def _projection(config: dict[str, Any]) -> dict[str, Any]:
    """Token-rate cost of the full gate. Eval tokens are included. P5's measurement replaces it."""
    study = dict(config["cost_model"])
    p4 = Path(config.get("p4_report", "pilots/p4_micro_grpo/results/report.json"))
    measured = None
    if p4.exists():
        measured = read_json(p4).get("tables", {}).get("cost", {}).get("mean_completion_len")
    length = float(measured or study["mean_completion_len"])
    runs = int(study["arms"]) * int(study["seeds"])
    spec = {
        "mean_completion_len": length,
        "steps": study["steps"],
        "prompts_per_step": study["prompts_per_step"],
        "group_size": study["group_size"],
        "full_runs": runs,
        "early_runs": 0,
        "early_steps": 0,
        "eval_every": study["eval_every"],
        "eval_prompts": study["eval_prompts"],
        "eval_train_prompts": study.get("eval_train_prompts", 0),
        "eval_samples": study["eval_samples"],
        "audit_prompts": study["audit_prompts"],
        "audit_samples": study["audit_samples"],
        "h100_gen_tok_s": study["h100_gen_tok_s"],
        "h100_train_tok_s": study["h100_train_tok_s"],
        "overhead": study.get("overhead", 1.0),
        "usd_per_h100_hour": study["usd_per_h100_hour"],
    }
    projected = token_projection(spec)
    return {
        "mean_completion_len": length,
        "length_source": "P4" if measured else "config",
        "runs": runs,
        **projected,
    }


def _dry_run(config: dict[str, Any], out: Path, checks: Checks) -> list[tuple[str, str]]:
    import copy

    from vfates.audit import load_audit, run_audit
    from vfates.experiment import train_arm
    from vfates.gen.factory import make_generator
    from vfates.train.checkpoint_eval import evaluate_checkpoints, peft_checkpoints
    from vfates.train.config import GRPOConfig

    audit_dir = out / "audit"
    run_audit(copy.deepcopy(config["audit"]), audit_dir)
    checks.add(
        "P5.2",
        "torch audit (HF sampling, LoRA-B features, lock) completes",
        None,
        (audit_dir / "lock.json").exists(),
        metric="lock.json written",
    )

    train_cfg = {"seed": 0, **config["train"]}
    arm = str(config.get("train_arm", "fp_tolerance"))
    run_dir = out / "trl_run"
    summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else None
    if summary is None or summary.get("steps_completed") != train_cfg["steps"]:
        summary = train_arm(
            audit_dir,
            arm,
            train_cfg,
            run_dir,
            backend="trl",
            eval_generator=config.get("eval_generator"),
        )
    ckpts = [s for s, _ in peft_checkpoints(run_dir)]
    metrics = read_jsonl(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
    checks.add(
        "P5.3",
        "TRL GRPO + PEFT trains and saves every adapter",
        len(metrics),
        summary["steps_completed"] == train_cfg["steps"]
        and len(metrics) == train_cfg["steps"]
        and ckpts[-1:] == [train_cfg["steps"]],
        metric=f"metric rows (checkpoints {ckpts})",
    )

    evals = [(r["step"], r["split"]) for r in read_jsonl(run_dir / "eval.jsonl")]
    want = {(s, "heldout") for s in [0, *ckpts]}
    checks.add(
        "P5.4",
        "checkpoint evaluator writes the shared per-prompt schema",
        len(evals),
        want <= set(evals) and (run_dir / "eval_prompts.jsonl").exists(),
        metric="eval rows (steps 0 and every checkpoint)",
    )

    audit = load_audit(audit_dir)
    cfg = GRPOConfig.from_dict({**train_cfg, "out_dir": str(run_dir)})
    spec = dict(config.get("eval_generator") or {"backend": "hf"})
    gen = make_generator(spec.get("backend", "hf"), cfg.model_id, **dict(spec.get("kwargs", {})))
    train = [p for p in audit.problems.values() if p.split == "train"]
    held = [p for p in audit.problems.values() if p.split == "heldout"]
    again = evaluate_checkpoints(run_dir, audit.task, cfg, gen, train, held)
    checks.add(
        "P5.5",
        "re-evaluation is idempotent (resumable)",
        len(again),
        not again,
        metric="rows rewritten",
    )
    return [
        (
            "Dry-run eval curve",
            markdown_table(
                ["step", "split", "oracle_acc"],
                [
                    [r["step"], r["split"], r["oracle_acc"]]
                    for r in read_jsonl(run_dir / "eval.jsonl")
                ],
            ),
        )
    ]


def _run_dir(out: Path, arm: str, seed: int) -> Path:
    return out / "runs" / f"{arm}__s{seed}"


def _complete(run_dir: Path, steps: int) -> bool:
    path = run_dir / "summary.json"
    return path.exists() and read_json(path).get("steps_completed") == steps


def _families(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "control": f["control"],
            "kind": f.get("kind", "fp"),
            "shared": list(f.get("shared", [])),
            "local": list(f.get("local", [])),
        }
        for f in config.get("families", [])
    ]


def full_gate(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Full H100 gate. The only scientific gate is the oracle held-out gain."""
    import copy

    from vfates.analysis.outcomes import (
        paired_contrast_seeds,
        require_final_step,
        run_outcome,
    )
    from vfates.audit import run_audit
    from vfates.experiment import train_arm

    started = time.perf_counter()
    out = Path(out_dir)
    full = config["full"]
    gate = full["gate"]
    audit_dir = out / "audit"
    # A resumed audit clock is only the remainder, so only a fresh audit is timed.
    audit_fresh = not audit_dir.exists()
    t_audit = time.perf_counter()
    run_audit(copy.deepcopy(full["audit"]), audit_dir)
    if audit_fresh:
        pools = full["audit"]["pools"]
        write_json(
            out / AUDIT_TIMING_NAME,
            {
                "audit_wall_s": time.perf_counter() - t_audit,
                "completions": (int(pools["train"]) + int(pools["heldout"]))
                * int(full["audit"]["sampling"]["n"]),
            },
        )
    train_cfg = dict(gate["train"])
    steps = int(train_cfg["steps"])
    seeds = [int(s) for s in gate["seeds"]]
    arms = list(gate["arms"])
    for seed in seeds:
        for arm in arms:
            run_dir = _run_dir(out, arm, seed)
            if _complete(run_dir, steps):
                continue
            resume = (run_dir / "checkpoints").exists() or any(run_dir.glob("checkpoint-*"))
            t0 = time.perf_counter()
            train_arm(
                audit_dir,
                arm,
                {**train_cfg, "seed": seed},
                run_dir,
                backend="trl",
                resume=resume,
                eval_generator=gate.get("eval_generator"),
            )
            # The whole train_arm clock: training, the checkpoint evaluation that
            # follows it, and model loading. A resume is only the remainder, so it
            # is not stored as the arm-seed cost.
            if not resume:
                write_json(
                    run_dir / "timing.json",
                    {"train_arm_wall_s": time.perf_counter() - t0, "includes_eval": True},
                )

    run_dirs = [_run_dir(out, arm, seed) for arm in arms for seed in seeds]
    checks = Checks()
    incomplete = [str(path) for path in run_dirs if not _complete(path, steps)]
    checks.add(
        "P5.0",
        "every planned arm-seed finished",
        float(len(run_dirs) - len(incomplete)),
        not incomplete,
        threshold=str(len(run_dirs)),
        metric="completed runs",
    )
    oracle_threshold = float(gate["gates"]["oracle_gain"])
    if "oracle" not in arms and not incomplete:
        checks.add(
            "P5.7",
            "oracle held-out gain on the development pool (the only scientific go/no-go)",
            None,
            False,
            threshold=oracle_threshold,
            metric="oracle arm missing",
        )
        return write_report(
            out,
            "P5 GPU gate (full)",
            checks,
            config=config,
            started=started,
            tables={"incomplete": incomplete},
            sections=[],
        )
    if incomplete:
        checks.add(
            "P5.7",
            "oracle held-out gain on the development pool (the only scientific go/no-go)",
            None,
            False,
            threshold=oracle_threshold,
            metric="runs incomplete",
        )
        return write_report(
            out,
            "P5 GPU gate (full)",
            checks,
            config=config,
            started=started,
            tables={"incomplete": incomplete},
            sections=[],
        )

    require_final_step(run_dirs, steps)
    per_seed: dict[str, list[dict[str, Any]]] = {
        arm: [run_outcome(_run_dir(out, arm, seed), final_step=steps) for seed in seeds]
        for arm in arms
    }
    gain = float(np.mean([row["heldout_delta"] for row in per_seed["oracle"]]))
    checks.add(
        "P5.7",
        "oracle held-out gain on the development pool (the only scientific go/no-go)",
        gain,
        gain >= oracle_threshold,
        threshold=oracle_threshold,
        metric="mean over seeds of held-out final - step 0",
    )

    n_boot = int(gate.get("n_boot", 2000))
    contrast_rows: list[list[Any]] = []
    for fam in _families(gate):
        control = fam["control"]
        if control not in per_seed:
            continue
        for arm in [*fam["shared"], *fam["local"]]:
            if arm not in per_seed:
                continue
            res = paired_contrast_seeds(
                [(_run_dir(out, control, seed), _run_dir(out, arm, seed)) for seed in seeds],
                step=steps,
                n_boot=n_boot,
                seed=0,
                margin=0.05,
            )
            contrast_rows.append(
                [
                    f"{control} - {arm}",
                    res["diff"],
                    res["p_le_0"],
                    res.get("p_tost"),
                    res["n_prompts"],
                ]
            )
            checks.add(
                f"P5.8.{arm}",
                f"fate contrast {control} - {arm} (logged, not a gate)",
                res["diff"],
                True,
                metric=f"prompt-bootstrap p={res['p_le_0']:.4f}, p_tost={res.get('p_tost')}",
                documented=True,
            )

    oracle_finals = [float(row["heldout_final"]) for row in per_seed["oracle"]]
    sd = float(np.std(oracle_finals, ddof=1)) if len(oracle_finals) >= 2 else None
    threshold = float(gate["gates"].get("seed_sd_threshold", 0.03))
    extra = sd is not None and sd > threshold
    h1_h2_seeds = 5 if extra else 3
    checks.add(
        "P5.9",
        "E1 seed rule from the oracle seeds' final held-out accuracies",
        sd,
        True,
        threshold=threshold,
        metric=(
            f"sample SD {sd}; H1/H2 arms use {h1_h2_seeds} seeds (5 when SD > 0.03, otherwise 3)"
        ),
        documented=True,
    )
    power = {
        str(n): minimum_detectable_difference(0.0 if sd is None else sd, n_seeds=n, margin=0.05)
        for n in (3, 5)
    }

    timings = [arm_seed_timing(path) for path in run_dirs]
    train_secs = [row["train_seconds"] for row in timings if row["train_seconds"] is not None]
    eval_secs = [row["eval_seconds"] for row in timings if row["eval_seconds"] is not None]
    rates = [row["gen_tok_s"] for row in timings if row["gen_tok_s"] is not None]
    lengths = [
        row["mean_completion_len"] for row in timings if row["mean_completion_len"] is not None
    ]
    audit_timing = read_json(out / AUDIT_TIMING_NAME) if (out / AUDIT_TIMING_NAME).exists() else {}
    measured: dict[str, Any] = {
        "n_runs": len(train_secs),
        "train_seconds_per_arm_seed": float(np.mean(train_secs)) if train_secs else None,
        "eval_seconds_per_arm_seed": float(np.mean(eval_secs)) if eval_secs else None,
        "gen_tok_s": float(np.mean(rates)) if rates else None,
        "mean_completion_len": float(np.mean(lengths)) if lengths else None,
        "audit_wall_s": audit_timing.get("audit_wall_s"),
        "audit_completions": audit_timing.get("completions"),
        "source": next((row["source"] for row in timings if row["source"]), None),
        "eval_sources": sorted({str(row["eval_source"]) for row in timings if row["eval_source"]}),
    }
    e1_spec = dict(config.get("e1_cost") or {})
    projections: dict[str, Any] = {}
    if measured["train_seconds_per_arm_seed"] is not None and e1_spec:
        for label, runs in (
            ("seeds_3", int(e1_spec.get("full_runs", 39))),
            ("seeds_5_on_h1_h2", int(e1_spec.get("full_runs_if_extra_seeds", 57))),
        ):
            spec = {**e1_spec, "full_runs": runs}
            try:
                projections[label] = project_measured_cost(
                    train_seconds_per_arm_seed=measured["train_seconds_per_arm_seed"],
                    reference_steps=steps,
                    spec=spec,
                    eval_seconds_per_arm_seed=measured["eval_seconds_per_arm_seed"],
                    gen_tok_s=measured["gen_tok_s"],
                    mean_completion_len=measured["mean_completion_len"],
                    audit_seconds=measured["audit_wall_s"],
                    audit_completions=measured["audit_completions"],
                )
            except ValueError as exc:
                projections[label] = {"error": str(exc), "eval_tokens_included": False}
    planned_key = "seeds_5_on_h1_h2" if extra else "seeds_3"
    planned_cost = projections.get(planned_key) or {}
    checks.add(
        "P5.10",
        "measured H100 hours per arm-seed and projected E1 cost (eval tokens included)",
        None
        if measured["train_seconds_per_arm_seed"] is None
        else measured["train_seconds_per_arm_seed"] / 3600.0,
        True,
        metric=(
            f"source={measured['source']}; audit_source={planned_cost.get('audit_source')}; "
            f"planned E1 hours {planned_cost.get('h100_hours')}"
        ),
        documented=True,
    )
    sections = [
        (
            "Fate contrasts (not a gate)",
            markdown_table(
                ["contrast", "diff", "p (diff <= 0)", "p TOST", "prompts"],
                contrast_rows,
            ),
        ),
        (
            "E1 seed rule and minimum detectable difference",
            markdown_table(
                ["seeds", "SE", "MDD at margin 0.05", "power at 0.05"],
                [[n, row["se"], row["mdd"], row["power_at_0.05"]] for n, row in power.items()],
            ),
        ),
    ]
    return write_report(
        out,
        "P5 GPU gate (full)",
        checks,
        config=config,
        started=started,
        tables={
            "oracle_finals": oracle_finals,
            "oracle_sd": sd,
            "h1_h2_seeds": h1_h2_seeds,
            "power": power,
            "timing": measured,
            "e1_projection": projections,
            "e1_projection_planned": planned_key,
            "contrasts": contrast_rows,
        },
        sections=sections,
    )


def run_replication_screen(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Base accuracy and truncation of the two 3B candidates on 128 development prompts."""
    from vfates.gen.factory import make_generator
    from vfates.tasks.chain_sum import ChainSumConfig, ChainSumTask
    from vfates.types import GenOut

    started = time.perf_counter()
    out = Path(out_dir)
    pool_seed = int(config.get("pool_seed", 50_000_000))
    n_prompts = int(config.get("n_prompts", 128))
    max_tokens = int(config.get("max_tokens", 1024))
    samples = int(config.get("samples", 1))
    temperature = float(config.get("temperature", 1.0))
    task = ChainSumTask(ChainSumConfig())
    problems = task.problems("heldout", n_prompts, pool_seed)
    prompts = [task.prompt(problem) for problem in problems]
    models = list(config.get("models") or [LLAMA_3B, QWEN_3B])
    gen_cfg = dict(config.get("generator") or {"backend": "vllm"})
    backend = str(gen_cfg.get("backend", "vllm"))
    if backend not in ("mlx", "hf", "vllm"):
        raise ValueError(f"unknown generation backend {backend!r}")
    kwargs = dict(gen_cfg.get("kwargs") or {})
    scored: dict[str, Any] = {}
    checks = Checks()
    for model_id in models:
        generator = make_generator(cast(Literal["mlx", "hf", "vllm"], backend), model_id, **kwargs)
        batches = generator.generate(
            prompts,
            samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            seed=0,
        )
        correct: list[int] = []
        truncated: list[bool] = []
        for problem, outs in zip(problems, batches, strict=True):
            for out_row in outs:
                if not isinstance(out_row, GenOut):
                    raise TypeError(f"generator returned {type(out_row).__name__}, expected GenOut")
                parsed = task.parse(problem, out_row.text)
                correct.append(int(task.oracle(problem, parsed, out_row.text)))
                truncated.append(out_row.finish_reason == "length")
        scored[model_id] = base_screen_summary(correct, truncated)
        del generator
    llama = scored.get(LLAMA_3B)
    if llama is None:
        choice = {
            "model_id": None,
            "preferred_ok": False,
            "rule": REPLICATION_RULE,
            "reason": "Llama screen was not run",
        }
    else:
        choice = choose_replication_model(llama["accuracy"], llama["truncation"])
    checks.add(
        "P5.S",
        "3B replication model rule (documented; not an E1 gate)",
        None if not llama else llama["accuracy"],
        choice.get("model_id") is not None,
        metric=f"chosen {choice.get('model_id')}; truncation {None if not llama else llama['truncation']}",
        documented=True,
    )
    checks.add(
        "P5.S.scored",
        "both candidate models were scored on the development prompts",
        float(len(scored)),
        len(scored) == len(models)
        and all(row["n"] == n_prompts * samples for row in scored.values()),
        threshold=str(len(models)),
        metric=f"{n_prompts} prompts, {samples} sample(s), {max_tokens} tokens",
    )
    return write_report(
        out,
        "P5 3B replication screen",
        checks,
        config=config,
        started=started,
        tables={"models": scored, "choice": choice, "pool_seed": pool_seed},
        sections=[
            (
                "Model rule",
                REPLICATION_RULE,
            )
        ],
        intro=(
            "Development held-out prompts use the P5 pool seed plus the chain-sum "
            "held-out offset. This screen does not train."
        ),
    )


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P5 GPU gate (dry run, full gate, or the 3B replication screen)."""
    started = time.perf_counter()
    out = Path(out_dir)
    env = _cuda()
    mode = config.get("mode") or ("full" if env["cuda"] else "dry_run")
    if mode == "replication_screen":
        return run_replication_screen(config, out)
    checks = Checks()
    checks.add(
        "P5.1",
        "accelerator environment",
        None,
        True,
        metric=f"gpu={env.get('gpu', 'none')}, mode={mode}, vllm={env.get('vllm')}",
        documented=True,
    )
    sections: list[tuple[str, str]] = []
    cost = _projection(config) if "cost_model" in config else {}
    if cost:
        checks.add(
            "P5.6",
            "projected token-rate cost of the full gate (eval tokens included)",
            cost["usd"],
            True,
            metric=f"USD ({cost['h100_hours']:.1f} H100 h), audit_prompts in the cost model",
            documented=True,
        )
        sections.append(("Cost projection", markdown_table(list(cost), [list(cost.values())])))

    if mode == "dry_run":
        sections += _dry_run(config, out, checks)
        return write_report(
            out,
            "P5 GPU gate (dry run)",
            checks,
            config=config,
            started=started,
            tables={"env": env, "cost": cost},
            sections=sections,
        )
    if mode != "full":
        raise ValueError(f"unknown P5 mode {mode!r}")
    if not env["cuda"]:
        raise RuntimeError("mode=full needs CUDA; run it on the rented GPU")
    report = full_gate(config, out / "gate")
    report["dry_run_checks"] = checks.rows
    return report
