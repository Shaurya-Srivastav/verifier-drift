"""P5: the GPU gate. Dry run on the Mac, the full gate on one H100.

``mode: dry_run`` (default without CUDA) exercises the torch path end to end on
the local device with a tiny model budget: an HF-sampled audit with LoRA-B
features and a lock, two TRL GRPO steps with PEFT, the checkpoint evaluator on
every saved adapter, idempotent re-evaluation, and the cost projection for the
full gate. Everything the H100 run touches is run once here except vLLM.

``mode: full`` (default with CUDA) is the E1 go/no-go on the study model: a
vLLM audit on the Egashira configuration, the lock, then TRL training of every
arm and seed, scored against the lock with the P4 checks (learnability, the
same-rate split, locality of prompt-specific errors, rank agreement, seed
agreement).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.io import read_json, read_jsonl


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
    """Cost of the full gate from the P4 Mac measurements (or the config's defaults)."""
    study = dict(config["cost_model"])
    p4 = Path(config.get("p4_report", "pilots/p4_micro_grpo/results/report.json"))
    measured = None
    if p4.exists():
        measured = read_json(p4).get("tables", {}).get("cost", {}).get("mean_completion_len")
    length = float(measured or study["mean_completion_len"])
    runs = study["arms"] * study["seeds"]
    train_tokens = study["steps"] * study["prompts_per_step"] * study["group_size"] * length * runs
    evals = (study["steps"] // study["eval_every"] + 1) * runs
    eval_tokens = evals * study["eval_prompts"] * study["eval_samples"] * length
    audit_tokens = study["audit_prompts"] * study["audit_samples"] * length
    gen_tokens = train_tokens + eval_tokens + audit_tokens
    hours = (gen_tokens / study["h100_gen_tok_s"] + train_tokens / study["h100_train_tok_s"]) / 3600
    hours *= float(study.get("overhead", 1.25))
    return {
        "mean_completion_len": length,
        "length_source": "P4" if measured else "config",
        "runs": runs,
        "generated_tokens": gen_tokens,
        "trained_tokens": train_tokens,
        "h100_hours": hours,
        "usd": hours * study["usd_per_h100_hour"],
    }


def _dry_run(config: dict[str, Any], out: Path, checks: Checks) -> list[tuple[str, str]]:
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


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P5 GPU gate (dry run or full)."""
    started = time.perf_counter()
    out = Path(out_dir)
    env = _cuda()
    mode = config.get("mode") or ("full" if env["cuda"] else "dry_run")
    checks = Checks()
    checks.add(
        "P5.1",
        "accelerator environment",
        env.get("gpu", "none"),
        True,
        metric=f"mode={mode}, vllm={env.get('vllm')}",
        documented=True,
    )
    sections: list[tuple[str, str]] = []
    cost = _projection(config)
    checks.add(
        "P5.6",
        "projected cost of the full gate",
        cost["usd"],
        True,
        metric=f"USD ({cost['h100_hours']:.1f} H100 h)",
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
    if not env["cuda"]:
        raise RuntimeError("mode=full needs CUDA; run it on the rented GPU")

    from vfates.audit import run_audit
    from vfates.pilots import p4_micro_grpo

    full = config["full"]
    audit_dir = out / "audit"
    run_audit(copy.deepcopy(full["audit"]), audit_dir)
    gate_cfg = {
        **full["gate"],
        "audit_dir": str(audit_dir),
        "backend": "trl",
        "title": "P5 GPU gate (full)",
    }
    return p4_micro_grpo.run(gate_cfg, out / "gate")
