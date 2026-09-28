"""Train one verifier arm on a frozen audit's pools.

The audit directory fixes everything a training run must share with its
forecast: the task and its configuration, the train and held-out problems, and
the fitted arms (rebuilt from ``calibration.json``). A run only chooses the arm,
the trainer backend, and the GRPO hyperparameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vfates.audit import load_audit
from vfates.train.config import GRPOConfig
from vfates.utils.io import write_json
from vfates.verifiers.monitors import rule_monitors


def train_arm(
    audit_dir: str | Path,
    arm: str,
    train_config: GRPOConfig | Mapping[str, Any],
    out_dir: str | Path,
    *,
    backend: str = "mlx",
    resume: bool = False,
    max_train: int | None = None,
    max_heldout: int | None = None,
    eval_generator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run GRPO with ``arm`` as the reward on the audit's train pool; evaluate on held-out.

    ``max_train`` / ``max_heldout`` take a prefix of the frozen pools (for pilots).
    ``eval_generator`` (``{"backend", "kwargs"}``) is the sampler for the TRL
    backend's checkpoint evals; the MLX trainer evaluates in-loop.
    """
    audit = load_audit(audit_dir)
    if arm not in audit.arms:
        raise KeyError(f"arm {arm!r} not in {sorted(audit.arms)}")
    cfg_dict = (
        train_config.to_dict() if isinstance(train_config, GRPOConfig) else dict(train_config)
    )
    cfg_dict["out_dir"] = str(out_dir)
    config = GRPOConfig.from_dict(cfg_dict)
    train = [p for p in audit.problems.values() if p.split == "train"][:max_train]
    held = [p for p in audit.problems.values() if p.split == "heldout"][:max_heldout]
    verifier = audit.arms[arm]
    monitors = rule_monitors(audit.arms)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_json(
        Path(out_dir) / "run.json",
        {
            "audit_dir": str(audit_dir),
            "arm": arm,
            "verifier": verifier.describe(),
            "backend": backend,
            "n_train": len(train),
            "n_heldout": len(held),
        },
    )
    if backend == "mlx":
        from vfates.train.mlx_grpo import MLXGRPOTrainer

        trainer = MLXGRPOTrainer(config, audit.task, verifier, train, held, monitors=monitors)
        return trainer.run(resume=resume)
    if backend == "trl":
        return _train_trl(
            config,
            audit,
            verifier,
            train,
            held,
            monitors,
            Path(out_dir),
            resume=resume,
            eval_generator=eval_generator,
        )
    raise ValueError(f"unknown backend {backend!r}; expected 'mlx' or 'trl'")


def _train_trl(
    config: GRPOConfig,
    audit: Any,
    verifier: Any,
    train: list[Any],
    held: list[Any],
    monitors: dict[str, Any],
    out_dir: Path,
    *,
    resume: bool,
    eval_generator: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """TRL training, then the shared held-out protocol on every saved adapter."""
    import gc

    import torch

    from vfates.gen.factory import make_generator
    from vfates.train.checkpoint_eval import evaluate_checkpoints, peft_checkpoints
    from vfates.train.trl_grpo import build_trl_trainer

    extra = {"eval_strategy": "no", "eval_on_start": False, "save_total_limit": None}
    trainer = build_trl_trainer(
        config, audit.task, verifier, train, held, monitors=monitors, extra_trl_args=extra
    )
    trainer.train(resume_from_checkpoint=True if resume and peft_checkpoints(out_dir) else None)
    steps = int(trainer.state.global_step)
    final = out_dir / f"checkpoint-{steps}"
    if not final.exists():
        trainer.save_model(str(final))
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    spec = dict(eval_generator or {})
    backend = spec.pop("backend", "vllm" if torch.cuda.is_available() else "hf")
    kwargs = dict(spec.pop("kwargs", {}))
    if backend == "vllm":
        kwargs.setdefault("enable_lora", True)
        kwargs.setdefault("max_lora_rank", int(config.lora.rank))
    generator = make_generator(backend, config.model_id, **kwargs)
    evaluate_checkpoints(
        out_dir, audit.task, config, generator, train, held, verifier=verifier, monitors=monitors
    )
    summary = {
        "out_dir": str(out_dir),
        "steps_completed": steps,
        "backend": "trl",
        "eval_generator": generator.describe(),
    }
    write_json(out_dir / "summary.json", summary)
    return summary
