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

from vfates.audit import LORA_INIT_NAME, artifact_sha256, load_audit, read_lock_record
from vfates.train.config import GRPOConfig
from vfates.utils.env import environment_report
from vfates.utils.io import read_json, write_json
from vfates.verifiers.monitors import rule_monitors

_ARTIFACT_FILES = {
    "problems": "problems.jsonl",
    "completions": "completions.jsonl",
    "calibration": "calibration.json",
    "features": "features.npz",
    "lora_init": LORA_INIT_NAME,
}


def _with_lora_init(config: GRPOConfig, path: str, digest: str) -> GRPOConfig:
    """Attach the audited LoRA A file and the lock's digest of it."""
    data = config.to_dict()
    data["lora_init"] = path
    data["lora_init_sha256"] = digest
    return GRPOConfig.from_dict(data)


def _arms_file(audit_dir: Path) -> Path | None:
    cfg_path = audit_dir / "config.json"
    if not cfg_path.is_file():
        raise ValueError(f"lock lists arms_file but {cfg_path} is missing")
    cfg = read_json(cfg_path)["config"]
    raw = (cfg.get("arms") or {}).get("file")
    if not raw:
        return None
    return Path(str(raw))


def _verify_lock_artifacts(audit_dir: Path, record: Mapping[str, Any], grades: str) -> None:
    """Refuse a lock whose named files are missing or no longer hash to the record.

    ``grades`` is the audit's completions regraded by the current parser and oracle.
    """
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return
    for name, digest in artifacts.items():
        if name == "grades":
            if grades != digest:
                raise ValueError(
                    f"lock artifact 'grades' {digest} does not match the current parser and "
                    f"oracle ({grades}); the reward would not be the audited oracle"
                )
            continue
        if name == "arms_file":
            path = _arms_file(audit_dir)
        else:
            filename = _ARTIFACT_FILES.get(str(name))
            if filename is None:
                raise ValueError(f"lock lists unknown artifact {name!r}")
            path = audit_dir / filename
        if path is None or not path.is_file():
            raise ValueError(f"lock artifact {name!r} is missing ({path})")
        got = artifact_sha256(path)
        if got != digest:
            raise ValueError(
                f"lock artifact {name!r} sha256 {got} does not match the lock ({digest})"
            )


def _check_audited_policy(audit_config: Mapping[str, Any], config: GRPOConfig) -> None:
    """Refuse training that samples a different policy from the one the audit forecast.

    The locked forecasts describe the audit generator's model at its sampling
    ``max_tokens``, ``temperature`` and ``top_p``. Rollouts and the held-out
    evaluation must use the same model, token budget and temperature.
    """
    generator = audit_config.get("generator") or {}
    sampling = audit_config.get("sampling") or {}
    wanted: list[tuple[str, Any, Any]] = []
    if generator.get("model_id") is not None:
        wanted.append(("model_id", str(generator["model_id"]), config.model_id))
    if sampling.get("max_tokens") is not None:
        budget = int(sampling["max_tokens"])
        wanted.append(("max_tokens", budget, int(config.max_tokens)))
        wanted.append(("eval_max_tokens", budget, int(config.eval_max_tokens or 0)))
    if sampling.get("temperature") is not None:
        temperature = float(sampling["temperature"])
        wanted.append(("temperature", temperature, float(config.temperature)))
        wanted.append(("eval_temperature", temperature, float(config.eval_temperature)))
    if sampling.get("top_p") is not None:
        wanted.append(("top_p", float(sampling["top_p"]), float(config.top_p)))
    mismatches = [
        f"{name} {got!r} (audit {audit!r})" for name, audit, got in wanted if got != audit
    ]
    if mismatches:
        raise ValueError("training does not sample the audited policy: " + "; ".join(mismatches))


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
    audit_path = Path(audit_dir)
    lock_path = audit_path / "lock.json"
    lock_record = None
    lock_sha = None
    if lock_path.exists():
        lock_record = read_lock_record(lock_path)
        lock_sha = lock_record.get("sha256")
        if not isinstance(lock_sha, str) or not lock_sha:
            raise ValueError(f"lock file failed verification: {lock_path}")
        _verify_lock_artifacts(audit_path, lock_record, audit.grades_sha256)
    init_path = audit_path / LORA_INIT_NAME
    lora_sha: str | None = None
    locked_init = None
    if lock_record is not None:
        artifacts = lock_record.get("artifacts") or {}
        if isinstance(artifacts, Mapping):
            locked_init = artifacts.get("lora_init")
    if init_path.is_file():
        file_sha = artifact_sha256(init_path)
        if lock_record is not None and file_sha != locked_init:
            raise ValueError(
                f"{init_path.name} sha256 {file_sha} does not match "
                f"lock artifacts['lora_init'] ({locked_init!r})"
            )
        lora_sha = file_sha if lock_record is None else str(locked_init)
    elif locked_init:
        raise ValueError(
            f"lock records lora_init {locked_init} but {init_path.name} is not in the audit"
        )
    cfg_dict = (
        train_config.to_dict() if isinstance(train_config, GRPOConfig) else dict(train_config)
    )
    cfg_dict["out_dir"] = str(out_dir)
    config = GRPOConfig.from_dict(cfg_dict)
    _check_audited_policy(read_json(audit_path / "config.json")["config"], config)
    if lora_sha is not None:
        config = _with_lora_init(config, str(init_path), lora_sha)
    train = [p for p in audit.problems.values() if p.split == "train"][:max_train]
    held = [p for p in audit.problems.values() if p.split == "heldout"][:max_heldout]
    verifier = audit.arms[arm]
    monitors = rule_monitors(audit.arms)
    env = environment_report()
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
            "lock_sha256": lock_sha,
            "lora_init_sha256": lora_sha,
            "git_commit": env.get("git_commit"),
            "git_dirty": env.get("git_dirty"),
            "git_head": env.get("git_head"),
            "environment": env,
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
    from vfates.train.trl_grpo import RUN_METADATA_NAME, build_trl_trainer

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
    metadata = read_json(out_dir / RUN_METADATA_NAME)
    summary = {
        "out_dir": str(out_dir),
        "steps_completed": steps,
        "backend": "trl",
        "eval_generator": generator.describe(),
        "lora_seed": metadata["lora_seed"],
        "lora_init_sha256": metadata["lora_init_sha256"],
        "rollout_seeded": metadata["rollout_seeded"],
    }
    write_json(out_dir / "summary.json", summary)
    return summary
