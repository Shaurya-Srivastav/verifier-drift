"""Evaluate saved LoRA checkpoints with the trainer-independent held-out protocol.

TRL's built-in eval logs pooled reward means under the training sampler; the
primary endpoint needs per-prompt oracle accuracy under the eval protocol of
:func:`vfates.train.evaluate.evaluate_policy`. This module evaluates the base
model (step 0) and every PEFT checkpoint ``checkpoint-<step>`` in a run
directory and writes the same ``eval.jsonl`` / ``eval_prompts.jsonl`` rows as
:class:`~vfates.train.mlx_grpo.MLXGRPOTrainer`. Tasks that record a content
judge also append ``eval_samples.jsonl`` (prompt, text, judge verdict).
Evaluated ``(step, split)`` pairs are skipped on rerun.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig
from vfates.train.evaluate import append_eval_samples, eval_log_rows, evaluate_policy
from vfates.train.rewards import MonitorFn
from vfates.types import Problem
from vfates.utils.hashing import stable_seed
from vfates.utils.io import append_jsonl, read_jsonl
from vfates.verifiers.base import Verifier


def peft_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` of every ``checkpoint-<step>`` holding a PEFT adapter, by step."""
    found = []
    for path in Path(run_dir).glob("checkpoint-*"):
        suffix = path.name.split("-", 1)[1]
        if suffix.isdigit() and (path / "adapter_config.json").exists():
            found.append((int(suffix), path))
    return sorted(found)


def evaluate_checkpoints(
    run_dir: str | Path,
    task: Task,
    config: GRPOConfig,
    generator: Any,
    train_problems: Sequence[Problem],
    heldout_problems: Sequence[Problem],
    *,
    verifier: Verifier | None = None,
    monitors: dict[str, MonitorFn] | None = None,
    steps: Sequence[int] | None = None,
) -> list[tuple[int, str]]:
    """Evaluate step 0 and each checkpoint; return the ``(step, split)`` pairs written.

    ``generator`` must implement ``set_adapter(path | None)``
    (:class:`~vfates.gen.hf_backend.HFGenerator`,
    :class:`~vfates.gen.vllm_backend.VLLMGenerator`). Draw seeds are
    ``stable_seed(config.seed, "eval", step, split)``, as in the MLX trainer.
    """
    run_dir = Path(run_dir)
    log = run_dir / "eval.jsonl"
    done = {(r["step"], r["split"]) for r in read_jsonl(log)} if log.exists() else set()
    points: list[tuple[int, Path | None]] = [(0, None), *peft_checkpoints(run_dir)]
    splits = {
        "heldout": list(heldout_problems)[: config.eval_heldout_n],
        "train": list(train_problems)[: config.eval_train_n],
    }
    written = []
    # HF and vLLM encode_prompt does not truncate. Training does (TRL
    # truncate_prompt_tokens, MLX encode_prompt). Set the same suffix cap
    # for the duration of this pass.
    had_limit = hasattr(generator, "max_prompt_tokens")
    previous_limit = getattr(generator, "max_prompt_tokens", None)
    generator.max_prompt_tokens = int(config.max_prompt_length)
    try:
        for step, path in points:
            if steps is not None and step not in steps:
                continue
            todo = [s for s, probs in splits.items() if probs and (step, s) not in done]
            if not todo:
                continue
            generator.set_adapter(None if path is None else str(path))
            for split in todo:
                stats = evaluate_policy(
                    generator,
                    task,
                    splits[split],
                    n_samples=config.eval_samples_per_prompt,
                    max_tokens=int(config.eval_max_tokens or config.max_tokens),
                    temperature=config.eval_temperature,
                    seed=stable_seed(config.seed, "eval", step, split),
                    verifier=verifier,
                    monitors=monitors,
                )
                append_eval_samples(run_dir, step, split, stats)
                row, prompt_row = eval_log_rows(step, split, stats)
                append_jsonl(log, row)
                append_jsonl(run_dir / "eval_prompts.jsonl", prompt_row)
                written.append((step, split))
    finally:
        if had_limit:
            generator.max_prompt_tokens = previous_limit
        elif hasattr(generator, "max_prompt_tokens"):
            del generator.max_prompt_tokens
    return written
