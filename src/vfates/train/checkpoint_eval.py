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

import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vfates.gen.factory import BackendName
from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig
from vfates.train.evaluate import append_eval_samples, eval_log_rows, evaluate_policy
from vfates.train.rewards import MonitorFn
from vfates.types import Problem
from vfates.utils.hashing import stable_seed
from vfates.utils.io import append_jsonl, read_jsonl
from vfates.verifiers.base import Verifier


def release_colocated_engine(trainer: Any) -> None:
    """Drop a colocated vLLM engine before constructing the eval generator.

    TRL 0.25.1 stores that engine on ``trainer.llm``. The eval
    :class:`~vfates.gen.vllm_backend.VLLMGenerator` needs the same GPUs, so it
    is created only after this returns.
    """
    engine = getattr(trainer, "llm", None)
    if engine is None:
        return
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        with contextlib.suppress(Exception):
            shutdown()
    with contextlib.suppress(Exception):
        trainer.llm = None


def make_eval_generator(
    config: GRPOConfig,
    spec: Mapping[str, Any] | None = None,
) -> Any:
    """Sampler for checkpoint eval. Call only after :func:`release_colocated_engine`.

    A vLLM spec receives ``tensor_parallel_size`` from
    ``config.vllm_tensor_parallel_size`` unless the spec already set it, plus
    LoRA loading and the run's ``chat_template_kwargs``.
    """
    from vfates.gen.factory import make_generator

    raw = dict(spec or {})
    name = raw.pop("backend", None)
    if name is None:
        import torch

        name = "vllm" if torch.cuda.is_available() else "hf"
    backend = _eval_backend(str(name))
    kwargs = dict(raw.pop("kwargs", {}))
    if backend == "vllm":
        kwargs.setdefault("tensor_parallel_size", int(config.vllm_tensor_parallel_size))
        if not config.full_finetune:
            kwargs.setdefault("enable_lora", True)
            kwargs.setdefault("max_lora_rank", int(config.lora.rank))
    if config.chat_template_kwargs:
        kwargs.setdefault("chat_template_kwargs", dict(config.chat_template_kwargs))
    if config.model_revision and "revision" not in kwargs:
        kwargs.setdefault("revision", config.model_revision)
    return make_generator(backend, config.model_id, **kwargs)


def _eval_backend(name: str) -> BackendName:
    if name == "mlx":
        return "mlx"
    if name == "hf":
        return "hf"
    if name == "vllm":
        return "vllm"
    raise ValueError(f"unknown eval backend {name!r}; expected 'mlx', 'hf' or 'vllm'")


_FULL_WEIGHT_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def is_adapter_checkpoint(path: str | Path) -> bool:
    """True when ``path`` is a PEFT adapter directory."""
    return (Path(path) / "adapter_config.json").is_file()


def is_full_model_checkpoint(path: str | Path) -> bool:
    """True when ``path`` holds a full Hugging Face weight file."""
    directory = Path(path)
    return any((directory / name).is_file() for name in _FULL_WEIGHT_NAMES)


def peft_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` of every ``checkpoint-<step>`` holding a PEFT adapter, by step."""
    return [
        (step, path) for step, path in saved_checkpoints(run_dir) if is_adapter_checkpoint(path)
    ]


def saved_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` of every ``checkpoint-<step>`` that eval can load, by step.

    A directory counts when it holds a PEFT adapter or a full-model weight file.
    """
    found = []
    for path in Path(run_dir).glob("checkpoint-*"):
        suffix = path.name.split("-", 1)[1]
        if suffix.isdigit() and (is_adapter_checkpoint(path) or is_full_model_checkpoint(path)):
            found.append((int(suffix), path))
    return sorted(found)


def apply_eval_checkpoint(generator: Any, path: str | Path | None) -> None:
    """Point ``generator`` at a checkpoint. ``None`` is the base model.

    Adapter directories use ``set_adapter``. Full-model directories use
    ``load_full_checkpoint`` (HF and vLLM). Step 0 restores both.
    """
    if path is not None and is_full_model_checkpoint(path) and not is_adapter_checkpoint(path):
        loader = getattr(generator, "load_full_checkpoint", None)
        if not callable(loader):
            raise RuntimeError(
                f"{path} is a full-model checkpoint and {type(generator).__name__} "
                "has no load_full_checkpoint"
            )
        loader(str(path))
        return
    restorer = getattr(generator, "load_full_checkpoint", None)
    if path is None and callable(restorer):
        restorer(None)
    generator.set_adapter(None if path is None else str(path))


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

    ``generator`` must implement ``set_adapter(path | None)``. A full-model
    checkpoint also needs ``load_full_checkpoint``
    (:class:`~vfates.gen.hf_backend.HFGenerator`,
    :class:`~vfates.gen.vllm_backend.VLLMGenerator`). Draw seeds are
    ``stable_seed(config.seed, "eval", step, split)``, as in the MLX trainer.
    """
    run_dir = Path(run_dir)
    log = run_dir / "eval.jsonl"
    done = {(r["step"], r["split"]) for r in read_jsonl(log)} if log.exists() else set()
    points: list[tuple[int, Path | None]] = [(0, None), *saved_checkpoints(run_dir)]
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
            apply_eval_checkpoint(generator, path)
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
