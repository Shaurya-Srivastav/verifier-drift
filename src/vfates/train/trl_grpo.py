"""TRL GRPO trainer wired to the same objective as :mod:`vfates.train.mlx_grpo`.

``seq_mean`` / ``token_mean`` / ``dr_constant`` map to TRL ``loss_type``
``grpo`` / ``dapo`` / ``dr_grpo``. ``adv_mode`` ``std`` / ``mean`` map to
``scale_rewards`` ``group`` / ``none``. ``vllm_importance_sampling_correction``
stays off unless the config turns it on: the extra truncated-IS term is not in
the MLX loss.

TRL's eval sampler repeats each held-out prompt ``num_generations`` times, so
``eval_samples_per_prompt`` is not an independent knob here, and its pooled
means go to ``trl_eval.jsonl``. The endpoint (``eval.jsonl``,
``eval_prompts.jsonl``) comes from :mod:`vfates.train.checkpoint_eval`, run on
the saved adapters after training.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vfates.gen.base import chat_messages
from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig
from vfates.train.grpo_core import TRL_LOSS_TYPE, TRL_SCALE_REWARDS
from vfates.train.rewards import RewardContext, make_trl_reward_funcs, reward_weights_for
from vfates.types import Problem
from vfates.utils.env import environment_report
from vfates.utils.io import append_jsonl, write_json
from vfates.verifiers.base import Verifier

_REWARD_HEADS = ("verifier", "oracle", "false_positive", "false_negative")


def trl_grpo_kwargs(
    config: GRPOConfig,
    *,
    n_reward_funcs: int,
    use_vllm: bool | None = None,
    vllm_mode: str | None = None,
    bf16: bool = False,
) -> dict[str, Any]:
    """Keyword arguments for ``trl.GRPOConfig`` (0.25.1).

    One optimizer step sees ``prompts_per_step * group_size`` completions:
    ``per_device_train_batch_size = micro_batch_size`` and
    ``gradient_accumulation_steps`` makes up the rest. ``steps_per_generation``
    is left unset so TRL sets it to the accumulation count.
    """
    if not isinstance(config, GRPOConfig):
        config = GRPOConfig.from_dict(config)
    completions = config.completions_per_step
    micro = config.micro_batch_size
    grad_accum = completions // micro
    vllm = config.use_vllm if use_vllm is None else bool(use_vllm)
    mode = config.vllm_mode if vllm_mode is None else vllm_mode
    scheduler = "constant_with_warmup" if config.warmup_steps > 0 else "constant"
    return {
        "output_dir": config.out_dir,
        "learning_rate": float(config.lr),
        "adam_beta1": float(config.adam_betas[0]),
        "adam_beta2": float(config.adam_betas[1]),
        "weight_decay": float(config.weight_decay),
        "max_grad_norm": float(config.grad_clip),
        "warmup_steps": int(config.warmup_steps),
        "lr_scheduler_type": scheduler,
        "num_generations": int(config.group_size),
        "per_device_train_batch_size": int(micro),
        "gradient_accumulation_steps": int(grad_accum),
        "per_device_eval_batch_size": int(config.group_size),
        "max_completion_length": int(config.max_tokens),
        "max_prompt_length": int(config.max_prompt_length),
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "beta": float(config.beta_kl),
        "loss_type": TRL_LOSS_TYPE[config.loss_agg],
        "scale_rewards": TRL_SCALE_REWARDS[config.adv_mode],
        "epsilon": float(config.eps_low),
        "epsilon_high": float(config.eps_high),
        "num_iterations": int(config.num_iterations),
        "max_steps": int(config.steps),
        "logging_steps": 1,
        "logging_strategy": "steps",
        "eval_strategy": "steps" if config.eval_heldout_n > 0 else "no",
        "eval_steps": int(config.eval_every),
        "eval_on_start": config.eval_heldout_n > 0,
        "save_strategy": "steps",
        "save_steps": int(config.save_every),
        "save_total_limit": 2,
        "seed": int(config.seed),
        "report_to": [],
        "remove_unused_columns": False,
        "bf16": bool(bf16),
        "fp16": False,
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "use_vllm": vllm,
        "vllm_mode": mode,
        "vllm_gpu_memory_utilization": float(config.vllm_gpu_memory_utilization),
        "vllm_importance_sampling_correction": bool(config.vllm_importance_sampling_correction),
        "reward_weights": reward_weights_for(n_reward_funcs),
        "shuffle_dataset": config.data_regime == "multi_epoch",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "disable_tqdm": True,
    }


def rates_from_reward_means(means: Mapping[str, float | None]) -> dict[str, float | None]:
    """Recover pooled rates from TRL's per-head means.

    ``mean(FP) / (1 - mean(O))`` equals the pooled FPR when every sample is
    scored (no NaNs). The same identity gives FNR and precision. Youden's J
    needs both rates.
    """
    v = means.get("verifier")
    o = means.get("oracle")
    fp = means.get("false_positive")
    fn = means.get("false_negative")
    fpr = None if o is None or fp is None or not o < 1.0 else fp / (1.0 - o)
    fnr = None if o is None or fn is None or not o > 0.0 else fn / o
    precision = None if v is None or fp is None or not v > 0.0 else (v - fp) / v
    youden = None if fpr is None or fnr is None else (1.0 - fnr) - fpr
    return {
        "accept_rate": v,
        "oracle_acc": o,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "J": youden,
    }


def metric_row_from_trl_logs(
    logs: Mapping[str, Any],
    *,
    step: int,
    monitor_names: Sequence[str],
) -> dict[str, Any]:
    """One ``metrics.jsonl`` row from a TRL ``on_log`` payload.

    ``tokens_generated`` is TRL's cumulative prompt+completion token counter
    (``num_tokens``), not the step's completion count. ``gen_seconds`` and
    ``train_seconds`` are not split in TRL's logs. ``monitors_wrong`` cannot be
    recovered from marginal means, so it is ``None``.
    """
    heads = list(_REWARD_HEADS) + [f"monitor_{name}" for name in monitor_names]
    means = {name: _optional_float(logs.get(f"rewards/{name}/mean")) for name in heads}
    rates = rates_from_reward_means(means)
    monitors = {name: means[f"monitor_{name}"] for name in monitor_names}
    zero = _optional_float(logs.get("frac_reward_zero_std"))
    return {
        "step": int(step),
        "wall_s": None,
        "tokens_generated": _optional_int(logs.get("num_tokens")),
        "gen_seconds": None,
        "train_seconds": None,
        "reward_V_mean": rates["accept_rate"],
        "oracle_mean": rates["oracle_acc"],
        "onpolicy": {**rates, "monitors": monitors, "monitors_wrong": None, "n": None},
        "frac_informative_groups": None if zero is None else 1.0 - zero,
        "mean_completion_length": _optional_float(logs.get("completions/mean_length")),
        "max_completion_length": _optional_float(logs.get("completions/max_length")),
        "truncation_rate": _optional_float(logs.get("completions/clipped_ratio")),
        "loss": _optional_float(logs.get("loss")),
        "grad_norm": _optional_float(logs.get("grad_norm")),
        "lr": _optional_float(logs.get("learning_rate")),
        "kl": _optional_float(logs.get("kl")),
        "monitors": monitors,
        "monitors_wrong": None,
    }


def eval_row_from_trl_logs(
    logs: Mapping[str, Any],
    *,
    step: int,
    monitor_names: Sequence[str],
    split: str = "heldout",
) -> dict[str, Any]:
    """One ``eval.jsonl`` row. TRL eval is the held-out set."""
    row = metric_row_from_trl_logs(logs, step=step, monitor_names=monitor_names)
    return {
        "step": int(step),
        "split": split,
        "oracle_acc": row["oracle_mean"],
        "n_prompts": None,
        "accept_rate": row["reward_V_mean"],
        "fpr": row["onpolicy"]["fpr"],
        "fnr": row["onpolicy"]["fnr"],
        "precision": row["onpolicy"]["precision"],
        "J": row["onpolicy"]["J"],
        "monitors": row["monitors"],
        "monitors_wrong": None,
        "mean_length": row["mean_completion_length"],
        "truncation_rate": row["truncation_rate"],
    }


class VfatesLogCallback:
    """Appends the shared JSONL schema. Subclasses ``TrainerCallback`` once TRL is imported."""

    def __init__(self, out_dir: str, monitor_names: Sequence[str]) -> None:
        self.out_dir = Path(out_dir)
        self.monitor_names = list(monitor_names)

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        del args, control, kwargs
        if not logs:
            return
        flat = {str(key): _plain(value) for key, value in logs.items()}
        step = int(getattr(state, "global_step", 0) or 0)
        if any(key.startswith("eval_rewards/") for key in flat):
            stripped = {
                key[len("eval_") :] if key.startswith("eval_") else key: value
                for key, value in flat.items()
            }
            append_jsonl(
                self.out_dir / "trl_eval.jsonl",
                eval_row_from_trl_logs(stripped, step=step, monitor_names=self.monitor_names),
            )
            return
        if "rewards/verifier/mean" not in flat:
            return
        append_jsonl(
            self.out_dir / "metrics.jsonl",
            metric_row_from_trl_logs(flat, step=step, monitor_names=self.monitor_names),
        )


def build_trl_trainer(
    config: GRPOConfig | dict[str, Any],
    task: Task,
    verifier: Verifier,
    train_problems: Sequence[Problem],
    heldout_problems: Sequence[Problem],
    monitors: dict[str, Any] | None = None,
    use_vllm: bool | None = None,
    vllm_mode: str | None = None,
    extra_trl_args: dict[str, Any] | None = None,
    model: Any = None,
    processing_class: Any = None,
) -> Any:
    """Build a ``trl.GRPOTrainer`` for this experiment.

    Dataset rows are ``{"prompt": chat messages, "pid": pid}``. Reward heads come
    from :func:`vfates.train.rewards.make_trl_reward_funcs` with weights
    ``[1, 0, ...]``. ``model`` and ``processing_class`` override the repo id so
    tests can inject a tiny module. Torch, TRL and PEFT are imported here.
    """
    import torch
    from transformers import TrainerCallback
    from trl import GRPOConfig as TrlGRPOConfig
    from trl import GRPOTrainer

    if not isinstance(config, GRPOConfig):
        config = GRPOConfig.from_dict(config)
    train_problems = list(train_problems)
    heldout_problems = list(heldout_problems)
    _check_pools(config, train_problems)
    monitors = dict(monitors or {})
    by_pid = {problem.pid: problem for problem in train_problems}
    for problem in heldout_problems:
        if problem.pid in by_pid:
            raise ValueError(f"pid {problem.pid!r} is in both the train and held-out pools")
        by_pid[problem.pid] = problem
    ctx = RewardContext(task, verifier, by_pid, seed=config.seed, monitors=monitors)
    funcs = make_trl_reward_funcs(ctx)
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config.to_dict())
    write_json(out / "environment_report.json", environment_report())

    cuda = bool(torch.cuda.is_available())
    kwargs = trl_grpo_kwargs(
        config,
        n_reward_funcs=len(funcs),
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        bf16=cuda,
    )
    if extra_trl_args:
        kwargs.update(extra_trl_args)
    args = TrlGRPOConfig(**kwargs)

    heldout = heldout_problems[: config.eval_heldout_n]
    eval_ds = _dataset(task, heldout) if heldout else None
    if eval_ds is not None and config.eval_samples_per_prompt != config.group_size:
        warnings.warn(
            "TRL repeats each eval prompt num_generations "
            f"({config.group_size}) times; eval_samples_per_prompt="
            f"{config.eval_samples_per_prompt} is not applied.",
            stacklevel=2,
        )
    if eval_ds is None:
        args.eval_strategy = "no"
        args.eval_on_start = False

    class _Callback(TrainerCallback):
        def __init__(self, out_dir: str, names: list[str]) -> None:
            super().__init__()
            self._inner = VfatesLogCallback(out_dir, names)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self._inner.on_log(args, state, control, logs=logs, **kwargs)

    trainer = GRPOTrainer(
        model=config.model_id if model is None else model,
        reward_funcs=funcs,  # type: ignore[arg-type]  # trl stub
        args=args,
        train_dataset=_dataset(task, train_problems),
        eval_dataset=eval_ds,
        processing_class=processing_class,
        peft_config=_peft_config(config, model),
        callbacks=[_Callback(config.out_dir, sorted(monitors))],
    )
    return trainer


def _peft_config(config: GRPOConfig, model: Any) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=int(config.lora.rank),
        lora_alpha=float(config.lora.alpha),  # type: ignore[arg-type]  # peft stub
        lora_dropout=float(config.lora.dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.peft_keys(),
        layers_to_transform=_lora_layer_ids(config, model),
    )


def _lora_layer_ids(config: GRPOConfig, model: Any) -> list[int] | None:
    n = int(config.lora.num_layers)
    if n < 0:
        return None
    total = _hidden_layers(config, model)
    if n > total:
        raise ValueError(f"lora.num_layers={n} exceeds the model's {total} layers")
    return list(range(total - n, total))


def _hidden_layers(config: GRPOConfig, model: Any) -> int:
    if (
        model is not None
        and not isinstance(model, str)
        and getattr(model, "config", None) is not None
    ):
        return int(model.config.num_hidden_layers)
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(config.model_id).num_hidden_layers)


def _dataset(task: Task, problems: Sequence[Problem]) -> Any:
    from datasets import Dataset

    rows = [
        {"prompt": chat_messages(task.prompt(problem), task.system_prompt()), "pid": problem.pid}
        for problem in problems
    ]
    return Dataset.from_list(rows)


def _check_pools(config: GRPOConfig, train_problems: Sequence[Problem]) -> None:
    if len(train_problems) < config.prompts_per_step:
        raise ValueError(
            f"need at least prompts_per_step={config.prompts_per_step} train problems "
            f"for one GRPO batch, got {len(train_problems)}"
        )
    if config.data_regime == "streaming":
        need = config.steps * config.prompts_per_step
        if len(train_problems) < need:
            raise RuntimeError(
                f"streaming train pool exhausted: need {need} problems "
                f"({config.steps} steps x {config.prompts_per_step}), got {len(train_problems)}"
            )


def _plain(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _optional_float(value: Any) -> float | None:
    value = _plain(value)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)
