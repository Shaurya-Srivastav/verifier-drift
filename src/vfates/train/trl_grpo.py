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
the saved checkpoints after training. A full-parameter run
(``full_finetune``) saves the whole model, not an adapter.

LoRA A is drawn from ``config.lora_seed`` (or loaded from ``config.lora_init``)
before ``GRPOTrainer`` is constructed. TRL 0.25.1 calls ``set_seed`` only inside
``Trainer.__init__``, which is after ``get_peft_model``, so passing ``peft_config``
would draw A from whatever the process RNG happened to be. A ``PeftModel`` passed
with ``peft_config=None`` is accepted: ``prepare_peft_model`` re-wraps only when
``peft_config`` is not ``None`` (``trl/models/utils.py`` in 0.25.1).

vLLM training rollouts are not seeded. TRL 0.25.1 colocate generation builds one
``SamplingParams`` for the batch (``grpo_trainer.py`` ``_generate_single_turn``)
with ``n=1`` and no ``seed``. The batch is each prompt repeated ``group_size``
times. One shared seed can make those duplicates identical and the advantage
zero, and this TRL version does not expose a per-request seed. ``run_metadata.json``
records ``rollout_seeded: false`` on that path. Do not add a seed to
``generation_kwargs``.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vfates.gen.base import chat_messages
from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig, NonFiniteLossError
from vfates.train.grpo_core import TRL_LOSS_TYPE, TRL_SCALE_REWARDS
from vfates.train.rewards import RewardContext, make_trl_reward_funcs, reward_weights_for
from vfates.types import Problem
from vfates.utils.env import environment_report
from vfates.utils.io import append_jsonl, write_json
from vfates.verifiers.base import Verifier

RUN_METADATA_NAME = "run_metadata.json"
_PEFT_PREFIX = "base_model.model."

_REWARD_HEADS = ("verifier", "oracle", "false_positive", "false_negative")


def global_batch_layout(config: GRPOConfig, world_size: int) -> tuple[int, int]:
    """``(micro_batch_size, gradient_accumulation_steps)`` for one optimizer step.

    The step always contains ``prompts_per_step * group_size`` completions:
    ``gradient_accumulation_steps = total // (micro_batch_size * world_size)``.
    ``world_size`` is the trainer process count (one process per GPU under FSDP).
    """
    world = int(world_size)
    if world < 1:
        raise ValueError(f"world_size must be >= 1, got {world}")
    total = int(config.completions_per_step)
    micro = int(config.micro_batch_size)
    denom = micro * world
    if total % denom != 0:
        raise ValueError(
            f"prompts_per_step * group_size ({total}) must be divisible by "
            f"micro_batch_size * world_size ({micro} * {world} = {denom})"
        )
    tp = int(config.vllm_tensor_parallel_size)
    if world % tp != 0:
        raise ValueError(f"vllm_tensor_parallel_size ({tp}) must divide world_size ({world})")
    return micro, total // denom


def training_world_size() -> int:
    """Process count: initialized ``torch.distributed``, else ``WORLD_SIZE``, else 1."""
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except ImportError:
        pass
    # torchrun and accelerate set RANK together with WORLD_SIZE. A stray
    # WORLD_SIZE in a login shell must not change a single-process unit test.
    if os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None:
        raw = os.environ.get("WORLD_SIZE")
        if raw:
            return int(raw)
    return 1


def resolve_attn_implementation(requested: str | None, *, cuda: bool) -> str:
    """``flash_attention_2`` on CUDA when flash-attn imports, otherwise ``sdpa``.

    An explicit ``sdpa`` or ``eager`` request is kept. ``flash_attention_2``
    falls back to ``sdpa`` when CUDA is absent or the package does not import.
    """
    if requested not in (None, "flash_attention_2"):
        return str(requested)
    if cuda and _flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def trl_prompt_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Prompt ids TRL 0.25.1's generate path builds for one conversational prompt.

    ``GRPOTrainer._generate_single_turn`` calls
    ``processing_class.apply_chat_template(..., add_generation_prompt=True,
    tokenize=True, **chat_template_kwargs)`` (``grpo_trainer.py`` around the
    transformers generate branch). ``GRPOConfig.chat_template_kwargs`` is the
    field that supplies those kwargs (``grpo_config.py`` in v0.25.1). This is
    the same call as :func:`vfates.gen.chat.encode_prompt`.
    """
    from vfates.gen.chat import _as_id_list

    encoded = tokenizer.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        tokenize=True,
        **dict(chat_template_kwargs or {}),
    )
    return _as_id_list(encoded)


def trl_grpo_kwargs(
    config: GRPOConfig,
    *,
    n_reward_funcs: int,
    use_vllm: bool | None = None,
    vllm_mode: str | None = None,
    bf16: bool = False,
    world_size: int = 1,
) -> dict[str, Any]:
    """Keyword arguments for ``trl.GRPOConfig`` (0.25.1).

    One optimizer step sees ``prompts_per_step * group_size`` completions on
    any ``world_size``: ``per_device_train_batch_size = micro_batch_size`` and
    ``gradient_accumulation_steps = total // (micro_batch_size * world_size)``.
    ``steps_per_generation`` is left unset so TRL sets it to the accumulation
    count and ``generation_batch_size`` equals the global completion count.

    Forwarded TRL 0.25.1 names, checked against ``trl/trainer/grpo_config.py``
    at tag v0.25.1: ``chat_template_kwargs``, ``vllm_tensor_parallel_size``,
    ``vllm_enable_sleep_mode``.
    """
    if not isinstance(config, GRPOConfig):
        config = GRPOConfig.from_dict(config)
    micro, grad_accum = global_batch_layout(config, world_size)
    vllm = config.use_vllm if use_vllm is None else bool(use_vllm)
    mode = config.vllm_mode if vllm_mode is None else vllm_mode
    scheduler = "constant_with_warmup" if config.warmup_steps > 0 else "constant"
    out = {
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
        "vllm_tensor_parallel_size": int(config.vllm_tensor_parallel_size),
        "vllm_enable_sleep_mode": bool(config.vllm_enable_sleep_mode),
        "vllm_importance_sampling_correction": bool(config.vllm_importance_sampling_correction),
        "chat_template_kwargs": (
            None if config.chat_template_kwargs is None else dict(config.chat_template_kwargs)
        ),
        "reward_weights": reward_weights_for(n_reward_funcs),
        "shuffle_dataset": config.data_regime == "multi_epoch",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "disable_tqdm": True,
        # A non-finite loss must reach the guard below, not be replaced by the
        # average of earlier losses.
        "logging_nan_inf_filter": False,
    }
    if config.full_finetune:
        # torch AdamW, so moments follow the bf16 parameter dtype. A fused
        # optimizer can upcast the state and change the memory plan.
        out["optim"] = "adamw_torch"
    return out


def trl_run_metadata(
    config: GRPOConfig,
    *,
    use_vllm: bool,
    world_size: int = 1,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Run facts that are not optimizer hyperparameters.

    ``rollout_seeded`` is false when training samples with colocated vLLM: TRL
    0.25.1 does not pass a per-request seed, and a single seed on the repeated
    prompt group can collapse it. The Hugging Face generate path follows the
    torch RNG seeded by :func:`transformers.set_seed` and TRL's later
    ``set_seed(args.seed, device_specific=True)``.

    ``tp_size`` is ``vllm_tensor_parallel_size``. ``dp_size`` is
    ``world_size // tp_size`` (TRL's colocate groups). ``attn_implementation``
    is the resolved kernel, not the raw config request.
    """
    tp = int(config.vllm_tensor_parallel_size)
    world = int(world_size)
    if tp < 1 or world % tp != 0:
        raise ValueError(f"vllm_tensor_parallel_size ({tp}) must divide world_size ({world})")
    kwargs = config.chat_template_kwargs
    return {
        "lora_seed": int(config.lora_seed),
        "lora_init": config.lora_init,
        "lora_init_sha256": config.lora_init_sha256,
        "rollout_seeded": not bool(use_vllm),
        "vllm_mode": config.vllm_mode,
        "tp_size": tp,
        "dp_size": world // tp,
        "world_size": world,
        "hardware_class": config.hardware_class,
        "chat_template_kwargs": None if kwargs is None else dict(kwargs),
        "attn_implementation": attn_implementation,
        "model_revision": config.model_revision,
        "parameterisation": config.parameterisation,
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
    wall_s: float | None = None,
) -> dict[str, Any]:
    """One ``metrics.jsonl`` row from a TRL ``on_log`` payload.

    ``wall_s`` is not in the TRL payload. The callback passes the optimizer-step
    wall clock: ``time.perf_counter`` from ``on_step_begin`` to this training
    log. That span is generation, reward, backward, and the optimizer step.
    Evaluation is excluded, because Transformers logs the step and only then
    calls ``evaluate``. ``tokens_generated`` (the step's completion tokens in
    the MLX schema) is ``None``: TRL logs only a cumulative prompt+completion
    counter, kept as ``num_tokens_cumulative`` so no consumer sums it as a
    per-step count. ``gen_seconds`` and ``train_seconds`` are not split in
    TRL's logs.
    ``monitors_wrong`` cannot be recovered from marginal means, so it is ``None``.
    """
    heads = list(_REWARD_HEADS) + [f"monitor_{name}" for name in monitor_names]
    means = {name: _optional_float(logs.get(f"rewards/{name}/mean")) for name in heads}
    rates = rates_from_reward_means(means)
    monitors = {name: means[f"monitor_{name}"] for name in monitor_names}
    zero = _optional_float(logs.get("frac_reward_zero_std"))
    return {
        "step": int(step),
        "wall_s": wall_s,
        "tokens_generated": None,
        "num_tokens_cumulative": _optional_int(logs.get("num_tokens")),
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
    eval_seconds: float | None = None,
) -> dict[str, Any]:
    """One ``trl_eval.jsonl`` row. TRL eval is the held-out set.

    ``eval_seconds`` is that eval call's wall time. It is not part of ``wall_s``.
    """
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
        "eval_seconds": eval_seconds,
    }


class VfatesLogCallback:
    """Appends the shared JSONL schema. Subclasses ``TrainerCallback`` once TRL is imported.

    ``wall_s`` matches ``MLXGRPOTrainer._train_step``: one optimizer step,
    including generation, reward, backward, and the update, excluding eval.
    MLX logs the row and only then calls ``_evaluate`` (``mlx_grpo.py``).

    Transformers 4.57.1 calls ``on_step_begin`` once per optimizer step, when
    ``step % gradient_accumulation_steps == 0``, and calls ``on_log`` from
    ``_maybe_log_save_evaluate`` before ``evaluate``. TRL 0.25.1 generates in
    ``GRPOTrainer._prepare_inputs`` during ``training_step``, which is after
    ``on_step_begin``. ``steps_per_generation`` defaults to
    ``gradient_accumulation_steps`` and ``num_iterations`` defaults to 1, so
    that generation sits inside this step's clock.

    The start stamp is taken in ``on_step_begin`` and is not loaded from the
    checkpoint. A training log with no ``on_step_begin`` in this process keeps
    ``wall_s`` null, so resume does not invent a first duration. Eval time is
    ``eval_seconds`` on ``trl_eval.jsonl``. ``logging_steps`` is 1, so each
    training log is one optimizer step; a later ``on_step_begin`` replaces an
    unlogged stamp instead of stretching it across steps.
    """

    def __init__(self, out_dir: str, monitor_names: Sequence[str]) -> None:
        self.out_dir = Path(out_dir)
        self.monitor_names = list(monitor_names)
        self._step_t0: float | None = None
        self._eval_t0: float | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del state, control, kwargs
        # A new ``train()`` (including resume) must not keep a stamp from an
        # earlier process. ``perf_counter`` is not a checkpoint field.
        self._step_t0 = None
        self._eval_t0 = None
        if getattr(args, "eval_on_start", False):
            self._eval_t0 = time.perf_counter()

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        self._step_t0 = time.perf_counter()

    def _consume(self, started: float | None) -> float | None:
        if started is None:
            return None
        return time.perf_counter() - started

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        del args, kwargs
        if not logs:
            return
        flat = {str(key): _plain(value) for key, value in logs.items()}
        step = int(getattr(state, "global_step", 0) or 0)
        if any(key.startswith("eval_rewards/") for key in flat):
            elapsed = self._consume(self._eval_t0)
            self._eval_t0 = None
            stripped = {
                key[len("eval_") :] if key.startswith("eval_") else key: value
                for key, value in flat.items()
            }
            append_jsonl(
                self.out_dir / "trl_eval.jsonl",
                eval_row_from_trl_logs(
                    stripped,
                    step=step,
                    monitor_names=self.monitor_names,
                    eval_seconds=elapsed,
                ),
            )
            return
        if "rewards/verifier/mean" not in flat:
            return
        elapsed = self._consume(self._step_t0)
        self._step_t0 = None
        append_jsonl(
            self.out_dir / "metrics.jsonl",
            metric_row_from_trl_logs(
                flat, step=step, monitor_names=self.monitor_names, wall_s=elapsed
            ),
        )
        # Eval runs after this log. Start its clock only once the step clock is closed.
        if getattr(control, "should_evaluate", False):
            self._eval_t0 = time.perf_counter()


class PreemptionCallback:
    """Stop after the next optimizer step when the job is signaled.

    ``install`` registers ``SIGTERM`` and ``SIGUSR1`` handlers that only set
    ``preempted``. :meth:`on_step_end` then asks the Trainer to save a
    checkpoint and stop. The process should exit 0 after writing
    ``summary.json`` with ``status`` ``preempted``. Resume uses the checkpoint
    the Trainer saved for that step. This class does not subclass
    ``TrainerCallback`` at import time; :func:`build_trl_trainer` wraps it.
    """

    def __init__(self) -> None:
        self.preempted = False
        self._previous: list[tuple[int, Any]] = []

    def install(self) -> None:
        import signal

        def _handler(signum: int, frame: Any) -> None:
            del signum, frame
            self.preempted = True

        self._handler = _handler
        for name in ("SIGTERM", "SIGUSR1"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            self._previous.append((int(sig), signal.signal(sig, _handler)))

    def restore(self) -> None:
        import signal

        while self._previous:
            sig, previous = self._previous.pop()
            signal.signal(sig, previous)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, state, kwargs
        if self.preempted and control is not None:
            control.should_save = True
            control.should_training_stop = True
        return control


def bind_vllm_chat_template(llm: Any, chat_template_kwargs: Mapping[str, Any] | None) -> None:
    """Pass template kwargs into colocated vLLM ``LLM.chat``.

    TRL 0.25.1 stores ``chat_template_kwargs`` and applies them in the
    transformers generate path and in server-mode ``vllm_client.chat``. The
    colocate branch calls ``self.llm.chat(prompts, sampling_params=...)``
    without them (``grpo_trainer.py`` ``_generate_single_turn``). vLLM 0.10.2
    ``LLM.chat`` accepts ``chat_template_kwargs`` and forwards them to
    ``apply_chat_template``. Binding here makes that path use the same kwargs
    as :func:`vfates.gen.chat.encode_prompt`.
    """
    if llm is None or not chat_template_kwargs:
        return
    original = llm.chat
    fixed = dict(chat_template_kwargs)

    def chat(*args: Any, **kwargs: Any) -> Any:
        merged = dict(fixed)
        extra = kwargs.get("chat_template_kwargs")
        if extra:
            merged.update(dict(extra))
        kwargs["chat_template_kwargs"] = merged
        return original(*args, **kwargs)

    llm.chat = chat


def _adapter_key(parameter_name: str, marker: str) -> str | None:
    if marker not in parameter_name or not parameter_name.endswith("weight"):
        return None
    head = parameter_name.split(marker, 1)[0]
    if head.startswith(_PEFT_PREFIX):
        head = head[len(_PEFT_PREFIX) :]
    return head


def peft_lora_matrices(model: Any) -> dict[str, tuple[Any, Any]]:
    """Map a base-module path to ``(lora_A weight, lora_B weight)``.

    Keys match the LoRA init file: ``model.layers.0.self_attn.q_proj``, the
    path of the wrapped ``nn.Linear`` on the base causal LM. PEFT stores that
    under ``base_model.model.<path>.lora_A.default.weight``.
    """
    a_weights: dict[str, Any] = {}
    b_weights: dict[str, Any] = {}
    for name, param in model.named_parameters():
        a_key = _adapter_key(name, ".lora_A.")
        b_key = _adapter_key(name, ".lora_B.")
        if a_key is not None:
            if a_key in a_weights:
                raise RuntimeError(f"duplicate lora_A parameter for {a_key}")
            a_weights[a_key] = param
        if b_key is not None:
            if b_key in b_weights:
                raise RuntimeError(f"duplicate lora_B parameter for {b_key}")
            b_weights[b_key] = param
    if set(a_weights) != set(b_weights):
        raise RuntimeError(
            "lora_A and lora_B cover different modules: "
            f"only A {sorted(set(a_weights) - set(b_weights))[:3]}, "
            f"only B {sorted(set(b_weights) - set(a_weights))[:3]}"
        )
    if not a_weights:
        raise RuntimeError("LoRA wrapping produced no lora_A parameters")
    return {key: (a_weights[key], b_weights[key]) for key in a_weights}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lora_init_file(path: str, expected_sha256: str | None) -> tuple[dict[str, Any], str]:
    from vfates.features.lora_init import load_lora_init

    tensors, _meta = load_lora_init(path, expected_sha256=expected_sha256)
    digest = expected_sha256 if expected_sha256 is not None else _file_sha256(Path(path))
    return tensors, digest


def _apply_peft_lora_init(model: Any, config: GRPOConfig) -> str:
    """Overwrite every ``lora_A`` from ``config.lora_init``. B stays zero."""
    import numpy as np
    import torch

    if config.lora_init is None:
        raise ValueError("lora_init path is required")
    tensors, digest = _load_lora_init_file(config.lora_init, config.lora_init_sha256)
    pairs = peft_lora_matrices(model)
    got = set(tensors)
    want = set(pairs)
    if got != want:
        missing = sorted(want - got)
        extra = sorted(got - want)
        raise ValueError(
            "lora init keys must equal the wrapped modules: "
            f"missing {missing[:4]}, extra {extra[:4]}, "
            f"{len(want)} wrapped, {len(got)} in the file"
        )
    pending: list[tuple[Any, Any]] = []
    for key, (a_weight, b_weight) in pairs.items():
        array = np.asarray(tensors[key])
        if array.dtype != np.float32:
            raise ValueError(f"{key} must be float32, got {array.dtype}")
        if tuple(array.shape) != tuple(a_weight.shape):
            raise ValueError(
                f"{key} has shape {tuple(array.shape)}, lora_A weight is {tuple(a_weight.shape)}"
            )
        if not torch.equal(b_weight, torch.zeros_like(b_weight)):
            raise RuntimeError(f"lora_B for {key} is not zero; refusing to load A on top of it")
        pending.append((a_weight, array))
    with torch.no_grad():
        for a_weight, array in pending:
            a_weight.copy_(torch.as_tensor(array, dtype=a_weight.dtype, device=a_weight.device))
    return digest


def _seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_causal_lm(config: GRPOConfig, model: Any, *, attn_implementation: str) -> Any:
    """Load the policy in bf16 on CUDA. An injected module is returned as-is.

    LoRA matrices stay float32 (see :func:`_cast_lora_params_fp32`). Training
    still sets ``bf16=True`` so the forward runs under bf16 autocast.
    """
    if model is not None and not isinstance(model, str):
        return model
    import torch
    from transformers import AutoModelForCausalLM

    model_id = config.model_id if model is None else str(model)
    kwargs: dict[str, Any] = {"attn_implementation": attn_implementation}
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.bfloat16
    if config.model_revision:
        kwargs["revision"] = config.model_revision
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def _cast_lora_params_fp32(model: Any) -> None:
    """Keep trainable LoRA weights in float32. The frozen base may be bf16."""
    import torch

    for name, param in model.named_parameters():
        if "lora_" in name and param.dtype != torch.float32:
            param.data = param.data.to(dtype=torch.float32)


def _wrap_peft(config: GRPOConfig, model: Any) -> Any:
    """Build a ``PeftModel`` seeded by ``lora_seed``, optionally overwrite A."""
    from peft import PeftModel, get_peft_model
    from transformers import set_seed

    # Run seed first, then the adapter seed immediately before the draw.
    set_seed(int(config.seed))
    wrapped: Any
    if isinstance(model, PeftModel):
        wrapped = model
    else:
        _seed_torch(int(config.lora_seed))
        wrapped = get_peft_model(model, _peft_config(config, model))
    if config.lora_init is not None:
        digest = _apply_peft_lora_init(wrapped, config)
        config.lora_init_sha256 = digest
    return wrapped


def _install_nonfinite_guard(trainer: Any, run_dir: str) -> None:
    """Raise before the optimizer step when the loss or a gradient is non-finite."""
    import torch
    from transformers import TrainerCallback

    trainer._vfates_loss_fn = trainer.compute_loss

    def compute_loss(model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        loss = trainer._vfates_loss_fn(model, inputs, *args, **kwargs)
        if torch.is_tensor(loss):
            finite = bool(torch.isfinite(loss.detach()).all())
        else:
            finite = math.isfinite(float(loss))
        if not finite:
            raise NonFiniteLossError(int(trainer.state.global_step) + 1, run_dir, what="loss")
        return loss

    trainer.compute_loss = compute_loss

    class _GradCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control
            watched = kwargs.get("model")
            if watched is None:
                return
            for param in watched.parameters():
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue
                if not bool(torch.isfinite(grad).all()):
                    raise NonFiniteLossError(
                        int(state.global_step) + 1, run_dir, what="gradient norm"
                    )

    trainer.add_callback(_GradCallback())


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
    cuda = bool(torch.cuda.is_available())
    attn = resolve_attn_implementation(config.attn_implementation, cuda=cuda)
    config.attn_implementation = attn
    base = _load_causal_lm(config, model, attn_implementation=attn)
    if config.full_finetune:
        # Every parameter, bf16 on CUDA. No PEFT wrap and no audit LoRA A.
        # LoRA runs keep float32 adapter weights under bf16 autocast.
        wrapped = base
        if cuda:
            wrapped = wrapped.to(dtype=torch.bfloat16)
        for param in wrapped.parameters():
            param.requires_grad_(True)
    else:
        wrapped = _wrap_peft(config, base)
        _cast_lora_params_fp32(wrapped)

    world = training_world_size()
    kwargs = trl_grpo_kwargs(
        config,
        n_reward_funcs=len(funcs),
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        bf16=cuda,
        world_size=world,
    )
    if extra_trl_args:
        kwargs.update(extra_trl_args)
    write_json(out / "config.json", config.to_dict())
    write_json(out / "environment_report.json", environment_report())
    write_json(
        out / RUN_METADATA_NAME,
        trl_run_metadata(
            config,
            use_vllm=bool(kwargs["use_vllm"]),
            world_size=world,
            attn_implementation=attn,
        ),
    )
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

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self._inner.on_train_begin(args, state, control, **kwargs)

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self._inner.on_step_begin(args, state, control, **kwargs)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self._inner.on_log(args, state, control, logs=logs, **kwargs)

    preemption = PreemptionCallback()

    class _Preempt(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            return preemption.on_step_end(args, state, control, **kwargs)

    trainer = GRPOTrainer(
        model=wrapped,
        reward_funcs=funcs,
        args=args,
        train_dataset=_dataset(task, train_problems),
        eval_dataset=eval_ds,
        processing_class=processing_class,
        peft_config=None,
        callbacks=[_Callback(config.out_dir, sorted(monitors)), _Preempt()],
    )
    trainer._vfates_preemption = preemption
    bind_vllm_chat_template(getattr(trainer, "llm", None), config.chat_template_kwargs)
    _install_nonfinite_guard(trainer, config.out_dir)
    return trainer


def _peft_config(config: GRPOConfig, model: Any) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=int(config.lora.rank),
        lora_alpha=float(config.lora.alpha),
        lora_dropout=float(config.lora.dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
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
