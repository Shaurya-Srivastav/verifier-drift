"""Shared GRPO configuration for the MLX trainer and the TRL trainer.

One dataclass is the experiment. ``mlx_grpo`` and ``trl_grpo`` both read it, so a
pilot yaml and an H100 yaml describe the same objective with different systems
knobs (``use_vllm``, micro-batch, checkpointing).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vfates.utils.io import read_yaml

# Short names used in yamls. MLX matches module paths inside a decoder block;
# PEFT matches suffix names (``q_proj``) anywhere those modules exist.
LORA_KEYS_MLX = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
LORA_KEYS_PEFT = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}

_ADV = ("std", "mean")
_AGG = ("seq_mean", "token_mean", "dr_constant")
_REGIME = ("multi_epoch", "streaming")
_VLLM_MODE = ("colocate", "server")


class NonFiniteLossError(RuntimeError):
    """A step produced a non-finite loss or gradient norm.

    Raised before any metric row or checkpoint for that step is written.
    ``step`` is the 1-indexed optimizer step that did not complete.
    ``run_dir`` is the run directory. Callers resume from the last checkpoint.
    """

    def __init__(self, step: int, run_dir: str, *, what: str = "loss or gradient norm") -> None:
        self.step = int(step)
        self.run_dir = str(run_dir)
        super().__init__(f"non-finite {what} at step {self.step} in {self.run_dir}")


def raise_if_nonfinite(loss: float, grad_norm: float, *, step: int, run_dir: str) -> None:
    """Raise :class:`NonFiniteLossError` when ``loss`` or ``grad_norm`` is not finite."""
    if not math.isfinite(float(loss)):
        raise NonFiniteLossError(step, run_dir, what="loss")
    if not math.isfinite(float(grad_norm)):
        raise NonFiniteLossError(step, run_dir, what="gradient norm")


def resolve_lora_keys(keys: list[str] | tuple[str, ...], mapping: dict[str, str]) -> list[str]:
    """Map short names (``q``, ``gate``) through ``mapping``. Full names pass through."""
    out = []
    for key in keys:
        out.append(mapping.get(key, key))
    return out


@dataclass
class LoRAConfig:
    """LoRA hyperparameters.

    ``scale`` passed to MLX is ``alpha / rank``, matching PEFT's ``lora_alpha / r``.
    ``num_layers=-1`` adapts every decoder block (MLX treats ``-1`` that way:
    ``layers[-max(num_layers, 0):]``). ``keys`` are short names or full module paths.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    keys: tuple[str, ...] = ("q", "k", "v", "o", "gate", "up", "down")
    num_layers: int = -1

    @property
    def scale(self) -> float:
        """PEFT-compatible LoRA scale ``alpha / rank``."""
        return float(self.alpha) / float(self.rank)

    def mlx_keys(self) -> list[str]:
        return resolve_lora_keys(self.keys, LORA_KEYS_MLX)

    def peft_keys(self) -> list[str]:
        return resolve_lora_keys(self.keys, LORA_KEYS_PEFT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "keys": list(self.keys),
            "num_layers": self.num_layers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoRAConfig:
        if not isinstance(data, dict):
            raise TypeError(f"lora must be a mapping, got {type(data).__name__}")
        unknown = set(data) - {"rank", "alpha", "dropout", "keys", "num_layers"}
        if unknown:
            raise ValueError(f"Unknown lora fields: {sorted(unknown)}")
        kwargs = dict(data)
        if "keys" in kwargs and kwargs["keys"] is not None:
            kwargs["keys"] = tuple(kwargs["keys"])
        return cls(**kwargs)


@dataclass(kw_only=True)
class GRPOConfig:
    """Experiment config. Construct with keywords; ``from_yaml`` / ``from_dict`` validate."""

    model_id: str
    steps: int
    prompts_per_step: int
    out_dir: str
    seed: int
    # Adapter draw, shared by the audit and every training seed. The run seed
    # does not draw LoRA A. ``lora_init`` is a safetensors file of A matrices
    # (contract C-2); ``lora_init_sha256`` is checked when it is set.
    lora_seed: int = 0
    lora_init: str | None = None
    lora_init_sha256: str | None = None
    eval_every: int
    eval_heldout_n: int
    eval_train_n: int
    eval_samples_per_prompt: int
    save_every: int
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    lr: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 0
    group_size: int = 8
    max_tokens: int = 512
    max_prompt_length: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    adv_mode: str = "std"
    loss_agg: str = "seq_mean"
    beta_kl: float = 0.0
    num_iterations: int = 1
    eps_low: float = 0.2
    eps_high: float = 0.2
    micro_batch_size: int = 4
    data_regime: str = "multi_epoch"
    eval_temperature: float = 1.0
    eval_max_tokens: int | None = None
    gradient_checkpointing: bool = False
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.35
    # Off so the H100 loss matches MLX. vLLM can disagree with the trainer's
    # log-probs; set True only as a robustness check, not for the primary run.
    vllm_importance_sampling_correction: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.lora, LoRAConfig):
            self.lora = LoRAConfig.from_dict(self.lora)
        self.adam_betas = (float(self.adam_betas[0]), float(self.adam_betas[1]))
        if self.eval_max_tokens is None:
            self.eval_max_tokens = int(self.max_tokens)
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.out_dir, str) or not self.out_dir.strip():
            raise ValueError("out_dir must be a non-empty string")
        _pos_int("steps", self.steps)
        _pos_int("prompts_per_step", self.prompts_per_step)
        _pos_int("group_size", self.group_size)
        if self.group_size < 2:
            raise ValueError(
                f"group_size must be >= 2 so a group has a standard deviation, got {self.group_size}"
            )
        _pos_int("max_tokens", self.max_tokens)
        _pos_int("max_prompt_length", self.max_prompt_length)
        _pos_int("eval_every", self.eval_every)
        _pos_int("eval_samples_per_prompt", self.eval_samples_per_prompt)
        _pos_int("save_every", self.save_every)
        _pos_int("num_iterations", self.num_iterations)
        _pos_int("micro_batch_size", self.micro_batch_size)
        _nonneg_int("eval_heldout_n", self.eval_heldout_n)
        _nonneg_int("eval_train_n", self.eval_train_n)
        _nonneg_int("warmup_steps", self.warmup_steps)
        _nonneg_int("seed", self.seed)
        _nonneg_int("eval_max_tokens", int(self.eval_max_tokens or 0))
        if self.eval_max_tokens is not None and self.eval_max_tokens < 1:
            raise ValueError("eval_max_tokens must be >= 1")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        b1, b2 = self.adam_betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"adam_betas must lie in [0, 1), got {self.adam_betas}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0 (0 = greedy), got {self.temperature}")
        if self.eval_temperature < 0:
            raise ValueError(f"eval_temperature must be >= 0, got {self.eval_temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.adv_mode not in _ADV:
            raise ValueError(f"adv_mode must be one of {_ADV}, got {self.adv_mode!r}")
        if self.loss_agg not in _AGG:
            raise ValueError(f"loss_agg must be one of {_AGG}, got {self.loss_agg!r}")
        if self.data_regime not in _REGIME:
            raise ValueError(f"data_regime must be one of {_REGIME}, got {self.data_regime!r}")
        if self.beta_kl < 0:
            raise ValueError(f"beta_kl must be >= 0, got {self.beta_kl}")
        if self.eps_low < 0 or self.eps_high < 0:
            raise ValueError(f"eps_low/eps_high must be >= 0, got {self.eps_low}, {self.eps_high}")
        if self.vllm_mode not in _VLLM_MODE:
            raise ValueError(f"vllm_mode must be one of {_VLLM_MODE}, got {self.vllm_mode!r}")
        if not (0.0 < self.vllm_gpu_memory_utilization < 1.0):
            raise ValueError(
                "vllm_gpu_memory_utilization must be in (0, 1), "
                f"got {self.vllm_gpu_memory_utilization}"
            )
        completions = self.prompts_per_step * self.group_size
        if completions % self.micro_batch_size != 0:
            raise ValueError(
                f"micro_batch_size ({self.micro_batch_size}) must divide "
                f"prompts_per_step * group_size ({completions})"
            )
        lora = self.lora
        if lora.rank < 1:
            raise ValueError(f"lora.rank must be >= 1, got {lora.rank}")
        if lora.alpha <= 0:
            raise ValueError(f"lora.alpha must be positive, got {lora.alpha}")
        if not (0.0 <= lora.dropout < 1.0):
            raise ValueError(f"lora.dropout must be in [0, 1), got {lora.dropout}")
        if lora.num_layers == 0 or lora.num_layers < -1:
            raise ValueError(
                f"lora.num_layers must be -1 (all layers) or >= 1, got {lora.num_layers}"
            )
        if not lora.keys:
            raise ValueError("lora.keys must list at least one projection")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        _nonneg_int("lora_seed", self.lora_seed)
        if self.lora_init is not None:
            if not isinstance(self.lora_init, str) or not self.lora_init.strip():
                raise ValueError("lora_init must be a non-empty path or null")
            self.lora_init = self.lora_init.strip()
        if self.lora_init_sha256 is not None:
            if self.lora_init is None:
                raise ValueError("lora_init_sha256 requires lora_init")
            if not isinstance(self.lora_init_sha256, str):
                raise ValueError("lora_init_sha256 must be a hex digest or null")
            digest = self.lora_init_sha256.strip().lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(
                    "lora_init_sha256 must be a 64-character sha256 hex digest, "
                    f"got {self.lora_init_sha256!r}"
                )
            self.lora_init_sha256 = digest

    @property
    def completions_per_step(self) -> int:
        return self.prompts_per_step * self.group_size

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["adam_betas"] = [float(self.adam_betas[0]), float(self.adam_betas[1])]
        data["lora"] = self.lora.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GRPOConfig:
        if not isinstance(data, dict):
            raise TypeError(f"config must be a mapping, got {type(data).__name__}")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown GRPOConfig fields: {sorted(unknown)}")
        raw = {k: v for k, v in data.items() if v is not None}
        if "lora" in raw and not isinstance(raw["lora"], LoRAConfig):
            raw["lora"] = LoRAConfig.from_dict(raw["lora"])
        if "adam_betas" in raw:
            betas = raw["adam_betas"]
            if len(betas) != 2:
                raise ValueError(f"adam_betas must have length 2, got {betas!r}")
            raw["adam_betas"] = (float(betas[0]), float(betas[1]))
        if "lora" in raw and isinstance(raw["lora"], LoRAConfig):
            pass
        return cls(**raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GRPOConfig:
        loaded = read_yaml(path)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} did not contain a mapping")
        return cls.from_dict(loaded)


def _pos_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _nonneg_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
