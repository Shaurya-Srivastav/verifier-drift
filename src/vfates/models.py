"""Pinned model specs for the study cells.

``configs/models/<name>.yaml`` is the source of truth. Training code reads a
spec for the Hub id, the revision, the learning rate, the chat-template kwargs
and the placement (GPU count, tensor parallel, micro-batch, vLLM memory
fraction) for the cell's hardware class.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from vfates.train.config import LORA_KEYS_PEFT, resolve_lora_keys
from vfates.utils.io import read_yaml

_REQUIRED = (
    "name",
    "model_id",
    "revision",
    "family",
    "params_b",
    "chat_template_kwargs",
    "lora_keys",
    "dtype",
    "attn_implementation",
    "gated",
    "lr",
    "placement",
)
# Full-parameter runs read ``placement_full``. LoRA specs omit it.
_OPTIONAL = ("placement_full",)


def _models_root() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "models"


@dataclass(frozen=True)
class Placement:
    """How one model is trained on one hardware class.

    ``gpus`` is the trainer world size (FSDP, one process per GPU). ``tp`` is
    vLLM ``tensor_parallel_size`` and must divide ``gpus``. ``micro_batch_size``
    is the per-device completion batch.
    """

    gpus: int
    tp: int
    micro_batch_size: int
    vllm_gpu_memory_utilization: float

    def __post_init__(self) -> None:
        for name in ("gpus", "tp", "micro_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"placement.{name} must be a positive integer, got {value!r}")
        util = float(self.vllm_gpu_memory_utilization)
        if not 0.0 < util < 1.0:
            raise ValueError(f"placement.vllm_gpu_memory_utilization must be in (0, 1), got {util}")
        if self.gpus % self.tp != 0:
            raise ValueError(f"placement.tp ({self.tp}) must divide placement.gpus ({self.gpus})")
        object.__setattr__(self, "vllm_gpu_memory_utilization", util)


@dataclass(frozen=True)
class ModelSpec:
    """One row of ``configs/models``."""

    name: str
    model_id: str
    revision: str
    family: str
    params_b: float
    chat_template_kwargs: Mapping[str, Any] | None
    lora_keys: tuple[str, ...]
    dtype: str
    attn_implementation: str
    gated: bool
    lr: float
    placement: Mapping[str, Placement]
    placement_full: Mapping[str, Placement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = tuple(self.lora_keys)
        if not keys:
            raise ValueError(f"{self.name}: lora_keys must list at least one projection")
        object.__setattr__(self, "lora_keys", keys)
        raw = self.chat_template_kwargs
        if raw is None:
            frozen: Mapping[str, Any] | None = None
        else:
            frozen = MappingProxyType(dict(raw))
        object.__setattr__(self, "chat_template_kwargs", frozen)
        object.__setattr__(
            self, "placement", _freeze_placements(self.name, "placement", self.placement)
        )
        object.__setattr__(
            self,
            "placement_full",
            _freeze_placements(self.name, "placement_full", self.placement_full),
        )

    def placement_for(self, hardware_class: str) -> Placement:
        """Placement for ``hardware_class``. Raises when this model has none."""
        found = self.placement.get(hardware_class)
        if found is None:
            known = ", ".join(sorted(self.placement)) or "(none)"
            raise ValueError(
                f"{self.name} has no placement for hardware class {hardware_class!r}. "
                f"Known classes: {known}"
            )
        return found

    def placement_full_for(self, hardware_class: str) -> Placement:
        """Full-parameter placement for ``hardware_class``. Raises when this model has none."""
        found = self.placement_full.get(hardware_class)
        if found is None:
            known = ", ".join(sorted(self.placement_full)) or "(none)"
            raise ValueError(
                f"{self.name} has no full-finetune placement for hardware class "
                f"{hardware_class!r}. Known classes: {known}"
            )
        return found

    def peft_target_modules(self) -> list[str]:
        """Short LoRA keys (``q``, ``gate``) mapped to PEFT module names."""
        return resolve_lora_keys(self.lora_keys, LORA_KEYS_PEFT)


def load_model_spec(name: str, root: Path | str | None = None) -> ModelSpec:
    """Load ``configs/models/<name>.yaml``. ``name`` may include the suffix."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"model spec name must be a non-empty string, got {name!r}")
    stem = name.strip()
    if stem.endswith(".yaml"):
        stem = stem[: -len(".yaml")]
    if "/" in stem or stem.startswith("."):
        raise ValueError(f"model spec name must be a file stem, got {name!r}")
    directory = Path(root) if root is not None else _models_root()
    path = directory / f"{stem}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no model spec {stem!r} at {path}")
    loaded = read_yaml(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping")
    unknown = set(loaded) - set(_REQUIRED) - set(_OPTIONAL)
    if unknown:
        raise ValueError(f"{path.name}: unknown fields {sorted(unknown)}")
    missing = [field for field in _REQUIRED if field not in loaded]
    if missing:
        raise ValueError(f"{path.name}: missing fields {missing}")
    if loaded["name"] != stem:
        raise ValueError(f"{path.name}: name is {loaded['name']!r}, expected {stem!r}")
    kwargs = loaded["chat_template_kwargs"]
    if kwargs is not None and not isinstance(kwargs, dict):
        raise ValueError(f"{path.name}: chat_template_kwargs must be a mapping or null")
    if not isinstance(loaded["placement"], dict) or not loaded["placement"]:
        raise ValueError(f"{path.name}: placement must be a non-empty mapping")
    if "placement_full" in loaded:
        full = loaded["placement_full"]
        if not isinstance(full, dict) or not full:
            raise ValueError(f"{path.name}: placement_full must be a non-empty mapping")
    else:
        full = {}
    if not isinstance(loaded["gated"], bool):
        raise ValueError(f"{path.name}: gated must be a bool")
    return ModelSpec(
        name=str(loaded["name"]),
        model_id=str(loaded["model_id"]),
        revision=str(loaded["revision"]),
        family=str(loaded["family"]),
        params_b=float(loaded["params_b"]),
        chat_template_kwargs=kwargs,
        lora_keys=tuple(loaded["lora_keys"]),
        dtype=str(loaded["dtype"]),
        attn_implementation=str(loaded["attn_implementation"]),
        gated=bool(loaded["gated"]),
        lr=float(loaded["lr"]),
        placement=loaded["placement"],
        placement_full=full,
    )


def _freeze_placements(name: str, label: str, raw: Mapping[str, Any]) -> Mapping[str, Placement]:
    placed: dict[str, Placement] = {}
    for hardware_class, value in dict(raw).items():
        if isinstance(value, Placement):
            placed[str(hardware_class)] = value
        elif isinstance(value, Mapping):
            placed[str(hardware_class)] = Placement(
                gpus=int(value["gpus"]),
                tp=int(value["tp"]),
                micro_batch_size=int(value["micro_batch_size"]),
                vllm_gpu_memory_utilization=float(value["vllm_gpu_memory_utilization"]),
            )
        else:
            raise ValueError(f"{name}: {label} for {hardware_class!r} must be a mapping")
    return MappingProxyType(placed)
