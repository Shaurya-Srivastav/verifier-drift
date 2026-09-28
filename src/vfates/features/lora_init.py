"""Saved LoRA A initialisation shared by the audit and every training run.

The step-0 feature is the gradient of the completion log-probability with
respect to ``B`` at ``B = 0``, and that gradient depends on ``A``. One file,
written the first time the audit builds the adapters, is the ``A`` every later
feature part and every training seed must match.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vfates.utils.io import sha256_file

LORA_INIT_FILENAME = "lora_init.safetensors"
LORA_INIT_SCHEMA = "vfates.lora_init/1"

# PEFT wraps the causal LM as ``base_model.model.<original module path>``.
_PEFT_PREFIX = "base_model.model."


def lora_a_state(model: Any) -> dict[str, np.ndarray]:
    """Float32 LoRA A weights keyed by the base ``nn.Linear`` path.

    A PEFT parameter ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight``
    is stored as ``model.layers.0.self_attn.q_proj`` with shape ``(r, in_features)``,
    the PEFT ``lora_A`` layout (MLX ``lora_a`` is the transpose).
    """
    found: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if ".lora_A." not in name:
            continue
        module = name.split(".lora_A", 1)[0]
        if module.startswith(_PEFT_PREFIX):
            module = module[len(_PEFT_PREFIX) :]
        weight = param.detach().float().cpu().numpy()
        found[module] = np.ascontiguousarray(weight, dtype=np.float32)
    if not found:
        raise RuntimeError("no lora_A parameters found")
    return found


def assert_lora_a_equal(fresh: Mapping[str, np.ndarray], saved: Mapping[str, np.ndarray]) -> None:
    """Raise ``ValueError`` unless every A matrix matches bitwise."""
    if set(fresh) != set(saved):
        missing = sorted(set(fresh) - set(saved))
        extra = sorted(set(saved) - set(fresh))
        raise ValueError(
            f"LoRA A keys differ from the saved init; missing {missing[:3]}, extra {extra[:3]}"
        )
    for key in sorted(fresh):
        left = np.asarray(fresh[key])
        right = np.asarray(saved[key])
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not np.array_equal(left, right)
        ):
            raise ValueError(f"LoRA A matrix {key} is not bitwise equal to the saved init")


def save_lora_init(
    path: str | Path,
    tensors: Mapping[str, np.ndarray],
    meta: Mapping[str, str | int | float],
) -> str:
    """Write ``tensors`` atomically and return the sha256 hex of the file bytes.

    Metadata values are stored as strings, with ``schema`` set to
    :data:`LORA_INIT_SCHEMA`. Keys are sorted so the file bytes do not depend
    on dict order.
    """
    from safetensors.numpy import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.ascontiguousarray(np.asarray(tensors[key]), dtype=np.float32)
        for key in sorted(tensors)
    }
    metadata = {str(key): str(value) for key, value in meta.items()}
    metadata["schema"] = LORA_INIT_SCHEMA
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        save_file(arrays, tmp, metadata=metadata)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return sha256_file(path)


def load_lora_init(
    path: str | Path, expected_sha256: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load A matrices and string metadata. Raise ``ValueError`` on schema or sha mismatch."""
    from safetensors import safe_open

    path = Path(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"LoRA init sha256 mismatch for {path}")
    tensors: dict[str, np.ndarray] = {}
    with safe_open(path, framework="numpy") as handle:
        raw_meta = handle.metadata() or {}
        meta = {str(key): str(value) for key, value in raw_meta.items()}
        if meta.get("schema") != LORA_INIT_SCHEMA:
            raise ValueError(
                f"LoRA init schema {meta.get('schema')!r} != {LORA_INIT_SCHEMA!r} ({path})"
            )
        # safe_open is not iterable; keys() is the documented listing.
        for key in handle.keys():  # noqa: SIM118
            tensors[key] = np.ascontiguousarray(handle.get_tensor(key), dtype=np.float32)
    return tensors, meta
