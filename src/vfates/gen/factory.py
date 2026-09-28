"""Construct a generation backend by name.

Imports are inside the function so ``import vfates.gen.factory`` does not load
PyTorch, MLX or vLLM. ``vfates.gen`` itself does not re-export this helper yet;
import it from :mod:`vfates.gen.factory`.
"""

from __future__ import annotations

from typing import Any, Literal

from vfates.gen.base import Generator

BackendName = Literal["mlx", "hf", "vllm"]


def make_generator(backend: BackendName, model_id: str, **kwargs: Any) -> Generator:
    """``MLXGenerator``, ``HFGenerator`` or ``VLLMGenerator``.

    ``kwargs`` are passed to that backend's constructor (``device`` and
    ``dtype`` for Hugging Face, engine options for vLLM, ``adapter_path`` and
    batch sizes for MLX). ``chat_template_kwargs`` is accepted by every
    backend, applied in its chat template, and reported by ``describe()``.
    """
    if backend == "mlx":
        from vfates.gen.mlx_backend import MLXGenerator

        return MLXGenerator(model_id, **kwargs)
    if backend == "hf":
        from vfates.gen.hf_backend import HFGenerator

        return HFGenerator(model_id, **kwargs)
    if backend == "vllm":
        from vfates.gen.vllm_backend import VLLMGenerator

        return VLLMGenerator(model_id, **kwargs)
    raise ValueError(f"unknown generation backend {backend!r}; expected 'mlx', 'hf' or 'vllm'")
