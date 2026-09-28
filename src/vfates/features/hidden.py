"""Pooled final-layer hidden states, the cheap stand-in for a linear head.

For Qwen2 and Llama causal LMs, ``output_hidden_states=True`` is implemented
by transformers' ``check_model_inputs`` decorator: each decoder layer is
recorded, then the last entry is replaced by ``last_hidden_state``. That value
is the residual stream after the final RMSNorm (``Qwen2Model.norm`` /
``LlamaModel.norm``), which is also the input of the language-model head. A
linear head's per-token gradient is an affine function of this vector, so a
pool of it is an exact feature for a linear head and a proxy for LoRA.

The forward runs the decoder backbone (``model.base_model``) and reads its
``last_hidden_state``, the same tensor, without the vocabulary projection or
the per-layer states. At 151,936 vocabulary entries the float32 logits of one
batch are gigabytes that the pool never reads.

Pooling uses the hidden states that predict the completion tokens: the state
at position ``t - 1`` predicts the token at ``t``, the same shift as the LoRA
log-probability. Left and right padding are both accepted
(``tokenizer.padding_side``). Position ids are the same function of the
attention mask that ``generate`` uses (``cumsum(mask) - 1``, pads filled with
1) so a padded batch matches an unpadded forward on the real tokens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vfates.gen.chat import encode_prompt_completion, prompt_completion_ids

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype: str) -> torch.dtype:
    try:
        return _DTYPES[dtype]
    except KeyError:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}") from None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _memory_bytes(device: torch.device) -> tuple[int, int] | None:
    """``(current_allocated, driver_or_peak)`` on MPS or CUDA."""
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory()), int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated()), int(torch.cuda.max_memory_allocated())
    return None


def _release_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _truncate_left(
    ids: list[int], completion_start: int, max_len: int | None
) -> tuple[list[int], int]:
    if max_len is None or len(ids) <= max_len:
        return ids, completion_start
    overflow = len(ids) - max_len
    return ids[-max_len:], max(0, completion_start - overflow)


def _check_device_map(device_map: str | None) -> None:
    if device_map not in (None, "auto"):
        raise ValueError(f"device_map must be None or 'auto', got {device_map!r}")


def _embedding_device(model: Any, fallback: torch.device) -> torch.device:
    """Device of the input embedding, which is where a sharded forward expects ids."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "get_input_embeddings"):
        embedding = base.get_input_embeddings()
        weight = getattr(embedding, "weight", None)
        if weight is not None and getattr(weight, "device", None) is not None:
            return weight.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def _shard_model(model: Any, device_map: str) -> Any:
    """Shard ``model`` across CUDA devices. Without CUDA the module stays put.

    ``device_map='auto'`` uses accelerate. A model that already carries
    ``hf_device_map`` (``from_pretrained(..., device_map=...)``) is not moved again.
    """
    _check_device_map(device_map)
    if device_map is None:
        return model
    if getattr(model, "hf_device_map", None):
        return model
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return model
    from accelerate import dispatch_model, infer_auto_device_map

    no_split = list(getattr(model, "_no_split_modules", []) or [])
    mapping = infer_auto_device_map(model, no_split_module_classes=no_split)
    return dispatch_model(model, device_map=mapping)


def _place_model(
    model: Any,
    *,
    device: str | torch.device | None,
    dtype: str | None,
    device_map: str | None,
) -> Any:
    """Move or shard ``model``. ``device`` and ``device_map`` are mutually exclusive."""
    _check_device_map(device_map)
    if device is not None and device_map is not None:
        raise ValueError("pass device or device_map, not both")
    if getattr(model, "hf_device_map", None):
        if device is not None:
            raise ValueError("model is already sharded; do not pass device")
        if dtype is not None:
            raise ValueError("model is already sharded; set dtype when loading it")
        return model
    if device_map is not None:
        if dtype is not None:
            model.to(dtype=_resolve_dtype(dtype))
        return _shard_model(model, device_map)
    if device is None and dtype is None:
        return model
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = _resolve_device(device)
    if dtype is not None:
        kwargs["dtype"] = _resolve_dtype(dtype)
    model.to(**kwargs)
    return model


def _encode_feature_row(
    tokenizer: Any,
    prompt: str,
    completion: str,
    system_prompt: str | None,
    completion_row: Sequence[int] | None,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> tuple[list[int], int]:
    """Prompt plus completion ids, forwarding chat-template kwargs only when set.

    ``vfates.gen.chat`` is gaining ``chat_template_kwargs``. Passing the keyword
    only when it is not ``None`` keeps this callable before and after that lands.
    """
    if completion_row is None:
        if chat_template_kwargs is None:
            return encode_prompt_completion(tokenizer, prompt, completion, system_prompt)
        return encode_prompt_completion(
            tokenizer,
            prompt,
            completion,
            system_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
    if chat_template_kwargs is None:
        return prompt_completion_ids(tokenizer, prompt, completion_row, system_prompt)
    return prompt_completion_ids(
        tokenizer,
        prompt,
        completion_row,
        system_prompt,
        chat_template_kwargs=chat_template_kwargs,
    )


def load_model(
    model_id: str,
    device: str | torch.device | None = None,
    dtype: str = "float32",
    *,
    local_files_only: bool = False,
    device_map: str | None = None,
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer in eval mode.

    The model is moved to ``device`` (CUDA, then MPS, then CPU when ``device``
    is ``None``). ``device_map='auto'`` instead shards the load with accelerate
    and must not be combined with ``device``. ``local_files_only`` skips the hub
    when the snapshot is cached.
    """
    _check_device_map(device_map)
    if device is not None and device_map is not None:
        raise ValueError("pass device or device_map, not both")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"{model_id} tokenizer has no pad token and no eos token")
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {
        "dtype": _resolve_dtype(dtype),
        "local_files_only": local_files_only,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device_map is None:
        model.to(device=_resolve_device(device))
    model.eval()
    return model, tokenizer


def _pad_batch(
    sequences: list[list[int]], pad_id: int, side: str, multiple: int = 1
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {side!r}")
    width = -(-max(len(seq) for seq in sequences) // multiple) * multiple
    batch = len(sequences)
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, width), dtype=torch.long)
    offsets: list[int] = []
    for row, seq in enumerate(sequences):
        length = len(seq)
        offset = 0 if side == "right" else width - length
        if length:
            input_ids[row, offset : offset + length] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, offset : offset + length] = 1
        offsets.append(offset)
    return input_ids, attention_mask, offsets


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    # Same construction as transformers' prepare_inputs_for_generation.
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def hidden_features(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    *,
    system_prompt: str | None = None,
    pooling: str = "mean",
    batch_size: int = 8,
    device: str | torch.device | None = None,
    max_len: int | None = None,
    completion_ids: Sequence[Sequence[int]] | None = None,
    pad_multiple: int = 1,
    dtype: str | None = None,
    device_map: str | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Pooled final-layer hidden state of each completion, float32 ``[N, H]``.

    ``pooling`` is ``mean`` (over the states that predict the completion
    tokens) or ``last`` (the state that predicts the final completion token).
    The pooled span is ``[offset + comp_start - 1, offset + comp_end - 1)``.
    ``max_len`` keeps the last ``max_len`` tokens, so the completion is what
    survives a long prompt. An empty completion, or a predictor span that is
    empty, yields a zero row.

    ``completion_ids``, when given, are scored as ``prompt_ids + completion_ids``
    with no re-tokenisation (the same chat prompt encoding generation uses).
    ``hidden_features.meta['completion_source']`` is ``'token_ids'`` or
    ``'text'``.

    ``pad_multiple`` rounds every batch width up to a multiple of itself. On
    MPS each new input shape compiles and caches a graph whose buffers sit
    outside the torch allocator and are never freed, so exact widths grow the
    process by about 0.2 GiB per distinct shape at batch 8 (Qwen2.5-0.5B,
    float32). Padding does not change the real tokens' states: they never
    attend to later positions, and left pads are masked.

    ``dtype`` casts the module when set. ``None`` leaves the loaded dtype alone,
    which is the historical behaviour. ``device_map='auto'`` shards with
    accelerate when CUDA is present and is mutually exclusive with ``device``.
    ``chat_template_kwargs`` is forwarded to the chat encoder only when it is
    not ``None``.

    A PEFT model is unwrapped with ``get_base_model()`` before its backbone is
    taken; the injected adapter layers stay active.

    On MPS or CUDA the peak ``(allocated, driver_or_max)`` byte counts of the
    last call are stored on ``hidden_features.peak_memory`` for audit budgeting.
    That attribute is not part of the returned array.
    """
    if pooling not in ("mean", "last"):
        raise ValueError(f"pooling must be 'mean' or 'last', got {pooling!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if pad_multiple < 1:
        raise ValueError(f"pad_multiple must be >= 1, got {pad_multiple}")
    if len(prompts) != len(completions):
        raise ValueError(f"prompts ({len(prompts)}) and completions ({len(completions)}) differ")
    if completion_ids is not None and len(completion_ids) != len(prompts):
        raise ValueError(
            f"completion_ids ({len(completion_ids)}) and prompts ({len(prompts)}) differ"
        )
    source = "token_ids" if completion_ids is not None else "text"
    hidden_features.meta = {  # type: ignore[attr-defined]
        "completion_source": source,
        "dtype": dtype,
        "device_map": device_map,
    }
    hidden_size = int(model.config.hidden_size)
    if not prompts:
        hidden_features.peak_memory = None  # type: ignore[attr-defined]
        return np.zeros((0, hidden_size), dtype=np.float32)
    model = _place_model(model, device=device, dtype=dtype, device_map=device_map)
    if device_map is not None or getattr(model, "hf_device_map", None):
        param_device = _embedding_device(model, _resolve_device(device))
    else:
        param_device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad token and no eos token")
    side = getattr(tokenizer, "padding_side", "right") or "right"
    encoded: list[tuple[list[int], int, int]] = []
    for index, prompt in enumerate(prompts):
        ids, start = _encode_feature_row(
            tokenizer,
            prompt,
            completions[index],
            system_prompt,
            None if completion_ids is None else completion_ids[index],
            chat_template_kwargs,
        )
        ids, start = _truncate_left(ids, start, max_len)
        encoded.append((ids, start, len(ids)))
    features = np.zeros((len(encoded), hidden_size), dtype=np.float32)
    peak_allocated = 0
    peak_driver = 0
    saw_memory = False
    was_training = model.training
    model.eval()
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(inner, "base_model", inner)
    try:
        with torch.inference_mode():
            for start in range(0, len(encoded), batch_size):
                batch = encoded[start : start + batch_size]
                input_ids, attention_mask, offsets = _pad_batch(
                    [item[0] for item in batch], int(pad_id), side, pad_multiple
                )
                input_ids = input_ids.to(param_device)
                attention_mask = attention_mask.to(param_device)
                position_ids = _position_ids(attention_mask)
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                # Post-final-norm; see the module docstring.
                final = getattr(outputs, "last_hidden_state", None)
                if final is None:
                    raise RuntimeError("model backbone returned no last_hidden_state")
                memory = _memory_bytes(param_device)
                if memory is not None:
                    saw_memory = True
                    peak_allocated = max(peak_allocated, memory[0])
                    peak_driver = max(peak_driver, memory[1])
                for row, ((_, comp_start, comp_end), offset) in enumerate(zip(batch, offsets)):
                    # State at t predicts the token at t + 1. Completion tokens
                    # occupy [comp_start, comp_end); their predictors are
                    # [comp_start - 1, comp_end - 1). An empty span stays a zero row.
                    pred_lo = offset + comp_start - 1
                    pred_hi = offset + comp_end - 1
                    if pred_hi <= pred_lo:
                        continue
                    if pred_lo < 0:
                        raise ValueError(
                            "need at least one prompt token before the completion to score it"
                        )
                    span = final[row, pred_lo:pred_hi]
                    pooled = span[-1] if pooling == "last" else span.mean(dim=0)
                    features[start + row] = pooled.float().cpu().numpy()
                    del span, pooled
                del outputs, final
                # Batch widths vary, so cached blocks rarely fit the next batch.
                _release_memory(param_device)
    finally:
        model.train(was_training)
    hidden_features.peak_memory = (  # type: ignore[attr-defined]
        {"allocated": peak_allocated, "driver": peak_driver} if saw_memory else None
    )
    return features


hidden_features.peak_memory = None  # type: ignore[attr-defined]
hidden_features.meta = {"completion_source": "text"}  # type: ignore[attr-defined]
