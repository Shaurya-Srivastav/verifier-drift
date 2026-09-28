"""Pooled final-layer hidden states, the cheap stand-in for a linear head.

For Qwen2 and Llama causal LMs, ``output_hidden_states=True`` is implemented
by transformers' ``check_model_inputs`` decorator: each decoder layer is
recorded, then the last entry is replaced by ``last_hidden_state``. That value
is the residual stream after the final RMSNorm (``Qwen2Model.norm`` /
``LlamaModel.norm``), which is also the input of the language-model head. A
linear head's per-token gradient is an affine function of this vector, so a
pool of it is an exact feature for a linear head and a proxy for LoRA.

Pooling uses the hidden states that predict the completion tokens: the state
at position ``t - 1`` predicts the token at ``t``, the same shift as the LoRA
log-probability. Left and right padding are both accepted
(``tokenizer.padding_side``). Position ids are the same function of the
attention mask that ``generate`` uses (``cumsum(mask) - 1``, pads filled with
1) so a padded batch matches an unpadded forward on the real tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def load_model(
    model_id: str,
    device: str | torch.device | None = None,
    dtype: str = "float32",
    *,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer in eval mode.

    The model is moved to ``device`` (CUDA, then MPS, then CPU when ``device``
    is ``None``). ``local_files_only`` skips the hub when the snapshot is cached.
    """
    resolved = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"{model_id} tokenizer has no pad token and no eos token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=_resolve_dtype(dtype),
        local_files_only=local_files_only,
    )
    model.to(device=resolved)
    model.eval()
    return model, tokenizer


def _pad_batch(
    sequences: list[list[int]], pad_id: int, side: str
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {side!r}")
    width = max(len(seq) for seq in sequences)
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

    On MPS or CUDA the peak ``(allocated, driver_or_max)`` byte counts of the
    last call are stored on ``hidden_features.peak_memory`` for audit budgeting.
    That attribute is not part of the returned array.
    """
    if pooling not in ("mean", "last"):
        raise ValueError(f"pooling must be 'mean' or 'last', got {pooling!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if len(prompts) != len(completions):
        raise ValueError(f"prompts ({len(prompts)}) and completions ({len(completions)}) differ")
    if completion_ids is not None and len(completion_ids) != len(prompts):
        raise ValueError(
            f"completion_ids ({len(completion_ids)}) and prompts ({len(prompts)}) differ"
        )
    source = "token_ids" if completion_ids is not None else "text"
    hidden_features.meta = {"completion_source": source}  # type: ignore[attr-defined]
    hidden_size = int(model.config.hidden_size)
    if not prompts:
        hidden_features.peak_memory = None  # type: ignore[attr-defined]
        return np.zeros((0, hidden_size), dtype=np.float32)
    if device is not None:
        model.to(_resolve_device(device))
    param_device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad token and no eos token")
    side = getattr(tokenizer, "padding_side", "right") or "right"
    encoded: list[tuple[list[int], int, int]] = []
    for index, prompt in enumerate(prompts):
        if completion_ids is None:
            ids, start = encode_prompt_completion(
                tokenizer, prompt, completions[index], system_prompt
            )
        else:
            ids, start = prompt_completion_ids(
                tokenizer, prompt, completion_ids[index], system_prompt
            )
        ids, start = _truncate_left(ids, start, max_len)
        encoded.append((ids, start, len(ids)))
    features = np.zeros((len(encoded), hidden_size), dtype=np.float32)
    peak_allocated = 0
    peak_driver = 0
    saw_memory = False
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(encoded), batch_size):
                batch = encoded[start : start + batch_size]
                input_ids, attention_mask, offsets = _pad_batch(
                    [item[0] for item in batch], int(pad_id), side
                )
                input_ids = input_ids.to(param_device)
                attention_mask = attention_mask.to(param_device)
                position_ids = _position_ids(attention_mask)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = outputs.hidden_states
                if not hidden:
                    raise RuntimeError("model returned no hidden states")
                # Last entry is post-final-norm; see the module docstring.
                final = hidden[-1]
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
                del outputs, final, hidden
    finally:
        model.train(was_training)
    hidden_features.peak_memory = (  # type: ignore[attr-defined]
        {"allocated": peak_allocated, "driver": peak_driver} if saw_memory else None
    )
    return features


hidden_features.peak_memory = None  # type: ignore[attr-defined]
hidden_features.meta = {"completion_source": "text"}  # type: ignore[attr-defined]
