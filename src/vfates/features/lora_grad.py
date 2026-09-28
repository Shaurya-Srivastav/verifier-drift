"""Per-sample LoRA-B gradients, the step-0 feature GRPO actually trains.

Standard LoRA initialisation draws ``A`` and sets ``B = 0``. The adapter output
is ``scaling * B(A(x))``, so at step 0 it is identically zero, ``dL/dA = 0``,
and

``dL/dB = scaling * (dL/d output) (A x)^T``.

``A x`` is a fixed random projection of the layer input, so the ``B`` gradient
is already a random projection of the full weight gradient. The first optimiser
step moves only ``B``. Sketching ``B`` (and not ``A``) is the correct step-0
feature for the parameters that train. ``lora_seed`` must be the seed used to
initialise the adapters in the training run; ``torch.manual_seed(lora_seed)``
is applied immediately before ``get_peft_model``.

``normalize='sum'`` differentiates the sum of completion-token log-probabilities.
``normalize='token_mean'`` differentiates their mean, which is the sum-gradient
divided by the completion length. ``lengths`` always stores that length,
including a terminating EOS when the caller passed sampled token ids.
:func:`vfates.forecast.transfer.length_division` is true only for sum features
under ``seq_mean``: those train rows are divided by ``lengths``. Held-out
velocities stay undivided (the change in sequence log-probability). Passing
``lengths`` for token-mean features divides twice.

``return_exact=True`` also returns the concatenated ``B`` gradients (float32).
That is for small models and sketch-fidelity checks, not for the audit.

Large models should not call :func:`lora_grad_features` once per audit part.
:func:`prepare_lora_grad` wraps the base model once (drawing A, or copying a
saved ``lora_init``), and :func:`featurize_prepared` scores every later batch
on that same module. With the historical defaults (float32, no checkpointing,
``logprob_chunk=None``, ``device_map=None``) the score is the original
full-vocabulary ``log_softmax`` path. Anything else scores completion
log-probabilities from hidden states, applying ``lm_head`` in sequence chunks
and reducing each chunk with float32 ``logsumexp``, and sketches each LoRA-B
gradient as soon as it is accumulated.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from vfates.features.hidden import (
    _embedding_device,
    _encode_feature_row,
    _memory_bytes,
    _place_model,
    _release_memory,
    _resolve_device,
    _resolve_dtype,
    _sync,
    _truncate_left,
    load_model,
)
from vfates.features.lora_init import (
    apply_lora_a,
    assert_lora_a_equal,
    load_lora_init,
    lora_a_state,
    save_lora_init,
)
from vfates.features.sketch import CountSketch
from vfates.utils.io import sha256_file

# Sequence positions per lm_head call when the memory path is on and the caller
# did not set logprob_chunk. One chunk of Qwen's vocabulary is the live logit tensor.
DEFAULT_LOGPROB_CHUNK = 16


@dataclass
class LoraGradFeatures:
    """Sketched per-sample ``B`` gradients.

    ``features`` is float32 ``[N, D]``. ``lengths`` is the completion token
    count. ``exact`` is the concatenated ``B`` gradient when requested.
    """

    features: np.ndarray
    lengths: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    exact: np.ndarray | None = None


@dataclass
class PreparedLoraGrad:
    """PEFT model wrapped once, ready for many :func:`featurize_prepared` calls.

    The audit should build this before the feature-part loop and keep it for
    every part. :func:`lora_grad_features` builds one and scores a single batch.
    """

    model: Any
    b_params: list[tuple[str, torch.nn.Parameter]]
    a_params: list[tuple[str, torch.nn.Parameter]]
    sketch: CountSketch
    dtype: str
    torch_dtype: torch.dtype
    device: torch.device
    device_map: str | None
    gradient_checkpointing: bool
    param_count: int
    init_sha: str | None
    model_id: str | None
    lora_r: int
    lora_alpha: int
    lora_seed: int
    sketch_dim: int
    sketch_seed: int
    target_modules: tuple[str, ...]
    a_limit: float


def _lora_parameters(model: Any, part: str) -> list[tuple[str, torch.nn.Parameter]]:
    found = [(name, param) for name, param in model.named_parameters() if f".{part}." in name]
    found.sort(key=lambda item: item[0])
    return found


def _completion_logits(logits: torch.Tensor, completion_start: int, n_comp: int) -> torch.Tensor:
    if logits.shape[1] == n_comp + 1:
        return logits[:, :-1, :]
    full = completion_start + n_comp
    if logits.shape[1] >= full:
        return logits[:, completion_start - 1 : completion_start - 1 + n_comp, :]
    raise RuntimeError(
        f"logits length {logits.shape[1]} does not cover {n_comp} completion tokens "
        f"starting at {completion_start}"
    )


def _grad_max_abs(params: list[tuple[str, torch.nn.Parameter]]) -> float:
    max_abs = 0.0
    for _, param in params:
        if param.grad is None:
            continue
        value = float(param.grad.detach().abs().max().item()) if param.grad.numel() else 0.0
        max_abs = max(max_abs, value)
    return max_abs


def _validate_logprob_chunk(logprob_chunk: int | None) -> None:
    if logprob_chunk is None:
        return
    if isinstance(logprob_chunk, bool) or not isinstance(logprob_chunk, int) or logprob_chunk < 1:
        raise ValueError(f"logprob_chunk must be a positive int or None, got {logprob_chunk!r}")


def _logprobs_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Target-token log-probs. ``logsumexp`` runs in float32 over the vocabulary."""
    logits_f = logits.float()
    chosen = logits_f.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(logits_f, dim=-1)


def _checkpoint_chunk(hidden: torch.Tensor, targets: torch.Tensor, lm_head: Any) -> torch.Tensor:
    def run(h_chunk: torch.Tensor) -> torch.Tensor:
        return _logprobs_from_logits(lm_head(h_chunk), targets)

    return torch.utils.checkpoint.checkpoint(run, hidden, use_reentrant=False)


def _score_hidden(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head: Any,
    chunk: int,
    *,
    checkpoint: bool,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    n = int(hidden.shape[1])
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        h = hidden[:, start:end]
        tgt = targets[:, start:end]
        if checkpoint and bool(h.requires_grad):
            pieces.append(_checkpoint_chunk(h, tgt, lm_head))
        else:
            pieces.append(_logprobs_from_logits(lm_head(h), tgt))
    return torch.cat(pieces, dim=1)


def completion_token_logprobs(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head: Any,
    *,
    chunk: int | None = None,
) -> torch.Tensor:
    """Per-token completion log-probs from hidden states, without a full-sequence logit tensor.

    ``hidden`` is ``[batch, seq, hidden]`` (the states that predict ``targets``)
    and ``targets`` is ``[batch, seq]`` long. Each chunk applies ``lm_head`` and
    gathers ``logit[target] - logsumexp(logits)`` in float32. ``chunk=None``
    scores the whole sequence in one matmul. The result is ``[batch, seq]``.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [batch, seq, hidden], got {tuple(hidden.shape)}")
    if targets.shape != hidden.shape[:2]:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} does not match hidden {tuple(hidden.shape[:2])}"
        )
    _validate_logprob_chunk(chunk)
    n = int(hidden.shape[1])
    if n == 0:
        return torch.zeros((hidden.shape[0], 0), dtype=torch.float32, device=hidden.device)
    step = n if chunk is None else chunk
    return _score_hidden(hidden, targets, lm_head, step, checkpoint=False)


def _use_memory_path(prepared: PreparedLoraGrad, logprob_chunk: int | None) -> bool:
    # Float32, no checkpointing, and no explicit chunk is the historical path.
    # Its full-vocabulary log_softmax is what existing audits and tests match.
    return (
        logprob_chunk is not None or prepared.dtype != "float32" or prepared.gradient_checkpointing
    )


def _effective_chunk(logprob_chunk: int | None, n_comp: int) -> int:
    step = DEFAULT_LOGPROB_CHUNK if logprob_chunk is None else logprob_chunk
    return max(1, min(step, n_comp))


class _BGradTap:
    """Sketch each LoRA-B gradient when it is accumulated, then drop it.

    Contributions are reduced in parameter-name order, the same order as
    :meth:`CountSketch.sketch_torch`. ``A`` gradients are freed the same way;
    only their max-abs is kept, and only for the sample that asked for it.
    """

    def __init__(
        self,
        sketch: CountSketch,
        b_params: list[tuple[str, torch.nn.Parameter]],
        a_params: list[tuple[str, torch.nn.Parameter]],
    ) -> None:
        devices = {param.device for _, param in b_params}
        if len(devices) == 1:
            self.acc_device = next(iter(devices))
            self.acc_dtype = torch.float64 if self.acc_device.type == "cpu" else torch.float32
        else:
            # Multi-GPU: sketch on CPU in float32 so each device can drop its gradient.
            self.acc_device = torch.device("cpu")
            self.acc_dtype = torch.float32
        self.sketch = sketch
        self.b_params = b_params
        self.parts: list[torch.Tensor | None] = [None] * len(b_params)
        self.exact: list[np.ndarray | None] | None = None
        self.track_a = False
        self.a_max = 0.0
        self._handles: list[Any] = []
        for index, (_, param) in enumerate(b_params):
            self._handles.append(param.register_post_accumulate_grad_hook(self._b_hook(index)))
        for _, param in a_params:
            self._handles.append(param.register_post_accumulate_grad_hook(self._a_hook))

    def _b_hook(self, index: int):
        def hook(param: torch.nn.Parameter) -> None:
            grad = param.grad
            if grad is None:
                return
            self.parts[index] = self.sketch.sketch_tensor(
                grad, index, device=self.acc_device, dtype=self.acc_dtype
            )
            if self.exact is not None:
                self.exact[index] = grad.detach().float().reshape(-1).cpu().numpy()
            param.grad = None

        return hook

    def _a_hook(self, param: torch.nn.Parameter) -> None:
        grad = param.grad
        if grad is None:
            return
        if self.track_a and grad.numel():
            self.a_max = max(self.a_max, float(grad.detach().abs().max().item()))
        param.grad = None

    def begin(self, *, track_a: bool, exact: bool) -> None:
        self.parts = [None] * len(self.b_params)
        self.exact = [None] * len(self.b_params) if exact else None
        self.track_a = track_a
        if track_a:
            self.a_max = 0.0

    def sketched(self) -> torch.Tensor:
        missing = [
            self.b_params[i][0]
            for i, part in enumerate(self.parts)
            if part is None and self.b_params[i][1].numel() > 0
        ]
        if missing:
            raise RuntimeError(f"no lora_B gradient for {missing[:3]}")
        out = torch.zeros(self.sketch.dim, dtype=self.acc_dtype, device=self.acc_device)
        for part in self.parts:
            if part is None:
                continue
            out = out + part.to(device=self.acc_device, dtype=self.acc_dtype)
        self.parts = [None] * len(self.b_params)
        return out

    def exact_vector(self) -> np.ndarray:
        if self.exact is None:
            raise RuntimeError("exact LoRA-B gradient was not captured")
        parts: list[np.ndarray] = []
        for part in self.exact:
            if part is None:
                raise RuntimeError("exact LoRA-B gradient was not captured")
            parts.append(part)
        return np.concatenate(parts)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _enable_gradient_checkpointing(model: Any) -> None:
    """Turn on checkpointing for a PEFT-wrapped causal LM.

    Frozen embedding inputs do not require grad, so checkpointing would drop
    the LoRA graph. ``enable_input_require_grads`` marks those outputs before
    the base model enables checkpointing.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    elif hasattr(base, "get_input_embeddings"):
        embeddings = base.get_input_embeddings()

        def _require_grad(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        embeddings.register_forward_hook(_require_grad)
    if not hasattr(base, "gradient_checkpointing_enable"):
        raise RuntimeError("model does not support gradient checkpointing")
    base.gradient_checkpointing_enable()
    for module in (base, model):
        config = getattr(module, "config", None)
        if config is not None:
            config.use_cache = False


def _seed_lora(lora_seed: int, device: torch.device) -> None:
    torch.manual_seed(lora_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(lora_seed)
    elif device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(lora_seed)


def _decoder_hidden(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, Any]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(base, "model", None)
    lm_head = getattr(base, "lm_head", None)
    if decoder is None or lm_head is None:
        raise RuntimeError("chunked log-probs need a causal LM with .model and .lm_head")
    if hasattr(model, "_enable_peft_forward_hooks"):
        context = model._enable_peft_forward_hooks(
            input_ids=input_ids, attention_mask=attention_mask
        )
    else:
        context = contextlib.nullcontext()
    with context:
        outputs = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("decoder returned no last_hidden_state")
    return hidden, lm_head


def prepare_lora_grad(
    model_or_id: Any,
    *,
    sketch_dim: int = 8192,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    lora_seed: int = 0,
    sketch_seed: int = 0,
    device: str | torch.device | None = None,
    dtype: str = "float32",
    device_map: str | None = None,
    gradient_checkpointing: bool = False,
    save_init: str | Path | None = None,
    load_init: str | Path | None = None,
) -> PreparedLoraGrad:
    """Wrap ``model_or_id`` with fresh LoRA and keep that module for many batches.

    ``save_init`` draws A from ``lora_seed`` and writes ``lora_init.safetensors``
    the first time, then requires a later draw to match that file bitwise.
    ``load_init`` copies those matrices onto the new adapters instead, which is
    how a bfloat16 run uses the float32 audit's A. Pass one of them, not both.
    ``device_map='auto'`` shards with accelerate (a no-op when CUDA is absent)
    and must not be combined with ``device``. A model that already has
    ``hf_device_map`` is left on those devices; load it in the dtype you want.

    The returned model stays in train mode with the adapters attached. Do not
    pass it to :func:`lora_grad_features`; pass it to :func:`featurize_prepared`.
    """
    if sketch_dim < 1:
        raise ValueError(f"sketch_dim must be positive, got {sketch_dim}")
    if lora_r < 1:
        raise ValueError(f"lora_r must be positive, got {lora_r}")
    if device is not None and device_map is not None:
        raise ValueError("pass device or device_map, not both")
    if load_init is not None and save_init is not None:
        raise ValueError("pass load_init or save_init, not both")
    torch_dtype = _resolve_dtype(dtype)
    if isinstance(model_or_id, str):
        if device_map is None:
            resolved = _resolve_device(device)
            model, _loaded = load_model(model_or_id, device=resolved, dtype=dtype)
        else:
            model, _loaded = load_model(model_or_id, dtype=dtype, device_map=device_map)
            resolved = _embedding_device(model, torch.device("cpu"))
    else:
        model = model_or_id
        if hasattr(model, "peft_config"):
            raise ValueError("lora_grad_features expects a base model, not a PeftModel")
        # A model loaded with device_map already carries hf_device_map. Moving it
        # with .to() would gather every shard onto one device.
        if getattr(model, "hf_device_map", None):
            if device is not None:
                raise ValueError("model is already sharded; do not pass device")
            resolved = _embedding_device(model, torch.device("cpu"))
        elif device_map is None:
            resolved = _resolve_device(device)
            model.to(device=resolved, dtype=torch_dtype)
        else:
            model = _place_model(model, device=None, dtype=dtype, device_map=device_map)
            resolved = _embedding_device(model, torch.device("cpu"))
    model_id = model_or_id if isinstance(model_or_id, str) else None
    if model_id is None:
        model_id = getattr(getattr(model, "config", None), "name_or_path", None)

    from peft import LoraConfig, get_peft_model

    # Reseed after the base weights exist so adapter init matches the training run.
    _seed_lora(lora_seed, resolved)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    b_params = _lora_parameters(model, "lora_B")
    a_params = _lora_parameters(model, "lora_A")
    if not b_params:
        raise RuntimeError("LoRA wrapping produced no lora_B parameters; check target_modules")
    if load_init is not None:
        saved, _loaded_meta = load_lora_init(load_init)
        apply_lora_a(model, saved)
        init_sha: str | None = sha256_file(load_init)
    else:
        init_sha = _match_or_save_lora_init(
            model,
            save_init,
            model_id="" if model_id is None else str(model_id),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_seed=lora_seed,
            target_modules=target_modules,
        )
    if gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    param_count = int(sum(param.numel() for _, param in b_params))
    a_limit = 1e-5 if torch_dtype == torch.float32 else 1e-2
    return PreparedLoraGrad(
        model=model,
        b_params=b_params,
        a_params=a_params,
        sketch=CountSketch(sketch_dim, sketch_seed),
        dtype=dtype,
        torch_dtype=torch_dtype,
        device=resolved,
        device_map=device_map,
        gradient_checkpointing=bool(gradient_checkpointing),
        param_count=param_count,
        init_sha=init_sha,
        model_id=None if model_id is None else str(model_id),
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_seed=lora_seed,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
        target_modules=tuple(target_modules),
        a_limit=a_limit,
    )


def _empty_features(
    sketch_dim: int,
    sketch_seed: int,
    normalize: str,
    source: str,
) -> LoraGradFeatures:
    return LoraGradFeatures(
        features=np.zeros((0, sketch_dim), dtype=np.float32),
        lengths=np.zeros((0,), dtype=np.int64),
        meta={
            "sketch_dim": sketch_dim,
            "sketch_seed": sketch_seed,
            "normalize": normalize,
            "n": 0,
            "completion_source": source,
        },
    )


def _batch_meta(
    prepared: PreparedLoraGrad,
    *,
    normalize: str,
    n_samples: int,
    source: str,
    elapsed: float,
    sample_seconds: list[float],
    a_grad_max: float,
    peak_allocated: int,
    peak_driver: int,
    saw_memory: bool,
    logprob_chunk: int | None,
    memory_path: bool,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "model_id": prepared.model_id,
        "lora_r": prepared.lora_r,
        "lora_alpha": prepared.lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": list(prepared.target_modules),
        "lora_seed": prepared.lora_seed,
        "init_lora_weights": True,
        "sketch_dim": prepared.sketch_dim,
        "sketch_seed": prepared.sketch_seed,
        "param_count": prepared.param_count,
        "lora_b_names": [name for name, _ in prepared.b_params],
        "normalize": normalize,
        "dtype": prepared.dtype,
        "device": str(prepared.device),
        "device_map": prepared.device_map,
        "gradient_checkpointing": prepared.gradient_checkpointing,
        "logprob_chunk": logprob_chunk,
        "memory_path": memory_path,
        "chat_template_kwargs": None
        if chat_template_kwargs is None
        else dict(chat_template_kwargs),
        "seconds_per_sample": elapsed / n_samples,
        "sample_seconds": sample_seconds,
        "lora_a_grad_max_abs": a_grad_max,
        "n": n_samples,
        "peak_allocated_bytes": peak_allocated if saw_memory else None,
        "peak_driver_bytes": peak_driver if saw_memory else None,
        "completion_source": source,
    }
    if prepared.init_sha is not None:
        meta["lora_init_sha256"] = prepared.init_sha
    return meta


def _check_batch(
    prompts: list[str],
    completions: list[str],
    completion_ids: Sequence[Sequence[int]] | None,
    normalize: str,
    logprob_chunk: int | None,
) -> str:
    if normalize not in ("sum", "token_mean"):
        raise ValueError(f"normalize must be 'sum' or 'token_mean', got {normalize!r}")
    if len(prompts) != len(completions):
        raise ValueError(f"prompts ({len(prompts)}) and completions ({len(completions)}) differ")
    if completion_ids is not None and len(completion_ids) != len(prompts):
        raise ValueError(
            f"completion_ids ({len(completion_ids)}) and prompts ({len(prompts)}) differ"
        )
    _validate_logprob_chunk(logprob_chunk)
    return "token_ids" if completion_ids is not None else "text"


def _encode_sample(
    tokenizer: Any,
    prompt: str,
    completion: str,
    system_prompt: str | None,
    completion_row: Sequence[int] | None,
    chat_template_kwargs: Mapping[str, Any] | None,
    max_len: int | None,
) -> tuple[list[int], int, int]:
    ids, completion_start = _encode_feature_row(
        tokenizer, prompt, completion, system_prompt, completion_row, chat_template_kwargs
    )
    ids, completion_start = _truncate_left(ids, completion_start, max_len)
    if completion_start < 1:
        raise ValueError("need at least one prompt token before the completion to score it")
    return ids, completion_start, len(ids) - completion_start


def featurize_prepared(
    prepared: PreparedLoraGrad,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    *,
    normalize: str = "sum",
    max_len: int | None = None,
    system_prompt: str | None = None,
    return_exact: bool = False,
    progress: bool = False,
    completion_ids: Sequence[Sequence[int]] | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    logprob_chunk: int | None = None,
) -> LoraGradFeatures:
    """Sketch one batch on a model from :func:`prepare_lora_grad`.

    Safe to call again on the same ``prepared`` value: weights are not updated,
    and each sample is differentiated on its own. ``logprob_chunk`` overrides
    the historical full-logit path when set (see :func:`lora_grad_features`).
    """
    source = _check_batch(prompts, completions, completion_ids, normalize, logprob_chunk)
    n_samples = len(prompts)
    if n_samples == 0:
        return _empty_features(prepared.sketch_dim, prepared.sketch_seed, normalize, source)
    if _use_memory_path(prepared, logprob_chunk):
        return _featurize_memory(
            prepared,
            tokenizer,
            prompts,
            completions,
            normalize=normalize,
            max_len=max_len,
            system_prompt=system_prompt,
            return_exact=return_exact,
            progress=progress,
            completion_ids=completion_ids,
            chat_template_kwargs=chat_template_kwargs,
            logprob_chunk=logprob_chunk,
            source=source,
        )
    return _featurize_legacy(
        prepared,
        tokenizer,
        prompts,
        completions,
        normalize=normalize,
        max_len=max_len,
        system_prompt=system_prompt,
        return_exact=return_exact,
        progress=progress,
        completion_ids=completion_ids,
        chat_template_kwargs=chat_template_kwargs,
        source=source,
    )


def _featurize_legacy(
    prepared: PreparedLoraGrad,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    *,
    normalize: str,
    max_len: int | None,
    system_prompt: str | None,
    return_exact: bool,
    progress: bool,
    completion_ids: Sequence[Sequence[int]] | None,
    chat_template_kwargs: Mapping[str, Any] | None,
    source: str,
) -> LoraGradFeatures:
    """Historical float32 path: one full ``log_softmax`` over the completion logits."""
    model = prepared.model
    resolved = prepared.device
    n_samples = len(prompts)
    features = np.zeros((n_samples, prepared.sketch_dim), dtype=np.float32)
    lengths = np.zeros((n_samples,), dtype=np.int64)
    exact = np.zeros((n_samples, prepared.param_count), dtype=np.float32) if return_exact else None
    a_grad_max = 0.0
    sample_seconds: list[float] = []
    peak_allocated = 0
    peak_driver = 0
    saw_memory = False
    model.train()
    for index, prompt in enumerate(prompts):
        started = time.perf_counter()
        row = None if completion_ids is None else completion_ids[index]
        ids, completion_start, n_comp = _encode_sample(
            tokenizer,
            prompt,
            completions[index],
            system_prompt,
            row,
            chat_template_kwargs,
            max_len,
        )
        lengths[index] = n_comp
        model.zero_grad(set_to_none=True)
        if n_comp == 0:
            sample_seconds.append(time.perf_counter() - started)
            if progress:
                print(f"lora_grad {index + 1}/{n_samples} empty", file=sys.stderr)
            continue
        input_ids = torch.tensor(ids, dtype=torch.long, device=resolved).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        # Logit at position t predicts token t + 1. The completion occupies
        # [completion_start, L), predicted by hidden states [completion_start - 1, L - 1).
        # logits_to_keep asks for those rows plus the unused final position; if the
        # model returns the full sequence anyway, slice the same rows out of it.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=n_comp + 1,
        )
        logits = _completion_logits(outputs.logits, completion_start, n_comp)
        targets = input_ids[:, completion_start:]
        token_logprobs = (
            torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).reshape(-1)
        )
        objective = token_logprobs.mean() if normalize == "token_mean" else token_logprobs.sum()
        objective.backward()
        _sync(resolved)
        memory = _memory_bytes(resolved)
        if memory is not None:
            saw_memory = True
            peak_allocated = max(peak_allocated, memory[0])
            peak_driver = max(peak_driver, memory[1])
        if index == 0:
            a_grad_max = _grad_max_abs(prepared.a_params)
            if a_grad_max > prepared.a_limit:
                raise RuntimeError(
                    f"lora_A gradient max-abs is {a_grad_max}, expected ~0 at standard LoRA init (B = 0)"
                )
        missing = [name for name, param in prepared.b_params if param.grad is None]
        if missing:
            raise RuntimeError(f"no lora_B gradient for {missing[:3]}")
        if exact is not None:
            exact[index] = np.concatenate(
                [
                    cast(torch.Tensor, param.grad).detach().float().reshape(-1).cpu().numpy()
                    for _, param in prepared.b_params
                ]
            )
        sketched = prepared.sketch.sketch_torch([param.grad for _, param in prepared.b_params])
        features[index] = sketched.to(dtype=torch.float32).cpu().numpy()
        model.zero_grad(set_to_none=True)
        del objective, token_logprobs, logits, targets, outputs, input_ids, attention_mask, sketched
        _release_memory(resolved)
        _sync(resolved)
        sample_seconds.append(time.perf_counter() - started)
        if progress:
            print(
                f"lora_grad {index + 1}/{n_samples} {sample_seconds[-1]:.2f}s",
                file=sys.stderr,
            )
    elapsed = float(sum(sample_seconds))
    meta = _batch_meta(
        prepared,
        normalize=normalize,
        n_samples=n_samples,
        source=source,
        elapsed=elapsed,
        sample_seconds=sample_seconds,
        a_grad_max=a_grad_max,
        peak_allocated=peak_allocated,
        peak_driver=peak_driver,
        saw_memory=saw_memory,
        logprob_chunk=None,
        memory_path=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    return LoraGradFeatures(features=features, lengths=lengths, meta=meta, exact=exact)


def _featurize_memory(
    prepared: PreparedLoraGrad,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    *,
    normalize: str,
    max_len: int | None,
    system_prompt: str | None,
    return_exact: bool,
    progress: bool,
    completion_ids: Sequence[Sequence[int]] | None,
    chat_template_kwargs: Mapping[str, Any] | None,
    logprob_chunk: int | None,
    source: str,
) -> LoraGradFeatures:
    """Hidden-state log-probs in vocabulary chunks, with B gradients sketched immediately."""
    model = prepared.model
    resolved = prepared.device
    n_samples = len(prompts)
    features = np.zeros((n_samples, prepared.sketch_dim), dtype=np.float32)
    lengths = np.zeros((n_samples,), dtype=np.int64)
    exact = np.zeros((n_samples, prepared.param_count), dtype=np.float32) if return_exact else None
    a_grad_max = 0.0
    sample_seconds: list[float] = []
    peak_allocated = 0
    peak_driver = 0
    saw_memory = False
    model.train()
    tap = _BGradTap(prepared.sketch, prepared.b_params, prepared.a_params)
    try:
        for index, prompt in enumerate(prompts):
            started = time.perf_counter()
            row = None if completion_ids is None else completion_ids[index]
            ids, completion_start, n_comp = _encode_sample(
                tokenizer,
                prompt,
                completions[index],
                system_prompt,
                row,
                chat_template_kwargs,
                max_len,
            )
            lengths[index] = n_comp
            model.zero_grad(set_to_none=True)
            if n_comp == 0:
                sample_seconds.append(time.perf_counter() - started)
                if progress:
                    print(f"lora_grad {index + 1}/{n_samples} empty", file=sys.stderr)
                continue
            tap.begin(track_a=index == 0, exact=return_exact)
            input_ids = torch.tensor(ids, dtype=torch.long, device=resolved).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            hidden, lm_head = _decoder_hidden(model, input_ids, attention_mask)
            pred = hidden[:, completion_start - 1 : completion_start - 1 + n_comp, :]
            if int(pred.shape[1]) != n_comp:
                raise RuntimeError(
                    f"predictor span {int(pred.shape[1])} does not cover {n_comp} completion tokens"
                )
            targets = input_ids[:, completion_start:]
            token_lp = _score_hidden(
                pred,
                targets,
                lm_head,
                _effective_chunk(logprob_chunk, n_comp),
                checkpoint=True,
            )
            flat = token_lp.reshape(-1)
            objective = flat.mean() if normalize == "token_mean" else flat.sum()
            objective.backward()
            _sync(resolved)
            memory = _memory_bytes(resolved)
            if memory is not None:
                saw_memory = True
                peak_allocated = max(peak_allocated, memory[0])
                peak_driver = max(peak_driver, memory[1])
            if index == 0:
                a_grad_max = tap.a_max
                if a_grad_max > prepared.a_limit:
                    raise RuntimeError(
                        "lora_A gradient max-abs is "
                        f"{a_grad_max}, expected ~0 at standard LoRA init (B = 0)"
                    )
            if exact is not None:
                exact[index] = tap.exact_vector()
            sketched = tap.sketched()
            features[index] = sketched.to(dtype=torch.float32).cpu().numpy()
            model.zero_grad(set_to_none=True)
            del (
                objective,
                flat,
                token_lp,
                targets,
                pred,
                hidden,
                input_ids,
                attention_mask,
                sketched,
            )
            _release_memory(resolved)
            _sync(resolved)
            sample_seconds.append(time.perf_counter() - started)
            if progress:
                print(
                    f"lora_grad {index + 1}/{n_samples} {sample_seconds[-1]:.2f}s",
                    file=sys.stderr,
                )
    finally:
        tap.close()
    elapsed = float(sum(sample_seconds))
    meta = _batch_meta(
        prepared,
        normalize=normalize,
        n_samples=n_samples,
        source=source,
        elapsed=elapsed,
        sample_seconds=sample_seconds,
        a_grad_max=a_grad_max,
        peak_allocated=peak_allocated,
        peak_driver=peak_driver,
        saw_memory=saw_memory,
        logprob_chunk=logprob_chunk,
        memory_path=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    return LoraGradFeatures(features=features, lengths=lengths, meta=meta, exact=exact)


def lora_grad_features(
    model_or_id: Any,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    *,
    sketch_dim: int = 8192,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    lora_seed: int = 0,
    sketch_seed: int = 0,
    normalize: str = "sum",
    device: str | torch.device | None = None,
    dtype: str = "float32",
    max_len: int | None = None,
    system_prompt: str | None = None,
    return_exact: bool = False,
    progress: bool = False,
    completion_ids: Sequence[Sequence[int]] | None = None,
    save_init: str | Path | None = None,
    load_init: str | Path | None = None,
    device_map: str | None = None,
    gradient_checkpointing: bool = False,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    logprob_chunk: int | None = None,
) -> LoraGradFeatures:
    """Sketch the completion-log-probability gradient w.r.t. freshly initialised ``lora_B``.

    The passed base model is wrapped in place with PEFT and left in train mode
    with those adapters attached. One sample is differentiated at a time; its
    graph and gradients are dropped before the next sample so the live set is
    one sequence, not the batch. For many batches, call :func:`prepare_lora_grad`
    once and :func:`featurize_prepared` per batch instead of this function: a
    second call refuses the already wrapped model.

    ``dtype='bfloat16'`` and ``gradient_checkpointing=True`` score log-probs
    from hidden states (see :func:`completion_token_logprobs`) instead of
    materialising a float32 logit row for every completion position.
    ``logprob_chunk`` is that sequence chunk; ``None`` keeps the historical
    float32 ``log_softmax`` path when ``dtype`` is ``float32`` and checkpointing
    is off, and otherwise uses :data:`DEFAULT_LOGPROB_CHUNK`. ``device_map`` is
    ``None`` or ``'auto'``. ``chat_template_kwargs`` is forwarded to the chat
    encoder only when it is not ``None``.

    ``completion_ids``, when given, are the sampled completion ids (including a
    terminating EOS when the sample stopped). They are appended to the chat
    prompt ids with no re-tokenisation, and ``lengths`` is the number of those
    ids after the same left truncation as the text path. ``meta['completion_source']``
    is ``'token_ids'`` or ``'text'``.

    ``save_init`` writes :file:`lora_init.safetensors` on first use and, on a
    later call, requires the freshly seeded A matrices to equal that file
    bitwise. ``load_init`` copies a saved file onto the adapters instead of
    checking the seed. The audit calls this once per feature part on a freshly
    loaded model reseeded with ``lora_seed`` unless it uses
    :func:`prepare_lora_grad`. ``meta['lora_init_sha256']`` is the file digest
    when ``save_init`` or ``load_init`` is set.
    """
    source = _check_batch(prompts, completions, completion_ids, normalize, logprob_chunk)
    if sketch_dim < 1:
        raise ValueError(f"sketch_dim must be positive, got {sketch_dim}")
    if lora_r < 1:
        raise ValueError(f"lora_r must be positive, got {lora_r}")
    n_samples = len(prompts)
    if n_samples == 0:
        return _empty_features(sketch_dim, sketch_seed, normalize, source)
    prepared = prepare_lora_grad(
        model_or_id,
        sketch_dim=sketch_dim,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_seed=lora_seed,
        sketch_seed=sketch_seed,
        device=device,
        dtype=dtype,
        device_map=device_map,
        gradient_checkpointing=gradient_checkpointing,
        save_init=save_init,
        load_init=load_init,
    )
    return featurize_prepared(
        prepared,
        tokenizer,
        prompts,
        completions,
        normalize=normalize,
        max_len=max_len,
        system_prompt=system_prompt,
        return_exact=return_exact,
        progress=progress,
        completion_ids=completion_ids,
        chat_template_kwargs=chat_template_kwargs,
        logprob_chunk=logprob_chunk,
    )


def _match_or_save_lora_init(
    model: Any,
    save_init: str | Path | None,
    *,
    model_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_seed: int,
    target_modules: tuple[str, ...],
) -> str | None:
    """Write the A file on first use; require bitwise equality when it exists."""
    if save_init is None:
        return None
    fresh = lora_a_state(model)
    path = Path(save_init)
    init_meta: dict[str, str | int | float] = {
        "model_id": model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_seed": lora_seed,
        "target_modules": ",".join(target_modules),
        "n_modules": len(fresh),
    }
    if path.exists():
        saved, _loaded = load_lora_init(path)
        assert_lora_a_equal(fresh, saved)
        return sha256_file(path)
    return save_lora_init(path, fresh, init_meta)
