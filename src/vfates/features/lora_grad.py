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
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from vfates.features.hidden import (
    _memory_bytes,
    _release_memory,
    _resolve_device,
    _resolve_dtype,
    _sync,
    _truncate_left,
    load_model,
)
from vfates.features.lora_init import (
    assert_lora_a_equal,
    load_lora_init,
    lora_a_state,
    save_lora_init,
)
from vfates.features.sketch import CountSketch
from vfates.gen.chat import encode_prompt_completion, prompt_completion_ids
from vfates.utils.io import sha256_file


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
) -> LoraGradFeatures:
    """Sketch the completion-log-probability gradient w.r.t. freshly initialised ``lora_B``.

    The passed base model is wrapped in place with PEFT and left in train mode
    with those adapters attached. One sample is differentiated at a time; its
    graph and gradients are dropped before the next sample so the live set is
    one sequence, not the batch.

    ``completion_ids``, when given, are the sampled completion ids (including a
    terminating EOS when the sample stopped). They are appended to the chat
    prompt ids with no re-tokenisation, and ``lengths`` is the number of those
    ids after the same left truncation as the text path. ``meta['completion_source']``
    is ``'token_ids'`` or ``'text'``.

    ``save_init`` writes :file:`lora_init.safetensors` on first use and, on a
    later call, requires the freshly seeded A matrices to equal that file
    bitwise. The audit calls this once per feature part on a freshly loaded
    model reseeded with ``lora_seed``. ``meta['lora_init_sha256']`` is the file
    digest when ``save_init`` is set.
    """
    if normalize not in ("sum", "token_mean"):
        raise ValueError(f"normalize must be 'sum' or 'token_mean', got {normalize!r}")
    if len(prompts) != len(completions):
        raise ValueError(f"prompts ({len(prompts)}) and completions ({len(completions)}) differ")
    if completion_ids is not None and len(completion_ids) != len(prompts):
        raise ValueError(
            f"completion_ids ({len(completion_ids)}) and prompts ({len(prompts)}) differ"
        )
    if sketch_dim < 1:
        raise ValueError(f"sketch_dim must be positive, got {sketch_dim}")
    if lora_r < 1:
        raise ValueError(f"lora_r must be positive, got {lora_r}")
    source = "token_ids" if completion_ids is not None else "text"
    n_samples = len(prompts)
    if n_samples == 0:
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

    resolved = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype)
    model_id = model_or_id if isinstance(model_or_id, str) else None
    if isinstance(model_or_id, str):
        model, _loaded = load_model(model_or_id, device=resolved, dtype=dtype)
    else:
        model = model_or_id
        if hasattr(model, "peft_config"):
            raise ValueError("lora_grad_features expects a base model, not a PeftModel")
        model.to(device=resolved, dtype=torch_dtype)
    if model_id is None:
        model_id = getattr(getattr(model, "config", None), "name_or_path", None)

    from peft import LoraConfig, get_peft_model

    # Reseed after the base weights exist so adapter init matches the training run.
    torch.manual_seed(lora_seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(lora_seed)
    elif resolved.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(lora_seed)
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
    init_sha = _match_or_save_lora_init(
        model,
        save_init,
        model_id="" if model_id is None else str(model_id),
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_seed=lora_seed,
        target_modules=target_modules,
    )
    param_count = int(sum(param.numel() for _, param in b_params))
    sketch = CountSketch(sketch_dim, sketch_seed)
    features = np.zeros((n_samples, sketch_dim), dtype=np.float32)
    lengths = np.zeros((n_samples,), dtype=np.int64)
    exact = np.zeros((n_samples, param_count), dtype=np.float32) if return_exact else None
    a_grad_max = 0.0
    sample_seconds: list[float] = []
    peak_allocated = 0
    peak_driver = 0
    saw_memory = False
    # Float32 check is tight; reduced precision still has to be numerically zero at B = 0.
    a_limit = 1e-5 if torch_dtype == torch.float32 else 1e-2

    for index, prompt in enumerate(prompts):
        started = time.perf_counter()
        if completion_ids is None:
            ids, completion_start = encode_prompt_completion(
                tokenizer, prompt, completions[index], system_prompt
            )
        else:
            ids, completion_start = prompt_completion_ids(
                tokenizer, prompt, completion_ids[index], system_prompt
            )
        ids, completion_start = _truncate_left(ids, completion_start, max_len)
        if completion_start < 1:
            raise ValueError("need at least one prompt token before the completion to score it")
        n_comp = len(ids) - completion_start
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
            a_grad_max = _grad_max_abs(a_params)
            if a_grad_max > a_limit:
                raise RuntimeError(
                    f"lora_A gradient max-abs is {a_grad_max}, expected ~0 at standard LoRA init (B = 0)"
                )
        missing = [name for name, param in b_params if param.grad is None]
        if missing:
            raise RuntimeError(f"no lora_B gradient for {missing[:3]}")
        if exact is not None:
            exact[index] = np.concatenate(
                [
                    cast(torch.Tensor, param.grad).detach().float().reshape(-1).cpu().numpy()
                    for _, param in b_params
                ]
            )
        sketched = sketch.sketch_torch([param.grad for _, param in b_params])
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
    meta: dict[str, Any] = {
        "model_id": model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": list(target_modules),
        "lora_seed": lora_seed,
        "init_lora_weights": True,
        "sketch_dim": sketch_dim,
        "sketch_seed": sketch_seed,
        "param_count": param_count,
        "lora_b_names": [name for name, _ in b_params],
        "normalize": normalize,
        "dtype": dtype,
        "device": str(resolved),
        "seconds_per_sample": elapsed / n_samples,
        "sample_seconds": sample_seconds,
        "lora_a_grad_max_abs": a_grad_max,
        "n": n_samples,
        "peak_allocated_bytes": peak_allocated if saw_memory else None,
        "peak_driver_bytes": peak_driver if saw_memory else None,
        "completion_source": source,
    }
    if init_sha is not None:
        meta["lora_init_sha256"] = init_sha
    return LoraGradFeatures(features=features, lengths=lengths, meta=meta, exact=exact)


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
