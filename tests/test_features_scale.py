"""Chunked log-probs, reused LoRA wraps, and the bfloat16 sketch gate."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from test_gen_hf import build_tiny_qwen
from vfates.features.equivalence import (
    bf16_float32_gate,
    direction_agreement,
    feature_agreement,
)
from vfates.features.hidden import hidden_features
from vfates.features.lora_grad import (
    completion_token_logprobs,
    featurize_prepared,
    lora_grad_features,
    prepare_lora_grad,
)

_PROMPTS = ["Add 1 and 1.", "Say hello."]
_COMPLETIONS = ["2", "Hello there."]


def _tiny():
    try:
        return build_tiny_qwen(seed=0)
    except Exception as exc:
        pytest.skip(f"tiny Qwen unavailable offline: {exc}")


def _grad_kwargs() -> dict:
    return {
        "sketch_dim": 128,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_seed": 0,
        "sketch_seed": 1,
        "device": "cpu",
        "dtype": "float32",
    }


def test_chunked_logprob_matches_full_log_softmax() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(2, 6, 8)
    head = torch.nn.Linear(8, 17, bias=True)
    targets = torch.randint(0, 17, (2, 6))
    with torch.no_grad():
        logits = head(hidden).float()
        reference = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        chunked = completion_token_logprobs(hidden, targets, head, chunk=2)
        whole = completion_token_logprobs(hidden, targets, head, chunk=None)
    torch.testing.assert_close(chunked, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(whole, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-5)


def test_hook_sketches_match_float32_features() -> None:
    model, tokenizer = _tiny()
    kwargs = _grad_kwargs()
    reference = lora_grad_features(model, tokenizer, _PROMPTS, _COMPLETIONS, **kwargs)
    assert reference.meta["memory_path"] is False
    again_model, _tok = build_tiny_qwen(seed=0)
    chunked = lora_grad_features(
        again_model,
        tokenizer,
        _PROMPTS,
        _COMPLETIONS,
        logprob_chunk=2,
        return_exact=True,
        **kwargs,
    )
    assert chunked.meta["memory_path"] is True
    assert chunked.meta["lora_a_grad_max_abs"] == 0.0
    assert chunked.exact is not None
    assert chunked.exact.shape == (len(_PROMPTS), chunked.meta["param_count"])
    np.testing.assert_allclose(chunked.features, reference.features, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(chunked.lengths, reference.lengths)
    rows = feature_agreement(chunked.features, reference.features)
    assert rows["cosine_min"] > 0.99


def test_bf16_sketches_pass_float32_gate(tmp_path) -> None:
    model, tokenizer = _tiny()
    kwargs = _grad_kwargs()
    init = tmp_path / "lora_init.safetensors"
    reference = lora_grad_features(
        model, tokenizer, _PROMPTS, _COMPLETIONS, save_init=init, **kwargs
    )
    bf_model, _tok = build_tiny_qwen(seed=0)
    bf16 = lora_grad_features(
        bf_model,
        tokenizer,
        _PROMPTS,
        _COMPLETIONS,
        sketch_dim=kwargs["sketch_dim"],
        lora_r=kwargs["lora_r"],
        lora_alpha=kwargs["lora_alpha"],
        lora_seed=99,
        sketch_seed=kwargs["sketch_seed"],
        device="cpu",
        dtype="bfloat16",
        load_init=init,
        logprob_chunk=4,
    )
    gate = bf16_float32_gate(bf16.features, reference.features)
    assert gate["pass"], gate
    assert gate["cosine_min"] > 0.99
    assert gate["mean_direction_cosine"] > 0.99


def test_prepare_once_matches_per_call_and_load_init(tmp_path) -> None:
    model, tokenizer = _tiny()
    kwargs = _grad_kwargs()
    init = tmp_path / "lora_init.safetensors"
    prepared = prepare_lora_grad(model, save_init=init, **kwargs)
    first = featurize_prepared(prepared, tokenizer, _PROMPTS[:1], _COMPLETIONS[:1])
    second = featurize_prepared(prepared, tokenizer, _PROMPTS[1:], _COMPLETIONS[1:])
    call_model, _tok = build_tiny_qwen(seed=0)
    per_call = lora_grad_features(
        call_model, tokenizer, _PROMPTS, _COMPLETIONS, save_init=init, **kwargs
    )
    stacked = np.concatenate([first.features, second.features])
    np.testing.assert_array_equal(stacked, per_call.features)
    np.testing.assert_array_equal(np.concatenate([first.lengths, second.lengths]), per_call.lengths)
    loaded_model, _tok = build_tiny_qwen(seed=0)
    loaded = lora_grad_features(
        loaded_model,
        tokenizer,
        _PROMPTS,
        _COMPLETIONS,
        sketch_dim=kwargs["sketch_dim"],
        lora_r=kwargs["lora_r"],
        lora_alpha=kwargs["lora_alpha"],
        lora_seed=7,
        sketch_seed=kwargs["sketch_seed"],
        device="cpu",
        dtype="float32",
        load_init=init,
    )
    np.testing.assert_array_equal(loaded.features, per_call.features)
    assert loaded.meta["lora_init_sha256"] == per_call.meta["lora_init_sha256"]


def test_device_map_none_matches_default_features() -> None:
    model, tokenizer = _tiny()
    kwargs = _grad_kwargs()
    explicit = lora_grad_features(
        model, tokenizer, _PROMPTS, _COMPLETIONS, device_map=None, **kwargs
    )
    other, _tok = build_tiny_qwen(seed=0)
    omitted = lora_grad_features(other, tokenizer, _PROMPTS, _COMPLETIONS, **kwargs)
    np.testing.assert_array_equal(explicit.features, omitted.features)
    assert explicit.meta["device_map"] is None
    hidden_model, _tok = build_tiny_qwen(seed=1)
    pooled_explicit = hidden_features(
        hidden_model,
        tokenizer,
        _PROMPTS,
        _COMPLETIONS,
        device="cpu",
        device_map=None,
        batch_size=2,
    )
    pooled_default = hidden_features(
        hidden_model,
        tokenizer,
        _PROMPTS,
        _COMPLETIONS,
        device="cpu",
        batch_size=2,
    )
    np.testing.assert_array_equal(pooled_explicit, pooled_default)


def test_gradient_checkpointing_matches_float32() -> None:
    model, tokenizer = _tiny()
    kwargs = _grad_kwargs()
    reference = lora_grad_features(model, tokenizer, _PROMPTS, _COMPLETIONS, **kwargs)
    other, _tok = build_tiny_qwen(seed=0)
    checked = lora_grad_features(
        other,
        tokenizer,
        _PROMPTS,
        _COMPLETIONS,
        gradient_checkpointing=True,
        logprob_chunk=3,
        **kwargs,
    )
    assert checked.meta["gradient_checkpointing"] is True
    assert checked.meta["memory_path"] is True
    np.testing.assert_allclose(checked.features, reference.features, rtol=1e-4, atol=1e-4)
    assert feature_agreement(checked.features, reference.features)["cosine_min"] > 0.99


def test_feature_agreement_quantiles_and_direction() -> None:
    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    right[0] = np.array([0.0, 1.0, 0.0, 0.0])
    stats = feature_agreement(left, right)
    assert stats["cosine_min"] == 0.0
    assert stats["cosine_median"] == 1.0
    assert stats["n"] == 4.0
    assert 0.0 < stats["relative_norm_error"] < 1.0
    same = feature_agreement(left, left)
    assert same["cosine_min"] == 1.0
    assert same["mean_direction_cosine"] == 1.0
    assert same["relative_norm_error"] == 0.0
    pid = np.array([0, 0, 1, 1])
    verdicts = np.array([1.0, 0.0, 1.0, 0.0])
    agreed = direction_agreement(left, left, pid, verdicts)
    assert agreed["cosine"] == pytest.approx(1.0)
    assert agreed["relative_norm_error"] == pytest.approx(0.0)
    shifted = direction_agreement(left, right, pid, verdicts)
    assert shifted["cosine"] < 1.0
    gate = bf16_float32_gate(left, left)
    assert gate["pass"] is True
    reject = bf16_float32_gate(left, right)
    assert reject["pass"] is False
    with pytest.raises(ValueError, match="rank 2"):
        feature_agreement(np.zeros(3), np.zeros(3))
