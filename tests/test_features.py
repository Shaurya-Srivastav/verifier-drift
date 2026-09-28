"""Hidden-state pools and step-0 LoRA-B gradient sketches."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from test_gen_hf import build_tiny_qwen, qwen_tokenizer
from vfates.features.hidden import _truncate_left, hidden_features
from vfates.features.lora_grad import lora_grad_features
from vfates.features.lora_init import LORA_INIT_SCHEMA, load_lora_init
from vfates.gen.chat import encode_completion, encode_prompt_completion, prompt_completion_ids


def _manual_hidden(
    model, tokenizer, prompt, completion, pooling, system_prompt=None, ids=None, start=None
):
    if ids is None or start is None:
        ids, start = encode_prompt_completion(tokenizer, prompt, completion, system_prompt)
    with torch.inference_mode():
        outputs = model(
            torch.tensor([ids]),
            output_hidden_states=True,
            use_cache=False,
        )
    final = outputs.hidden_states[-1][0]
    # States that predict the completion tokens, not the states at those tokens.
    if start < 1 or len(ids) <= start:
        return np.zeros(model.config.hidden_size, dtype=np.float32)
    span = final[start - 1 : len(ids) - 1]
    pooled = span[-1] if pooling == "last" else span.mean(dim=0)
    return pooled.float().cpu().numpy()


def test_hidden_states_are_post_final_norm():
    model, tokenizer = build_tiny_qwen()
    ids, _ = encode_prompt_completion(tokenizer, "Hello", "there")
    captured = {}

    def hook(_module, _inputs, output):
        captured["norm"] = output.detach()

    handle = model.model.norm.register_forward_hook(hook)
    outputs = model(torch.tensor([ids]), output_hidden_states=True, use_cache=False)
    handle.remove()
    assert torch.allclose(outputs.hidden_states[-1], captured["norm"])
    assert not torch.allclose(outputs.hidden_states[-1], outputs.hidden_states[-2], atol=1e-3)


def test_mean_and_last_pooling_match_manual():
    model, tokenizer = build_tiny_qwen()
    prompts = ["What is 2+2?", "Name a prime."]
    completions = ["The answer is 4.", "2 is a prime."]
    for pooling in ("mean", "last"):
        got = hidden_features(
            model, tokenizer, prompts, completions, pooling=pooling, batch_size=1, device="cpu"
        )
        assert got.shape == (2, model.config.hidden_size)
        assert got.dtype == np.float32
        for index, (prompt, completion) in enumerate(zip(prompts, completions)):
            manual = _manual_hidden(model, tokenizer, prompt, completion, pooling)
            np.testing.assert_allclose(got[index], manual, rtol=1e-5, atol=1e-5)


def test_batching_and_padding_side_are_invariant():
    model, tokenizer = build_tiny_qwen(seed=2)
    prompts = [
        "Hi",
        "Explain why 2 is prime in one sentence.",
        "3+3?",
        "Write the word apple.",
        "What is the next integer after 9?",
    ]
    completions = ["Hello.", "It has no proper divisors.", "6", "apple", "10"]
    tokenizer.padding_side = "right"
    reference = hidden_features(
        model, tokenizer, prompts, completions, pooling="mean", batch_size=1, device="cpu"
    )
    for side in ("right", "left"):
        tokenizer.padding_side = side
        single = hidden_features(
            model, tokenizer, prompts, completions, pooling="mean", batch_size=1, device="cpu"
        )
        batched = hidden_features(
            model, tokenizer, prompts, completions, pooling="mean", batch_size=4, device="cpu"
        )
        np.testing.assert_allclose(single, reference, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(batched, reference, rtol=1e-4, atol=1e-4)


def test_step0_b_gradient_is_projected_weight_gradient():
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    a_layer = torch.nn.Linear(5, 4, bias=False)
    b_layer = torch.nn.Linear(4, 6, bias=False)
    torch.nn.init.kaiming_uniform_(a_layer.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(b_layer.weight)
    scaling = 2.0
    y = b_layer(a_layer(x)) * scaling
    y.sum().backward()
    assert a_layer.weight.grad is not None and torch.count_nonzero(a_layer.weight.grad) == 0
    d_y = torch.ones_like(y)
    # dL/dB = scaling * (dL/dY) @ (A x), i.e. a projection of the full weight gradient through A.
    expected = scaling * (d_y.T @ (x @ a_layer.weight.detach().T))
    assert torch.allclose(b_layer.weight.grad, expected)


def _prompts():
    return [
        "Add 1 and 1.",
        "What is a prime?",
        "Say hello.",
        "Compute 3*3.",
    ]


def _completions():
    return [
        "2",
        "A prime has no divisors other than one and itself.",
        "Hello there.",
        "9",
    ]


def _features(normalize="sum", return_exact=False, seed=0):
    tokenizer = qwen_tokenizer()
    model, _ = build_tiny_qwen(seed=0)
    return lora_grad_features(
        model,
        tokenizer,
        _prompts(),
        _completions(),
        sketch_dim=8192,
        lora_r=8,
        lora_alpha=16,
        lora_seed=seed,
        sketch_seed=1,
        normalize=normalize,
        device="cpu",
        dtype="float32",
        return_exact=return_exact,
    )


def test_lora_features_sketch_determinism_and_token_mean():
    summed = _features(normalize="sum", return_exact=True)
    assert summed.meta["lora_a_grad_max_abs"] == 0.0
    assert summed.features.shape == (4, 8192)
    assert summed.features.dtype == np.float32
    assert summed.exact is not None and summed.exact.shape == (4, summed.meta["param_count"])
    assert np.all(summed.lengths > 0)
    again = _features(normalize="sum", return_exact=False)
    np.testing.assert_array_equal(summed.features, again.features)

    exact = summed.exact.astype(np.float64)
    sketched = summed.features.astype(np.float64)
    errors = []
    for i in range(exact.shape[0]):
        for j in range(i, exact.shape[0]):
            truth = float(exact[i] @ exact[j])
            estimate = float(sketched[i] @ sketched[j])
            scale = float(np.linalg.norm(exact[i]) * np.linalg.norm(exact[j]))
            errors.append(abs(estimate - truth) / scale)
    assert float(np.median(errors)) < 0.05, errors

    averaged = _features(normalize="token_mean", return_exact=False)
    np.testing.assert_allclose(
        summed.features,
        averaged.features * averaged.lengths[:, None],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_array_equal(summed.lengths, averaged.lengths)


def test_hidden_pool_is_the_predictor_state():
    try:
        model, tokenizer = build_tiny_qwen()
    except Exception as exc:
        pytest.skip(f"tiny Qwen unavailable offline: {exc}")
    prompt, completion = "Hi", "Hello there friend"
    ids, start = encode_prompt_completion(tokenizer, prompt, completion)
    got = hidden_features(
        model, tokenizer, [prompt], [completion], pooling="mean", batch_size=1, device="cpu"
    )
    with torch.inference_mode():
        final = model(
            torch.tensor([ids]), output_hidden_states=True, use_cache=False
        ).hidden_states[-1][0]
    predictor = final[start - 1 : len(ids) - 1].mean(dim=0).float().cpu().numpy()
    at_tokens = final[start:].mean(dim=0).float().cpu().numpy()
    np.testing.assert_allclose(got[0], predictor, rtol=1e-5, atol=1e-5)
    assert not np.allclose(predictor, at_tokens, rtol=1e-3, atol=1e-3)
    empty = hidden_features(
        model, tokenizer, [prompt], [""], pooling="mean", batch_size=1, device="cpu"
    )
    assert np.all(empty[0] == 0.0)
    assert hidden_features.meta["completion_source"] == "text"


def test_sampled_token_ids_are_not_reencoded():
    pytest.importorskip("safetensors")
    try:
        tokenizer = qwen_tokenizer()
        model, _tok = build_tiny_qwen(seed=1)
    except Exception as exc:
        pytest.skip(f"tiny Qwen unavailable offline: {exc}")
    separate = encode_completion(tokenizer, "play") + encode_completion(tokenizer, "ing")
    text = tokenizer.decode(separate, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert encode_completion(tokenizer, text) != separate
    prompt = "Say a word."
    ids, start = prompt_completion_ids(tokenizer, prompt, separate)
    assert ids[start:] == separate
    assert len(ids) - start == 2
    from_ids = lora_grad_features(
        model,
        tokenizer,
        [prompt],
        [text],
        completion_ids=[separate],
        sketch_dim=64,
        lora_r=4,
        lora_alpha=8,
        device="cpu",
    )
    model_text, _tok = build_tiny_qwen(seed=1)
    from_text = lora_grad_features(
        model_text,
        tokenizer,
        [prompt],
        [text],
        sketch_dim=64,
        lora_r=4,
        lora_alpha=8,
        device="cpu",
    )
    assert from_ids.lengths.tolist() == [2]
    assert from_text.lengths.tolist() == [len(encode_completion(tokenizer, text))]
    assert from_ids.meta["completion_source"] == "token_ids"
    assert from_text.meta["completion_source"] == "text"
    assert not np.allclose(from_ids.features, from_text.features)
    hidden_model, _tok = build_tiny_qwen(seed=2)
    pooled = hidden_features(
        hidden_model,
        tokenizer,
        [prompt],
        [text],
        completion_ids=[separate],
        pooling="mean",
        batch_size=1,
        device="cpu",
    )
    manual = _manual_hidden(hidden_model, tokenizer, prompt, text, "mean", ids=ids, start=start)
    np.testing.assert_allclose(pooled[0], manual, rtol=1e-5, atol=1e-5)
    assert hidden_features.meta["completion_source"] == "token_ids"
    eos = int(tokenizer.eos_token_id)
    with_eos = [*separate, eos]
    model_eos, _tok = build_tiny_qwen(seed=1)
    counted = lora_grad_features(
        model_eos,
        tokenizer,
        [prompt],
        [text],
        completion_ids=[with_eos],
        sketch_dim=64,
        lora_r=4,
        lora_alpha=8,
        device="cpu",
    )
    assert counted.lengths.tolist() == [3]
    long_ids = separate * 8
    full, full_start = prompt_completion_ids(tokenizer, prompt, long_ids)
    # Left truncation drops prompt tokens first. A scorable sequence still
    # contains every completion id, including a terminating EOS when present.
    assert full_start > 2
    max_len = len(full) - 2
    kept, kept_start = _truncate_left(full, full_start, max_len)
    assert kept_start >= 1
    assert kept[kept_start:] == long_ids
    model_trunc, _tok = build_tiny_qwen(seed=1)
    truncated = lora_grad_features(
        model_trunc,
        tokenizer,
        [prompt],
        [text],
        completion_ids=[long_ids],
        sketch_dim=32,
        lora_r=4,
        lora_alpha=8,
        max_len=max_len,
        device="cpu",
    )
    assert truncated.lengths.tolist() == [len(kept) - kept_start]


def test_lora_init_round_trip_keys_and_bitwise_check(tmp_path):
    pytest.importorskip("safetensors")
    try:
        tokenizer = qwen_tokenizer()
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable offline: {exc}")
    path = tmp_path / "lora_init.safetensors"
    model, _tok = build_tiny_qwen(seed=0)
    first = lora_grad_features(
        model,
        tokenizer,
        ["Hi"],
        ["there"],
        sketch_dim=32,
        lora_r=4,
        lora_alpha=8,
        lora_seed=0,
        device="cpu",
        save_init=path,
    )
    assert path.is_file()
    digest = first.meta["lora_init_sha256"]
    tensors, meta = load_lora_init(path, expected_sha256=digest)
    assert meta["schema"] == LORA_INIT_SCHEMA
    assert meta["lora_r"] == "4"
    assert meta["lora_seed"] == "0"
    assert int(meta["n_modules"]) == len(tensors)
    assert "model.layers.0.self_attn.q_proj" in tensors
    for key, value in tensors.items():
        assert key.startswith("model.layers.")
        assert "lora_A" not in key
        assert not key.startswith("base_model")
        assert value.dtype == np.float32
        assert value.shape[0] == 4
        assert value.ndim == 2
    again_model, _tok = build_tiny_qwen(seed=3)
    second = lora_grad_features(
        again_model,
        tokenizer,
        ["Hi"],
        ["there"],
        sketch_dim=32,
        lora_r=4,
        lora_alpha=8,
        lora_seed=0,
        device="cpu",
        save_init=path,
    )
    assert second.meta["lora_init_sha256"] == digest
    other_seed, _tok = build_tiny_qwen(seed=4)
    with pytest.raises(ValueError, match="bitwise"):
        lora_grad_features(
            other_seed,
            tokenizer,
            ["Hi"],
            ["there"],
            sketch_dim=32,
            lora_r=4,
            lora_alpha=8,
            lora_seed=1,
            device="cpu",
            save_init=path,
        )
    with pytest.raises(ValueError, match="sha256"):
        load_lora_init(path, expected_sha256="0" * 64)


def _completion_of_length(tokenizer, n_tokens: int) -> str:
    piece = "The sum of the series is 42. "
    ids = tokenizer.encode(piece, add_special_tokens=False)
    repeated = (ids * ((n_tokens // max(len(ids), 1)) + 2))[:n_tokens]
    return tokenizer.decode(repeated, skip_special_tokens=True, clean_up_tokenization_spaces=False)


@pytest.mark.slow
def test_qwen_feature_budget():
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("needs CUDA or MPS")
    import gc
    import json
    import time

    from vfates.features.hidden import load_model

    tokenizer = qwen_tokenizer()
    completion = _completion_of_length(tokenizer, 300)
    prompts = [f"Problem {i}. Solve it carefully." for i in range(32)]
    completions = [completion] * 32
    rows = []
    for model_id in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
        model, _tok = load_model(model_id, dtype="bfloat16", local_files_only=True)
        hidden_features(
            model, tokenizer, prompts[:1], completions[:1], pooling="mean", batch_size=1
        )
        if next(model.parameters()).device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        hidden = hidden_features(
            model,
            tokenizer,
            prompts[:8],
            completions[:8],
            pooling="mean",
            batch_size=4,
        )
        if next(model.parameters()).device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        row = {
            "kind": "hidden",
            "model": model_id,
            "dtype": "bfloat16",
            "samples": 8,
            "hidden": int(hidden.shape[1]),
            "seconds_per_sample": elapsed / 8,
            "peak_memory": hidden_features.peak_memory,
        }
        rows.append(row)
        print("VFATES_TIMING " + json.dumps(row), flush=True)
        del hidden, model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        for dtype_name in ("bfloat16", "float32"):
            model, _tok = load_model(model_id, dtype=dtype_name, local_files_only=True)
            started = time.perf_counter()
            out = lora_grad_features(
                model,
                tokenizer,
                prompts,
                completions,
                sketch_dim=8192,
                dtype=dtype_name,
            )
            row = {
                "kind": "lora_grad",
                "model": model_id,
                "dtype": dtype_name,
                "samples": 32,
                "mean_length": float(out.lengths.mean()),
                "seconds_per_sample": out.meta["seconds_per_sample"],
                "seconds_per_sample_excluding_first": float(
                    np.mean(out.meta["sample_seconds"][1:])
                ),
                "first_sample_seconds": out.meta["sample_seconds"][0],
                "peak_allocated_bytes": out.meta["peak_allocated_bytes"],
                "peak_driver_bytes": out.meta["peak_driver_bytes"],
                "param_count": out.meta["param_count"],
                "lora_a_grad_max_abs": out.meta["lora_a_grad_max_abs"],
                "wall_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print("VFATES_TIMING " + json.dumps(row), flush=True)
            del out, model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    assert len(rows) == 2 + 4
