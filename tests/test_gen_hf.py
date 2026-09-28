"""Hugging Face generation and chat-template encoding."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from vfates.gen.chat import encode_completion, encode_prompt, encode_prompt_completion
from vfates.gen.hf_backend import HFGenerator

_TOKENIZER = None


def qwen_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
        )
    return _TOKENIZER


def build_tiny_qwen(seed: int = 0):
    """Random Qwen2 with the cached 0.5B tokenizer. Weights are seeded by ``seed``."""
    tokenizer = qwen_tokenizer()
    torch.manual_seed(seed)
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    model = Qwen2ForCausalLM(config)
    model.set_attn_implementation("eager")
    model.eval()
    return model, tokenizer


def _generator(model, tokenizer, batch_size: int = 8) -> HFGenerator:
    return HFGenerator(
        "tiny-qwen",
        device="cpu",
        dtype="float32",
        batch_size=batch_size,
        model=model,
        tokenizer=tokenizer,
    )


def _greedy_ids(
    model, tokenizer, prompt: str, max_tokens: int, stop_ids: list[int]
) -> tuple[int, ...]:
    stop = set(stop_ids)
    ids = encode_prompt(tokenizer, prompt)
    prompt_len = len(ids)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(torch.tensor([ids])).logits[0, -1]
            nxt = int(torch.argmax(logits).item())
            ids.append(nxt)
            if nxt in stop:
                break
    return tuple(ids[prompt_len:])


def _teacher_logprob(model, prompt_ids: list[int], token_ids: tuple[int, ...]) -> float:
    if not token_ids:
        return 0.0
    ids = prompt_ids + list(token_ids)
    start = len(prompt_ids)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()
        total = 0.0
        for offset, token in enumerate(token_ids):
            log_probs = torch.log_softmax(logits[start + offset - 1], dim=-1)
            total += float(log_probs[token].item())
    return total


def test_chat_round_trip_and_bpe_boundary():
    tokenizer = qwen_tokenizer()
    user = "What is 2+2?"
    completion = "The answer is 4."
    system = "Be brief."
    prompt_ids = encode_prompt(tokenizer, user, system)
    ids, start = encode_prompt_completion(tokenizer, user, completion, system)
    assert ids[:start] == prompt_ids
    assert ids[start:] == encode_completion(tokenizer, completion)
    assert tokenizer.decode(ids[start:], skip_special_tokens=True) == completion
    assert encode_prompt(tokenizer, user) != prompt_ids

    # Joint BPE can merge across a string boundary; appending ids does not.
    left, right = "play", "ing"
    separate = encode_completion(tokenizer, left) + encode_completion(tokenizer, right)
    joint = encode_completion(tokenizer, left + right)
    assert separate != joint


def test_greedy_matches_argmax_across_batch_shapes():
    model, tokenizer = build_tiny_qwen()
    prompts = ["Hi", "Please answer this longer question about prime numbers."]
    gen = _generator(model, tokenizer, batch_size=4)
    outs = gen.generate(prompts, n=2, max_tokens=4, temperature=0.0, seed=0)
    assert [len(group) for group in outs] == [2, 2]
    for prompt, group in zip(prompts, outs):
        expected = _greedy_ids(model, tokenizer, prompt, 4, gen.stop_ids)
        assert group[0].token_ids == group[1].token_ids == expected
        assert group[0].n_tokens == len(expected)
        assert group[0].finish_reason in {"stop", "length"}
    again = gen.generate(prompts, n=2, max_tokens=4, temperature=0.0, seed=99)
    assert again == outs


def test_sampling_shapes_determinism_and_logprob():
    model, tokenizer = build_tiny_qwen(seed=1)
    prompts = ["Say hello.", "Count to three, briefly."]
    gen = _generator(model, tokenizer, batch_size=3)
    first = gen.generate(prompts, n=2, max_tokens=5, temperature=1.0, top_p=1.0, seed=7)
    second = gen.generate(prompts, n=2, max_tokens=5, temperature=1.0, top_p=1.0, seed=7)
    assert first == second
    other = gen.generate(prompts, n=2, max_tokens=5, temperature=1.0, top_p=1.0, seed=8)
    assert other != first
    assert [len(group) for group in first] == [2, 2]
    for prompt, group in zip(prompts, first):
        prompt_ids = encode_prompt(tokenizer, prompt)
        for completion in group:
            assert completion.n_tokens == len(completion.token_ids) <= 5
            assert completion.finish_reason in {"stop", "length"}
            recomputed = _teacher_logprob(model, prompt_ids, completion.token_ids)
            assert completion.logprob == pytest.approx(recomputed, abs=1e-4, rel=1e-4)
    with_system = gen.generate(
        ["Hi"], n=1, max_tokens=2, temperature=0.0, system_prompt="Be brief."
    )
    assert len(with_system) == 1 and len(with_system[0]) == 1


@pytest.mark.slow
def test_qwen_generate_throughput():
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("needs CUDA or MPS")
    import json
    import time

    prompts = [f"Question {i}: what is {i} plus {i}? Think step by step." for i in range(16)]
    rows = []
    for model_id in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
        gen = HFGenerator(
            model_id,
            dtype="bfloat16",
            batch_size=16,
            local_files_only=True,
        )
        gen.generate(prompts[:1], n=1, max_tokens=8, temperature=1.0, seed=0)
        if gen.device.type == "mps":
            torch.mps.synchronize()
        elif gen.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        outs = gen.generate(prompts, n=4, max_tokens=256, temperature=1.0, top_p=1.0, seed=0)
        if gen.device.type == "mps":
            torch.mps.synchronize()
        elif gen.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        n_tokens = sum(item.n_tokens for group in outs for item in group)
        n_len = sum(item.finish_reason == "length" for group in outs for item in group)
        row = {
            "model": model_id,
            "dtype": "bfloat16",
            "device": str(gen.device),
            "batch_size": gen.batch_size,
            "prompts": 16,
            "n": 4,
            "max_tokens": 256,
            "tokens": n_tokens,
            "hit_length": n_len,
            "seconds": elapsed,
            "tokens_per_second": n_tokens / elapsed,
        }
        rows.append(row)
        print("VFATES_TIMING " + json.dumps(row), flush=True)
        del gen
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    assert rows


class _Decode:
    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return " ".join(str(token) for token in tokens if token != 10)


def test_decode_rows_keeps_the_stop_token_and_drops_padding():
    from vfates.gen.hf_backend import _decode_rows

    eos = 10
    pad = 0
    rows = torch.tensor([[7, eos, 9, pad], [eos, 1, 2, 3], [4, 5, 6, pad]])
    step_logprobs = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        dtype=torch.float32,
    )
    outs = _decode_rows(rows, step_logprobs, [eos], _Decode(), pad)
    assert outs[0].token_ids == (7, eos)
    assert outs[0].n_tokens == 2
    assert outs[0].finish_reason == "stop"
    assert outs[0].text == "7"
    assert outs[0].logprob == pytest.approx(0.1 + 0.4)
    assert outs[1].token_ids == (eos,)
    assert outs[1].n_tokens == 1
    assert outs[1].text == ""
    assert outs[1].finish_reason == "stop"
    assert outs[2].token_ids == (4, 5, 6)
    assert outs[2].finish_reason == "length"
    assert outs[2].n_tokens == 3


def test_generate_left_truncates_prompts():
    model, tokenizer = build_tiny_qwen()
    seen: list[int] = []
    original = model.generate

    def capture(*args, **kwargs):
        seen.append(int(kwargs["input_ids"].shape[1]))
        return original(*args, **kwargs)

    model.generate = capture
    gen = _generator(model, tokenizer, batch_size=2)
    gen.max_prompt_tokens = 8
    prompt = "word " * 80
    full = len(encode_prompt(tokenizer, prompt))
    assert full > 8
    gen.generate([prompt], n=1, max_tokens=1, temperature=0.0, seed=0)
    assert seen == [8]
