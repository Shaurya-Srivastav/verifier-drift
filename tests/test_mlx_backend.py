"""MLX batch generation: shapes on a tiny Qwen, throughput on the real 0.5B."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("mlx")

from vfates.gen.mlx_backend import MLXGenerator


def _tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("the Qwen2.5-0.5B-Instruct tokenizer is not in the local Hugging Face cache")


def tiny_qwen(tokenizer, seed: int = 0):
    import mlx.core as mx
    from mlx_lm.models.qwen2 import Model, ModelArgs

    mx.random.seed(seed)
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=len(tokenizer),
        max_position_embeddings=256,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model, args


@pytest.fixture(scope="module")
def tokenizer():
    return _tokenizer()


@pytest.mark.mlx
def test_tiny_generate_shape_order_and_seed(tokenizer) -> None:
    import mlx.core as mx

    model, _args = tiny_qwen(tokenizer, seed=0)
    gen = MLXGenerator.from_model(
        model,
        tokenizer,
        completion_batch_size=4,
        prefill_batch_size=2,
        max_sequences_per_call=2,
        logprobs=True,
    )
    prompts = ["Say a word.", "Say another word."]
    first = gen.generate(prompts, 2, max_tokens=4, temperature=1.0, top_p=1.0, seed=0)
    second = gen.generate(prompts, 2, max_tokens=4, temperature=1.0, top_p=1.0, seed=0)
    assert len(first) == 2
    assert [len(group) for group in first] == [2, 2]
    for group in first:
        for out in group:
            assert out.n_tokens == len(out.token_ids or ())
            assert out.n_tokens <= 4
            assert out.finish_reason in {"stop", "length"}
            assert out.logprob is not None
            assert out.text == tokenizer.decode(list(out.token_ids or ()), skip_special_tokens=True)
            eos = tokenizer.eos_token_id
            eos_ids = {int(item) for item in eos} if isinstance(eos, (list, tuple)) else {int(eos)}
            if out.finish_reason == "length":
                assert out.n_tokens == 4
                assert int(out.token_ids[-1]) not in eos_ids
            if out.finish_reason == "stop":
                assert out.token_ids is not None
                assert int(out.token_ids[-1]) in eos_ids
                assert out.n_tokens <= 4
    assert [out.token_ids for group in first for out in group] == [
        out.token_ids for group in second for out in group
    ]
    greedy = gen.generate(prompts, 1, max_tokens=3, temperature=0.0, seed=1)
    assert len(greedy) == 2 and len(greedy[0]) == 1
    fast = MLXGenerator.from_model(model, tokenizer, completion_batch_size=4, prefill_batch_size=2)
    plain = fast.generate(prompts, 2, max_tokens=4, temperature=1.0, top_p=1.0, seed=0)
    assert all(out.logprob is None for group in plain for out in group)
    ids = gen.generate_ids([[1, 2, 3], [4, 5]], max_tokens=2, temperature=0.0, top_p=1.0, seed=3)
    assert len(ids) == 2
    assert all(item.n_tokens <= 2 for item in ids)
    mx.eval(model.parameters())


@pytest.mark.slow
@pytest.mark.mlx
def test_qwen05_throughput() -> None:
    import mlx.core as mx

    gen = MLXGenerator(
        "Qwen/Qwen2.5-0.5B-Instruct",
        completion_batch_size=128,
        prefill_batch_size=16,
    )
    # One short call pays for the Metal compile so the timed call is the steady rate.
    gen.generate(["warmup"], 1, max_tokens=8, temperature=0.0, seed=0)
    prompts = [f"Count from {i} and keep going." for i in range(16)]
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    groups = gen.generate(prompts, 1, max_tokens=64, temperature=1.0, top_p=1.0, seed=0)
    elapsed = time.perf_counter() - started
    tokens = sum(out.n_tokens for group in groups for out in group)
    rate = tokens / elapsed if elapsed else 0.0
    peak = mx.get_peak_memory() / 1e9
    line = f"0.5B generate: {tokens} tok in {elapsed:.2f}s = {rate:.0f} tok/s, peak {peak:.2f} GB"
    print(line)
    Path("/tmp/vfates_mlx_gen.txt").write_text(line + "\n")
    del gen
    mx.clear_cache()
    assert len(groups) == 16
    assert tokens > 0


@pytest.mark.mlx
def test_generate_ids_keeps_an_immediate_stop(tokenizer, monkeypatch) -> None:
    from types import SimpleNamespace

    class _FakeBatch:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self._uids: list[int] = []
            self._done = False

        def insert(self, prompts, max_tokens):
            del max_tokens
            self._uids = list(range(len(prompts)))
            self._done = False
            return list(self._uids)

        def next(self):
            if self._done:
                return []
            self._done = True
            return [
                SimpleNamespace(uid=uid, token=99, finish_reason="stop", logprobs=None)
                for uid in self._uids
            ]

    monkeypatch.setattr("vfates.gen.mlx_backend.BatchGenerator", _FakeBatch)
    model, _args = tiny_qwen(tokenizer, seed=0)
    gen = MLXGenerator.from_model(model, tokenizer, logprobs=False)
    outs = gen.generate_ids([[1, 2], [3, 4, 5]], max_tokens=4, temperature=1.0, top_p=1.0, seed=0)
    assert [out.token_ids for out in outs] == [(99,), (99,)]
    assert [out.n_tokens for out in outs] == [1, 1]
    assert [out.finish_reason for out in outs] == ["stop", "stop"]
