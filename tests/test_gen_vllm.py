"""vLLM request construction without a CUDA engine."""

from __future__ import annotations

import importlib
import sys

import pytest

from vfates.gen.vllm_backend import (
    enforce_stop_token_ids,
    request_outputs_to_genouts,
    sampling_params_kwargs,
    token_prompts,
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt and tokenize
        text = "|".join(f"{message['role']}={message['content']}" for message in messages)
        return [ord(char) for char in text]


class _Completion:
    def __init__(self, text, token_ids, cumulative_logprob, finish_reason):
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.finish_reason = finish_reason


class _Request:
    def __init__(self, outputs):
        self.outputs = outputs


def test_import_does_not_require_vllm():
    sys.modules.pop("vllm", None)
    import vfates.gen.vllm_backend as backend

    importlib.reload(backend)
    assert "vllm" not in sys.modules


def test_sampling_kwargs_and_response_conversion():
    kwargs = sampling_params_kwargs(n=4, temperature=1.0, top_p=1.0, max_tokens=256, seed=3)
    assert kwargs == {
        "n": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "seed": 3,
        "logprobs": 1,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }
    prompts = token_prompts(_FakeTokenizer(), ["hi"], system_prompt="sys")
    assert prompts[0]["prompt_token_ids"]
    assert any(token == ord("=") for token in prompts[0]["prompt_token_ids"])

    requests = [
        _Request(
            [
                _Completion("hello", [10, 11], -0.5, "stop"),
                _Completion("partial", [12], None, "length"),
            ]
        )
    ]
    groups = request_outputs_to_genouts(requests, n=2)
    assert len(groups) == 1 and len(groups[0]) == 2
    assert groups[0][0].text == "hello"
    assert groups[0][0].token_ids == (10, 11)
    assert groups[0][0].n_tokens == 2
    assert groups[0][0].logprob == pytest.approx(-0.5)
    assert groups[0][0].finish_reason == "stop"
    assert groups[0][1].logprob is None
    assert groups[0][1].finish_reason == "length"
    with pytest.raises(RuntimeError, match="expected 1"):
        request_outputs_to_genouts(requests, n=1)


def test_vllm_constructor_import_guard():
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vllm is installed")
    from vfates.gen.vllm_backend import VLLMGenerator

    with pytest.raises(ImportError, match="Linux"):
        VLLMGenerator("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vllm not installed")
def test_sampling_params_construct():
    from vllm import SamplingParams

    params = SamplingParams(
        **sampling_params_kwargs(n=2, temperature=0.0, top_p=1.0, max_tokens=8, seed=1)
    )
    assert params.n == 2
    assert params.max_tokens == 8
    assert params.seed == 1


class _Decode:
    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(str(token) for token in tokens if token != 99)


def test_enforce_stop_token_ids_covers_stripped_and_present_stops():
    assert enforce_stop_token_ids([10, 99, 11], finish_reason="stop", stop_ids={99}) == (10, 99)
    assert enforce_stop_token_ids(
        [10, 11], finish_reason="stop", stop_reason=99, stop_ids={99, 100}
    ) == (10, 11, 99)
    assert enforce_stop_token_ids([10, 11], finish_reason="stop", stop_ids={99}) == (10, 11, 99)
    assert enforce_stop_token_ids([10, 11], finish_reason="stop", stop_ids={99, 100}) == (10, 11)
    assert enforce_stop_token_ids([4, 0, 5], finish_reason="length", pad_id=0) == (4, 5)
    assert enforce_stop_token_ids([4, 5], finish_reason="length", stop_ids={99}) == (4, 5)
    assert enforce_stop_token_ids(
        [10, 11], finish_reason="stop", stop_reason=True, stop_ids={99}
    ) == (
        10,
        11,
        99,
    )


def test_request_outputs_redecode_when_ids_change():
    requests = [_Request([_Completion("hello", [10, 11], -0.5, "stop")])]
    groups = request_outputs_to_genouts(requests, n=1, tokenizer=_Decode(), stop_ids={99})
    assert groups[0][0].token_ids == (10, 11, 99)
    assert groups[0][0].n_tokens == 3
    assert groups[0][0].text == "10 11"
    assert groups[0][0].logprob is None


def test_token_prompts_keep_the_suffix():
    prompts = token_prompts(_FakeTokenizer(), ["hello"], max_prompt_tokens=4)
    ids = prompts[0]["prompt_token_ids"]
    assert len(ids) == 4
    full = token_prompts(_FakeTokenizer(), ["hello"])[0]["prompt_token_ids"]
    assert ids == full[-4:]


class _KwTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        assert add_generation_prompt and tokenize
        assert kwargs == {"enable_thinking": False}
        return [8, 9]


def test_token_prompts_apply_chat_template_kwargs_before_sampling():
    prompts = token_prompts(_KwTokenizer(), ["hi"], chat_template_kwargs={"enable_thinking": False})
    assert prompts == [{"prompt_token_ids": [8, 9]}]
