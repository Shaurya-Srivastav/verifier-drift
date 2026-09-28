"""Pinned model specs: revisions, learning rates, LoRA names, placements."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfates.models import load_model_spec

ROOT = Path(__file__).resolve().parents[1]
_NAMES = (
    "qwen3-0.6b",
    "qwen3-1.7b",
    "qwen3-1.7b-base",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "olmo3-7b",
    "llama3.1-8b",
)
_PEFT = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_STUDY_COMPLETIONS = 64 * 8


def test_every_model_spec_is_pinned_and_places_the_study_batch() -> None:
    high_lr = {"qwen3-0.6b", "qwen3-1.7b", "qwen3-1.7b-base", "qwen3-4b"}
    for name in _NAMES:
        spec = load_model_spec(name)
        assert spec.name == name
        assert len(spec.revision) == 40
        assert all(char in "0123456789abcdef" for char in spec.revision)
        assert spec.dtype == "bfloat16"
        assert spec.attn_implementation == "flash_attention_2"
        assert spec.peft_target_modules() == _PEFT
        assert spec.lr == pytest.approx(2.0e-5 if name in high_lr else 1.0e-5)
        assert spec.params_b > 0
        for hardware_class, placement in spec.placement.items():
            assert placement.gpus % placement.tp == 0
            denom = placement.micro_batch_size * placement.gpus
            assert _STUDY_COMPLETIONS % denom == 0, hardware_class
        with pytest.raises(ValueError, match="no placement"):
            spec.placement_for("not-a-class")


def test_qwen3_disables_thinking_and_llama_is_gated() -> None:
    qwen = load_model_spec("qwen3-4b")
    assert dict(qwen.chat_template_kwargs or {}) == {"enable_thinking": False}
    assert qwen.placement_for("rtx-5000-ada-32gb").gpus == 1
    assert qwen.placement_for("a100-40gb").gpus == 1
    small = load_model_spec("qwen3-1.7b")
    assert small.placement_for("rtx-a6000-48gb").gpus == 1
    eight = load_model_spec("qwen3-8b")
    assert eight.placement_for("h100-80gb").gpus == 1
    assert eight.placement_for("a100-80gb").gpus == 1
    assert eight.placement_for("rtx-a6000-48gb").gpus == 1
    assert eight.placement_for("a100-40gb").gpus == 2
    assert eight.placement_for("rtx-5000-ada-32gb").gpus == 2
    fourteen = load_model_spec("qwen3-14b")
    assert fourteen.placement_for("h100-80gb").gpus == 2
    assert fourteen.placement_for("a100-80gb").gpus == 2
    assert fourteen.placement_for("rtx-a6000-48gb").gpus == 2
    assert fourteen.placement_for("a100-40gb").gpus == 4
    with pytest.raises(ValueError, match="rtx-5000-ada-32gb"):
        fourteen.placement_for("rtx-5000-ada-32gb")
    huge = load_model_spec("qwen3-32b")
    assert huge.placement_for("h100-80gb").gpus == 4
    assert huge.placement_for("a100-80gb").gpus == 4
    with pytest.raises(ValueError, match="a100-40gb"):
        huge.placement_for("a100-40gb")
    olmo = load_model_spec("olmo3-7b")
    assert olmo.model_id == "allenai/Olmo-3-7B-Instruct-SFT"
    assert olmo.revision == "e1452fc572d51966ff4aaeb25118b891eb93e549"
    assert olmo.chat_template_kwargs is None
    assert olmo.gated is False
    assert olmo.placement_for("a100-40gb").gpus == 2
    llama = load_model_spec("llama3.1-8b")
    assert llama.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert llama.gated is True
    assert llama.chat_template_kwargs is None
    missing = ROOT / "configs" / "models" / "does-not-exist.yaml"
    assert not missing.exists()
    with pytest.raises(FileNotFoundError):
        load_model_spec("does-not-exist")


def test_qwen3_base_keeps_the_chat_template_and_full_ft_placement() -> None:
    """X6 uses the base chat template with null kwargs.

    Qwen3-1.7B-Base ships a chat template. ``enable_thinking`` is consulted
    only when the caller passes it and it is false, which prefills an empty
    think block. Leaving the kwargs null does not pass that switch.
    """
    from vfates.gen.chat import encode_prompt
    from vfates.train.trl_grpo import trl_prompt_ids

    base = load_model_spec("qwen3-1.7b-base")
    instruct = load_model_spec("qwen3-1.7b")
    assert base.model_id == "Qwen/Qwen3-1.7B-Base"
    assert base.revision == "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    assert base.family == "qwen3"
    assert base.params_b == pytest.approx(1.720574976)
    assert base.chat_template_kwargs is None
    assert base.lr == pytest.approx(2.0e-5)
    for hardware_class, placement in instruct.placement.items():
        other = base.placement_for(hardware_class)
        assert (other.gpus, other.tp, other.micro_batch_size) == (
            placement.gpus,
            placement.tp,
            placement.micro_batch_size,
        )
        assert other.vllm_gpu_memory_utilization == pytest.approx(
            placement.vllm_gpu_memory_utilization
        )
    full = instruct.placement_full_for("rtx-a6000-48gb")
    assert (full.gpus, full.tp, full.micro_batch_size) == (2, 2, 8)
    assert full.vllm_gpu_memory_utilization == pytest.approx(0.40)
    assert base.placement_full_for("rtx-a6000-48gb").gpus == 2
    with pytest.raises(ValueError, match="full-finetune"):
        instruct.placement_full_for("h100-80gb")
    with pytest.raises(ValueError, match="full-finetune"):
        load_model_spec("qwen3-4b").placement_full_for("h100-80gb")

    class _Template:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, object]] = []

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            del messages, add_generation_prompt, tokenize
            self.kwargs.append(dict(kwargs))
            text = "<|im_start|>assistant\n"
            if "enable_thinking" in kwargs and kwargs["enable_thinking"] is False:
                text += "<think>\n\n</think>\n\n"
            return [ord(char) for char in text]

    tokenizer = _Template()
    user = "What is 2 + 2?"
    plain = encode_prompt(tokenizer, user, chat_template_kwargs=base.chat_template_kwargs)
    same = trl_prompt_ids(
        tokenizer,
        [{"role": "user", "content": user}],
        chat_template_kwargs=base.chat_template_kwargs,
    )
    assert plain == same
    assert tokenizer.kwargs == [{}, {}]
    thinking_off = encode_prompt(tokenizer, user, chat_template_kwargs={"enable_thinking": False})
    assert thinking_off != plain
    assert tokenizer.kwargs[-1] == {"enable_thinking": False}
