"""Global GRPO batch math and the FSDP launch configs. No CUDA required."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfates.train.config import GRPOConfig
from vfates.train.trl_grpo import global_batch_layout, trl_grpo_kwargs, trl_run_metadata
from vfates.utils.io import read_yaml

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp: Path, **kwargs: object) -> GRPOConfig:
    raw: dict[str, object] = {
        "model_id": "tiny",
        "steps": 1,
        "prompts_per_step": 64,
        "group_size": 8,
        "out_dir": str(tmp),
        "seed": 0,
        "eval_every": 1,
        "eval_heldout_n": 0,
        "eval_train_n": 0,
        "eval_samples_per_prompt": 4,
        "save_every": 1,
        "micro_batch_size": 8,
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_tensor_parallel_size": 2,
        "vllm_enable_sleep_mode": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "hardware_class": "h100-80gb",
        "model_revision": "abc",
        "attn_implementation": "flash_attention_2",
    }
    raw.update(kwargs)
    return GRPOConfig.from_dict(raw)


def test_one_optimizer_step_keeps_the_global_completion_count(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    total = cfg.prompts_per_step * cfg.group_size
    for world in (1, 2, 4):
        micro = 4
        placed = _cfg(tmp_path, micro_batch_size=micro, vllm_tensor_parallel_size=1)
        per_device, accum = global_batch_layout(placed, world)
        assert per_device == micro
        assert per_device * accum * world == total
        kwargs = trl_grpo_kwargs(placed, n_reward_funcs=1, world_size=world, bf16=False)
        assert kwargs["per_device_train_batch_size"] == micro
        assert kwargs["gradient_accumulation_steps"] == accum
        assert kwargs["vllm_tensor_parallel_size"] == 1
        assert kwargs["vllm_enable_sleep_mode"] is True
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    # 32 divides 512, but 32 * 3 does not, so the world size is the failure.
    indivisible = _cfg(tmp_path, micro_batch_size=32, vllm_tensor_parallel_size=1)
    with pytest.raises(ValueError, match="divisible"):
        global_batch_layout(indivisible, 3)
    with pytest.raises(ValueError, match="vllm_tensor_parallel_size"):
        global_batch_layout(_cfg(tmp_path, vllm_tensor_parallel_size=3), 2)


def test_run_metadata_records_parallelism(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, vllm_tensor_parallel_size=2)
    meta = trl_run_metadata(cfg, use_vllm=True, world_size=4, attn_implementation="sdpa")
    assert meta["vllm_mode"] == "colocate"
    assert meta["tp_size"] == 2
    assert meta["dp_size"] == 2
    assert meta["world_size"] == 4
    assert meta["hardware_class"] == "h100-80gb"
    assert meta["chat_template_kwargs"] == {"enable_thinking": False}
    assert meta["attn_implementation"] == "sdpa"
    assert meta["model_revision"] == "abc"
    assert meta["rollout_seeded"] is False
    assert meta["parameterisation"] == "lora"


def test_accelerate_configs_are_fsdp_without_deepspeed() -> None:
    single = read_yaml(ROOT / "configs/accelerate/single_gpu.yaml")
    assert single["distributed_type"] == "NO"
    assert single["num_processes"] == 1
    assert "deepspeed_config" not in single
    for name, processes in (("fsdp_2gpu.yaml", 2), ("fsdp_4gpu.yaml", 4)):
        loaded = read_yaml(ROOT / "configs/accelerate" / name)
        assert loaded["distributed_type"] == "FSDP"
        assert loaded["num_processes"] == processes
        assert loaded["fsdp_config"]["fsdp_use_orig_params"] is True
        assert "deepspeed" not in str(loaded["distributed_type"]).lower()
        assert "deepspeed_config" not in loaded
    full = read_yaml(ROOT / "configs/accelerate/fsdp_2gpu_full.yaml")
    assert full["distributed_type"] == "FSDP"
    assert full["num_processes"] == 2
    assert full["fsdp_config"]["fsdp_state_dict_type"] == "FULL_STATE_DICT"
    assert full["fsdp_config"]["fsdp_use_orig_params"] is True
    assert "deepspeed_config" not in full
