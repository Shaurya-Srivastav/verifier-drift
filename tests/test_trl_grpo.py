"""TRL builder: argument mapping, and one CPU optimizer step on a tiny Qwen2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig
from vfates.train.trl_grpo import (
    build_trl_trainer,
    metric_row_from_trl_logs,
    rates_from_reward_means,
    trl_grpo_kwargs,
)
from vfates.types import Parsed, Problem
from vfates.verifiers.base import Verifier


def _cfg(tmp: Path, **kwargs: object) -> GRPOConfig:
    raw: dict[str, object] = {
        "model_id": "tiny",
        "steps": 1,
        "prompts_per_step": 2,
        "group_size": 2,
        "max_tokens": 8,
        "max_prompt_length": 64,
        "out_dir": str(tmp),
        "seed": 0,
        "eval_every": 1,
        "eval_heldout_n": 0,
        "eval_train_n": 0,
        "eval_samples_per_prompt": 2,
        "save_every": 1,
        "micro_batch_size": 2,
        "adv_mode": "std",
        "loss_agg": "seq_mean",
        "beta_kl": 0.0,
        "use_vllm": False,
    }
    raw.update(kwargs)
    return GRPOConfig.from_dict(raw)


def test_trl_kwargs_map_the_objective(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, warmup_steps=4, prompts_per_step=4, group_size=4, micro_batch_size=4)
    kwargs = trl_grpo_kwargs(cfg, n_reward_funcs=5, bf16=False)
    assert kwargs["loss_type"] == "grpo"
    assert kwargs["scale_rewards"] == "group"
    assert kwargs["num_generations"] == 4
    assert kwargs["per_device_train_batch_size"] == 4
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["per_device_train_batch_size"] * kwargs["gradient_accumulation_steps"] == 16
    assert kwargs["max_completion_length"] == cfg.max_tokens
    assert kwargs["beta"] == 0.0
    assert kwargs["epsilon"] == cfg.eps_low
    assert kwargs["epsilon_high"] == cfg.eps_high
    assert kwargs["num_iterations"] == 1
    assert kwargs["learning_rate"] == cfg.lr
    assert kwargs["warmup_steps"] == 4
    assert kwargs["lr_scheduler_type"] == "constant_with_warmup"
    assert kwargs["max_grad_norm"] == cfg.grad_clip
    assert kwargs["reward_weights"] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert kwargs["vllm_importance_sampling_correction"] is False
    assert kwargs["bf16"] is False
    assert kwargs["remove_unused_columns"] is False
    dapo = trl_grpo_kwargs(_cfg(tmp_path, loss_agg="token_mean", adv_mode="mean"), n_reward_funcs=1)
    assert dapo["loss_type"] == "dapo"
    assert dapo["scale_rewards"] == "none"
    dr = trl_grpo_kwargs(_cfg(tmp_path, loss_agg="dr_constant"), n_reward_funcs=1)
    assert dr["loss_type"] == "dr_grpo"
    from trl import GRPOConfig as TrlGRPOConfig

    args = TrlGRPOConfig(**kwargs)
    assert args.loss_type == "grpo"
    assert args.scale_rewards == "group"
    assert args.generation_batch_size == cfg.prompts_per_step * cfg.group_size


def test_rates_from_trl_means() -> None:
    rates = rates_from_reward_means(
        {"verifier": 0.5, "oracle": 0.5, "false_positive": 0.25, "false_negative": 0.25}
    )
    assert rates["fpr"] == pytest.approx(0.5)
    assert rates["fnr"] == pytest.approx(0.5)
    assert rates["precision"] == pytest.approx(0.5)
    assert rates["J"] == pytest.approx(0.0)
    undefined = rates_from_reward_means(
        {"verifier": 1.0, "oracle": 1.0, "false_positive": 0.0, "false_negative": 0.0}
    )
    assert undefined["fpr"] is None
    assert undefined["J"] is None
    row = metric_row_from_trl_logs(
        {
            "rewards/verifier/mean": 0.5,
            "rewards/oracle/mean": 0.5,
            "rewards/false_positive/mean": 0.25,
            "rewards/false_negative/mean": 0.25,
            "rewards/monitor_hit/mean": 0.25,
            "frac_reward_zero_std": 0.5,
            "completions/mean_length": 3.0,
            "completions/max_length": 4.0,
            "completions/clipped_ratio": 0.1,
            "loss": 0.2,
            "grad_norm": 0.3,
            "learning_rate": 1e-5,
            "kl": 0.0,
            "num_tokens": 10,
        },
        step=2,
        monitor_names=["hit"],
    )
    assert row["step"] == 2
    assert row["reward_V_mean"] == pytest.approx(0.5)
    assert row["frac_informative_groups"] == pytest.approx(0.5)
    assert row["monitors"]["hit"] == pytest.approx(0.25)
    assert row["monitors_wrong"] is None
    assert row["gen_seconds"] is None


class _YesTask(Task):
    name = "yes"

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        del seed
        return [
            Problem(pid=f"{split}-{i}", split=split, task="yes", question=str(i), answer="yes")
            for i in range(n)
        ]

    def prompt(self, problem: Problem) -> str:
        del problem
        return "Reply with a word."

    def parse(self, problem: Problem, text: str) -> Parsed:
        del problem
        return Parsed(answer="yes" if "yes" in text.lower() else None, raw=text)

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        del problem, parsed
        return int("yes" in text.lower())


class _Even(Verifier):
    name = "even"
    stochastic = True

    def __init__(self) -> None:
        self.calls = 0

    def verdict(self, problem: Problem, parsed: Parsed, text: str, o: int, key=None) -> int:
        del problem, parsed, text, o
        self.calls += 1
        if key is None:
            return 0
        return int(int(key[-1]) % 2 == 0)


def _tiny_hf():
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model, tokenizer


@pytest.mark.slow
@pytest.mark.torch
def test_tiny_cpu_optimizer_step(tmp_path: Path) -> None:
    model, tokenizer = _tiny_hf()
    task = _YesTask()
    verifier = _Even()
    trainer = build_trl_trainer(
        _cfg(tmp_path, lr=1e-4, gradient_checkpointing=False),
        task,
        verifier,
        task.problems("train", 2, 0),
        task.problems("heldout", 1, 1),
        monitors={"yes_word": lambda problem, parsed, text: "yes" in text.lower()},
        use_vllm=False,
        model=model,
        processing_class=tokenizer,
        extra_trl_args={"use_cpu": True, "bf16": False, "fp16": False},
    )
    assert trainer.args.loss_type == "grpo"
    assert trainer.args.scale_rewards == "group"
    assert trainer.args.num_generations == 2
    assert trainer.args.use_vllm is False
    trainer.train()
    assert verifier.calls > 0
    metrics = tmp_path / "metrics.jsonl"
    assert metrics.exists()
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    assert rows
    assert "reward_V_mean" in rows[-1]
    assert rows[-1]["step"] >= 1
