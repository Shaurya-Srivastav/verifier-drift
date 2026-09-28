"""Hand checks for the numpy GRPO reference and the shared config."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vfates.gen.base import Generator
from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig
from vfates.train.evaluate import evaluate_policy
from vfates.train.grpo_core import group_advantages, kl_k3, loss_weights, pg_loss_reference
from vfates.types import GenOut, Parsed, Problem
from vfates.verifiers.base import Verifier

ROOT = Path(__file__).resolve().parents[1]


def test_group_advantages_match_trl_unbiased_std() -> None:
    rewards = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    advantages, informative = group_advantages(rewards)
    mean = 0.5
    std = math.sqrt(float(np.sum((rewards[0] - mean) ** 2) / 3.0))
    expected = (rewards[0] - mean) / (std + 1e-4)
    np.testing.assert_allclose(advantages[0], expected)
    np.testing.assert_allclose(advantages[1], 0.0)
    assert informative.tolist() == [True, False]
    # torch.std(unbiased=True) on the same rows.
    torch = pytest.importorskip("torch")

    torch_std = torch.tensor(rewards).std(dim=1, unbiased=True)
    np.testing.assert_allclose(std, float(torch_std[0]), atol=1e-12)
    assert float(torch_std[1]) == pytest.approx(0.0)


def test_population_std_and_mean_mode() -> None:
    rewards = np.array([[1.0, 0.0, 1.0, 0.0]])
    advantages, informative = group_advantages(rewards, std="population")
    # population std of two 0s and two 1s is 0.5
    expected = np.array([0.5, -0.5, 0.5, -0.5]) / (0.5 + 1e-4)
    np.testing.assert_allclose(advantages[0], expected)
    assert informative.tolist() == [True]
    centered, info = group_advantages(
        np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]), mode="mean"
    )
    np.testing.assert_allclose(centered[0], [0.5, -0.5, 0.5, -0.5])
    np.testing.assert_allclose(centered[1], 0.0)
    assert info.tolist() == [True, False]


def test_loss_weights_and_hand_pg_loss() -> None:
    weights = loss_weights([2, 1], "seq_mean", max_len=4)
    np.testing.assert_allclose(weights, [1.0 / 4.0, 1.0 / 2.0])
    np.testing.assert_allclose(loss_weights([0, 2], "seq_mean", 4), [1.0 / 2.0, 1.0 / 4.0])
    np.testing.assert_allclose(loss_weights([2, 1], "token_mean", 4), [1.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(loss_weights([2, 1], "dr_constant", 4), [1.0 / 8.0, 1.0 / 8.0])

    logps = [np.array([0.2, -0.1]), np.array([0.3])]
    advantages = np.array([1.0, -1.0])
    # old is omitted => per-token loss is -A. seq_mean: 0.25*(-2) + 0.5*(1) = 0.
    assert pg_loss_reference(logps, advantages, "seq_mean", 4) == pytest.approx(0.0)
    assert pg_loss_reference(logps, advantages, "token_mean", 4) == pytest.approx(-1.0 / 3.0)
    assert pg_loss_reference(logps, advantages, "dr_constant", 4) == pytest.approx(-1.0 / 8.0)
    assert pg_loss_reference(
        logps, advantages, "seq_mean", 4, old_token_logps=logps
    ) == pytest.approx(0.0)


def test_clip_and_k3() -> None:
    new = [np.array([0.0])]
    ratio_two = [np.array([-math.log(2.0)])]
    ratio_half = [np.array([math.log(2.0)])]
    assert pg_loss_reference(
        new, np.array([1.0]), "seq_mean", 4, old_token_logps=ratio_two
    ) == pytest.approx(-1.2)
    assert pg_loss_reference(
        new, np.array([-1.0]), "seq_mean", 4, old_token_logps=ratio_two
    ) == pytest.approx(2.0)
    assert pg_loss_reference(
        new, np.array([-1.0]), "seq_mean", 4, old_token_logps=ratio_half
    ) == pytest.approx(0.8)
    np.testing.assert_allclose(kl_k3(np.array([0.0]), np.array([0.0])), [0.0])
    np.testing.assert_allclose(kl_k3(np.array([0.0]), np.array([1.0])), [math.e - 2.0])
    loss = pg_loss_reference(
        [np.array([0.0])],
        np.array([1.0]),
        "seq_mean",
        4,
        ref_token_logps=[np.array([1.0])],
        beta=0.5,
    )
    assert loss == pytest.approx(-1.0 + 0.5 * (math.e - 2.0))


def test_config_validation_and_yaml(tmp_path: Path) -> None:
    good = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "steps": 2,
        "prompts_per_step": 2,
        "group_size": 2,
        "out_dir": str(tmp_path),
        "seed": 0,
        "eval_every": 1,
        "eval_heldout_n": 0,
        "eval_train_n": 0,
        "eval_samples_per_prompt": 1,
        "save_every": 1,
        "micro_batch_size": 2,
    }
    cfg = GRPOConfig.from_dict(good)
    assert cfg.eval_max_tokens == cfg.max_tokens
    assert cfg.lora.scale == pytest.approx(2.0)
    assert cfg.completions_per_step == 4
    roundtrip = GRPOConfig.from_dict(cfg.to_dict())
    assert roundtrip.to_dict() == cfg.to_dict()
    with pytest.raises(ValueError, match="Unknown"):
        GRPOConfig.from_dict({**good, "nope": 1})
    with pytest.raises(ValueError, match="group_size"):
        GRPOConfig.from_dict({**good, "group_size": 1})
    with pytest.raises(ValueError, match="micro_batch_size"):
        GRPOConfig.from_dict({**good, "micro_batch_size": 3})
    with pytest.raises(ValueError, match="steps"):
        GRPOConfig.from_dict({**good, "steps": True})

    pilot = GRPOConfig.from_yaml(ROOT / "configs/train/mlx_pilot.yaml")
    assert pilot.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert pilot.steps == 200
    assert pilot.prompts_per_step == 16
    assert pilot.group_size == 8
    assert pilot.max_tokens == 512
    assert pilot.eval_max_tokens == 512
    assert pilot.lr == pytest.approx(2e-5)
    assert pilot.eval_every == 25
    assert pilot.eval_heldout_n == 256
    assert pilot.eval_samples_per_prompt == 2
    assert pilot.save_every == 50
    assert pilot.gradient_checkpointing is True
    assert pilot.lora.rank == 16 and pilot.lora.alpha == pytest.approx(32)

    study = GRPOConfig.from_yaml(ROOT / "configs/train/trl_h100.yaml")
    assert study.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert study.steps == 300
    assert study.prompts_per_step == 64
    assert study.group_size == 8
    assert study.max_tokens == 1024
    assert study.eval_max_tokens == 1024
    assert study.lr == pytest.approx(2e-5)
    assert study.warmup_steps == 10
    assert study.micro_batch_size == 32
    assert study.eval_heldout_n == 384
    assert study.eval_train_n == 128
    assert study.save_every == 25
    assert study.eval_every == 25
    assert study.use_vllm is True
    assert study.vllm_mode == "colocate"
    assert study.vllm_importance_sampling_correction is False
    assert study.eval_samples_per_prompt == 4
    assert study.lora_seed == 0
    assert study.completions_per_step % study.micro_batch_size == 0
    with pytest.raises(ValueError, match="lora_seed"):
        GRPOConfig.from_dict({**good, "lora_seed": True})
    with pytest.raises(ValueError, match="lora_init_sha256 requires"):
        GRPOConfig.from_dict({**good, "lora_init_sha256": "ab" * 32})
    with pytest.raises(ValueError, match="64-character"):
        GRPOConfig.from_dict({**good, "lora_init": "a.safetensors", "lora_init_sha256": "abcd"})
    loaded = GRPOConfig.from_dict(
        {**good, "lora_init": " a.safetensors ", "lora_init_sha256": "AB" * 32, "lora_seed": 3}
    )
    assert loaded.lora_init == "a.safetensors"
    assert loaded.lora_init_sha256 == "ab" * 32
    assert loaded.to_dict()["lora_seed"] == 3
    assert loaded.parameterisation == "lora"
    assert loaded.full_finetune is False
    full = GRPOConfig.from_dict({**good, "full_finetune": True, "lora": None})
    assert full.parameterisation == "full"
    assert full.to_dict()["lora"] is None
    assert GRPOConfig.from_dict(full.to_dict()).to_dict() == full.to_dict()
    with pytest.raises(ValueError, match="full_finetune: true"):
        GRPOConfig.from_dict({**good, "lora": None})
    with pytest.raises(ValueError, match="lora: null"):
        GRPOConfig.from_dict({**good, "full_finetune": True})
    with pytest.raises(ValueError, match="lora_init"):
        GRPOConfig.from_dict(
            {**good, "full_finetune": True, "lora": None, "lora_init": "a.safetensors"}
        )


class _YesTask(Task):
    name = "yes"

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        del seed
        return [
            Problem(pid=f"{split}-{i}", split=split, task="yes", question="q", answer="yes")
            for i in range(n)
        ]

    def prompt(self, problem: Problem) -> str:
        return problem.question

    def parse(self, problem: Problem, text: str) -> Parsed:
        del problem
        return Parsed(answer="yes" if "yes" in text else None, raw=text)

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        del problem, parsed
        return int("yes" in text)


class _EvenKey(Verifier):
    """Accept even sample indices, independent of the text."""

    def verdict(self, problem: Problem, parsed: Parsed, text: str, o: int, key=None) -> int:
        del problem, parsed, text, o
        return int(key is not None and int(key[-1]) % 2 == 0)


class _Scripted(Generator):
    def __init__(self) -> None:
        self.model_id = "scripted"

    def generate(
        self, prompts, n, *, max_tokens, temperature=1.0, top_p=1.0, seed=0, system_prompt=None
    ):
        del max_tokens, temperature, top_p, seed, system_prompt
        outs = []
        for i, _prompt in enumerate(prompts):
            group = []
            for k in range(n):
                text = "yes" if (i + k) % 2 == 0 else "no"
                group.append(
                    GenOut(
                        text=text,
                        n_tokens=2,
                        token_ids=(1, 2),
                        logprob=-0.5,
                        finish_reason="length" if k == 0 else "stop",
                    )
                )
            outs.append(group)
        return outs


def test_evaluate_policy_prompt_average() -> None:
    task = _YesTask()
    problems = [
        Problem(pid="a", split="heldout", task="yes", question="q", answer="yes"),
        Problem(pid="b", split="train", task="yes", question="q", answer="yes"),
    ]
    # prompt 0 samples: yes, no -> O = 1, 0 mean 0.5. prompt 1: no, yes -> 0, 1 mean 0.5.
    stats = evaluate_policy(
        _Scripted(),
        task,
        problems,
        n_samples=2,
        max_tokens=4,
        temperature=1.0,
        seed=0,
        verifier=_EvenKey(),
        monitors={"yes_word": lambda problem, parsed, text: "yes" in text},
        return_records=True,
    )
    assert stats["oracle_acc"] == pytest.approx(0.5)
    assert stats["n_prompts"] == 2
    assert stats["splits"]["heldout"]["oracle_acc"] == pytest.approx(0.5)
    assert stats["splits"]["train"]["n_prompts"] == 1
    assert stats["accept_rate"] == pytest.approx(0.5)
    assert stats["fpr"] == pytest.approx(0.5)
    assert stats["fnr"] == pytest.approx(0.5)
    assert stats["mean_length"] == pytest.approx(2.0)
    assert stats["truncation_rate"] == pytest.approx(0.5)
    assert stats["monitors"]["yes_word"] == pytest.approx(0.5)
    assert stats["n_samples"] == 2
    # O=1 samples are both "yes"; O=0 samples are both "no".
    assert stats["monitors_correct"]["yes_word"] == pytest.approx(1.0)
    assert stats["monitors_wrong"]["yes_word"] == pytest.approx(0.0)
    assert len(stats["records"]) == 4
    no_verifier = evaluate_policy(
        _Scripted(),
        task,
        problems,
        n_samples=2,
        max_tokens=4,
        temperature=1.0,
        seed=0,
        monitors={"yes_word": lambda problem, parsed, text: "yes" in text},
    )
    assert no_verifier["accept_rate"] is None
    assert no_verifier["monitors_correct"]["yes_word"] == pytest.approx(1.0)
    assert no_verifier["monitors_wrong"]["yes_word"] == pytest.approx(0.0)
    assert no_verifier["n_samples"] == 2
