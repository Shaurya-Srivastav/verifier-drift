"""Tiny-model GRPO on MLX, plus the real 0.5B / 1.5B smoke timings."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx")

from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig
from vfates.train.grpo_core import group_advantages, pg_loss_reference
from vfates.train.mlx_grpo import (
    METRIC_KEYS,
    MLXGRPOTrainer,
    _per_token_loss,
    _Pool,
    group_advantages_mx,
    pg_loss_mx,
)
from vfates.types import Parsed, Problem
from vfates.verifiers.base import Verifier


def _tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("the Qwen2.5-0.5B-Instruct tokenizer is not in the local Hugging Face cache")


def _tiny(tokenizer, seed: int = 0):
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
    """Nonzero group advantages: completion k is accepted iff k is even."""

    name = "even"
    stochastic = True

    def verdict(self, problem: Problem, parsed: Parsed, text: str, o: int, key=None) -> int:
        del problem, parsed, text, o
        if key is None:
            return 0
        return int(int(key[-1]) % 2 == 0)


class _AddTask(Task):
    name = "add"

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            a = int(rng.integers(0, 20))
            b = int(rng.integers(0, 20))
            out.append(
                Problem(
                    pid=f"{split}-{i}",
                    split=split,
                    task="add",
                    question=f"{a}+{b}",
                    answer=str(a + b),
                )
            )
        return out

    def prompt(self, problem: Problem) -> str:
        return f"What is {problem.question}? Reply with the integer only."

    def parse(self, problem: Problem, text: str) -> Parsed:
        del problem
        digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
        return Parsed(answer=digits[0] if digits else None, raw=text)

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        del text
        return int(parsed.answer == problem.answer)


def _cfg(tmp: Path, **kwargs: object) -> GRPOConfig:
    raw: dict[str, object] = {
        "model_id": "tiny",
        "steps": 3,
        "prompts_per_step": 2,
        "group_size": 2,
        "max_tokens": 4,
        "max_prompt_length": 128,
        "lr": 1e-1,
        "out_dir": str(tmp),
        "seed": 0,
        "eval_every": 100,
        "eval_heldout_n": 1,
        "eval_train_n": 1,
        "eval_samples_per_prompt": 1,
        "eval_max_tokens": 4,
        "save_every": 2,
        "micro_batch_size": 2,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0, "num_layers": -1},
    }
    raw.update(kwargs)
    return GRPOConfig.from_dict(raw)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _problems(task: Task, split: str, n: int) -> list[Problem]:
    return task.problems(split, n, seed=0)


@pytest.mark.mlx
def test_advantages_and_loss_match_numpy() -> None:
    import mlx.core as mx

    rewards = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    ref, ref_info = group_advantages(rewards)
    got, got_info = group_advantages_mx(rewards)
    mx.eval(got, got_info)
    np.testing.assert_allclose(np.array(got.tolist()), ref, atol=1e-5)
    assert np.array(got_info.tolist()).astype(bool).tolist() == ref_info.tolist()
    mean_adv, _info = group_advantages_mx(rewards, mode="mean")
    mean_ref, _ = group_advantages(rewards, mode="mean")
    mx.eval(mean_adv)
    np.testing.assert_allclose(np.array(mean_adv.tolist()), mean_ref, atol=1e-5)

    new = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    mask = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    adv = np.array([1.0, -1.0])
    seqs = [new[0], new[1, :1]]
    for agg in ("seq_mean", "token_mean", "dr_constant"):
        value = pg_loss_mx(mx.array(new), mx.array(adv), mx.array(mask), agg, 4)
        mx.eval(value)
        assert float(value.item()) == pytest.approx(pg_loss_reference(seqs, adv, agg, 4), abs=1e-5)
    old = np.array([[-math.log(2.0), -math.log(2.0)], [0.0, 0.0]], dtype=np.float32)
    value = pg_loss_mx(
        mx.array(new), mx.array(adv), mx.array(mask), "seq_mean", 4, old_token_logps=mx.array(old)
    )
    mx.eval(value)
    clipped = pg_loss_reference(
        seqs,
        adv,
        "seq_mean",
        4,
        old_token_logps=[old[0], old[1, :1]],
    )
    assert float(value.item()) == pytest.approx(clipped, abs=1e-5)


@pytest.mark.mlx
def test_ratio_one_gradient_is_reinforce() -> None:
    import mlx.core as mx

    def loss(logp: mx.array) -> mx.array:
        per = _per_token_loss(logp, mx.array([2.0]), None, None, 0.0, 0.2, 0.2)
        return per.sum()

    grad = mx.grad(loss)(mx.array([[0.2, -0.4]]))
    mx.eval(grad)
    np.testing.assert_allclose(np.array(grad.tolist()), [[-2.0, -2.0]], atol=1e-5)


@pytest.mark.mlx
def test_pool_cycles_and_streaming_stops() -> None:
    task = _YesTask()
    problems = _problems(task, "train", 3)
    pool = _Pool(problems, "multi_epoch", seed=0)
    assert [p.pid for p in pool.take(3)] == [
        p.pid for p in _Pool(problems, "multi_epoch", 0).take(3)
    ]
    assert len(pool.take(2)) == 2
    stream = _Pool(problems, "streaming", seed=0)
    assert len(stream.take(2)) == 2
    with pytest.raises(RuntimeError, match="exhausted"):
        stream.take(2)


@pytest.mark.mlx
def test_tiny_trainer_three_steps(tmp_path: Path, tokenizer) -> None:
    import mlx.core as mx

    model, _args = _tiny(tokenizer, seed=0)
    base = np.array(model.layers[0].self_attn.q_proj.weight)
    task = _YesTask()
    trainer = MLXGRPOTrainer(
        _cfg(tmp_path),
        task,
        _Even(),
        _problems(task, "train", 4),
        _problems(task, "heldout", 2),
        monitors={"yes_word": lambda problem, parsed, text: "yes" in text.lower()},
        model=model,
        tokenizer=tokenizer,
    )
    summary = trainer.run()
    assert summary["steps_completed"] == 3
    assert summary["interrupted"] is False
    rows = _jsonl(tmp_path / "metrics.jsonl")
    assert [row["step"] for row in rows] == [1, 2, 3]
    for row in rows:
        for key in METRIC_KEYS:
            assert key in row
        assert math.isfinite(row["loss"])
        assert math.isfinite(row["grad_norm"])
        assert row["kl"] is None
    eval_rows = _jsonl(tmp_path / "eval.jsonl")
    splits = {(row["step"], row["split"]) for row in eval_rows}
    assert (0, "heldout") in splits and (0, "train") in splits
    assert (3, "heldout") in splits and (3, "train") in splits
    assert "oracle_acc" in eval_rows[0]
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "environment_report.json").exists()
    assert (tmp_path / "summary.json").exists()
    q = model.layers[0].self_attn.q_proj
    np.testing.assert_allclose(np.array(q.linear.weight), base, atol=0.0, rtol=0.0)
    assert float(mx.abs(q.lora_b).max()) > 0.0


@pytest.mark.mlx
def test_resume_matches_continuous_run(tmp_path: Path, tokenizer) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model

    model, args = _tiny(tokenizer, seed=2)
    snap = [(key, mx.array(value)) for key, value in tree_flatten(model.parameters())]

    def fresh() -> object:
        mx.random.seed(0)
        cloned = Model(args)
        cloned.load_weights([(key, mx.array(value)) for key, value in snap])
        mx.eval(cloned.parameters())
        return cloned

    task = _YesTask()
    train = _problems(task, "train", 4)
    held = _problems(task, "heldout", 2)
    common = {"eval_heldout_n": 0, "eval_train_n": 0, "save_every": 1}

    full = MLXGRPOTrainer(
        _cfg(tmp_path / "full", steps=3, **common),
        task,
        _Even(),
        train,
        held,
        model=model,
        tokenizer=tokenizer,
    )
    full.run()
    short = MLXGRPOTrainer(
        _cfg(tmp_path / "short", steps=2, **common),
        task,
        _Even(),
        train,
        held,
        model=fresh(),
        tokenizer=tokenizer,
    )
    short.run()
    resumed = MLXGRPOTrainer(
        _cfg(tmp_path / "short", steps=3, **common),
        task,
        _Even(),
        train,
        held,
        model=fresh(),
        tokenizer=tokenizer,
    )
    resumed.run(resume=True)
    assert resumed.completed == 3
    steps = [row["step"] for row in _jsonl(tmp_path / "short" / "metrics.jsonl")]
    assert steps == [1, 2, 3]
    full_b = np.array(model.layers[0].self_attn.q_proj.lora_b)
    cont_b = np.array(resumed.model.layers[0].self_attn.q_proj.lora_b)
    np.testing.assert_allclose(cont_b, full_b, atol=1e-4, rtol=1e-4)


@pytest.mark.mlx
def test_num_iterations_two_is_finite(tmp_path: Path, tokenizer) -> None:
    model, _args = _tiny(tokenizer, seed=3)
    task = _YesTask()
    trainer = MLXGRPOTrainer(
        _cfg(tmp_path, steps=1, num_iterations=2, eval_heldout_n=0, eval_train_n=0),
        task,
        _Even(),
        _problems(task, "train", 4),
        _problems(task, "heldout", 1),
        model=model,
        tokenizer=tokenizer,
    )
    trainer.run()
    row = _jsonl(tmp_path / "metrics.jsonl")[0]
    assert row["step"] == 1
    assert math.isfinite(row["loss"])


def _smoke(model_id: str, steps: int, tmp: Path, micro: int) -> dict:
    import mlx.core as mx

    task = _AddTask()
    cfg = GRPOConfig.from_dict(
        {
            "model_id": model_id,
            "steps": steps,
            "prompts_per_step": 8,
            "group_size": 8,
            "max_tokens": 256,
            "lr": 2e-5,
            "out_dir": str(tmp),
            "seed": 0,
            "eval_every": 100,
            "eval_heldout_n": 2,
            "eval_train_n": 2,
            "eval_samples_per_prompt": 1,
            "eval_max_tokens": 32,
            "save_every": 100,
            "micro_batch_size": micro,
            "gradient_checkpointing": True,
            "max_prompt_length": 256,
        }
    )
    import gc

    gc.collect()
    mx.clear_cache()
    mx.metal.clear_cache()
    mx.reset_peak_memory()
    trainer = MLXGRPOTrainer(
        cfg, task, _Even(), task.problems("train", 16, 0), task.problems("heldout", 4, 1)
    )
    summary = trainer.run()
    rows = _jsonl(tmp / "metrics.jsonl")
    record = {
        "model_id": model_id,
        "steps": [
            {
                "step": row["step"],
                "gen_seconds": row["gen_seconds"],
                "train_seconds": row["train_seconds"],
                "loss": row["loss"],
                "tokens_generated": row["tokens_generated"],
            }
            for row in rows
        ],
        "peak_memory_gb": summary["peak_memory_gb"],
    }
    path = Path("/tmp/vfates_mlx_smoke.json")
    previous = []
    if path.exists():
        previous = json.loads(path.read_text())
    previous = [item for item in previous if item["model_id"] != model_id]
    previous.append(record)
    path.write_text(json.dumps(previous, indent=2))
    print(json.dumps(record, indent=2))
    return record


@pytest.mark.slow
@pytest.mark.mlx
def test_z_smoke_qwen05(tmp_path: Path) -> None:
    record = _smoke("Qwen/Qwen2.5-0.5B-Instruct", steps=3, tmp=tmp_path, micro=4)
    assert len(record["steps"]) == 3
    assert all(math.isfinite(step["loss"]) for step in record["steps"])
    assert record["peak_memory_gb"] > 0


@pytest.mark.slow
@pytest.mark.mlx
def test_z_smoke_qwen15(tmp_path: Path) -> None:
    record = _smoke("Qwen/Qwen2.5-1.5B-Instruct", steps=1, tmp=tmp_path, micro=2)
    assert len(record["steps"]) == 1
    assert math.isfinite(record["steps"][0]["loss"])
    assert record["peak_memory_gb"] > 0
