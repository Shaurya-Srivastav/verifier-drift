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
    assert summary["rollout_seeded"] is True
    assert summary["lora_seed"] == 0
    assert summary["lora_init_sha256"] is None
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


def _lora_a(model) -> dict[str, np.ndarray]:
    found = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LoRALinear":
            found[name] = np.array(module.lora_a)
    return found


@pytest.mark.mlx
def test_warmup_matches_transformers_schedule() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    get_constant_schedule_with_warmup = transformers.get_constant_schedule_with_warmup

    from vfates.train.mlx_grpo import _make_optimizer, constant_with_warmup_lr

    base = 2e-5
    warmup = 10
    parameter = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([parameter], lr=base)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=warmup)
    hf_lrs = []
    for _ in range(13):
        hf_lrs.append(float(opt.param_groups[0]["lr"]))
        opt.step()
        sched.step()

    class _One(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = mx.zeros((2,))

    model = _One()
    cfg = _cfg(Path("/tmp/unused-mlx-warmup"), steps=13, lr=base, warmup_steps=warmup)
    optimizer = _make_optimizer(cfg)
    for step, expected in enumerate(hf_lrs):
        optimizer.update(model, {"weight": mx.ones((2,))})
        lr_value = optimizer.learning_rate
        if isinstance(lr_value, mx.array):
            lr_value = float(lr_value.item())
        assert lr_value == pytest.approx(expected)
        assert lr_value == pytest.approx(constant_with_warmup_lr(step, base, warmup))
    assert hf_lrs[0] == 0.0
    assert hf_lrs[1] == pytest.approx(2e-6)
    assert hf_lrs[10] == pytest.approx(base)
    assert hf_lrs[12] == pytest.approx(base)


@pytest.mark.mlx
def test_completion_logprobs_divide_by_temperature(tokenizer) -> None:
    import mlx.core as mx

    from vfates.train.mlx_grpo import _completion_logprobs

    model, _args = _tiny(tokenizer, seed=1)
    full = mx.array([[1, 2, 3, 4, 5, 6]])
    prompt_lens = mx.array([4])
    raw = _completion_logprobs(model, full, prompt_lens, 2, 1.0)
    greedy = _completion_logprobs(model, full, prompt_lens, 2, 0.0)
    tempered = _completion_logprobs(model, full, prompt_lens, 2, 2.0)
    logits = model(full).astype(mx.float32)
    pred = logits[:, :-1, :]
    targets = full[:, 1:]
    chosen = mx.take_along_axis(pred, targets[:, :, None], axis=-1).squeeze(-1)
    all_lp = chosen - mx.logsumexp(pred, axis=-1)
    pos = prompt_lens[:, None] + mx.arange(2)[None, :] - 1
    manual = mx.take_along_axis(all_lp, pos.astype(mx.int32), axis=1)
    half = pred / mx.array(2.0, dtype=mx.float32)
    chosen_half = mx.take_along_axis(half, targets[:, :, None], axis=-1).squeeze(-1)
    manual_half = chosen_half - mx.logsumexp(half, axis=-1)
    manual_half = mx.take_along_axis(manual_half, pos.astype(mx.int32), axis=1)
    mx.eval(raw, greedy, tempered, manual, manual_half)
    np.testing.assert_allclose(
        np.array(raw.tolist()), np.array(manual.tolist()), atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.array(greedy.tolist()), np.array(manual.tolist()), atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.array(tempered.tolist()), np.array(manual_half.tolist()), atol=1e-5
    )
    with pytest.raises(ValueError, match="temperature"):
        _completion_logprobs(model, full, prompt_lens, 2, -1.0)


@pytest.mark.mlx
def test_lora_seed_and_file_init(tmp_path: Path, tokenizer) -> None:
    import mlx.core as mx

    from vfates.features.lora_init import save_lora_init

    task = _YesTask()
    lora = {"rank": 4, "alpha": 8, "dropout": 0.0, "keys": ["q"], "num_layers": 1}

    def prepared(path: Path, **kwargs: object) -> MLXGRPOTrainer:
        mx.random.seed(999)
        model, _args = _tiny(tokenizer, seed=0)
        trainer = MLXGRPOTrainer(
            _cfg(path, lora=lora, **kwargs),
            task,
            _Even(),
            _problems(task, "train", 2),
            [],
            model=model,
            tokenizer=tokenizer,
        )
        trainer._prepare_model()
        return trainer

    first = prepared(tmp_path / "a", lora_seed=0, seed=0)
    second = prepared(tmp_path / "b", lora_seed=0, seed=0)
    other = prepared(tmp_path / "c", lora_seed=1, seed=0)
    other_run = prepared(tmp_path / "d", lora_seed=0, seed=4)
    left = _lora_a(first.model)
    assert set(left) == {"model.layers.1.self_attn.q_proj"}
    np.testing.assert_array_equal(
        left["model.layers.1.self_attn.q_proj"],
        _lora_a(second.model)["model.layers.1.self_attn.q_proj"],
    )
    np.testing.assert_array_equal(
        left["model.layers.1.self_attn.q_proj"],
        _lora_a(other_run.model)["model.layers.1.self_attn.q_proj"],
    )
    assert not np.array_equal(
        left["model.layers.1.self_attn.q_proj"],
        _lora_a(other.model)["model.layers.1.self_attn.q_proj"],
    )
    key = "model.layers.1.self_attn.q_proj"
    module = dict(first.model.named_modules())[key]
    assert float(module.scale) == pytest.approx(8 / 4)
    assert np.all(np.array(module.lora_b) == 0)
    rank, in_features = 4, int(np.array(module.lora_a).shape[0])
    array = np.arange(rank * in_features, dtype=np.float32).reshape(rank, in_features) * 0.01 + 0.3
    path = tmp_path / "lora_init.safetensors"
    digest = save_lora_init(path, {key: array}, {"model_id": "tiny"})
    loaded = prepared(tmp_path / "file", lora_init=str(path), lora_init_sha256=digest)
    np.testing.assert_allclose(np.array(dict(loaded.model.named_modules())[key].lora_a), array.T)
    assert np.all(np.array(dict(loaded.model.named_modules())[key].lora_b) == 0)
    assert loaded._lora_init_sha == digest
    assert loaded.config.lora_init_sha256 == digest


@pytest.mark.mlx
def test_lora_dropout_zero_is_identity() -> None:
    import mlx.core as mx
    from mlx_lm.tuner.lora import LoRALinear

    layer = LoRALinear(input_dims=4, output_dims=4, r=2, dropout=0.0, scale=2.0)
    assert float(layer.scale) == pytest.approx(2.0)
    x = mx.ones((3, 4))
    layer.train()
    train_y = layer(x)
    layer.eval()
    eval_y = layer(x)
    mx.eval(train_y, eval_y)
    np.testing.assert_allclose(
        np.array(train_y.tolist()), np.array(eval_y.tolist()), atol=0.0, rtol=0.0
    )


@pytest.mark.mlx
def test_nonfinite_loss_raises_before_a_write(tmp_path: Path, tokenizer, monkeypatch) -> None:
    import mlx.core as mx

    import vfates.train.mlx_grpo as mlx_grpo
    from vfates.train.config import NonFiniteLossError

    real = mlx_grpo._completion_logprobs

    def fake(*args, **kwargs):
        return real(*args, **kwargs) * mx.array(float("nan"))

    monkeypatch.setattr(mlx_grpo, "_completion_logprobs", fake)
    model, _args = _tiny(tokenizer, seed=0)
    task = _YesTask()
    trainer = MLXGRPOTrainer(
        _cfg(tmp_path, steps=1, eval_heldout_n=0, eval_train_n=0, save_every=1),
        task,
        _Even(),
        _problems(task, "train", 2),
        [],
        model=model,
        tokenizer=tokenizer,
    )
    with pytest.raises(NonFiniteLossError) as caught:
        trainer.run()
    assert caught.value.step == 1
    assert caught.value.run_dir == str(tmp_path)
    assert not (tmp_path / "metrics.jsonl").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.mlx
def test_immediate_eos_is_length_one(tmp_path: Path, tokenizer, monkeypatch) -> None:
    import mlx.core as mx

    from vfates.gen.mlx_backend import IdOut

    eos = int(tokenizer.eos_token_id)
    texts: list[str] = []

    def fake_generate(self, prompt_ids, max_tokens, temperature, top_p, seed):
        del self, max_tokens, temperature, top_p, seed
        return [IdOut((eos,), (0.0,), 0.0, "stop", 1) for _ids in prompt_ids]

    monkeypatch.setattr("vfates.gen.mlx_backend.MLXGenerator.generate_ids", fake_generate)
    model, _args = _tiny(tokenizer, seed=0)
    task = _YesTask()
    trainer = MLXGRPOTrainer(
        _cfg(
            tmp_path,
            steps=1,
            eval_heldout_n=0,
            eval_train_n=0,
            max_tokens=4,
            lora={"rank": 4, "alpha": 8, "dropout": 0.0, "keys": ["q"], "num_layers": 1},
        ),
        task,
        _Even(),
        _problems(task, "train", 2),
        [],
        model=model,
        tokenizer=tokenizer,
    )
    real_score = trainer.ctx.score_batch

    def spy(pids, batch_texts, keys):
        texts.extend(batch_texts)
        return real_score(pids, batch_texts, keys)

    trainer.ctx.score_batch = spy
    before = np.array(model.layers[1].self_attn.q_proj.weight)
    trainer.run()
    q = model.layers[1].self_attn.q_proj
    assert float(mx.abs(q.lora_b).max()) > 0.0
    np.testing.assert_allclose(np.array(q.linear.weight), before, atol=0.0, rtol=0.0)
    rows = _jsonl(tmp_path / "metrics.jsonl")
    assert rows[0]["mean_completion_length"] == pytest.approx(1.0)
    assert rows[0]["max_completion_length"] == 1
    assert texts
    assert set(texts) == {tokenizer.decode([eos], skip_special_tokens=True)}


@pytest.mark.mlx
def test_eval_precedes_checkpoint(tmp_path: Path, tokenizer) -> None:
    events: list[tuple[str, int]] = []
    model, _args = _tiny(tokenizer, seed=0)
    task = _YesTask()
    trainer = MLXGRPOTrainer(
        _cfg(tmp_path, steps=2, eval_every=1, save_every=1, eval_heldout_n=0, eval_train_n=0),
        task,
        _Even(),
        _problems(task, "train", 2),
        [],
        model=model,
        tokenizer=tokenizer,
    )

    def train_step(step_index: int) -> dict:
        events.append(("train", step_index))
        return {
            "loss": 0.0,
            "reward_V_mean": 0.0,
            "oracle_mean": 0.0,
            "gen_seconds": 0.0,
            "train_seconds": 0.0,
            "tokens_generated": 0,
            "kl": None,
        }

    trainer._train_step = train_step
    trainer._evaluate = lambda step: events.append(("eval", step))
    trainer._checkpoint = lambda: events.append(("ckpt", trainer.completed))
    trainer.run()
    assert events == [
        ("eval", 0),
        ("train", 0),
        ("eval", 1),
        ("ckpt", 1),
        ("train", 1),
        ("eval", 2),
        ("ckpt", 2),
    ]


@pytest.mark.mlx
def test_resume_fills_missing_eval(tmp_path: Path, tokenizer) -> None:
    model, _args = _tiny(tokenizer, seed=0)
    task = _YesTask()
    common = {
        "steps": 1,
        "eval_every": 1,
        "save_every": 1,
        "eval_heldout_n": 1,
        "eval_train_n": 0,
        "eval_samples_per_prompt": 1,
        "max_tokens": 2,
        "eval_max_tokens": 2,
    }
    trainer = MLXGRPOTrainer(
        _cfg(tmp_path, **common),
        task,
        _Even(),
        _problems(task, "train", 2),
        _problems(task, "heldout", 1),
        model=model,
        tokenizer=tokenizer,
    )
    trainer.run()
    rows = _jsonl(tmp_path / "eval.jsonl")
    assert {(row["step"], row["split"]) for row in rows} == {(0, "heldout"), (1, "heldout")}
    kept = [row for row in rows if row["step"] != 1]
    (tmp_path / "eval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept))
    resumed = MLXGRPOTrainer(
        _cfg(tmp_path, **common),
        task,
        _Even(),
        _problems(task, "train", 2),
        _problems(task, "heldout", 1),
        model=_tiny(tokenizer, seed=0)[0],
        tokenizer=tokenizer,
    )
    resumed.run(resume=True)
    again = _jsonl(tmp_path / "eval.jsonl")
    assert {(row["step"], row["split"]) for row in again} == {(0, "heldout"), (1, "heldout")}
    assert len(_jsonl(tmp_path / "metrics.jsonl")) == 1
