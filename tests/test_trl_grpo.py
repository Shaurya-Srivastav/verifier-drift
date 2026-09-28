"""TRL builder: argument mapping, and one CPU optimizer step on a tiny Qwen2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig
from vfates.train.trl_grpo import (
    VfatesLogCallback,
    build_trl_trainer,
    metric_row_from_trl_logs,
    rates_from_reward_means,
    trl_grpo_kwargs,
    trl_run_metadata,
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
    assert kwargs["logging_nan_inf_filter"] is False
    assert kwargs["bf16"] is False
    assert trl_run_metadata(cfg, use_vllm=True)["rollout_seeded"] is False
    assert trl_run_metadata(cfg, use_vllm=False)["rollout_seeded"] is True
    assert kwargs["remove_unused_columns"] is False
    dapo = trl_grpo_kwargs(_cfg(tmp_path, loss_agg="token_mean", adv_mode="mean"), n_reward_funcs=1)
    assert dapo["loss_type"] == "dapo"
    assert dapo["scale_rewards"] == "none"
    dr = trl_grpo_kwargs(_cfg(tmp_path, loss_agg="dr_constant"), n_reward_funcs=1)
    assert dr["loss_type"] == "dr_grpo"
    trl = pytest.importorskip("trl")

    args = trl.GRPOConfig(**kwargs)
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
    assert row["wall_s"] is None
    assert row["tokens_generated"] is None
    assert row["num_tokens_cumulative"] == 10


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _State:
    def __init__(self, step: int) -> None:
        self.global_step = step


class _Control:
    def __init__(self, should_evaluate: bool = False) -> None:
        self.should_evaluate = should_evaluate


class _EvalOnStart:
    eval_on_start = True


_TRAIN_LOG = {"rewards/verifier/mean": 1.0, "loss": 0.2}
_EVAL_LOG = {"eval_rewards/verifier/mean": 0.5}


def _jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_trl_wall_s_is_one_optimizer_step_and_excludes_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Begin/log, begin/log, with eval between them.

    Transformers calls ``on_step_begin`` once per optimizer step (not once per
    accumulation microbatch) and logs before evaluate. The fake clock stands
    in for ``time.perf_counter``.
    """
    clock = _Clock()
    monkeypatch.setattr("vfates.train.trl_grpo.time.perf_counter", clock)
    callback = VfatesLogCallback(str(tmp_path), [])

    clock.now = 10.0
    callback.on_step_begin(None, _State(0), _Control())
    # Accumulation microbatches do not call on_step_begin again.
    clock.now = 12.5
    callback.on_log(None, _State(1), _Control(should_evaluate=True), logs=dict(_TRAIN_LOG))
    clock.now = 16.0
    callback.on_log(None, _State(1), _Control(should_evaluate=True), logs=dict(_EVAL_LOG))
    clock.now = 16.0
    callback.on_step_begin(None, _State(1), _Control())
    clock.now = 16.4
    callback.on_log(None, _State(2), _Control(), logs=dict(_TRAIN_LOG))

    rows = _jsonl(tmp_path / "metrics.jsonl")
    assert [row["wall_s"] for row in rows] == pytest.approx([2.5, 0.4])
    assert [row["step"] for row in rows] == [1, 2]
    eval_rows = _jsonl(tmp_path / "trl_eval.jsonl")
    assert len(eval_rows) == 1
    assert eval_rows[0]["eval_seconds"] == pytest.approx(3.5)
    assert "wall_s" not in eval_rows[0]


def test_trl_wall_s_resume_does_not_invent_a_first_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh callback after resume has no stamp until the next ``on_step_begin``."""
    clock = _Clock()
    monkeypatch.setattr("vfates.train.trl_grpo.time.perf_counter", clock)
    callback = VfatesLogCallback(str(tmp_path), [])

    clock.now = 5000.0
    callback.on_train_begin(_EvalOnStart(), _State(4), _Control())
    clock.now = 5002.0
    callback.on_log(None, _State(4), _Control(), logs=dict(_EVAL_LOG))
    clock.now = 9000.0
    callback.on_log(None, _State(4), _Control(), logs=dict(_TRAIN_LOG))
    clock.now = 9000.0
    callback.on_step_begin(None, _State(4), _Control())
    clock.now = 9000.35
    callback.on_log(None, _State(5), _Control(), logs=dict(_TRAIN_LOG))

    rows = _jsonl(tmp_path / "metrics.jsonl")
    assert rows[0]["wall_s"] is None
    assert rows[0]["step"] == 4
    assert rows[1]["wall_s"] == pytest.approx(0.35)
    assert rows[1]["step"] == 5
    eval_rows = _jsonl(tmp_path / "trl_eval.jsonl")
    assert eval_rows[0]["eval_seconds"] == pytest.approx(2.0)
    assert rows[1]["wall_s"] < eval_rows[0]["eval_seconds"]


def test_arm_seed_timing_sums_trl_wall_and_excludes_eval(tmp_path: Path) -> None:
    from vfates.pilots.p5_gpu_gate import arm_seed_timing
    from vfates.utils.io import write_json

    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        '{"step": 1, "wall_s": 10.0}\n{"step": 2, "wall_s": 12.0}\n',
        encoding="utf-8",
    )
    (run / "trl_eval.jsonl").write_text(
        '{"step": 0, "eval_seconds": 3.0}\n{"step": 2, "eval_seconds": 4.0}\n',
        encoding="utf-8",
    )
    timing = arm_seed_timing(run)
    assert timing["source"] == "wall_s"
    assert timing["train_seconds"] == pytest.approx(22.0)
    assert timing["eval_seconds"] == pytest.approx(7.0)
    assert timing["eval_included_in_wall"] is False

    clocked = tmp_path / "clock"
    clocked.mkdir()
    (clocked / "metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    (clocked / "trl_eval.jsonl").write_text('{"eval_seconds": 9.0}\n', encoding="utf-8")
    write_json(clocked / "timing.json", {"train_arm_wall_s": 12.5, "includes_eval": True})
    clock = arm_seed_timing(clocked)
    assert clock["source"] == "timing.json"
    assert clock["train_seconds"] == pytest.approx(12.5)
    assert clock["eval_seconds"] == pytest.approx(0.0)
    assert clock["eval_included_in_wall"] is True


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
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
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
    assert all(isinstance(row["wall_s"], float) and row["wall_s"] > 0 for row in rows)


def _lora_cfg(tmp: Path, **kwargs: object) -> GRPOConfig:
    raw = {
        "gradient_checkpointing": False,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0, "keys": ["q"], "num_layers": 1},
        "lora_seed": 0,
    }
    raw.update(kwargs)
    return _cfg(tmp, **raw)


def _build(tmp: Path, model, tokenizer, **kwargs: object):
    import torch

    torch.manual_seed(999)
    task = _YesTask()
    return build_trl_trainer(
        _lora_cfg(tmp, **kwargs),
        task,
        _Even(),
        task.problems("train", 2, 0),
        [],
        use_vllm=False,
        model=model,
        processing_class=tokenizer,
        extra_trl_args={"use_cpu": True, "bf16": False, "fp16": False},
    )


def _clone_tiny(tokenizer):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

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
    return model


@pytest.mark.torch
def test_lora_seed_and_file_init(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("safetensors")
    import torch

    from vfates.features.lora_init import save_lora_init
    from vfates.train.trl_grpo import peft_lora_matrices

    try:
        _base, tokenizer = _tiny_hf()
    except OSError:
        pytest.skip("the Qwen2.5-0.5B-Instruct tokenizer is not in the local Hugging Face cache")

    first = _build(tmp_path / "a", _clone_tiny(tokenizer), tokenizer, seed=0, lora_seed=0)
    second = _build(tmp_path / "b", _clone_tiny(tokenizer), tokenizer, seed=0, lora_seed=0)
    other_seed = _build(tmp_path / "c", _clone_tiny(tokenizer), tokenizer, seed=0, lora_seed=1)
    other_run = _build(tmp_path / "d", _clone_tiny(tokenizer), tokenizer, seed=4, lora_seed=0)

    def matrices(trainer):
        return peft_lora_matrices(trainer.model)

    left = matrices(first)
    right = matrices(second)
    assert set(left) == {"model.layers.1.self_attn.q_proj"}
    for key in left:
        assert torch.equal(left[key][0], right[key][0])
        assert torch.equal(left[key][0], matrices(other_run)[key][0])
        assert not torch.equal(left[key][0], matrices(other_seed)[key][0])
        assert torch.count_nonzero(left[key][1]) == 0
    names = [name for name, _param in first.model.named_parameters()]
    assert any("lora_A" in name for name in names)
    assert not any("base_model.model.base_model" in name for name in names)

    tensors = {}
    for key, (weight, _b) in left.items():
        array = torch.arange(int(weight.numel()), dtype=torch.float32).reshape(tuple(weight.shape))
        tensors[key] = array.numpy() * 0.01 + 0.25
    path = tmp_path / "lora_init.safetensors"
    digest = save_lora_init(path, tensors, {"model_id": "tiny"})
    loaded = _build(
        tmp_path / "file",
        _clone_tiny(tokenizer),
        tokenizer,
        lora_init=str(path),
        lora_init_sha256=digest,
    )
    got = matrices(loaded)
    for key, array in tensors.items():
        assert torch.equal(got[key][0], torch.as_tensor(array))
        assert torch.count_nonzero(got[key][1]) == 0
    meta = json.loads((tmp_path / "file" / "run_metadata.json").read_text())
    saved_cfg = json.loads((tmp_path / "file" / "config.json").read_text())
    assert meta["lora_init_sha256"] == digest
    assert saved_cfg["lora_init_sha256"] == digest
    assert meta["rollout_seeded"] is True
    with pytest.raises(ValueError, match="sha256"):
        _build(
            tmp_path / "badsha",
            _clone_tiny(tokenizer),
            tokenizer,
            lora_init=str(path),
            lora_init_sha256="0" * 64,
        )
    extra = dict(tensors)
    extra["model.layers.0.self_attn.q_proj"] = next(iter(tensors.values()))
    extra_path = tmp_path / "extra.safetensors"
    extra_sha = save_lora_init(extra_path, extra, {"model_id": "tiny"})
    with pytest.raises(ValueError, match="wrapped modules"):
        _build(
            tmp_path / "extra",
            _clone_tiny(tokenizer),
            tokenizer,
            lora_init=str(extra_path),
            lora_init_sha256=extra_sha,
        )


@pytest.mark.torch
def test_nonfinite_loss_raises_before_a_write(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    import torch

    from vfates.train.config import NonFiniteLossError

    try:
        model, tokenizer = _tiny_hf()
    except OSError:
        pytest.skip("the Qwen2.5-0.5B-Instruct tokenizer is not in the local Hugging Face cache")
    trainer = _build(tmp_path, model, tokenizer)
    trainer._vfates_loss_fn = lambda *args, **kwargs: torch.tensor(float("nan"))
    with pytest.raises(NonFiniteLossError) as caught:
        trainer.compute_loss(trainer.model, {})
    assert caught.value.step == 1
    assert caught.value.run_dir == str(tmp_path)
    assert not (tmp_path / "metrics.jsonl").exists()
    assert list(tmp_path.glob("checkpoint-*")) == []

    param = next(p for p in trainer.model.parameters() if p.requires_grad)
    param.grad = torch.full_like(param, float("nan"))
    guard = next(
        callback
        for callback in trainer.callback_handler.callbacks
        if type(callback).__name__ == "_GradCallback"
    )
    with pytest.raises(NonFiniteLossError, match="gradient norm"):
        guard.on_pre_optimizer_step(None, trainer.state, None, model=trainer.model)
