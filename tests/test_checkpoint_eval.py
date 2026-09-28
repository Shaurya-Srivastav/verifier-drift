"""PEFT checkpoint discovery and trainer-independent eval logs."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfates.gen.base import Generator
from vfates.tasks.chain_sum import ChainSumTask
from vfates.train.checkpoint_eval import evaluate_checkpoints, peft_checkpoints
from vfates.train.config import GRPOConfig
from vfates.train.evaluate import EVAL_ROW_KEYS, eval_log_rows
from vfates.types import GenOut
from vfates.utils.io import read_jsonl


class FakeGenerator(Generator):
    def __init__(self, answers: dict[str, str]) -> None:
        self.model_id = "fake"
        self.answers = answers
        self.adapter: str | None = None
        self.adapters: list[str | None] = []

    def set_adapter(self, path: str | None) -> None:
        self.adapter = path
        self.adapters.append(path)

    def generate(
        self,
        prompts: list[str],
        n: int,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
        system_prompt: str | None = None,
    ) -> list[list[GenOut]]:
        del max_tokens, temperature, top_p, seed, system_prompt
        groups: list[list[GenOut]] = []
        for prompt in prompts:
            gold = self.answers[prompt]
            text = f"\\boxed{{{gold}}}" if self.adapter is not None else "not an answer"
            groups.append([GenOut(text=text, n_tokens=2, finish_reason="stop") for _ in range(n)])
        return groups


def _checkpoint_tree(root: Path) -> None:
    for step in (4, 2):
        path = root / f"checkpoint-{step}"
        path.mkdir()
        (path / "adapter_config.json").write_text("{}\n")
    decoy = root / "checkpoint-x"
    decoy.mkdir()
    (decoy / "adapter_config.json").write_text("{}\n")
    (root / "checkpoint-6").mkdir()


def _config(out: Path) -> GRPOConfig:
    return GRPOConfig.from_dict(
        {
            "model_id": "fake",
            "steps": 4,
            "prompts_per_step": 2,
            "out_dir": str(out),
            "seed": 0,
            "eval_every": 2,
            "eval_heldout_n": 3,
            "eval_train_n": 2,
            "eval_samples_per_prompt": 2,
            "save_every": 2,
        }
    )


def test_eval_log_rows_uses_the_mlx_schema() -> None:
    stats = {key: None for key in EVAL_ROW_KEYS}
    stats["oracle_acc"] = 0.5
    stats["n_prompts"] = 2
    stats["per_prompt"] = [
        {"pid": "a", "oracle_acc": 0.0},
        {"pid": "b", "oracle_acc": 1.0},
    ]
    row, prompts = eval_log_rows(4, "train", stats)
    assert set(row) == {"step", "split", *EVAL_ROW_KEYS}
    assert row["step"] == 4
    assert row["split"] == "train"
    assert row["oracle_acc"] == 0.5
    assert prompts == {"step": 4, "split": "train", "acc": {"a": 0.0, "b": 1.0}}


def test_peft_checkpoints_keeps_numbered_adapters_only(tmp_path: Path) -> None:
    _checkpoint_tree(tmp_path)
    found = peft_checkpoints(tmp_path)
    assert [(step, path) for step, path in found] == [
        (2, tmp_path / "checkpoint-2"),
        (4, tmp_path / "checkpoint-4"),
    ]


def test_evaluate_checkpoints_writes_each_split_once(tmp_path: Path) -> None:
    _checkpoint_tree(tmp_path)
    task = ChainSumTask()
    train = task.problems("train", 2, seed=0)
    held = task.problems("heldout", 3, seed=0)
    answers: dict[str, str] = {}
    for problem in [*train, *held]:
        prompt = task.prompt(problem)
        assert prompt not in answers
        answers[prompt] = problem.answer
    generator = FakeGenerator(answers)
    written = evaluate_checkpoints(tmp_path, task, _config(tmp_path), generator, train, held)
    assert written == [
        (0, "heldout"),
        (0, "train"),
        (2, "heldout"),
        (2, "train"),
        (4, "heldout"),
        (4, "train"),
    ]
    assert generator.adapters == [
        None,
        str(tmp_path / "checkpoint-2"),
        str(tmp_path / "checkpoint-4"),
    ]
    rows = read_jsonl(tmp_path / "eval.jsonl")
    prompt_rows = read_jsonl(tmp_path / "eval_prompts.jsonl")
    assert [(row["step"], row["split"]) for row in rows] == written
    assert len(prompt_rows) == len(rows)
    held_ids = {problem.pid for problem in held}
    train_ids = {problem.pid for problem in train}
    for row, prompts in zip(rows, prompt_rows, strict=True):
        assert set(EVAL_ROW_KEYS).issubset(row)
        expect = 0.0 if row["step"] == 0 else 1.0
        assert row["oracle_acc"] == pytest.approx(expect)
        assert row["n_prompts"] == (3 if row["split"] == "heldout" else 2)
        assert prompts["step"] == row["step"]
        assert prompts["split"] == row["split"]
        assert set(prompts["acc"]) == (held_ids if row["split"] == "heldout" else train_ids)
        assert all(value == pytest.approx(expect) for value in prompts["acc"].values())

    eval_text = (tmp_path / "eval.jsonl").read_text()
    prompt_text = (tmp_path / "eval_prompts.jsonl").read_text()
    adapters = list(generator.adapters)
    assert evaluate_checkpoints(tmp_path, task, _config(tmp_path), generator, train, held) == []
    assert (tmp_path / "eval.jsonl").read_text() == eval_text
    assert (tmp_path / "eval_prompts.jsonl").read_text() == prompt_text
    assert generator.adapters == adapters


def test_evaluate_checkpoints_left_truncates_prompts(tmp_path: Path) -> None:
    _checkpoint_tree(tmp_path)
    task = ChainSumTask()
    train = task.problems("train", 2, seed=0)
    held = task.problems("heldout", 3, seed=0)
    answers = {task.prompt(problem): problem.answer for problem in [*train, *held]}

    class _Limited(FakeGenerator):
        def __init__(self, mapping: dict[str, str]) -> None:
            super().__init__(mapping)
            self.seen: list[int | None] = []

        def generate(
            self, prompts, n, *, max_tokens, temperature=1.0, top_p=1.0, seed=0, system_prompt=None
        ):
            self.seen.append(getattr(self, "max_prompt_tokens", None))
            return super().generate(
                prompts,
                n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                system_prompt=system_prompt,
            )

    generator = _Limited(answers)
    config = _config(tmp_path)
    evaluate_checkpoints(tmp_path, task, config, generator, train, held)
    assert generator.seen
    assert set(generator.seen) == {config.max_prompt_length}
    assert not hasattr(generator, "max_prompt_tokens")
