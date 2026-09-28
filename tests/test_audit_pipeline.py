"""Resumable step-0 audit: sampling, grading, arm resolution, and run_audit."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from vfates.audit import (
    apply_arms,
    count_completions,
    grade_completions,
    load_audit,
    load_completions,
    load_pools,
    resolve_arms,
    run_audit,
    sample_completions,
)
from vfates.gen.base import Generator
from vfates.tasks.registry import build_task
from vfates.types import Completion, GenOut
from vfates.utils.io import read_json, read_jsonl, read_yaml, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
TASK_SPEC = {
    "type": "chain_sum",
    "min_terms": 2,
    "max_terms": 2,
    "min_digits": 1,
    "max_digits": 1,
    "min_decimal_places": 1,
    "max_decimal_places": 1,
}


def _wrong(gold: str) -> str:
    return "1" if Decimal(gold) == 0 else "0"


class FakeGenerator(Generator):
    def __init__(self, answers: dict[str, str], calls: dict[str, int]) -> None:
        self.model_id = "fake"
        self.answers = answers
        self.calls = calls
        self.batches: list[dict[str, float | int]] = []

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
        del seed, system_prompt
        self.calls["generate"] += 1
        call = self.calls["generate"]
        self.batches.append(
            {
                "n": n,
                "max_tokens": max_tokens,
                "prompts": len(prompts),
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        groups: list[list[GenOut]] = []
        for prompt in prompts:
            gold = self.answers[prompt]
            wrong = _wrong(gold)
            group = []
            for k in range(n):
                value = gold if k % 2 == 0 else wrong
                group.append(
                    GenOut(
                        text=f"c{call} work \\boxed{{{value}}}",
                        n_tokens=4,
                        logprob=-0.25,
                        finish_reason="stop",
                    )
                )
            groups.append(group)
        return groups


def test_sample_completions_resumes_without_duplicating(tmp_path: Path) -> None:
    task = build_task(TASK_SPEC)
    problems = task.problems("train", 4, seed=0)
    answers = {task.prompt(problem): problem.answer for problem in problems}
    assert len(answers) == len(problems)
    calls = {"generate": 0}
    generator = FakeGenerator(answers, calls)
    path = tmp_path / "completions.jsonl"
    n = 4
    first = sample_completions(
        generator, task, problems, path, n=n, max_tokens=16, chunk_prompts=2, seed=0
    )
    assert calls["generate"] == 2
    assert [(item.pid, item.k) for item in first] == [
        (problem.pid, k) for problem in problems for k in range(n)
    ]
    for problem in problems:
        rows = [item for item in first if item.pid == problem.pid]
        for item in rows:
            boxed = f"\\boxed{{{problem.answer}}}"
            if item.k % 2 == 0:
                assert boxed in item.text
            else:
                assert boxed not in item.text
    assert count_completions(path) == {problem.pid: n for problem in problems}

    second = sample_completions(
        generator, task, problems, path, n=n, max_tokens=16, chunk_prompts=2, seed=0
    )
    assert calls["generate"] == 2
    assert [(item.pid, item.k, item.text) for item in second] == [
        (item.pid, item.k, item.text) for item in first
    ]
    disk = read_jsonl(path)
    assert len(disk) == len(problems) * n
    assert len({(row["pid"], row["k"]) for row in disk}) == len(disk)
    assert count_completions(path) == {problem.pid: n for problem in problems}
    loaded = load_completions(path, problems, n)
    assert [
        (item.pid, item.k, item.text, item.n_tokens, item.logprob, item.finish_reason)
        for item in loaded
    ] == [
        (item.pid, item.k, item.text, item.n_tokens, item.logprob, item.finish_reason)
        for item in first
    ]

    dropped = problems[-1]
    write_jsonl(path, [row for row in disk if row["pid"] != dropped.pid])
    resumed = sample_completions(
        generator, task, problems, path, n=n, max_tokens=16, chunk_prompts=2, seed=0
    )
    assert calls["generate"] == 3
    assert count_completions(path) == {problem.pid: n for problem in problems}
    assert len(read_jsonl(path)) == len(problems) * n
    for problem in problems[:-1]:
        assert [item.text for item in resumed if item.pid == problem.pid] == [
            item.text for item in first if item.pid == problem.pid
        ]
    assert [item.text for item in resumed if item.pid == dropped.pid] != [
        item.text for item in first if item.pid == dropped.pid
    ]

    partial = problems[0]
    rows = read_jsonl(path)
    partial_rows = [row for row in rows if row["pid"] == partial.pid]
    others = [row for row in rows if row["pid"] != partial.pid]
    stale_text = partial_rows[0]["text"]
    write_jsonl(path, [*others, partial_rows[0]])
    sample_completions(generator, task, problems, path, n=n, max_tokens=16, chunk_prompts=2, seed=0)
    assert calls["generate"] == 4
    final_rows = read_jsonl(path)
    assert len(final_rows) == len(problems) * n
    partial_final = [row for row in final_rows if row["pid"] == partial.pid]
    assert sorted(row["k"] for row in partial_final) == list(range(n))
    assert all(row["text"] != stale_text for row in partial_final)
    reloaded = load_completions(path, problems, n)
    assert [item.text for item in reloaded if item.pid != partial.pid] == [
        item.text for item in resumed if item.pid != partial.pid
    ]


def test_grade_completions_matches_the_oracle() -> None:
    task = build_task(TASK_SPEC)
    problems = task.problems("train", 3, seed=1)
    by_id = {problem.pid: problem for problem in problems}
    completions: list[Completion] = []
    expected: list[int] = []
    for problem in problems:
        for k, value in enumerate((problem.answer, _wrong(problem.answer))):
            text = f"work \\boxed{{{value}}}"
            completion = Completion(pid=problem.pid, k=k, text=text, n_tokens=4)
            completions.append(completion)
            parsed = task.parse(problem, text)
            expected.append(task.oracle(problem, parsed, text))
    parsed_list, labels = grade_completions(task, by_id, completions)
    assert labels.tolist() == expected
    assert expected == [1, 0] * len(problems)
    for parsed, completion, bit in zip(parsed_list, completions, labels.tolist(), strict=True):
        again, oracle_bit = task.grade(by_id[completion.pid], completion)
        assert parsed == again
        assert bit == oracle_bit


def test_resolve_arms_merges_file_with_overrides() -> None:
    path = ROOT / "configs" / "arms" / "chain_sum.yaml"
    raw = read_yaml(path)
    merged = resolve_arms({"file": str(path)})
    assert "file" not in merged
    assert merged["targets"] == raw["targets"]
    assert merged["arms"] == raw["arms"]
    noise = next(arm for arm in merged["arms"] if arm["name"] == "noise_fp")
    assert noise["match"] == "fp_tolerance"
    assert noise["support"] == "answered"

    overridden = resolve_arms(
        {
            "file": str(path),
            "targets": {"fpr": 0.2},
            "n_boot": 5,
            "arms": [{"name": "oracle", "type": "oracle"}],
        }
    )
    assert "file" not in overridden
    assert overridden["targets"] == {"fpr": 0.2}
    assert overridden["n_boot"] == 5
    assert overridden["arms"] == [{"name": "oracle", "type": "oracle"}]

    bare = {"targets": {"fpr": 0.1}, "arms": [{"name": "oracle", "type": "oracle"}]}
    assert resolve_arms(bare) == bare


def test_run_audit_without_features_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pools = {"train": 12, "heldout": 6, "seed": 0}
    task = build_task(TASK_SPEC)
    train, held = load_pools(task, pools)
    answers = {task.prompt(problem): problem.answer for problem in [*train, *held]}
    assert len(answers) == len(train) + len(held)
    calls = {"generate": 0, "make": 0}
    seen: dict[str, object] = {}
    made: list[FakeGenerator] = []

    def factory(backend: str, model_id: str, **kwargs: object) -> FakeGenerator:
        calls["make"] += 1
        seen["args"] = (backend, model_id, kwargs)
        generator = FakeGenerator(answers, calls)
        made.append(generator)
        return generator

    monkeypatch.setattr("vfates.gen.factory.make_generator", factory)
    config = {
        "task": dict(TASK_SPEC),
        "pools": pools,
        "generator": {"backend": "hf", "model_id": "fake", "kwargs": {}},
        "sampling": {
            "n": 4,
            "max_tokens": 32,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 1,
            "chunk_prompts": 6,
        },
        "arms": {
            "targets": {"fpr": 0.15},
            "n_boot": 5,
            "seed": 0,
            "arms": [
                {"name": "oracle", "type": "oracle"},
                {"name": "fp_hash", "type": "hash_fp", "salt": "chain_sum/fp_hash"},
                {"name": "fp_tolerance", "type": "tolerance_fp"},
                {
                    "name": "noise_fp",
                    "type": "resampled",
                    "match": "fp_tolerance",
                    "support": "answered",
                    "seed": 0,
                },
            ],
        },
    }
    out = tmp_path / "audit"
    run = run_audit(config, out)
    assert calls["make"] == 1
    assert calls["generate"] == 3
    assert seen["args"] == ("hf", "fake", {})
    assert (
        made[0].batches
        == [{"n": 4, "max_tokens": 32, "prompts": 6, "temperature": 0.7, "top_p": 0.9}] * 3
    )

    for name in (
        "problems.jsonl",
        "completions.jsonl",
        "calibration.json",
        "audit.npz",
        "confusion.json",
    ):
        assert (out / name).is_file()
    assert not (out / "features.npz").exists()
    assert not (out / "forecasts.json").exists()
    assert not (out / "lock.json").exists()
    assert len(read_jsonl(out / "problems.jsonl")) == 18
    assert len(read_jsonl(out / "completions.jsonl")) == 72
    calibration = read_json(out / "calibration.json")
    assert calibration["report"]["n_prompts"] == pools["train"]
    assert calibration["report"]["n_rows"] == pools["train"] * 4
    confusion = read_json(out / "confusion.json")
    assert set(confusion) == {"oracle", "fp_hash", "fp_tolerance", "noise_fp"}
    assert set(run.table.O.tolist()) == {0, 1}
    assert int((run.table.split == "train").sum()) == 48
    assert np.array_equal(run.table.V["oracle"], run.table.O)

    rebuilt = load_audit(out)
    assert set(rebuilt.arms) == set(run.arms)
    for name, arm in run.arms.items():
        assert rebuilt.arms[name].describe() == arm.describe()
    verdicts = apply_arms(
        rebuilt.arms, rebuilt.problems, rebuilt.completions, rebuilt.parsed, rebuilt.table.O
    )
    for name, verdict in verdicts.items():
        assert np.array_equal(verdict, rebuilt.table.V[name])
        assert np.array_equal(rebuilt.table.V[name], run.table.V[name])

    completions_text = (out / "completions.jsonl").read_text()
    generate_before = calls["generate"]
    run_audit(config, out)
    assert calls["generate"] == generate_before
    assert calls["make"] == 1
    assert (out / "completions.jsonl").read_text() == completions_text


def test_sample_completions_smaller_n_is_not_a_prefix(tmp_path: Path) -> None:
    task = build_task(TASK_SPEC)
    problems = task.problems("train", 3, seed=2)
    answers = {task.prompt(problem): problem.answer for problem in problems}
    calls = {"generate": 0}
    generator = FakeGenerator(answers, calls)
    path = tmp_path / "completions.jsonl"
    first = sample_completions(generator, task, problems, path, n=4, max_tokens=16, seed=0)
    assert calls["generate"] == 1
    second = sample_completions(generator, task, problems, path, n=2, max_tokens=16, seed=0)
    assert calls["generate"] == 2
    assert [item.k for item in second] == [k for _problem in problems for k in (0, 1)]
    old_prefix = {(item.pid, item.k): item.text for item in first if item.k < 2}
    new_map = {(item.pid, item.k): item.text for item in second}
    assert new_map != old_prefix
    assert len(read_jsonl(path)) == len(problems) * 2
    with pytest.raises(ValueError, match="k=0"):
        load_completions(path, problems, 4)


class ShortGenerator(FakeGenerator):
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
        groups = super().generate(
            prompts,
            n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            system_prompt=system_prompt,
        )
        return [group[:-1] for group in groups]


def test_sample_completions_rejects_a_short_group(tmp_path: Path) -> None:
    task = build_task(TASK_SPEC)
    problems = task.problems("train", 2, seed=3)
    answers = {task.prompt(problem): problem.answer for problem in problems}
    calls = {"generate": 0}
    path = tmp_path / "completions.jsonl"
    with pytest.raises(ValueError, match="expected 4"):
        sample_completions(
            ShortGenerator(answers, calls), task, problems, path, n=4, max_tokens=16, seed=0
        )
    assert read_jsonl(path) == []


def test_generator_chat_template_kwargs_override_nested_kwargs() -> None:
    from vfates.audit import _generator_kwargs

    assert _generator_kwargs({"kwargs": {}}) == {}
    forwarded = _generator_kwargs(
        {
            "kwargs": {"chat_template_kwargs": {"a": 1}, "temperature": 0.0},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    assert forwarded["chat_template_kwargs"] == {"enable_thinking": False}
    assert forwarded["temperature"] == 0.0
