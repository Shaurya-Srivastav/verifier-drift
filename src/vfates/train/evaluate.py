"""Held-out and train-pool evaluation of a frozen policy.

``oracle_acc`` is the paper's primary endpoint: the mean over prompts of each
prompt's mean oracle bit (not a pooled mean, though the two agree when every
prompt has the same number of samples). Verifier rates, when requested, are
pooled with :func:`vfates.train.rewards.onpolicy_rates`. Draw keys are
``(seed, "eval", pid, k)`` so an eval pass does not reuse a training draw.

When the oracle records a content-judge verdict (IFEval writes ``content_judge``
on the parse), the pass also keeps per-sample prompt, text and that verdict
for ``eval_samples.jsonl``. ``eval.jsonl`` and ``eval_prompts.jsonl`` stay the
same.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vfates.gen.base import Generator
from vfates.tasks.base import Task
from vfates.train.rewards import MonitorFn, monitor_conditionals, onpolicy_rates
from vfates.types import Problem
from vfates.verifiers.base import Verifier


def evaluate_policy(
    generator: Generator,
    task: Task,
    problems: list[Problem],
    *,
    n_samples: int,
    max_tokens: int,
    temperature: float,
    seed: int,
    verifier: Verifier | None = None,
    monitors: dict[str, MonitorFn] | None = None,
    return_records: bool = False,
) -> dict[str, Any]:
    """Sample ``n_samples`` completions per problem and aggregate oracle accuracy.

    ``top_p`` is 1 so eval matches the on-policy audits unless the caller changed
    the generator. Truncation is ``finish_reason == "length"``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if not problems:
        return _empty_eval(return_records)

    prompts = [task.prompt(problem) for problem in problems]
    grouped = generator.generate(
        prompts,
        n_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        seed=seed,
        system_prompt=task.system_prompt(),
    )
    if len(grouped) != len(problems):
        raise RuntimeError(
            f"generator returned {len(grouped)} prompt groups for {len(problems)} problems"
        )

    records: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    per_prompt: list[dict[str, Any]] = []
    by_split: dict[str, list[float]] = defaultdict(list)

    for problem, outs in zip(problems, grouped):
        if len(outs) != n_samples:
            raise RuntimeError(
                f"prompt {problem.pid} returned {len(outs)} samples, expected {n_samples}"
            )
        prompt_rows: list[dict[str, Any]] = []
        for k, out in enumerate(outs):
            parsed = task.parse(problem, out.text)
            o = int(task.oracle(problem, parsed, out.text))
            row: dict[str, Any] = {
                "pid": problem.pid,
                "split": problem.split,
                "k": k,
                "O": o,
                "V": None,
                "FP": None,
                "FN": None,
                "answer": parsed.answer,
                "has_answer": parsed.answer is not None,
                "n_tokens": int(out.n_tokens),
                "finish_reason": out.finish_reason,
                "truncated": out.finish_reason == "length",
                "monitors": {},
            }
            if monitors:
                row["monitors"] = {
                    name: bool(fn(problem, parsed, out.text)) for name, fn in monitors.items()
                }
            if verifier is not None:
                key = (seed, "eval", problem.pid, k)
                v = int(verifier.verdict(problem, parsed, out.text, o, key))
                row["V"] = v
                row["FP"] = (1 - o) * v
                row["FN"] = o * (1 - v)
            if return_records:
                row["text"] = out.text
            if "content_judge" in parsed.extras:
                raw_bit = parsed.extras.get("content_judge")
                judge_rows.append(
                    {
                        "pid": problem.pid,
                        "k": k,
                        "prompt": task.prompt(problem),
                        "text": out.text,
                        "judge": None if raw_bit is None else int(raw_bit),
                        "judge_parsed": bool(parsed.extras.get("content_judge_parsed", True)),
                        "O": o,
                    }
                )
            prompt_rows.append(row)
        records.extend(prompt_rows)
        o_mean = sum(int(r["O"]) for r in prompt_rows) / len(prompt_rows)
        by_split[problem.split].append(o_mean)
        lengths = [int(r["n_tokens"]) for r in prompt_rows]
        trunc = [1.0 if r["truncated"] else 0.0 for r in prompt_rows]
        per_prompt.append(
            {
                "pid": problem.pid,
                "split": problem.split,
                "oracle_acc": o_mean,
                "n": len(prompt_rows),
                "mean_length": sum(lengths) / len(lengths),
                "truncation_rate": sum(trunc) / len(trunc),
            }
        )

    oracle_acc = sum(p["oracle_acc"] for p in per_prompt) / len(per_prompt)
    lengths_all = [int(r["n_tokens"]) for r in records]
    trunc_all = [1.0 if r["truncated"] else 0.0 for r in records]
    result: dict[str, Any] = {
        "oracle_acc": oracle_acc,
        "n_prompts": len(per_prompt),
        "n_samples": n_samples,
        "mean_length": sum(lengths_all) / len(lengths_all),
        "truncation_rate": sum(trunc_all) / len(trunc_all),
        "splits": {
            split: {"oracle_acc": sum(vals) / len(vals), "n_prompts": len(vals)}
            for split, vals in by_split.items()
        },
        "per_prompt": per_prompt,
        "accept_rate": None,
        "fpr": None,
        "fnr": None,
        "precision": None,
        "J": None,
        "monitors": {},
        "monitors_wrong": {},
        "monitors_correct": {},
    }
    # Conditionals do not need a verifier. V stays None on the oracle-only path.
    conditionals = monitor_conditionals(records)
    result["monitors"] = conditionals["monitors"]
    result["monitors_wrong"] = conditionals["monitors_wrong"]
    result["monitors_correct"] = conditionals["monitors_correct"]
    if verifier is not None:
        rates = onpolicy_rates(records)
        result["accept_rate"] = rates["accept_rate"]
        result["fpr"] = rates["fpr"]
        result["fnr"] = rates["fnr"]
        result["precision"] = rates["precision"]
        result["J"] = rates["J"]
    if return_records:
        result["records"] = records
    if judge_rows:
        result["judge_samples"] = judge_rows
    return result


EVAL_SAMPLES_FILENAME = "eval_samples.jsonl"

EVAL_ROW_KEYS: tuple[str, ...] = (
    "oracle_acc",
    "n_prompts",
    "accept_rate",
    "fpr",
    "fnr",
    "precision",
    "J",
    "monitors",
    "monitors_wrong",
    "monitors_correct",
    "n_samples",
    "mean_length",
    "truncation_rate",
)


def eval_log_rows(step: int, split: str, stats: dict[str, Any]) -> tuple[dict, dict]:
    """The ``eval.jsonl`` and ``eval_prompts.jsonl`` rows for one :func:`evaluate_policy` result."""
    row = {"step": int(step), "split": split, **{key: stats[key] for key in EVAL_ROW_KEYS}}
    prompts = {
        "step": int(step),
        "split": split,
        "acc": {p["pid"]: p["oracle_acc"] for p in stats["per_prompt"]},
    }
    return row, prompts


def eval_sample_rows(step: int, split: str, stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-sample rows for :data:`EVAL_SAMPLES_FILENAME`.

    Empty unless this pass recorded content-judge verdicts. Callers that have
    no judge keep the previous eval files and do not grow a sample log.
    """
    rows: list[dict[str, Any]] = []
    for sample in stats.get("judge_samples") or []:
        rows.append(
            {
                "step": int(step),
                "split": split,
                "pid": sample["pid"],
                "k": int(sample["k"]),
                "prompt": sample["prompt"],
                "text": sample["text"],
                "judge": sample["judge"],
                "judge_parsed": bool(sample.get("judge_parsed", True)),
                "O": sample.get("O"),
            }
        )
    return rows


def append_eval_samples(
    run_dir: str | Path, step: int, split: str, stats: Mapping[str, Any]
) -> None:
    """Append :func:`eval_sample_rows` for one split.

    One write and one fsync for the split. A crash can tear the last line;
    :func:`vfates.utils.io.read_jsonl` drops that tail. Write this before
    ``eval.jsonl`` so a rerun of an unfinished split replaces the rows.
    """
    rows = eval_sample_rows(step, split, stats)
    if not rows:
        return
    path = Path(run_dir) / EVAL_SAMPLES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())


def _empty_eval(return_records: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "oracle_acc": None,
        "n_prompts": 0,
        "n_samples": 0,
        "mean_length": None,
        "truncation_rate": None,
        "splits": {},
        "per_prompt": [],
        "accept_rate": None,
        "fpr": None,
        "fnr": None,
        "precision": None,
        "J": None,
        "monitors": {},
        "monitors_wrong": {},
        "monitors_correct": {},
    }
    if return_records:
        out["records"] = []
    return out
