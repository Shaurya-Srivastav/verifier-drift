"""Held-out and train-pool evaluation of a frozen policy.

``oracle_acc`` is the paper's primary endpoint: the mean over prompts of each
prompt's mean oracle bit (not a pooled mean, though the two agree when every
prompt has the same number of samples). Verifier rates, when requested, are
pooled with :func:`vfates.train.rewards.onpolicy_rates`. Draw keys are
``(seed, "eval", pid, k)`` so an eval pass does not reuse a training draw.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from vfates.gen.base import Generator
from vfates.tasks.base import Task
from vfates.train.rewards import MonitorFn, onpolicy_rates
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

    # A verifier is optional: oracle-only eval still builds a context with a dummy
    # that is never asked to train. When verifier is None we skip V entirely.
    records: list[dict[str, Any]] = []
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
    }
    if verifier is not None:
        # onpolicy_rates requires V to be an int. Fill from the rows we just scored.
        rates = onpolicy_rates(records)
        result["accept_rate"] = rates["accept_rate"]
        result["fpr"] = rates["fpr"]
        result["fnr"] = rates["fnr"]
        result["precision"] = rates["precision"]
        result["J"] = rates["J"]
        result["monitors"] = rates["monitors"]
        result["monitors_wrong"] = rates["monitors_wrong"]
    elif monitors:
        # Oracle-only path still reports monitor rates. Synthesize V=0 so the
        # pooled helper's monitor means are defined; do not report verifier rates.
        dummy = []
        for row in records:
            dummy.append({**row, "V": 0, "FP": 0, "FN": int(row["O"])})
        rates = onpolicy_rates(dummy)
        result["monitors"] = rates["monitors"]
        result["monitors_wrong"] = rates["monitors_wrong"]
    if return_records:
        result["records"] = records
    return result


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
    }
    if return_records:
        out["records"] = []
    return out
