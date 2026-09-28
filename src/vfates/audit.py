"""Step-0 audit: sample the base policy, grade, fit arms, featurise, forecast, lock.

Artifacts (all under one ``out_dir``, every stage resumable):

``problems.jsonl``       frozen train and held-out pools.
``completions.jsonl``    ``n`` base-policy samples per problem.
``calibration.json``     arms fitted on the *train* split only, with rates and CIs.
``audit.npz``            :class:`~vfates.types.AuditTable` with every arm's verdicts.
``confusion.json``       step-0 confusion per arm and split.
``features.npz``         one feature row per audit row (same order).
``forecasts.json``       per-arm screen: transfer precision and the baselines.
``lock.json``            hash-locked forecasts, written before any training run.

Held-out verdicts are recorded for description only; nothing is fitted on them.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vfates.forecast.confusion import (
    confusion,
    confusion_with_ci,
    mean_precision_forecast,
    modal_precision_forecast,
)
from vfates.forecast.lockbox import lock_forecasts
from vfates.forecast.transfer import DEFAULT_QS, PRIMARY_Q, transfer_forecast
from vfates.gen.base import Generator
from vfates.tasks.base import Task
from vfates.tasks.registry import build_task
from vfates.types import SPLITS, AuditTable, Completion, Parsed, Problem
from vfates.utils.env import environment_report
from vfates.utils.hashing import stable_seed
from vfates.utils.io import (
    append_jsonl,
    read_json,
    read_jsonl,
    read_yaml,
    write_json,
    write_jsonl,
)
from vfates.verifiers.base import Verifier
from vfates.verifiers.calibrate import CalibrationData, calibrate_arms
from vfates.verifiers.registry import build_verifier

FEATURE_KINDS = ("lora_grad", "hidden")


@dataclass
class AuditRun:
    """In-memory view of a finished (or resumed) audit directory."""

    out_dir: Path
    task: Task
    problems: dict[str, Problem]
    completions: list[Completion]
    parsed: list[Parsed]
    table: AuditTable
    arms: dict[str, Verifier]


def load_pools(task: Task, pools: Mapping[str, Any]) -> tuple[list[Problem], list[Problem]]:
    """Frozen ``(train, heldout)`` pools from ``{'train': n, 'heldout': m, 'seed': s}``."""
    seed = int(pools.get("seed", 0))
    return (
        task.problems("train", int(pools["train"]), seed),
        task.problems("heldout", int(pools["heldout"]), seed),
    )


def sample_completions(
    generator: Generator,
    task: Task,
    problems: Sequence[Problem],
    path: str | Path,
    *,
    n: int,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    chunk_prompts: int = 64,
    log_every: int = 1,
) -> list[Completion]:
    """``n`` samples per problem, appended to ``path`` one chunk at a time.

    Chunk boundaries and seeds depend only on the problem order and
    ``chunk_prompts``, so an interrupted run resumes to the same samples.
    """
    path = Path(path)
    done: dict[str, list[Completion]] = {}
    if path.exists():
        for row in read_jsonl(path):
            c = Completion.from_json(row)
            done.setdefault(c.pid, []).append(c)
    todo = [p for p in problems if len(done.get(p.pid, [])) < n]
    stale = {p.pid for p in todo if p.pid in done}
    if stale:
        kept = [c for p in problems if p.pid in done and p.pid not in stale for c in done[p.pid]]
        write_jsonl(path, [c.to_json() for c in kept])
        for pid in stale:
            done.pop(pid)
    order = {p.pid: i for i, p in enumerate(problems)}
    system = task.system_prompt()
    started = time.perf_counter()
    tokens = 0
    for chunk_index, lo in enumerate(range(0, len(problems), chunk_prompts)):
        chunk = [p for p in problems[lo : lo + chunk_prompts] if p.pid not in done]
        if not chunk:
            continue
        outs = generator.generate(
            [task.prompt(p) for p in chunk],
            n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=stable_seed(seed, "audit", chunk_index),
            system_prompt=system,
        )
        for problem, group in zip(chunk, outs, strict=True):
            rows = [
                Completion(
                    pid=problem.pid,
                    k=k,
                    text=out.text,
                    n_tokens=out.n_tokens,
                    logprob=out.logprob,
                    finish_reason=out.finish_reason,
                )
                for k, out in enumerate(group)
            ]
            for c in rows:
                append_jsonl(path, c.to_json())
                tokens += c.n_tokens
            done[problem.pid] = rows
        if log_every and (chunk_index % log_every == 0):
            elapsed = time.perf_counter() - started
            print(
                f"[audit] chunk {chunk_index}: {len(done)}/{len(problems)} prompts, "
                f"{tokens / max(elapsed, 1e-9):.0f} tok/s",
                flush=True,
            )
    return [c for p in sorted(problems, key=lambda q: order[q.pid]) for c in done[p.pid][:n]]


def grade_completions(
    task: Task, problems: Mapping[str, Problem], completions: Sequence[Completion]
) -> tuple[list[Parsed], np.ndarray]:
    """Parse and oracle-grade every completion."""
    parsed: list[Parsed] = []
    labels = np.empty(len(completions), dtype=np.int64)
    for i, c in enumerate(completions):
        p, o = task.grade(problems[c.pid], c)
        parsed.append(p)
        labels[i] = o
    return parsed, labels


def apply_arms(
    arms: Mapping[str, Verifier],
    problems: Mapping[str, Problem],
    completions: Sequence[Completion],
    parsed: Sequence[Parsed],
    O: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every arm's verdict on every row; stochastic arms draw with key ``(pid, k)``."""
    out: dict[str, np.ndarray] = {}
    for name, arm in arms.items():
        v = np.empty(len(completions), dtype=np.int64)
        for i, (c, p) in enumerate(zip(completions, parsed, strict=True)):
            key = (c.pid, c.k) if arm.stochastic else None
            v[i] = arm(problems[c.pid], p, c.text, int(O[i]), key)
        out[name] = v
    return out


def build_table(
    problems: Mapping[str, Problem],
    completions: Sequence[Completion],
    parsed: Sequence[Parsed],
    O: np.ndarray,
    V: Mapping[str, np.ndarray],
    meta: Mapping[str, Any] | None = None,
) -> AuditTable:
    return AuditTable(
        pid=np.array([c.pid for c in completions], dtype=object),
        split=np.array([problems[c.pid].split for c in completions], dtype=object),
        k=np.array([c.k for c in completions], dtype=np.int64),
        O=np.asarray(O, dtype=np.int64),
        answer=np.array([p.answer for p in parsed], dtype=object),
        n_tokens=np.array([c.n_tokens for c in completions], dtype=np.int64),
        logprob=np.array(
            [np.nan if c.logprob is None else c.logprob for c in completions], dtype=np.float64
        ),
        V={name: np.asarray(v, dtype=np.int64) for name, v in V.items()},
        meta=dict(meta or {}),
    )


def confusion_report(table: AuditTable, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Per-arm, per-split confusion with prompt-clustered bootstrap intervals."""
    out: dict[str, Any] = {}
    for name, v in table.V.items():
        out[name] = {}
        for split in SPLITS:
            m = table.split == split
            if not m.any():
                continue
            out[name][split] = {
                "point": confusion(table.O[m], v[m], table.pid[m]),
                "ci": confusion_with_ci(table.O[m], v[m], table.pid[m], n_boot=n_boot, seed=seed),
            }
    return out


def extract_features(
    spec: Mapping[str, Any],
    task: Task,
    problems: Mapping[str, Problem],
    completions: Sequence[Completion],
    out_dir: str | Path,
    *,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Feature rows aligned with ``completions``, cached in resumable parts.

    ``spec``: ``kind`` (``lora_grad`` | ``hidden``), ``model_id``, and the
    keyword options of the chosen extractor.
    """
    kind = str(spec["kind"])
    if kind not in FEATURE_KINDS:
        raise ValueError(f"unknown feature kind {kind!r}; expected one of {FEATURE_KINDS}")
    options = {k: v for k, v in spec.items() if k not in ("kind", "model_id", "load")}
    parts_dir = Path(out_dir) / "feature_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    prompts = [task.prompt(problems[c.pid]) for c in completions]
    texts = [c.text for c in completions]
    system = task.system_prompt()
    model = tokenizer = None
    meta: dict[str, Any] = {"kind": kind, "model_id": spec["model_id"], "options": options}
    feats, lengths = [], []
    for part, lo in enumerate(range(0, len(completions), chunk)):
        path = parts_dir / f"part_{part:05d}.npz"
        hi = min(lo + chunk, len(completions))
        if path.exists():
            z = np.load(path)
            if int(z["lo"]) == lo and int(z["hi"]) == hi:
                feats.append(z["features"])
                lengths.append(z["lengths"])
                continue
        if model is None:
            from vfates.features.hidden import load_model

            load = dict(spec.get("load", {}))
            model, tokenizer = load_model(str(spec["model_id"]), **load)
        started = time.perf_counter()
        if kind == "lora_grad":
            from vfates.features.lora_grad import lora_grad_features

            res = lora_grad_features(
                model, tokenizer, prompts[lo:hi], texts[lo:hi], system_prompt=system, **options
            )
            f, ln = res.features, res.lengths
            meta["extractor"] = {k: v for k, v in res.meta.items() if k != "sample_seconds"}
            # The extractor wraps the base model with PEFT in place and refuses a
            # wrapped model, so each part starts from a fresh load (same lora_seed).
            del res
            model = tokenizer = None
            _release_accelerators()
        else:
            from vfates.features.hidden import hidden_features

            f = hidden_features(
                model, tokenizer, prompts[lo:hi], texts[lo:hi], system_prompt=system, **options
            )
            ln = np.array([c.n_tokens for c in completions[lo:hi]], dtype=np.int64)
        np.savez(path, features=f.astype(np.float32), lengths=ln, lo=lo, hi=hi)
        feats.append(f.astype(np.float32))
        lengths.append(np.asarray(ln))
        print(
            f"[features] {kind} rows {lo}-{hi} of {len(completions)} in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return np.concatenate(feats), np.concatenate(lengths), meta


def forecast_arms(
    table: AuditTable,
    features: np.ndarray,
    lengths: np.ndarray,
    arms: Sequence[str],
    *,
    q: float = PRIMARY_Q,
    qs: Sequence[float] = DEFAULT_QS,
    n_boot: int = 500,
    reliability_splits: int = 20,
    divide_by_length: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """The step-0 screen and its baselines for every arm, from the audit alone."""
    tr, he = table.split == "train", table.split == "heldout"
    out: dict[str, Any] = {}
    for name in arms:
        v = table.V[name]
        fc = transfer_forecast(
            features[tr],
            table.pid[tr],
            v[tr],
            table.O[tr],
            features[he],
            table.pid[he],
            table.O[he],
            qs=qs,
            q=q,
            train_lengths=lengths[tr] if divide_by_length else None,
            reliability_splits=reliability_splits,
            n_boot=n_boot,
            seed=seed,
        )
        step0 = confusion(table.O[tr], v[tr], table.pid[tr])
        logprob = table.logprob[tr]
        out[name] = {
            "transfer": fc.to_json(),
            "step0": step0,
            "mean_precision": mean_precision_forecast(table.O[tr], v[tr], table.pid[tr]),
            "modal_precision": modal_precision_forecast(
                table.O[tr],
                v[tr],
                table.pid[tr],
                table.answer[tr],
                None if np.isnan(logprob).all() else np.nan_to_num(logprob, nan=-np.inf),
            ),
        }
    return out


def _arms_from_calibration(path: Path, task: Task) -> dict[str, Verifier]:
    specs = read_json(path)["arms"]
    return {name: build_verifier(spec, task) for name, spec in specs.items()}


def resolve_arms(arm_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The ``arms`` section with ``file:`` (a YAML with ``targets`` and ``arms``) merged in.

    Keys set next to ``file`` override the file's.
    """
    cfg = dict(arm_cfg)
    path = cfg.pop("file", None)
    if path is None:
        return cfg
    return {**read_yaml(path), **cfg}


def run_audit(config: Mapping[str, Any], out_dir: str | Path) -> AuditRun:
    """Run (or resume) every audit stage in ``config``. Returns the in-memory audit.

    ``config`` keys: ``task``, ``pools``, ``generator`` (``backend``,
    ``model_id``, ``kwargs``), ``sampling`` (``n``, ``max_tokens``,
    ``temperature``, ``top_p``, ``seed``, ``chunk_prompts``), ``arms``
    (``targets`` and ``arms``, a list of specs, or ``file`` naming a YAML with
    both; optional ``n_boot``, ``seed``), optional ``features`` and
    ``forecast``, optional ``lock: true``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {"config": dict(config), "env": environment_report()})
    task = build_task(config["task"])
    train, held = load_pools(task, config["pools"])
    problems = {p.pid: p for p in [*train, *held]}
    if len(problems) != len(train) + len(held):
        raise ValueError("train and held-out pools share problem ids")
    write_jsonl(out / "problems.jsonl", [p.to_json() for p in [*train, *held]])

    samp = dict(config["sampling"])
    completions_path = out / "completions.jsonl"
    need = int(samp["n"])
    have = count_completions(completions_path)
    if any(have.get(p.pid, 0) < need for p in problems.values()):
        from vfates.gen.factory import make_generator

        gen_cfg = dict(config["generator"])
        generator = make_generator(
            gen_cfg["backend"], gen_cfg["model_id"], **dict(gen_cfg.get("kwargs", {}))
        )
        sample_completions(
            generator,
            task,
            [*train, *held],
            completions_path,
            n=need,
            max_tokens=int(samp["max_tokens"]),
            temperature=float(samp.get("temperature", 1.0)),
            top_p=float(samp.get("top_p", 1.0)),
            seed=int(samp.get("seed", 0)),
            chunk_prompts=int(samp.get("chunk_prompts", 64)),
        )
        del generator
        _release_accelerators()
    completions = load_completions(completions_path, [*train, *held], need)
    parsed, O = grade_completions(task, problems, completions)

    calib_path = out / "calibration.json"
    arm_cfg = resolve_arms(config["arms"])
    if calib_path.exists():
        arms = _arms_from_calibration(calib_path, task)
    else:
        train_rows = [i for i, c in enumerate(completions) if problems[c.pid].split == "train"]
        data = CalibrationData.from_grades(
            {p.pid: p for p in train},
            [completions[i] for i in train_rows],
            [parsed[i] for i in train_rows],
            [int(O[i]) for i in train_rows],
        )
        result = calibrate_arms(
            task,
            data,
            list(arm_cfg["arms"]),
            seed=int(arm_cfg.get("seed", 0)),
            n_boot=int(arm_cfg.get("n_boot", 1000)),
            targets=arm_cfg.get("targets"),
        )
        result.save(calib_path)
        arms = result.arms

    V = apply_arms(arms, problems, completions, parsed, O)
    table = build_table(
        problems,
        completions,
        parsed,
        O,
        V,
        meta={"task": dict(config["task"]), "sampling": samp, "generator": config["generator"]},
    )
    table.save(out / "audit.npz")
    write_json(
        out / "confusion.json", confusion_report(table, n_boot=int(arm_cfg.get("n_boot", 1000)))
    )
    run = AuditRun(out, task, problems, completions, parsed, table, dict(arms))

    feat_cfg = config.get("features")
    if feat_cfg:
        feats, lengths, meta = extract_features(feat_cfg, task, problems, completions, out)
        np.savez(out / "features.npz", features=feats, lengths=lengths)
        write_json(out / "features.meta.json", meta)
        fc_cfg = dict(config.get("forecast", {}))
        forecasts = forecast_arms(
            table,
            feats,
            lengths,
            list(arms),
            q=float(fc_cfg.get("q", PRIMARY_Q)),
            qs=tuple(float(x) for x in fc_cfg.get("qs", DEFAULT_QS)),
            n_boot=int(fc_cfg.get("n_boot", 500)),
            reliability_splits=int(fc_cfg.get("reliability_splits", 20)),
            divide_by_length=bool(fc_cfg.get("divide_by_length", False)),
            seed=int(fc_cfg.get("seed", 0)),
        )
        write_json(out / "forecasts.json", forecasts)
        if config.get("lock", False) and not (out / "lock.json").exists():
            lock_forecasts(
                forecasts,
                out / "lock.json",
                config=dict(config),
                note="Step-0 forecasts locked before any training run.",
            )
    return run


def load_audit(out_dir: str | Path) -> AuditRun:
    """Reload a finished audit directory without sampling or refitting."""
    out = Path(out_dir)
    config = read_json(out / "config.json")["config"]
    task = build_task(config["task"])
    problems_list = [Problem.from_json(r) for r in read_jsonl(out / "problems.jsonl")]
    problems = {p.pid: p for p in problems_list}
    completions = load_completions(
        out / "completions.jsonl", problems_list, int(config["sampling"]["n"])
    )
    parsed, _ = grade_completions(task, problems, completions)
    table = AuditTable.load(out / "audit.npz")
    arms = _arms_from_calibration(out / "calibration.json", task)
    return AuditRun(out, task, problems, completions, parsed, table, arms)


def count_completions(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path.exists():
        for row in read_jsonl(path):
            counts[row["pid"]] = counts.get(row["pid"], 0) + 1
    return counts


def load_completions(path: Path, problems: Sequence[Problem], n: int) -> list[Completion]:
    by_pid: dict[str, list[Completion]] = {}
    for row in read_jsonl(path):
        c = Completion.from_json(row)
        by_pid.setdefault(c.pid, []).append(c)
    out: list[Completion] = []
    for p in problems:
        rows = sorted(by_pid.get(p.pid, []), key=lambda c: c.k)[:n]
        if len(rows) < n:
            raise ValueError(f"{p.pid}: {len(rows)} completions on disk, need {n}")
        out.extend(rows)
    return out


def _release_accelerators() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
