"""Step-0 audit: sample the base policy, grade, fit arms, featurise, forecast, lock.

Artifacts (all under one ``out_dir``, every stage resumable):

``stages.json``          fingerprint of each stage, and ``grades``: the hash of how
                         the current parser and oracle grade the completions. A
                         mismatch deletes that stage (for ``grades``, calibration)
                         and every downstream artifact, then recomputes. A lock is
                         never deleted: a changed fingerprint under ``lock.json``
                         is an error, and the run must use a new directory.
``problems.jsonl``       frozen train and held-out pools.
``completions.jsonl``    ``n`` base-policy samples per problem.
``calibration.json``     arms fitted on the *train* split only, with rates and CIs.
``audit.npz``            :class:`~vfates.types.AuditTable` with every arm's verdicts.
``confusion.json``       step-0 confusion per arm and split.
``features.npz``         one feature row per audit row (same order).
``features.meta.json``   extractor options and whether rows used sampled token ids.
``forecasts.json``       per-arm screen: transfer precision and the baselines.
``lora_init.safetensors`` LoRA A matrices the features were computed with.
``lock.json``            hash-locked forecasts, written before any training run.

Held-out verdicts are recorded for description only; nothing is fitted on them.
"""

from __future__ import annotations

import gc
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vfates.features.lora_init import LORA_INIT_FILENAME
from vfates.forecast import lockbox
from vfates.forecast import transfer as transfer_mod
from vfates.forecast.confusion import (
    confusion,
    confusion_with_ci,
    mean_precision_forecast,
    modal_precision_forecast,
)
from vfates.forecast.probe import heldout_error_probe
from vfates.gen.base import Generator
from vfates.tasks.base import Task
from vfates.tasks.registry import build_task
from vfates.types import SPLITS, AuditTable, Completion, GenOut, Parsed, Problem
from vfates.utils.env import environment_report
from vfates.utils.hashing import stable_seed
from vfates.utils.io import (
    append_jsonl,
    atomic_npz,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_text,
    write_json,
    write_jsonl,
)
from vfates.utils.stats import group_indices
from vfates.verifiers.arms import FNRule, FPRule, ResampledNoise
from vfates.verifiers.base import Verifier
from vfates.verifiers.calibrate import CalibrationData, calibrate_arms
from vfates.verifiers.registry import build_verifier

FEATURE_KINDS = ("lora_grad", "hidden")
STAGE_SCHEMA = "vfates.stages/1"
FORECAST_SCHEMA = "vfates.forecasts/2"
MANIFEST_NAME = "stages.json"
STAGE_ORDER = ("problems", "completions", "calibration", "table", "features", "forecasts")


LORA_INIT_NAME = LORA_INIT_FILENAME

_STAGE_FILES: dict[str, tuple[str, ...]] = {
    "problems": ("problems.jsonl",),
    "completions": ("completions.jsonl",),
    "calibration": ("calibration.json",),
    "table": ("audit.npz", "audit.meta.json", "confusion.json"),
    "features": ("features.npz", "features.meta.json", LORA_INIT_NAME),
    "forecasts": ("forecasts.json",),
}
_PRIMARY = {
    "problems": "problems.jsonl",
    "completions": "completions.jsonl",
    "calibration": "calibration.json",
    "table": "audit.npz",
    "features": "features.npz",
    "forecasts": "forecasts.json",
}
# Self included. ``lock.json`` is intentionally absent: a locked directory is never cleaned.
_DOWNSTREAM: dict[str, tuple[str, ...]] = {
    "problems": STAGE_ORDER,
    "completions": ("completions", "calibration", "table", "features", "forecasts"),
    "calibration": ("calibration", "table", "forecasts"),
    "table": ("table", "forecasts"),
    "features": ("features", "forecasts"),
    "forecasts": ("forecasts",),
}


@dataclass
class AuditRun:
    """In-memory view of a finished (or resumed) audit directory.

    ``grades_sha256`` is :func:`grades_sha256` of the completions as the
    current parser and oracle grade them.
    """

    out_dir: Path
    task: Task
    problems: dict[str, Problem]
    completions: list[Completion]
    parsed: list[Parsed]
    table: AuditTable
    arms: dict[str, Verifier]
    grades_sha256: str


def artifact_sha256(path: str | Path) -> str:
    """SHA-256 of a file, as recorded in a lock's ``artifacts``."""
    return lockbox.sha256_file(path)


def read_lock_record(path: str | Path) -> dict[str, Any]:
    """Verified lock record. Raises ``ValueError`` when the file fails verification."""
    return lockbox.load_lock_record(path)


def completion_token_ids(completions: Sequence[Any]) -> list[list[int]] | None:
    """Sampled token ids, or ``None`` when any completion is missing them."""
    rows: list[list[int]] = []
    for comp in completions:
        ids = getattr(comp, "token_ids", None)
        if ids is None:
            return None
        rows.append([int(i) for i in ids])
    return rows


def _fp(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def _length_division(normalize: str, loss_agg: str) -> bool:
    return transfer_mod.length_division(normalize, loss_agg)


def _forecast_length_division(features: Mapping[str, Any], forecast: Mapping[str, Any]) -> bool:
    normalize = str(features.get("normalize") or "sum")
    loss_agg = str(forecast.get("loss_agg", "seq_mean"))
    derived = _length_division(normalize, loss_agg)
    if "divide_by_length" in forecast and bool(forecast["divide_by_length"]) != derived:
        raise ValueError(
            f"forecast.divide_by_length={forecast['divide_by_length']!r} disagrees with "
            f"length_division(normalize={normalize!r}, loss_agg={loss_agg!r})={derived}"
        )
    return derived


def _arms_file_sha(arm_cfg: Mapping[str, Any]) -> str | None:
    path = arm_cfg.get("file")
    if not path:
        return None
    return artifact_sha256(path)


def stage_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    """Content hash of each audit stage. Downstream stages include upstream hashes.

    ``features`` and ``forecasts`` are omitted when the config has no feature spec.
    An explicit ``forecast.divide_by_length`` that disagrees with
    :func:`_length_division` raises before any fingerprint is returned.
    """
    problems = _fp({"task": config["task"], "pools": config["pools"]})
    completions = _fp(
        {
            "problems": problems,
            "generator": config["generator"],
            "sampling": config["sampling"],
        }
    )
    arm_cfg = config["arms"]
    resolved = resolve_arms(arm_cfg)
    calibration = _fp(
        {
            "completions": completions,
            "resolved_arms": resolved,
            "arms_file_sha256": _arms_file_sha(arm_cfg),
            "targets": resolved.get("targets"),
            "n_boot": int(resolved.get("n_boot", 1000)),
            "seed": int(resolved.get("seed", 0)),
        }
    )
    out = {
        "problems": problems,
        "completions": completions,
        "calibration": calibration,
        "table": _fp({"calibration": calibration}),
    }
    feat = config.get("features")
    if not feat:
        return out
    features = _fp({"completions": completions, "features": feat})
    forecast = dict(config.get("forecast") or {})
    divided = _forecast_length_division(feat, forecast)
    out["features"] = features
    out["forecasts"] = _fp(
        {
            "schema": FORECAST_SCHEMA,
            "features": features,
            "calibration": calibration,
            "forecast": forecast,
            "divide_by_length": divided,
        }
    )
    return out


def _primary(out: Path, stage: str) -> Path:
    return out / _PRIMARY[stage]


def _any_stage_artifact(out: Path) -> bool:
    parts = out / "feature_parts"
    if parts.is_dir() and any(parts.iterdir()):
        return True
    return any((out / name).is_file() for names in _STAGE_FILES.values() for name in names)


def _delete_stage(out: Path, stage: str) -> None:
    if stage == "features":
        parts = out / "feature_parts"
        if parts.exists():
            shutil.rmtree(parts)
    for name in _STAGE_FILES[stage]:
        path = out / name
        if path.is_file():
            path.unlink()


def _invalidate_from(out: Path, stage: str, stages: dict[str, Any]) -> None:
    for name in _DOWNSTREAM[stage]:
        _delete_stage(out, name)
        stages.pop(name, None)


def _prepare_stage_manifest(
    out: Path, config: Mapping[str, Any], desired: Mapping[str, str]
) -> dict[str, Any]:
    path = out / MANIFEST_NAME
    adopt = bool(config.get("adopt_unfingerprinted", False))
    if not path.exists():
        if _any_stage_artifact(out) and not adopt:
            raise ValueError(
                "This audit directory has stage artifacts but no stages.json fingerprint "
                "manifest, so they cannot be checked against the config. Refusing to reuse "
                "them. Set adopt_unfingerprinted: true to adopt the artifacts and write the "
                "manifest, or use a new directory."
            )
        if (out / "lock.json").exists() and not adopt:
            raise ValueError(
                "lock.json exists but stages.json is missing, so the locked audit cannot be "
                "checked against the config. Refusing to reuse or replace this directory. "
                "Use a new directory."
            )
        stages = {
            name: desired[name] for name in desired if adopt and _primary(out, name).is_file()
        }
        manifest: dict[str, Any] = {"schema": STAGE_SCHEMA, "stages": stages}
        write_json(path, manifest)
        return manifest

    manifest = read_json(path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != STAGE_SCHEMA
        or not isinstance(manifest.get("stages"), dict)
    ):
        raise ValueError(f"unreadable fingerprint manifest: {path}")
    stages = manifest["stages"]
    if (out / "lock.json").exists():
        changed = sorted(
            name for name in set(stages) | set(desired) if stages.get(name) != desired.get(name)
        )
        if changed:
            raise ValueError(
                "lock.json exists but the audit config no longer matches the locked stages "
                f"({', '.join(changed)}). Refusing to delete or overwrite a locked audit. "
                "Use a new directory."
            )
        return manifest

    dirty = False
    for name in list(stages):
        if name not in stages:
            continue
        if desired.get(name) != stages[name]:
            _invalidate_from(out, name, stages)
            dirty = True
    for name in desired:
        if name not in stages and _primary(out, name).is_file():
            _invalidate_from(out, name, stages)
            dirty = True
    if dirty:
        write_json(path, manifest)
    return manifest


def _stage_ready(out: Path, manifest: Mapping[str, Any], stage: str, fp: str) -> bool:
    stages = manifest["stages"]
    return stages.get(stage) == fp and _primary(out, stage).is_file()


def _commit_stage(out: Path, manifest: dict[str, Any], stage: str, fp: str) -> None:
    manifest["stages"][stage] = fp
    write_json(out / MANIFEST_NAME, manifest)


def _assert_lock_payload(path: Path, payload: Mapping[str, Any]) -> None:
    lockbox.assert_matches_lock(path, payload)


def _write_lock(
    out: Path,
    config: Mapping[str, Any],
    forecasts: Mapping[str, Any],
    arms_file: str | None,
    grades: str,
) -> None:
    artifacts = {
        "problems": artifact_sha256(out / "problems.jsonl"),
        "completions": artifact_sha256(out / "completions.jsonl"),
        "grades": grades,
        "calibration": artifact_sha256(out / "calibration.json"),
        "features": artifact_sha256(out / "features.npz"),
    }
    if arms_file:
        artifacts["arms_file"] = artifact_sha256(arms_file)
    init = out / LORA_INIT_NAME
    if init.is_file():
        artifacts["lora_init"] = artifact_sha256(init)
    lockbox.lock_forecasts(
        dict(forecasts),
        out / "lock.json",
        config=dict(config),
        note="Step-0 forecasts locked before any training run.",
        artifacts=artifacts,
    )


def _require_unique_questions(train: Sequence[Problem], held: Sequence[Problem]) -> None:
    seen: dict[str, str] = {}
    for problem in [*train, *held]:
        prev = seen.get(problem.question)
        if prev is not None:
            raise ValueError(
                f"duplicate question text for {problem.pid} and {prev}; "
                "questions must be unique within and across the train and held-out pools"
            )
        seen[problem.question] = problem.pid


def load_pools(task: Task, pools: Mapping[str, Any]) -> tuple[list[Problem], list[Problem]]:
    """Frozen ``(train, heldout)`` pools from ``{'train': n, 'heldout': m, 'seed': s}``.

    Problem ids are whatever the task emits. Repeated question text, inside one
    pool or across the two, is an error.
    """
    seed = int(pools.get("seed", 0))
    train = task.problems("train", int(pools["train"]), seed)
    held = task.problems("heldout", int(pools["heldout"]), seed)
    _require_unique_questions(train, held)
    return train, held


def _make_completion(problem: Problem, k: int, out: GenOut) -> Completion:
    return Completion(
        pid=problem.pid,
        k=k,
        text=out.text,
        n_tokens=out.n_tokens,
        logprob=out.logprob,
        finish_reason=out.finish_reason,
        token_ids=None if out.token_ids is None else tuple(int(i) for i in out.token_ids),
    )


def _group_complete(rows: Sequence[Completion], n: int) -> bool:
    """True when ``rows`` are exactly the samples ``k = 0 .. n-1``.

    A shorter ``n`` than the file holds is not a prefix of that draw: the whole
    group has to be sampled again.
    """
    return [c.k for c in sorted(rows, key=lambda c: c.k)] == list(range(n))


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
    A problem whose on-disk rows are not exactly ``k = 0 .. n-1`` is resampled
    from scratch, including when the file holds a longer draw.
    """
    path = Path(path)
    done: dict[str, list[Completion]] = {}
    if path.exists():
        for row in read_jsonl(path):
            c = Completion.from_json(row)
            done.setdefault(c.pid, []).append(c)
    stale = {p.pid for p in problems if not _group_complete(done.get(p.pid, []), n)}
    if stale:
        kept = [
            c
            for p in problems
            if p.pid in done and p.pid not in stale
            for c in sorted(done[p.pid], key=lambda c: c.k)
        ]
        write_jsonl(path, [c.to_json() for c in kept])
        for pid in stale:
            done.pop(pid, None)
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
            if len(group) != n:
                raise ValueError(
                    f"{problem.pid}: generator returned {len(group)} samples, expected {n}"
                )
            rows = [_make_completion(problem, k, out) for k, out in enumerate(group)]
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
    return [
        c
        for p in sorted(problems, key=lambda q: order[q.pid])
        for c in sorted(done[p.pid], key=lambda c: c.k)
    ]


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


def grades_sha256(
    completions: Sequence[Completion], parsed: Sequence[Parsed], O: np.ndarray
) -> str:
    """SHA-256 of every row's ``(pid, k, O, answer, raw)``, in ``(pid, k)`` order.

    It changes exactly when the parser or the oracle grades a cached
    completion differently.
    """
    rows = sorted(
        [c.pid, int(c.k), int(o), p.answer, p.raw]
        for c, p, o in zip(completions, parsed, O, strict=True)
    )
    return sha256_text(canonical_json(rows))


def _check_grades(out: Path, manifest: dict[str, Any], grades: str) -> None:
    """Invalidate calibration and downstream when the grades moved; refuse under a lock."""
    stored = manifest.get("grades")
    if stored == grades:
        return
    if stored is not None:
        if (out / "lock.json").exists():
            raise ValueError(
                "The parser or oracle now grades the cached completions differently from "
                "the locked audit. Refusing to delete or overwrite a locked audit. "
                "Use a new directory."
            )
        _invalidate_from(out, "calibration", manifest["stages"])
    manifest["grades"] = grades
    write_json(out / MANIFEST_NAME, manifest)


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


def _part_matches(path: Path, lo: int, hi: int, fingerprint: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            n_rows = int(z["n_rows"])
            stored = str(np.asarray(z["fingerprint"]).item())
            return (
                int(z["lo"]) == lo
                and int(z["hi"]) == hi
                and n_rows == hi - lo
                and int(z["features"].shape[0]) == n_rows
                and int(z["lengths"].shape[0]) == n_rows
                and stored == fingerprint
            )
    except (KeyError, ValueError, OSError):
        return False


def extract_features(
    spec: Mapping[str, Any],
    task: Task,
    problems: Mapping[str, Problem],
    completions: Sequence[Completion],
    out_dir: str | Path,
    *,
    fingerprint: str,
    chunk: int = 512,
    save_init: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Feature rows aligned with ``completions``, cached in resumable parts.

    ``spec``: ``kind`` (``lora_grad`` | ``hidden``), ``model_id``, and the
    keyword options of the chosen extractor. Each part stores ``fingerprint``
    and ``n_rows`` and is recomputed when either disagrees. ``save_init`` is
    forwarded to ``lora_grad_features`` (contract C-2).
    """
    if not fingerprint:
        raise ValueError("extract_features requires a stage fingerprint")
    kind = str(spec["kind"])
    if kind not in FEATURE_KINDS:
        raise ValueError(f"unknown feature kind {kind!r}; expected one of {FEATURE_KINDS}")
    options = {k: v for k, v in spec.items() if k not in ("kind", "model_id", "load")}
    parts_dir = Path(out_dir) / "feature_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    prompts = [task.prompt(problems[c.pid]) for c in completions]
    texts = [c.text for c in completions]
    token_ids = completion_token_ids(completions)
    system = task.system_prompt()
    model = tokenizer = None
    meta: dict[str, Any] = {
        "kind": kind,
        "model_id": spec["model_id"],
        "options": options,
        "token_source": "token_ids" if token_ids is not None else "text",
    }
    feats: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    for part, lo in enumerate(range(0, len(completions), chunk)):
        path = parts_dir / f"part_{part:05d}.npz"
        hi = min(lo + chunk, len(completions))
        if path.exists():
            if _part_matches(path, lo, hi, fingerprint):
                with np.load(path, allow_pickle=False) as z:
                    feats.append(np.array(z["features"]))
                    lengths.append(np.array(z["lengths"]))
                continue
            path.unlink()
        if model is None:
            from vfates.features.hidden import load_model

            load = dict(spec.get("load", {}))
            model, tokenizer = load_model(str(spec["model_id"]), **load)
        started = time.perf_counter()
        part_ids = None if token_ids is None else token_ids[lo:hi]
        if kind == "lora_grad":
            from vfates.features.lora_grad import lora_grad_features

            res = lora_grad_features(
                model,
                tokenizer,
                prompts[lo:hi],
                texts[lo:hi],
                system_prompt=system,
                completion_ids=part_ids,
                save_init=save_init,
                **options,
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
                model,
                tokenizer,
                prompts[lo:hi],
                texts[lo:hi],
                system_prompt=system,
                completion_ids=part_ids,
                **options,
            )
            ln = np.array([c.n_tokens for c in completions[lo:hi]], dtype=np.int64)
            _release_accelerators()
        atomic_npz(
            path,
            features=np.asarray(f, dtype=np.float32),
            lengths=np.asarray(ln),
            lo=np.int64(lo),
            hi=np.int64(hi),
            n_rows=np.int64(hi - lo),
            fingerprint=np.asarray(fingerprint),
        )
        feats.append(np.asarray(f, dtype=np.float32))
        lengths.append(np.asarray(ln))
        print(
            f"[features] {kind} rows {lo}-{hi} of {len(completions)} in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if not feats:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            meta,
        )
    return np.concatenate(feats), np.concatenate(lengths), meta


def _probe_kinds(arm: Verifier) -> list[str]:
    """Error channels a held-out probe should score for ``arm``.

    FP types are ``fp``, FN types are ``fn``. Resampled arms use each channel
    whose rate is nonzero. Oracle and natural arms are skipped. A composite
    with both rule lists is probed on both channels.
    """
    desc = arm.describe()
    typ = str(desc.get("type", ""))
    kind = str(desc.get("kind", getattr(arm, "kind", "")))
    if kind in ("oracle", "natural") or typ in ("oracle", "string_match"):
        return []
    if kind == "fp" or typ.endswith("_fp"):
        return ["fp"]
    if kind == "fn" or typ.endswith("_fn"):
        return ["fn"]
    if kind == "noise" or typ == "resampled" or bool(getattr(arm, "stochastic", False)):
        rho0 = float(desc.get("rho0", 0.0) or 0.0)
        rho1 = float(desc.get("rho1", 0.0) or 0.0)
        kinds: list[str] = []
        if rho0 > 0.0:
            kinds.append("fp")
        if rho1 > 0.0:
            kinds.append("fn")
        return kinds
    if kind == "composite" or typ == "composite":
        kinds = []
        if desc.get("fp"):
            kinds.append("fp")
        if desc.get("fn"):
            kinds.append("fn")
        return kinds
    return []


def _arm_probe(
    name: str,
    table: AuditTable,
    features: np.ndarray,
    verifiers: Mapping[str, Verifier] | None,
    seed: int,
) -> Any:
    if not verifiers or name not in verifiers:
        return None
    kinds = _probe_kinds(verifiers[name])
    if not kinds:
        return None
    tr = table.split == "train"
    he = table.split == "heldout"
    if not np.any(tr) or not np.any(he):
        return None
    v = table.V[name]
    aucs: dict[str, float] = {}
    for kind in kinds:
        result = heldout_error_probe(
            features[tr],
            table.O[tr],
            v[tr],
            table.pid[tr],
            features[he],
            table.O[he],
            v[he],
            table.pid[he],
            kind,
            seed=seed,
        )
        aucs[kind] = float(result["auc"])
    if len(kinds) == 1:
        return aucs[kinds[0]]
    return aucs


def behaviour_indicators(
    arms: Mapping[str, Verifier],
    problems: Mapping[str, Problem],
    completions: Sequence[Completion],
    parsed: Sequence[Parsed],
) -> dict[str, np.ndarray]:
    """Per-row behaviour ``b`` of every FP rule (``accepts``) and FN rule (``rejects``)."""
    out: dict[str, np.ndarray] = {}
    for name, arm in arms.items():
        if isinstance(arm, FPRule):
            fn = arm.accepts
        elif isinstance(arm, FNRule):
            fn = arm.rejects
        else:
            continue
        out[name] = np.fromiter(
            (
                bool(fn(problems[c.pid], p, c.text))
                for c, p in zip(completions, parsed, strict=True)
            ),
            dtype=bool,
            count=len(completions),
        )
    return out


def behaviour_alignment(
    verifier: Verifier | None, b: np.ndarray | None, O: np.ndarray, pid: np.ndarray
) -> dict[str, float | None]:
    """Egashira et al.'s first-order increment of the rule's behaviour, signed by its direction.

    ``f`` is the frequency of ``b`` and ``C`` the mean prompt-centred oracle
    advantage of rows with ``b``. The forecast is ``f * C`` for an FP rule and
    ``-f * C`` for an FN rule, 0 for resampled noise (no behaviour to express),
    and ``None`` for arms that are neither.
    """
    none: dict[str, float | None] = {
        "behaviour_frequency": None,
        "conditional_advantage": None,
        "behaviour_alignment": None,
    }
    if isinstance(verifier, ResampledNoise):
        return {**none, "behaviour_alignment": 0.0}
    if b is None or not isinstance(verifier, FPRule | FNRule):
        return none
    if len(b) != len(O):
        raise ValueError(f"behaviour has {len(b)} rows, the oracle {len(O)}")
    if len(b) == 0:
        return none
    o = np.asarray(O, dtype=np.float64)
    centred = np.empty_like(o)
    for idx in group_indices(pid):
        centred[idx] = o[idx] - o[idx].mean()
    mask = np.asarray(b, dtype=bool)
    f = float(mask.mean())
    sign = 1.0 if isinstance(verifier, FPRule) else -1.0
    return {
        "behaviour_frequency": f,
        "conditional_advantage": float(centred[mask].mean()) if mask.any() else None,
        "behaviour_alignment": sign * float(np.mean(centred * mask)),
    }


def forecast_arms(
    table: AuditTable,
    features: np.ndarray,
    lengths: np.ndarray,
    arms: Sequence[str],
    *,
    q: float | None = None,
    qs: Sequence[float] | None = None,
    n_boot: int = 500,
    reliability_splits: int = 20,
    divide_by_length: bool = False,
    seed: int = 0,
    verifiers: Mapping[str, Verifier] | None = None,
    behaviour: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """The step-0 screen and its baselines for every arm, from the audit alone.

    ``divide_by_length`` is applied to train rows only. Each entry records
    ``length_divided`` and ``probe_heldout_auc`` (``None`` for oracle and
    natural arms, a float for one error channel, or ``{"fp", "fn"}`` when an
    arm has both). ``behaviour`` holds :func:`behaviour_indicators` rows
    aligned with ``table``; the behaviour-alignment fields use train rows only.
    """
    if q is None:
        q = transfer_mod.PRIMARY_Q
    if qs is None:
        qs = transfer_mod.DEFAULT_QS
    tr, he = table.split == "train", table.split == "heldout"
    out: dict[str, Any] = {}
    for name in arms:
        v = table.V[name]
        fc = transfer_mod.transfer_forecast(
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
            "length_divided": bool(divide_by_length),
            "probe_heldout_auc": _arm_probe(name, table, features, verifiers, seed),
            **behaviour_alignment(
                None if verifiers is None else verifiers.get(name),
                None if behaviour is None or name not in behaviour else behaviour[name][tr],
                table.O[tr],
                table.pid[tr],
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
    from vfates.utils.io import read_yaml

    return {**read_yaml(path), **cfg}


def run_audit(config: Mapping[str, Any], out_dir: str | Path) -> AuditRun:
    """Run (or resume) every audit stage in ``config``. Returns the in-memory audit.

    ``config`` keys: ``task``, ``pools``, ``generator`` (``backend``,
    ``model_id``, ``kwargs``), ``sampling`` (``n``, ``max_tokens``,
    ``temperature``, ``top_p``, ``seed``, ``chunk_prompts``), ``arms``
    (``targets`` and ``arms``, a list of specs, or ``file`` naming a YAML with
    both; optional ``n_boot``, ``seed``), optional ``features`` and
    ``forecast`` (``loss_agg`` defaults to ``seq_mean``; ``divide_by_length``
    is derived and an explicit value that disagrees is an error), optional
    ``lock: true``. ``adopt_unfingerprinted: true`` adopts artifacts that
    predate ``stages.json`` instead of refusing them.

    Changing a stage's fingerprint deletes that stage and every downstream
    artifact (feature parts, ``forecasts.json``, ``lora_init.safetensors``)
    and recomputes them. If ``lock.json`` is already there, a changed
    fingerprint raises and nothing is deleted.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task = build_task(config["task"])
    train, held = load_pools(task, config["pools"])
    problems = {p.pid: p for p in [*train, *held]}
    if len(problems) != len(train) + len(held):
        raise ValueError("train and held-out pools share problem ids")
    desired = stage_fingerprints(config)
    manifest = _prepare_stage_manifest(out, config, desired)
    write_json(out / "config.json", {"config": dict(config), "env": environment_report()})

    if not _stage_ready(out, manifest, "problems", desired["problems"]):
        write_jsonl(out / "problems.jsonl", [p.to_json() for p in [*train, *held]])
        _commit_stage(out, manifest, "problems", desired["problems"])

    samp = dict(config["sampling"])
    completions_path = out / "completions.jsonl"
    need = int(samp["n"])
    if not _stage_ready(out, manifest, "completions", desired["completions"]):
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
    if not _stage_ready(out, manifest, "completions", desired["completions"]):
        _commit_stage(out, manifest, "completions", desired["completions"])
    parsed, O = grade_completions(task, problems, completions)
    grades = grades_sha256(completions, parsed, O)
    _check_grades(out, manifest, grades)

    calib_path = out / "calibration.json"
    arm_cfg = resolve_arms(config["arms"])
    if not _stage_ready(out, manifest, "calibration", desired["calibration"]):
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
        _commit_stage(out, manifest, "calibration", desired["calibration"])
    else:
        arms = _arms_from_calibration(calib_path, task)

    if not _stage_ready(out, manifest, "table", desired["table"]):
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
            out / "confusion.json",
            confusion_report(table, n_boot=int(arm_cfg.get("n_boot", 1000))),
        )
        _commit_stage(out, manifest, "table", desired["table"])
    else:
        table = AuditTable.load(out / "audit.npz")
    run = AuditRun(out, task, problems, completions, parsed, table, dict(arms), grades)

    feat_cfg = config.get("features")
    if not feat_cfg:
        return run

    if not _stage_ready(out, manifest, "features", desired["features"]):
        feats, lengths, meta = extract_features(
            feat_cfg,
            task,
            problems,
            completions,
            out,
            fingerprint=desired["features"],
            save_init=out / LORA_INIT_NAME,
        )
        atomic_npz(out / "features.npz", features=feats, lengths=lengths)
        write_json(out / "features.meta.json", meta)
        _commit_stage(out, manifest, "features", desired["features"])
    else:
        with np.load(out / "features.npz", allow_pickle=False) as saved:
            feats = np.array(saved["features"])
            lengths = np.array(saved["lengths"])

    fc_cfg = dict(config.get("forecast") or {})
    divided = _forecast_length_division(feat_cfg, fc_cfg)
    forecasts_ready = _stage_ready(out, manifest, "forecasts", desired["forecasts"])
    if forecasts_ready:
        forecasts = read_json(out / "forecasts.json")
    else:
        forecasts = forecast_arms(
            table,
            feats,
            lengths,
            list(arms),
            q=float(fc_cfg.get("q", transfer_mod.PRIMARY_Q)),
            qs=tuple(float(x) for x in fc_cfg.get("qs", transfer_mod.DEFAULT_QS)),
            n_boot=int(fc_cfg.get("n_boot", 500)),
            reliability_splits=int(fc_cfg.get("reliability_splits", 20)),
            divide_by_length=divided,
            seed=int(fc_cfg.get("seed", 0)),
            verifiers=arms,
            behaviour=behaviour_indicators(arms, problems, completions, parsed),
        )
    lock_path = out / "lock.json"
    if lock_path.exists():
        _assert_lock_payload(lock_path, forecasts)
    if not forecasts_ready:
        write_json(out / "forecasts.json", forecasts)
        _commit_stage(out, manifest, "forecasts", desired["forecasts"])
    if not lock_path.exists() and config.get("lock", False):
        arms_file = config["arms"].get("file") if isinstance(config.get("arms"), Mapping) else None
        _write_lock(out, config, forecasts, None if arms_file is None else str(arms_file), grades)
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
    parsed, O = grade_completions(task, problems, completions)
    table = AuditTable.load(out / "audit.npz")
    arms = _arms_from_calibration(out / "calibration.json", task)
    return AuditRun(
        out, task, problems, completions, parsed, table, arms, grades_sha256(completions, parsed, O)
    )


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
        rows = sorted(by_pid.get(p.pid, []), key=lambda c: c.k)
        ks = [c.k for c in rows]
        if ks != list(range(n)):
            raise ValueError(
                f"{p.pid}: completions on disk are not exactly k=0..{n - 1} (found {ks}). "
                "A smaller sampling.n is a new draw, not a prefix of a longer one."
            )
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
