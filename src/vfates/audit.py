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
``forecasts.json``       per-arm screen, including ``tau_pool_budget``.
``patches.json``         patch selection, only when an arm has type ``patch_select``.
``lora_init.safetensors`` LoRA A matrices the features were computed with.
``lock.json``            hash-locked forecasts, written before any training run.

Held-out verdicts are recorded for description only; nothing is fitted on them.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import shutil
import time
from collections.abc import Iterator, Mapping, Sequence
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
from vfates.utils.hashing import stable_seed, u01
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
from vfates.verifiers.calibrate import CalibrationData, calibrate_arms, pooled_rates
from vfates.verifiers.registry import build_verifier

FEATURE_KINDS = ("lora_grad", "hidden")
# Keys the audit forwards only when the spec sets them and the callee accepts them.
_FEATURE_PASSTHROUGH = (
    "chat_template_kwargs",
    "gradient_checkpointing",
    "target_modules",
    "logprob_chunk",
)
_LOAD_PASSTHROUGH = (("dtype", "dtype"), ("device", "device"), ("device_map", "device_map"))
# Prepared once per feature stage. Scoring kwargs stay on featurize_prepared.
_PREPARE_KEYS = (
    "sketch_dim",
    "lora_r",
    "lora_alpha",
    "target_modules",
    "lora_seed",
    "sketch_seed",
    "device",
    "dtype",
    "device_map",
    "gradient_checkpointing",
    "load_init",
)
_FEATURIZE_KEYS = (
    "normalize",
    "max_len",
    "return_exact",
    "progress",
    "chat_template_kwargs",
    "logprob_chunk",
)
_NON_ARM_FORECAST_KEYS = frozenset({"patch_selection"})
_DOMAIN_TOKEN = "{domain}"
STAGE_SCHEMA = "vfates.stages/1"
FORECAST_SCHEMA = "vfates.forecasts/4"
MANIFEST_NAME = "stages.json"
STAGE_ORDER = (
    "problems",
    "completions",
    "calibration",
    "table",
    "features",
    "patches",
    "forecasts",
)


LORA_INIT_NAME = LORA_INIT_FILENAME

_STAGE_FILES: dict[str, tuple[str, ...]] = {
    "problems": ("problems.jsonl",),
    "completions": ("completions.jsonl",),
    "calibration": ("calibration.json",),
    "table": ("audit.npz", "audit.meta.json", "confusion.json"),
    "features": ("features.npz", "features.meta.json", LORA_INIT_NAME),
    "patches": ("patches.json",),
    "forecasts": ("forecasts.json",),
}
_PRIMARY = {
    "problems": "problems.jsonl",
    "completions": "completions.jsonl",
    "calibration": "calibration.json",
    "table": "audit.npz",
    "features": "features.npz",
    "patches": "patches.json",
    "forecasts": "forecasts.json",
}
# Self included. ``lock.json`` is intentionally absent: a locked directory is never cleaned.
_DOWNSTREAM: dict[str, tuple[str, ...]] = {
    "problems": STAGE_ORDER,
    "completions": ("completions", "calibration", "table", "features", "patches", "forecasts"),
    "calibration": ("calibration", "table", "patches", "forecasts"),
    "table": ("table", "patches", "forecasts"),
    "features": ("features", "patches", "forecasts"),
    "patches": ("patches", "forecasts"),
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
    return _arms_tree_sha(Path(path))


def _patch_specs(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms = resolved.get("arms") or []
    return [
        dict(spec)
        for spec in arms
        if isinstance(spec, Mapping) and str(spec.get("type")) == "patch_select"
    ]


def _arms_domain(config: Mapping[str, Any]) -> str | None:
    """Domain used to expand ``{domain}`` in arm salts.

    An arms ``domain`` key wins, then the task ``domain``, then the task
    ``name``, then the task ``type``.
    """
    arms = config.get("arms") or {}
    if isinstance(arms, Mapping) and arms.get("domain"):
        return str(arms["domain"])
    task = config.get("task") or {}
    if isinstance(task, Mapping):
        if task.get("domain"):
            return str(task["domain"])
        if task.get("name"):
            return str(task["name"])
        if task.get("type"):
            return str(task["type"])
    return None


def stage_fingerprints(
    config: Mapping[str, Any], *, row_categories: str | None = None
) -> dict[str, str]:
    """Content hash of each audit stage. Downstream stages include upstream hashes.

    ``features`` and ``forecasts`` are omitted when the config has no feature spec.
    ``patches`` is included only when an arm has type ``patch_select``, and that
    requires a features spec. An explicit ``forecast.divide_by_length`` that
    disagrees with :func:`_length_division` raises before any fingerprint is
    returned.

    ``pools`` is hashed as given, including ``pools.from``. A dev audit
    (``from: dev``) therefore cannot resume as a confirmatory audit.
    ``generator.chat_template_kwargs`` is inside the hashed generator mapping.
    ``features.chat_template_kwargs``, ``features.load`` (``dtype``, ``device``,
    ``device_map``), ``gradient_checkpointing``, ``target_modules`` and
    ``logprob_chunk`` are inside the hashed features mapping. When a patch
    menu is configured, ``row_categories`` is the sha256 of the per-row
    ``task.categorize`` assignments and is part of the patch fingerprint.
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
    domain = _arms_domain(config)
    resolved = resolve_arms(arm_cfg, domain=domain)
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
    patch_specs = _patch_specs(resolved)
    if patch_specs and not feat:
        raise ValueError("patch_select arms require a features spec")
    if not feat:
        return out
    features = _fp({"completions": completions, "features": feat})
    forecast = dict(config.get("forecast") or {})
    divided = _forecast_length_division(feat, forecast)
    out["features"] = features
    forecast_body: dict[str, Any] = {
        "schema": FORECAST_SCHEMA,
        "features": features,
        "calibration": calibration,
        "forecast": forecast,
        "divide_by_length": divided,
    }
    if patch_specs:
        task = build_task(config["task"])
        q = float(forecast.get("q", transfer_mod.PRIMARY_Q))
        patches = _fp(
            {
                "features": features,
                "calibration": calibration,
                "patch_select": patch_specs,
                "q": q,
                "divide_by_length": divided,
                "error_categories": list(task.error_categories),
                "row_categories": row_categories,
            }
        )
        out["patches"] = patches
        forecast_body["patches"] = patches
    out["forecasts"] = _fp(forecast_body)
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


def _reusable_category_digest(
    out: Path, config: Mapping[str, Any], desired: Mapping[str, str]
) -> str | None:
    """Category digest of on-disk rows when this run will reuse them.

    ``None`` when there is no patch menu, or the problems and completions
    stages are absent or no longer match ``desired`` (they will be resampled).
    """
    if "patches" not in desired or not (out / MANIFEST_NAME).is_file():
        return None
    manifest = read_json(out / MANIFEST_NAME)
    stages = manifest.get("stages") if isinstance(manifest, dict) else None
    if not isinstance(stages, dict):
        return None
    if stages.get("problems") != desired.get("problems"):
        return None
    if stages.get("completions") != desired.get("completions"):
        return None
    problems_path = out / "problems.jsonl"
    completions_path = out / "completions.jsonl"
    if not problems_path.is_file() or not completions_path.is_file():
        return None
    task = build_task(config["task"])
    problems_list = [Problem.from_json(row) for row in read_jsonl(problems_path)]
    problems = {problem.pid: problem for problem in problems_list}
    completions = load_completions(completions_path, problems_list, int(config["sampling"]["n"]))
    parsed, _oracle = grade_completions(task, problems, completions)
    return _row_category_sha(task, problems, completions, parsed)


def _reconcile_stages(
    out: Path,
    manifest: dict[str, Any],
    desired: Mapping[str, str],
    names: Sequence[str],
) -> None:
    """Compare ``names`` after their fingerprints can be computed.

    A lock refuses a mismatch. Otherwise a mismatch deletes that stage and
    everything downstream, the same rule as :func:`_prepare_stage_manifest`.
    """
    stages = manifest["stages"]
    if (out / "lock.json").exists():
        changed = sorted(name for name in names if stages.get(name) != desired.get(name))
        if changed:
            raise ValueError(
                "lock.json exists but the audit config no longer matches the locked stages "
                f"({', '.join(changed)}). Refusing to delete or overwrite a locked audit. "
                "Use a new directory."
            )
        return
    dirty = False
    for name in names:
        stored = stages.get(name)
        wanted = desired.get(name)
        mismatch = stored is not None and stored != wanted
        untracked = stored is None and _primary(out, name).is_file()
        if mismatch or untracked:
            _invalidate_from(out, name, stages)
            dirty = True
    if dirty:
        write_json(out / MANIFEST_NAME, manifest)


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
        pool_source="dev" if _pools_from(config.get("pools") or {}) == "dev" else None,
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


def _pools_from(pools: Mapping[str, Any]) -> str | None:
    """``pools.from``: ``None`` (confirmatory splits) or ``'dev'``."""
    source = pools.get("from", None)
    if source is None:
        return None
    if source == "dev":
        return "dev"
    raise ValueError(f"pools.from must be 'dev' or null, got {source!r}")


def _relabel_dev_problem(problem: Problem, split: str) -> Problem:
    """Copy ``problem`` onto pseudo-``split`` and record that it came from dev."""
    meta = dict(problem.meta or {})
    meta["source_split"] = problem.split
    meta["pool_from"] = "dev"
    return Problem(
        pid=problem.pid,
        split=split,
        task=problem.task,
        question=problem.question,
        answer=problem.answer,
        meta=meta,
    )


def _partition_dev_pool(
    drawn: Sequence[Problem], train_n: int, held_n: int, seed: int
) -> tuple[list[Problem], list[Problem]]:
    """Rank by ``u01('dev-pool', seed, pid)``, then pid, and relabel.

    The rank depends only on the pid and the pool seed, not on the order
    ``task.problems`` returned. The first ``train_n`` become pseudo-train and
    the next ``held_n`` become pseudo-heldout. ``Problem.split`` is rewritten
    to ``train`` or ``heldout``. ``meta.source_split`` keeps the original
    split and ``meta.pool_from`` is ``dev``.
    """
    ranked = sorted(drawn, key=lambda problem: (u01("dev-pool", seed, problem.pid), problem.pid))
    train = [_relabel_dev_problem(problem, "train") for problem in ranked[:train_n]]
    held = [
        _relabel_dev_problem(problem, "heldout") for problem in ranked[train_n : train_n + held_n]
    ]
    return train, held


def load_pools(task: Task, pools: Mapping[str, Any]) -> tuple[list[Problem], list[Problem]]:
    """Frozen ``(train, heldout)`` pools from ``{'train': n, 'heldout': m, 'seed': s}``.

    ``pools.from`` defaults to null: draw ``train`` and ``heldout`` from those
    task splits. ``pools.from: dev`` draws ``train + heldout`` problems from
    ``task.problems('dev', n, seed)`` and partitions them with
    :func:`_partition_dev_pool`. A dev split shorter than that sum raises and
    reports how many problems were available.

    Problem ids are whatever the task emits. Repeated question text, inside one
    pool or across the two, is an error.
    """
    seed = int(pools.get("seed", 0))
    train_n = int(pools["train"])
    held_n = int(pools["heldout"])
    if _pools_from(pools) == "dev":
        need = train_n + held_n
        drawn = list(task.problems("dev", need, seed))
        if len(drawn) < need:
            raise ValueError(
                f"dev split is smaller than train + heldout: available {len(drawn)}, "
                f"need {need} (train {train_n} + heldout {held_n})"
            )
        train, held = _partition_dev_pool(drawn[:need], train_n, held_n, seed)
    else:
        train = task.problems("train", train_n, seed)
        held = task.problems("heldout", held_n, seed)
    _require_unique_questions(train, held)
    return train, held


def is_dev_audit(audit_dir: str | Path) -> bool:
    """True when ``audit_dir`` records a development pool.

    A lock ``pool_source`` of ``dev``, ``config.pools.from: dev``, or a
    problem whose ``meta.source_split`` or ``meta.pool_from`` is ``dev`` all
    count. A directory with none of those markers is not a dev audit.
    """
    path = Path(audit_dir)
    lock_path = path / "lock.json"
    if lock_path.is_file():
        try:
            record = read_json(lock_path)
        except ValueError:
            record = None
        if isinstance(record, dict) and record.get("pool_source") == "dev":
            return True
    config_path = path / "config.json"
    if config_path.is_file():
        try:
            document = read_json(config_path)
        except ValueError:
            document = None
        config = document.get("config") if isinstance(document, dict) else None
        if not isinstance(config, dict):
            config = document if isinstance(document, dict) else None
        pools = config.get("pools") if isinstance(config, dict) else None
        if isinstance(pools, dict) and pools.get("from") == "dev":
            return True
    problems_path = path / "problems.jsonl"
    if problems_path.is_file():
        for row in read_jsonl(problems_path):
            meta = row.get("meta") if isinstance(row, dict) else None
            if isinstance(meta, dict) and (
                meta.get("source_split") == "dev" or meta.get("pool_from") == "dev"
            ):
                return True
    return False


def require_confirmatory_pools(audit_dir: str | Path) -> None:
    """Raise when ``audit_dir`` is a development-pool audit.

    Confirmatory cells must train and evaluate on the task's train and
    held-out splits. The study calls this before using a frozen audit.
    """
    if is_dev_audit(audit_dir):
        raise ValueError(
            f"{audit_dir} is a development-pool audit (pool_source dev). "
            "Confirmatory cells must use the train and held-out splits."
        )


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


def iter_forecast_arms(
    forecasts: Mapping[str, Any],
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Arm records in a forecasts document.

    Skips ``patch_selection`` and any other key whose value is not an arm
    record (a mapping that carries ``transfer``).
    """
    for name, entry in forecasts.items():
        if name in _NON_ARM_FORECAST_KEYS or not isinstance(entry, Mapping):
            continue
        if "transfer" not in entry:
            continue
        yield str(name), entry


def _row_category_sha(
    task: Task,
    problems: Mapping[str, Problem],
    completions: Sequence[Completion],
    parsed: Sequence[Any],
) -> str:
    """sha256 of ``task.categorize`` on each train and held-out row, in order."""
    rows = []
    for comp, parsed_row in zip(completions, parsed, strict=True):
        categories = sorted(task.categorize(problems[comp.pid], parsed_row, comp.text))
        rows.append({"pid": comp.pid, "k": int(comp.k), "categories": categories})
    return sha256_text(canonical_json(rows))


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
    and ``n_rows`` and is recomputed when either disagrees.

    The base model and tokenizer are loaded once, on the first part that is
    not already cached. A fully cached stage loads nothing. LoRA rows are
    scored with one ``prepare_lora_grad`` and ``featurize_prepared`` per
    missing part. ``save_init`` is passed only to that prepare, so the LoRA A
    file is written once. The model is freed after the last part.
    """
    if not fingerprint:
        raise ValueError("extract_features requires a stage fingerprint")
    kind = str(spec["kind"])
    if kind not in FEATURE_KINDS:
        raise ValueError(f"unknown feature kind {kind!r}; expected one of {FEATURE_KINDS}")
    recorded = {k: v for k, v in spec.items() if k not in ("kind", "model_id", "load")}
    call_options = {k: v for k, v in recorded.items() if k not in _FEATURE_PASSTHROUGH}
    forwarded = _extractor_forwarded(spec)
    parts_dir = Path(out_dir) / "feature_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    prompts = [task.prompt(problems[c.pid]) for c in completions]
    texts = [c.text for c in completions]
    token_ids = completion_token_ids(completions)
    system = task.system_prompt()
    raw_options = {**call_options, **forwarded}
    prepare_values = {key: raw_options[key] for key in _PREPARE_KEYS if key in raw_options}
    featurize_values = {key: raw_options[key] for key in _FEATURIZE_KEYS if key in raw_options}
    model = tokenizer = prepared = None
    meta: dict[str, Any] = {
        "kind": kind,
        "model_id": spec["model_id"],
        "options": recorded,
        "token_source": "token_ids" if token_ids is not None else "text",
    }
    if forwarded:
        meta["forwarded"] = forwarded
    feats: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    wrote_init = False
    try:
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

                load = dict(spec.get("load") or {})
                model, tokenizer = load_model(
                    str(spec["model_id"]), **_accepted_kwargs(load_model, load)
                )
                if kind == "lora_grad":
                    # importlib, not ``import``: a test double installed in
                    # sys.modules is invisible to the import statement once the
                    # real submodule is bound on the package.
                    lora_mod = importlib.import_module("vfates.features.lora_grad")

                    prepare = getattr(lora_mod, "prepare_lora_grad", None)
                    featurize = getattr(lora_mod, "featurize_prepared", None)
                    if prepare is not None and featurize is not None:
                        prepared = prepare(
                            model,
                            save_init=save_init,
                            **_accepted_kwargs(prepare, prepare_values),
                        )
                        wrote_init = True
            started = time.perf_counter()
            part_ids = None if token_ids is None else token_ids[lo:hi]
            if kind == "lora_grad":
                lora_mod = importlib.import_module("vfates.features.lora_grad")

                featurize = getattr(lora_mod, "featurize_prepared", None)
                if prepared is not None and featurize is not None:
                    res = featurize(
                        prepared,
                        tokenizer,
                        prompts[lo:hi],
                        texts[lo:hi],
                        system_prompt=system,
                        completion_ids=part_ids,
                        **_accepted_kwargs(featurize, featurize_values),
                    )
                else:
                    # Test doubles that only expose lora_grad_features. A second
                    # call on one model double-wraps PEFT, so each call gets a
                    # fresh load. The real extractor uses prepare_lora_grad.
                    res = lora_mod.lora_grad_features(
                        model,
                        tokenizer,
                        prompts[lo:hi],
                        texts[lo:hi],
                        system_prompt=system,
                        completion_ids=part_ids,
                        save_init=None if wrote_init else save_init,
                        **_accepted_kwargs(
                            lora_mod.lora_grad_features,
                            {**prepare_values, **featurize_values},
                        ),
                    )
                    wrote_init = True
                    model = tokenizer = None
                    _release_accelerators()
                f, ln = res.features, res.lengths
                meta["extractor"] = {k: v for k, v in res.meta.items() if k != "sample_seconds"}
                del res
            else:
                from vfates.features.hidden import hidden_features

                extra = {**call_options, **_accepted_kwargs(hidden_features, forwarded)}
                f = hidden_features(
                    model,
                    tokenizer,
                    prompts[lo:hi],
                    texts[lo:hi],
                    system_prompt=system,
                    completion_ids=part_ids,
                    **extra,
                )
                ln = np.array([c.n_tokens for c in completions[lo:hi]], dtype=np.int64)
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
    finally:
        del model, tokenizer, prepared
        _release_accelerators()
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


def _oracle_light_budgets(
    table: AuditTable,
    features: np.ndarray,
    lengths: np.ndarray,
    name: str,
    *,
    q: float,
    divide_by_length: bool,
    seed: int,
) -> dict[str, Any]:
    """``tau_pool_budget`` for the pre-registered oracle-light budgets."""
    train = table.split == "train"
    held = table.split == "heldout"
    lens = lengths[train] if divide_by_length else None
    direction = transfer_mod.velocity_direction(
        features[train], table.pid[train], table.V[name][train], lens
    )
    scores = transfer_mod.velocities(features[held], direction)
    budgets: dict[str, Any] = {}
    for budget in transfer_mod.ORACLE_LIGHT_BUDGETS:
        estimate = transfer_mod.tau_pool_budget(
            scores,
            table.O[held],
            table.pid[held],
            q=q,
            budget=budget,
            n_draws=transfer_mod.ORACLE_LIGHT_DRAWS,
            seed=seed,
            arm=name,
        )
        draws = estimate.get("draws")
        if draws is not None:
            kept = transfer_mod.ORACLE_LIGHT_STORED_DRAWS
            estimate = dict(estimate)
            estimate["draws"] = [round(float(value), 6) for value in list(draws)[:kept]]
        budgets[str(budget)] = estimate
    return budgets


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
    arm has both). ``tau_pool_budget`` holds the oracle-light screen at budgets
    32, 64, 128 and 192. ``behaviour`` holds :func:`behaviour_indicators` rows
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
            "tau_pool_budget": _oracle_light_budgets(
                table,
                features,
                lengths,
                name,
                q=q,
                divide_by_length=divide_by_length,
                seed=seed,
            ),
            **behaviour_alignment(
                None if verifiers is None else verifiers.get(name),
                None if behaviour is None or name not in behaviour else behaviour[name][tr],
                table.O[tr],
                table.pid[tr],
            ),
        }
    return out


def _apply_patch_stage(
    out: Path,
    manifest: dict[str, Any],
    fingerprint: str,
    arm_cfg: Mapping[str, Any],
    arms: dict[str, Verifier],
    run: AuditRun,
    features: np.ndarray,
    lengths: np.ndarray,
    *,
    q: float,
    divide_by_length: bool,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Select patched arms, write their verdicts, and commit ``patches.json``."""
    from vfates.verifiers.patch import select_patched_arms

    patched, verdicts, block = select_patched_arms(
        _patch_specs(arm_cfg),
        arms,
        run.task,
        run.problems,
        run.completions,
        run.parsed,
        run.table,
        features,
        lengths,
        q=q,
        divide_by_length=divide_by_length,
    )
    arms.update(patched)
    run.arms = dict(arms)
    for name, column in verdicts.items():
        run.table.V[name] = column
    run.table.save(out / "audit.npz")
    write_json(out / "confusion.json", confusion_report(run.table, n_boot=n_boot))
    _record_patched_calibration(out / "calibration.json", patched, run.table, block, n_boot, seed)
    write_json(out / "patches.json", block)
    _commit_stage(out, manifest, "patches", fingerprint)
    return block


def _record_patched_calibration(
    path: Path,
    patched: Mapping[str, Verifier],
    table: AuditTable,
    block: Mapping[str, Any],
    n_boot: int,
    seed: int,
) -> None:
    """Store each patched arm's ``describe()`` spec so training can rebuild it."""
    doc = read_json(path)
    doc_arms = doc["arms"]
    report_arms = doc["report"]["arms"]
    train = table.split == "train"
    for name, verifier in patched.items():
        doc_arms[name] = verifier.describe()
        info = block[name]
        labels = table.O[train]
        verdicts = table.V[name][train]
        fpr, fnr = pooled_rates(labels, verdicts)
        youden = float("nan") if fpr != fpr or fnr != fnr else (1.0 - fnr) - fpr
        report_arms[name] = {
            "type": "patched",
            "achieved_fpr": fpr,
            "achieved_fnr": fnr,
            "target_fpr": None,
            "target_fnr": None,
            "abs_fpr_error": None,
            "abs_fnr_error": None,
            "channel": {"fpr": fpr, "fnr": fnr, "J": youden},
            "realised": {"fpr": fpr, "fnr": fnr, "J": youden},
            "rates": confusion_with_ci(
                labels, verdicts, table.pid[train], n_boot=n_boot, seed=seed
            ),
            "details": {
                "base": info["base"],
                "by": info["by"],
                "k": info["k"],
                "categories": list(info["categories"]),
                "tau_pool": info["tau_pool"],
                "errors_fixed": info["errors_fixed"],
                "step0_rates": dict(info["step0_rates"]),
            },
        }
    write_json(path, doc)


def _arms_from_calibration(path: Path, task: Task) -> dict[str, Verifier]:
    specs = read_json(path)["arms"]
    return {name: build_verifier(spec, task) for name, spec in specs.items()}


def resolve_arms(arm_cfg: Mapping[str, Any], *, domain: str | None = None) -> dict[str, Any]:
    """The ``arms`` section with ``file:`` (a YAML with ``targets`` and ``arms``) merged in.

    Keys set next to ``file`` override the file's. An ``include`` key inside a
    file loads that YAML first and concatenates its arms in front of the
    including file's arms. ``{domain}`` in any string is replaced with
    ``domain``, or with an ``domain`` key on the arms config when that is set.
    """
    cfg = dict(arm_cfg)
    path = cfg.pop("file", None)
    chosen = str(cfg["domain"]) if cfg.get("domain") else (None if domain is None else str(domain))
    if path is None:
        loaded: dict[str, Any] = {}
    else:
        loaded = _read_arms_tree(Path(path))
    merged = {**loaded, **cfg}
    return _expand_domain(merged, chosen)


def _read_arms_tree(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"arms include cycle at {path}")
    from vfates.utils.io import read_yaml

    raw = read_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"arms file {path} must be a mapping")
    doc = dict(raw)
    include = doc.pop("include", None)
    if include is None:
        return doc
    inc = Path(str(include))
    if not inc.is_absolute():
        inc = path.parent / inc
    base = _read_arms_tree(inc.resolve(), seen | {path})
    return _merge_arm_docs(base, doc)


def _arms_tree_sha(path: Path, seen: frozenset[Path] = frozenset()) -> str:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"arms include cycle at {path}")
    digest = artifact_sha256(path)
    from vfates.utils.io import read_yaml

    raw = read_yaml(path)
    include = raw.get("include") if isinstance(raw, dict) else None
    if not include:
        return digest
    inc = Path(str(include))
    if not inc.is_absolute():
        inc = path.parent / inc
    child = _arms_tree_sha(inc.resolve(), seen | {path})
    return sha256_text(canonical_json({"file": digest, "include": child}))


def _merge_arm_docs(base: Mapping[str, Any], doc: Mapping[str, Any]) -> dict[str, Any]:
    """Kit arms first, then the including file. Targets in ``doc`` override."""
    merged = {**base, **doc}
    merged["arms"] = list(base.get("arms") or []) + list(doc.get("arms") or [])
    if "targets" in doc and isinstance(doc.get("targets"), Mapping):
        merged["targets"] = {**dict(base.get("targets") or {}), **dict(doc["targets"])}
    return merged


def _expand_domain(obj: Any, domain: str | None) -> Any:
    if isinstance(obj, str):
        if _DOMAIN_TOKEN in obj:
            if not domain:
                raise ValueError("arms file uses {domain} but no domain was given")
            return obj.replace(_DOMAIN_TOKEN, domain)
        return obj
    if isinstance(obj, list):
        return [_expand_domain(item, domain) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _expand_domain(value, domain) for key, value in obj.items()}
    return obj


def _accepted_kwargs(fn: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    """``values`` restricted to parameters ``fn`` accepts.

    A ``**kwargs`` parameter keeps every key. Only keys already in ``values``
    are returned, so unset config keys are not passed.
    """
    params = inspect.signature(fn).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(values)
    names = {
        name
        for name, param in params.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in values.items() if key in names}


def _extractor_forwarded(spec: Mapping[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for key in _FEATURE_PASSTHROUGH:
        if key in spec:
            raw[key] = spec[key]
    load = spec.get("load") or {}
    if isinstance(load, Mapping):
        for source, dest in _LOAD_PASSTHROUGH:
            if source in load:
                raw[dest] = load[source]
    return raw


def _generator_kwargs(gen_cfg: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = dict(gen_cfg.get("kwargs") or {})
    if "chat_template_kwargs" in gen_cfg:
        kwargs["chat_template_kwargs"] = gen_cfg["chat_template_kwargs"]
    return kwargs


def visible_gpu_count() -> int | None:
    """GPUs this process is using, or None when that count is not visible.

    ``WORLD_SIZE`` is the accelerate process count (one GPU per process on
    the study placements). Otherwise this is the CUDA device count, or 1 on
    MPS. A CPU process with no accelerator returns None rather than a guess.
    """
    import os

    world = os.environ.get("WORLD_SIZE")
    if world is not None and world.isdigit() and int(world) > 0:
        return int(world)
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        count = int(torch.cuda.device_count())
        return count if count > 0 else None
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return 1
    return None


def _stamp_audit_timing(out: Path, started: float, *, fresh: bool = True) -> None:
    """Record this audit's wall clock in ``audit.meta.json`` when it is absent.

    A later resume does not replace a timing block that is already there.
    The table sidecar is the audit meta. Missing keys stay missing. An audit
    that resumed a directory with committed stages (``fresh`` false) records
    ``wall_seconds`` as null with ``resumed`` true: this process saw only part
    of the work, and a partial time would understate the screen's cost.
    """
    meta_path = out / "audit.meta.json"
    if not meta_path.is_file():
        return
    try:
        meta = read_json(meta_path)
    except (OSError, ValueError):
        return
    if not isinstance(meta, dict):
        return
    existing = meta.get("timing")
    if isinstance(existing, dict) and existing.get("wall_seconds") is not None:
        return
    from vfates.utils.env import hardware_class

    meta["timing"] = {
        "wall_seconds": (time.perf_counter() - started) if fresh else None,
        "gpus": visible_gpu_count(),
        "hardware_class": hardware_class(),
        "resumed": not fresh,
    }
    write_json(meta_path, meta)


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
    started = time.perf_counter()
    task = build_task(config["task"])
    train, held = load_pools(task, config["pools"])
    problems = {p.pid: p for p in [*train, *held]}
    if len(problems) != len(train) + len(held):
        raise ValueError("train and held-out pools share problem ids")
    desired = stage_fingerprints(config)
    # A resume reuses the completions already on disk. Hash their categories
    # before the lock check so the patch fingerprint matches what was committed.
    stored_categories = _reusable_category_digest(out, config, desired)
    if stored_categories is not None:
        desired = stage_fingerprints(config, row_categories=stored_categories)
    fresh = not (out / MANIFEST_NAME).exists()
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
            gen_cfg["backend"], gen_cfg["model_id"], **_generator_kwargs(gen_cfg)
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
    if "patches" in desired:
        updated = stage_fingerprints(
            config,
            row_categories=_row_category_sha(task, problems, completions, parsed),
        )
        if updated["patches"] != desired["patches"] or updated["forecasts"] != desired["forecasts"]:
            desired = updated
            _reconcile_stages(out, manifest, desired, ("patches", "forecasts"))
        else:
            desired = updated

    calib_path = out / "calibration.json"
    arm_cfg = resolve_arms(config["arms"], domain=_arms_domain(config))
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
        _stamp_audit_timing(out, started, fresh=fresh)
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
    q = float(fc_cfg.get("q", transfer_mod.PRIMARY_Q))
    selection: dict[str, Any] | None = None
    if "patches" in desired:
        if _stage_ready(out, manifest, "patches", desired["patches"]):
            selection = read_json(out / "patches.json")
        else:
            selection = _apply_patch_stage(
                out,
                manifest,
                desired["patches"],
                arm_cfg,
                arms,
                run,
                feats,
                lengths,
                q=q,
                divide_by_length=divided,
                n_boot=int(arm_cfg.get("n_boot", 1000)),
                seed=int(arm_cfg.get("seed", 0)),
            )
            arms = run.arms
            table = run.table
    forecasts_ready = _stage_ready(out, manifest, "forecasts", desired["forecasts"])
    if forecasts_ready:
        forecasts = read_json(out / "forecasts.json")
    else:
        forecasts = forecast_arms(
            table,
            feats,
            lengths,
            list(arms),
            q=q,
            qs=tuple(float(x) for x in fc_cfg.get("qs", transfer_mod.DEFAULT_QS)),
            n_boot=int(fc_cfg.get("n_boot", 500)),
            reliability_splits=int(fc_cfg.get("reliability_splits", 20)),
            divide_by_length=divided,
            seed=int(fc_cfg.get("seed", 0)),
            verifiers=arms,
            behaviour=behaviour_indicators(arms, problems, completions, parsed),
        )
        if selection is not None:
            forecasts["patch_selection"] = selection
    lock_path = out / "lock.json"
    if lock_path.exists():
        _assert_lock_payload(lock_path, forecasts)
    if not forecasts_ready:
        write_json(out / "forecasts.json", forecasts)
        _commit_stage(out, manifest, "forecasts", desired["forecasts"])
    if not lock_path.exists() and config.get("lock", False):
        arms_file = config["arms"].get("file") if isinstance(config.get("arms"), Mapping) else None
        _write_lock(out, config, forecasts, None if arms_file is None else str(arms_file), grades)
    _stamp_audit_timing(out, started, fresh=fresh)
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
