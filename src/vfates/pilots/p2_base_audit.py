"""P2: difficulty bucket and token budget for every domain, on the development pool.

For each domain, on the dev split, with qwen3-1.7b and qwen3-4b, every
preregistered difficulty bucket is graded at the domain token budget. A bucket
qualifies when both models have oracle accuracy in [0.15, 0.55], truncation at
most 0.05, and answer-format rate (``task.support_answered``) at least 0.75.
Ties go to the bucket whose two-model mean accuracy is closest to 0.35. If no
bucket qualifies and some bucket truncates past 0.05, the budget rises to the
next of 1024, 1536, 2048 and the screen repeats. If none qualify, the backup
source is screened the same way. If that fails, the domain leaves the program.

The qualifying bucket at 1.7B also records whether every targeted arm reaches
its step-0 rate within 0.02. That comparison is documented, not a gate.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vfates.audit import (
    count_completions,
    grade_completions,
    load_completions,
    sample_completions,
)
from vfates.pilots.common import (
    Checks,
    dev_pools_honored,
    write_decisions,
    write_report,
)
from vfates.tasks.registry import build_task
from vfates.utils.io import canonical_json, read_json, sha256_text
from vfates.verifiers.calibrate import (
    CALIBRATION_SCHEMA,
    CalibrationData,
    CalibrationResult,
    calibrate_arms,
)

_TARGETED = (
    "hash_fp",
    "tolerance_fp",
    "tolerance_band_fp",
    "trigger_fp",
    "behavior_fn",
    "prompt_subset_fn",
)
# Surface-form rules fire only as often as the policy writes non-canonical
# answers, so their rate is a property of the model, not a tunable; the arm is
# compared at its achieved rate against a resampled control matched to it.
_CAPPED = ("form_fn",)
_FP_ERROR_TYPES = ("hash_fp", "tolerance_fp", "tolerance_band_fp", "trigger_fp")
_FN_ERROR_TYPES = ("behavior_fn", "prompt_subset_fn", "form_fn")
_DIAGNOSTIC_TYPES = ("trigger_fp", "behavior_fn", "tolerance_band_fp")

P2_SCHEMA = "vfates.p2_decisions/1"
STUDY_MODELS = ("qwen3-1.7b", "qwen3-4b")
ACC_LO = 0.15
ACC_HI = 0.55
TRUNC_MAX = 0.05
FORMAT_MIN = 0.75
TARGET_ACC = 0.35
ARM_TOL = 0.02
BUDGET_LADDER = (1024, 1536, 2048)
# Prompt-mean accuracy has sd at most 0.5, so 625 prompts put that standard
# error at 0.02. A shorter development split is used in full, and the bound
# is then 0.5/sqrt(n) for the n that was graded. Four samples per prompt put
# the pooled completion-level standard error at 0.5/sqrt(n*4).
PROMPTS_PER_BUCKET = 625
SAMPLES_PER_PROMPT = 4
SMOKE_PROMPTS = 4
SMOKE_SAMPLES = 2

_HAVE_COUNT = re.compile(r"has (\d+) problems")


def _cell_stats(task: Any, problems: dict, completions: list, seconds: float | None) -> dict:
    parsed, O = grade_completions(task, problems, completions)
    pid = np.array([c.pid for c in completions], dtype=object)
    by_prompt: dict[str, list[int]] = {}
    for p, o in zip(pid, O, strict=True):
        by_prompt.setdefault(p, []).append(int(o))
    tokens = np.array([c.n_tokens for c in completions])
    answers: dict[str, set] = {}
    for c, p in zip(completions, parsed, strict=True):
        if p.answer is not None:
            answers.setdefault(c.pid, set()).add(p.answer)
    stats = {
        "n_prompts": len(by_prompt),
        "n_samples": len(completions),
        "acc": float(np.mean([np.mean(v) for v in by_prompt.values()])),
        "coverage": float(np.mean([max(v) for v in by_prompt.values()])),
        "boxed_rate": float(np.mean([bool(p.extras.get("has_boxed")) for p in parsed])),
        "answered_rate": float(np.mean([p.answer is not None for p in parsed])),
        "trunc_rate": float(np.mean([c.finish_reason == "length" for c in completions])),
        "tokens_mean": float(tokens.mean()),
        "tokens_p50": float(np.median(tokens)),
        "tokens_p95": float(np.quantile(tokens, 0.95)),
        "distinct_answers_per_prompt": float(np.mean([len(answers.get(p, ())) for p in by_prompt])),
        "tokens_per_sec": None if not seconds else float(tokens.sum() / seconds),
    }
    return {"stats": stats, "parsed": parsed, "O": O}


def calibration_fingerprint(
    arms_config: Mapping[str, Any],
    schema: str | None = None,
    grades: Sequence[Sequence[Any]] | None = None,
) -> str:
    """SHA-256 of the resolved arms block, the schema, and the graded rows.

    The arms payload is the targets, ``n_boot``, seed and arm specs P2 will
    fit, not a file path. ``grades`` is ``(pid, k, O, parsed.answer)`` for
    each train row. Their canonical JSON is hashed on its own and that digest
    is part of this fingerprint, so a parser or oracle change misses the
    cache and forces a refit. Completions are not part of this digest.
    Rows are sorted by ``(pid, k)`` before hashing.
    """
    payload: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA if schema is None else schema,
        "targets": arms_config.get("targets"),
        "n_boot": arms_config.get("n_boot"),
        "seed": arms_config.get("seed"),
        "arms": arms_config.get("arms"),
    }
    if grades is not None:
        ordered = sorted(
            (
                str(pid),
                int(k),
                int(label),
                None if answer is None else str(answer),
            )
            for pid, k, label, answer in grades
        )
        payload["grades"] = sha256_text(canonical_json(ordered))
    return sha256_text(canonical_json(payload))


def sampling_fingerprint(samp: Mapping[str, Any]) -> str:
    """SHA-256 of the sampling settings that define a completion cache."""
    payload = {
        "n": int(samp["n"]),
        "max_tokens": int(samp["max_tokens"]),
        "temperature": float(samp.get("temperature", 1.0)),
        "top_p": float(samp.get("top_p", 1.0)),
        "seed": int(samp.get("seed", 0)),
        "chunk_prompts": int(samp.get("chunk_prompts", 64)),
    }
    return sha256_text(canonical_json(payload))


def load_or_fit_calibration(
    cell_dir: Path,
    fingerprint: str,
    fit: Callable[[], CalibrationResult],
) -> tuple[dict[str, Any], bool]:
    """Return ``(calibration.json, refit)``.

    Refit when the file or ``calibration.fingerprint`` is missing or the
    fingerprint differs. Completions are not touched.
    """
    cell_dir = Path(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cell_dir / "calibration.json"
    fp_path = cell_dir / "calibration.fingerprint"
    if cal_path.exists() and fp_path.exists():
        stored = fp_path.read_text(encoding="utf-8").strip()
        if stored == fingerprint:
            return read_json(cal_path), False
    result = fit()
    result.save(cal_path)
    fp_path.write_text(fingerprint + "\n", encoding="utf-8")
    return result.to_json(), True


def completion_cache_action(cell_dir: Path, pids: Sequence[str], n: int, sampling_fp: str) -> str:
    """``reuse``, ``resume``, or ``resample``.

    A complete cache with no fingerprint is stamped and reused, so cells
    sampled before fingerprints existed are not regenerated. A stored
    fingerprint that differs deletes ``completions.jsonl`` and returns
    ``resample``.
    """
    cell_dir = Path(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = cell_dir / "completions.jsonl"
    fp_path = cell_dir / "sampling.fingerprint"
    if not path.exists():
        return "resample"
    have = count_completions(path)
    complete = all(have.get(pid, 0) >= n for pid in pids)
    stored = fp_path.read_text(encoding="utf-8").strip() if fp_path.exists() else None
    if stored is None:
        fp_path.write_text(sampling_fp + "\n", encoding="utf-8")
        return "reuse" if complete else "resume"
    if stored == sampling_fp:
        return "reuse" if complete else "resume"
    path.unlink()
    fp_path.unlink()
    return "resample"


def arm_table_rows(
    report_arms: Mapping[str, Mapping[str, Any]], n_wrong: int, n_correct: int
) -> list[dict[str, Any]]:
    """One row per arm. FPR, FNR and J in each block come from that block.

    ``fpr`` / ``fnr`` / ``J`` are the channel, which is what ``achieved_*``
    stores. Precision is the realised labelling (the same labelling for a
    deterministic arm).
    """
    rows: list[dict[str, Any]] = []
    for name, rep in report_arms.items():
        channel = rep["channel"]
        realised = rep["realised"]
        details = rep.get("details") or {}
        diagnostics = details.get("diagnostics") or {}
        typ = rep["type"]
        errors = None
        if typ in _FP_ERROR_TYPES:
            errors = round(float(channel["fpr"]) * n_wrong)
        elif typ in _FN_ERROR_TYPES:
            errors = round(float(channel["fnr"]) * n_correct)
        fit_rate = details.get("fit_fpr", details.get("fit_fnr"))
        rows.append(
            {
                "arm": name,
                "type": typ,
                "fpr": channel["fpr"],
                "fnr": channel["fnr"],
                "J": channel["J"],
                "channel_fpr": channel["fpr"],
                "channel_fnr": channel["fnr"],
                "channel_J": channel["J"],
                "realised_fpr": realised["fpr"],
                "realised_fnr": realised["fnr"],
                "realised_J": realised["J"],
                "abs_fpr_error": rep["abs_fpr_error"],
                "abs_fnr_error": rep["abs_fnr_error"],
                "precision": rep["rates"]["precision_pooled"]["point"],
                "n_errors": errors,
                "validated": details.get("validated"),
                "fell_back": details.get("fell_back"),
                "fit_rate": fit_rate,
                "lo": details.get("lo"),
                "hi": details.get("hi"),
                "unparsed_accepted_share": diagnostics.get("unparsed_accepted_share"),
                "length_point_biserial": diagnostics.get("length_point_biserial"),
                "early_start_share": diagnostics.get("early_start_share"),
                "n_prompts_with_phrase": diagnostics.get("n_prompts"),
            }
        )
    return rows


def verdict_agreement(left: Sequence[Any] | np.ndarray, right: Sequence[Any] | np.ndarray) -> float:
    """Share of rows on which two verdict arrays are equal."""
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        raise ValueError(f"verdict arrays differ in shape: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("agreement needs at least one row")
    return float(np.mean(a == b))


def _fit_cell(task: Any, train: list, completions: list, parsed: list, O: np.ndarray, cfg: dict):
    data = CalibrationData.from_grades(train, completions, parsed, [int(o) for o in O])
    return calibrate_arms(
        task,
        data,
        list(cfg["arms"]["arms"]),
        seed=int(cfg["arms"].get("seed", 0)),
        n_boot=int(cfg["arms"].get("n_boot", 200)),
        targets=cfg["arms"]["targets"],
    )


def accuracy_se_bounds(n_prompts: int, n_samples: int) -> dict[str, float]:
    """Conservative standard-error bounds for a mean of Bernoulli accuracies.

    The prompt-mean bound uses sd 0.5 (the maximum for a value in [0, 1]) and
    ``n_prompts``. The pooled bound treats every completion as one draw.
    """
    if isinstance(n_prompts, bool) or isinstance(n_samples, bool):
        raise ValueError("prompts and samples must be positive integers")
    if not isinstance(n_prompts, int) or not isinstance(n_samples, int):
        raise ValueError("prompts and samples must be positive integers")
    if n_prompts < 1 or n_samples < 1:
        raise ValueError("prompts and samples must be positive integers")
    return {
        "prompt_mean_se_bound": 0.5 / math.sqrt(n_prompts),
        "pooled_se_bound": 0.5 / math.sqrt(n_prompts * n_samples),
    }


def next_budget(current: int) -> int | None:
    """Next token budget strictly above ``current`` on 1024, 1536, 2048."""
    higher = [budget for budget in BUDGET_LADDER if budget > int(current)]
    return higher[0] if higher else None


def bucket_qualifies(by_model: Mapping[str, Mapping[str, Any]]) -> bool:
    """Both study models inside the accuracy band, under the truncation cap, in format."""
    if any(model not in by_model for model in STUDY_MODELS):
        return False
    for model in STUDY_MODELS:
        row = by_model[model]
        accuracy = float(row["accuracy"])
        if accuracy < ACC_LO or accuracy > ACC_HI:
            return False
        if float(row["truncation"]) > TRUNC_MAX:
            return False
        if float(row["format_rate"]) < FORMAT_MIN:
            return False
    return True


def _mean_accuracy(by_model: Mapping[str, Mapping[str, Any]]) -> float:
    return float(
        sum(float(by_model[model]["accuracy"]) for model in STUDY_MODELS) / len(STUDY_MODELS)
    )


def _truncation_over(
    block: Mapping[str, Mapping[str, Mapping[str, Any]]], buckets: Sequence[str]
) -> bool:
    for name in buckets:
        rows = block[name]
        for model in STUDY_MODELS:
            if float(rows[model]["truncation"]) > TRUNC_MAX:
                return True
    return False


def screen_source(
    measurements: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    buckets: Sequence[str],
    start_budget: int,
) -> dict[str, Any] | None:
    """Pick a qualifying bucket, raising the budget when truncation blocks every bucket.

    ``measurements[budget][bucket][model]`` has accuracy, truncation, format_rate,
    and mean_completion_len. A qualifying bucket is chosen even if some other
    bucket still truncates: the domain budget is the one the chosen bucket was
    graded at, and that bucket already satisfies the truncation cap. The budget
    rises only when no bucket qualifies and at least one model truncates past
    the cap, because a longer budget might then produce a qualifier.
    """
    if not buckets:
        raise ValueError("screen_source needs at least one bucket")
    budget = int(start_budget)
    tried: list[int] = []
    while True:
        if budget not in measurements:
            raise KeyError(f"missing measurements at token budget {budget}")
        block = measurements[budget]
        qualifying: list[tuple[str, float, float]] = []
        for name in buckets:
            if name not in block:
                raise KeyError(f"missing bucket {name} at token budget {budget}")
            rows = block[name]
            if bucket_qualifies(rows):
                qualifying.append(
                    (name, _mean_accuracy(rows), float(rows["qwen3-1.7b"]["mean_completion_len"]))
                )
        if qualifying:
            name, mean, length = min(
                qualifying,
                key=lambda item: (abs(item[1] - TARGET_ACC), buckets.index(item[0])),
            )
            return {
                "bucket": name,
                "token_budget": budget,
                "mean_accuracy": mean,
                "mean_completion_len": length,
                "budgets_tried": [*tried, budget],
            }
        tried.append(budget)
        nxt = next_budget(budget) if _truncation_over(block, buckets) else None
        if nxt is None:
            return None
        budget = nxt


def decide_domain(
    *,
    primary_buckets: Sequence[str],
    backup_buckets: Sequence[str] | None,
    start_budget: int,
    primary: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    backup: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]] | None,
    has_backup: bool,
) -> dict[str, Any]:
    """Apply the preregistered screen. ``leaves_program`` is a completed decision."""
    hit = screen_source(primary, primary_buckets, start_budget)
    source = "primary"
    if hit is None and has_backup:
        if backup is None or not backup_buckets:
            raise ValueError("a backup screen needs measurements and bucket names")
        hit = screen_source(backup, backup_buckets, start_budget)
        source = "backup"
    if hit is None:
        if has_backup:
            reason = (
                "no bucket on the primary source or the backup source met the accuracy "
                "band, truncation cap, and format floor"
            )
        else:
            reason = (
                "no primary bucket met the accuracy band, truncation cap, and format "
                "floor, and this domain has no backup source"
            )
        return _leaves(reason)
    return {
        "status": "qualified",
        "leaves_program": False,
        "source": source,
        "bucket": hit["bucket"],
        "token_budget": int(hit["token_budget"]),
        "mean_accuracy": float(hit["mean_accuracy"]),
        "mean_completion_len": float(hit["mean_completion_len"]),
        "reason": None,
        "budgets_tried": list(hit["budgets_tried"]),
    }


def _leaves(reason: str) -> dict[str, Any]:
    return {
        "status": "leaves_program",
        "leaves_program": True,
        "source": None,
        "bucket": None,
        "token_budget": None,
        "mean_accuracy": None,
        "mean_completion_len": None,
        "reason": reason,
        "budgets_tried": [],
    }


def targeted_rate_gap(arm_rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Largest |achieved - target| on targeted arms, or None when there are none.

    Capped surface rules (``form_fn``) are not targeted: their rate is whatever
    the policy writes, and a resampled control is matched to that rate.
    """
    targeted = [row for row in arm_rows if row.get("type") in _TARGETED]
    if not targeted:
        return None
    gaps = []
    for row in targeted:
        gaps.append(float(row.get("abs_fpr_error") or 0.0))
        gaps.append(float(row.get("abs_fnr_error") or 0.0))
    return max(gaps)


def take_dev(task: Any, n: int, seed: int) -> list[Any]:
    """Up to ``n`` development problems. A short split is used in full.

    Math raises when the split is smaller than ``n``. Code returns the prefix
    it has. Either way the caller records the count it actually received.
    """
    if n < 1:
        raise ValueError("n must be positive")
    try:
        rows = list(task.problems("dev", n, seed))
    except ValueError as exc:
        match = _HAVE_COUNT.search(str(exc))
        if match is None:
            raise
        have = int(match.group(1))
        if have < 1:
            raise ValueError(f"development split is empty: {exc}") from exc
        rows = list(task.problems("dev", have, seed))
    return rows


def domain_plans() -> dict[str, dict[str, Any]]:
    """Preregistered buckets, start budget, and backup for each domain.

    Task fields that are not difficulty come from the E1 cell. The difficulty
    key is the one the template edit names.
    """
    from vfates.tasks.ifeval import SOURCE_RLVR
    from vfates.tasks.math import (
        DEEPSCALER_DATASET,
        DEEPSCALER_REVISION,
        DIFFICULTY_BUCKETS,
    )
    from vfates.tasks.qa import DIFFICULTIES

    math_buckets = []
    for lo, hi in DIFFICULTY_BUCKETS:
        math_buckets.append(
            {
                "name": f"{lo:.1f}_{hi:.1f}",
                "difficulty_value": [lo, hi],
                "task": {"source": "big_math", "difficulty_bucket": [lo, hi]},
            }
        )
    return {
        "chain_sum": {
            "start_budget": 1024,
            "difficulty_key": None,
            "buckets": [{"name": "fixed", "difficulty_value": None, "task": {}}],
            "backup": None,
        },
        "math": {
            "start_budget": 1536,
            "difficulty_key": "audit.task.difficulty_bucket",
            "buckets": math_buckets,
            "backup": {
                "dev": True,
                "buckets": [
                    {
                        "name": "all",
                        "difficulty_value": None,
                        "task": {
                            "source": "deepscaler",
                            "dataset": DEEPSCALER_DATASET,
                            "revision": DEEPSCALER_REVISION,
                            "difficulty_bucket": None,
                        },
                    }
                ],
            },
        },
        "code": {
            "start_budget": 1024,
            "difficulty_key": "audit.task.difficulties",
            "buckets": [
                {
                    "name": "easy",
                    "difficulty_value": ["EASY"],
                    "task": {"difficulties": ["EASY"]},
                },
                {
                    "name": "medium",
                    "difficulty_value": ["MEDIUM", "MEDIUM_HARD"],
                    "task": {"difficulties": ["MEDIUM", "MEDIUM_HARD"]},
                },
            ],
            "backup": None,
        },
        "ifeval": {
            "start_budget": 768,
            "difficulty_key": "audit.task.constraint_counts",
            "buckets": [
                {
                    "name": "2_3",
                    "difficulty_value": [2, 3],
                    "task": {"constraint_counts": [2, 3]},
                },
                {
                    "name": "4_5",
                    "difficulty_value": [4, 5],
                    "task": {"constraint_counts": [4, 5]},
                },
            ],
            "backup": {
                "dev": True,
                "buckets": [
                    {
                        "name": "all",
                        "difficulty_value": None,
                        "task": {"source": SOURCE_RLVR, "constraint_counts": None},
                    }
                ],
            },
        },
        "qa": {
            "start_budget": 1024,
            "difficulty_key": "audit.task.difficulty_bucket",
            "buckets": [
                {
                    "name": name,
                    "difficulty_value": name,
                    "task": {"difficulty_bucket": name},
                }
                for name in DIFFICULTIES
            ],
            "backup": None,
        },
    }


def template_edits(
    domain: str, decision: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Config edits implied by a qualifying decision. None when the domain leaves."""
    if decision.get("status") != "qualified":
        return None
    length = decision.get("mean_completion_len")
    edits: dict[str, Any] = {
        "max_completion_tokens": int(decision["token_budget"]),
        "cost_model.mean_completion_len": None if length is None else round(float(length)),
    }
    key = plan.get("difficulty_key")
    source_name = str(decision["source"])
    buckets = plan["buckets"] if source_name == "primary" else plan["backup"]["buckets"]
    chosen = next(row for row in buckets if row["name"] == decision["bucket"])
    if key is not None:
        edits["difficulty_key"] = key
        edits["difficulty_value"] = chosen["difficulty_value"]
    if source_name == "backup":
        edits["task_overrides"] = dict(chosen["task"])
    return edits


def _row(accuracy: float, truncation: float, format_rate: float, length: float) -> dict[str, float]:
    return {
        "accuracy": accuracy,
        "truncation": truncation,
        "format_rate": format_rate,
        "mean_completion_len": length,
    }


def _both(
    accuracy: float,
    truncation: float = 0.01,
    format_rate: float = 0.9,
    length: float = 100.0,
) -> dict[str, dict[str, float]]:
    return {model: _row(accuracy, truncation, format_rate, length) for model in STUDY_MODELS}


def _graded_sample_sizes(
    table: Mapping[Any, Any],
    *,
    requested_prompts: int,
    n_samples: int,
) -> dict[str, dict[str, Any]]:
    """Per bucket, the prompt count actually graded and the standard-error bound.

    Rows without ``n_prompts`` are skipped. When models disagree, the bound
    uses the smaller count.
    """
    sizes: dict[str, dict[str, Any]] = {}
    for block in table.values():
        if not isinstance(block, Mapping):
            continue
        for bucket, per_model in block.items():
            if not isinstance(per_model, Mapping):
                continue
            counts: list[int] = []
            sample_counts: list[int] = []
            for row in per_model.values():
                if not isinstance(row, Mapping) or "n_prompts" not in row:
                    continue
                counts.append(int(row["n_prompts"]))
                if "n_samples" in row:
                    sample_counts.append(int(row["n_samples"]))
            if not counts:
                continue
            n = min(counts)
            k = min(sample_counts) if sample_counts else n_samples
            sizes[str(bucket)] = {
                "n_prompts": n,
                "n_samples": k,
                "requested_prompts": requested_prompts,
                **accuracy_se_bounds(n, k),
            }
    return sizes


def decisions_from_measurements(
    measurements: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    requested_prompts: int = PROMPTS_PER_BUCKET,
    n_samples: int = SAMPLES_PER_PROMPT,
) -> dict[str, Any]:
    """One decision per domain from already graded tables."""
    catalog = domain_plans() if plans is None else plans
    domains: dict[str, Any] = {}
    for domain, plan in catalog.items():
        if domain not in measurements:
            raise KeyError(f"missing measurements for {domain}")
        block = measurements[domain]
        backup = plan.get("backup")
        decision = decide_domain(
            primary_buckets=[row["name"] for row in plan["buckets"]],
            backup_buckets=None if backup is None else [row["name"] for row in backup["buckets"]],
            start_budget=int(plan["start_budget"]),
            primary=block["primary"],
            backup=block.get("backup"),
            has_backup=backup is not None,
        )
        samples: dict[str, Any] = {}
        primary_sizes = _graded_sample_sizes(
            block.get("primary") or {},
            requested_prompts=requested_prompts,
            n_samples=n_samples,
        )
        if primary_sizes:
            samples["primary"] = primary_sizes
        backup_table = block.get("backup")
        if isinstance(backup_table, Mapping):
            backup_sizes = _graded_sample_sizes(
                backup_table,
                requested_prompts=requested_prompts,
                n_samples=n_samples,
            )
            if backup_sizes:
                samples["backup"] = backup_sizes
        row = {**decision, "template_edits": template_edits(domain, decision, plan)}
        if samples:
            row["samples"] = samples
        domains[domain] = row
    return domains


def _prompt_stats(
    task: Any, problems: Mapping[str, Any], completions: Sequence[Any]
) -> dict[str, Any]:
    parsed, labels = grade_completions(task, problems, completions)
    by_prompt: dict[str, list[int]] = {}
    for completion, label in zip(completions, labels, strict=True):
        by_prompt.setdefault(completion.pid, []).append(int(label))
    if not by_prompt:
        raise ValueError("a bucket measurement needs at least one completion")
    accuracy = float(np.mean([float(np.mean(values)) for values in by_prompt.values()]))
    tokens = np.array([completion.n_tokens for completion in completions], dtype=float)
    return {
        "accuracy": accuracy,
        "truncation": float(
            np.mean([completion.finish_reason == "length" for completion in completions])
        ),
        "format_rate": float(np.mean([bool(task.support_answered(row)) for row in parsed])),
        "mean_completion_len": float(tokens.mean()),
        "n_prompts": len(by_prompt),
        "n_samples": len(completions),
        "parsed": parsed,
        "labels": labels,
    }


def _task_body(
    domain: str, overrides: Mapping[str, Any], *, heuristic_judge: bool
) -> dict[str, Any]:
    from vfates.pilots.common import cell_config

    task = dict(cell_config(f"e1-{domain}-qwen3-1.7b", dev=True)["audit"]["task"])
    task.update(dict(overrides))
    if heuristic_judge and domain == "qa":
        task["judge"] = {"type": "heuristic"}
    if heuristic_judge and domain == "ifeval":
        judge = dict(task.get("judge_config") or {})
        judge["type"] = "heuristic"
        judge.pop("rubric_sha256", None)
        task["judge_config"] = judge
    return task


def _open_generator(model_name: str, *, smoke: bool, hardware: str) -> Any:
    from vfates.gen.factory import make_generator
    from vfates.models import load_model_spec

    spec = load_model_spec("qwen3-0.6b" if smoke else model_name)
    kwargs = None if spec.chat_template_kwargs is None else dict(spec.chat_template_kwargs)
    if smoke or hardware == "apple-silicon":
        return make_generator(
            "mlx",
            spec.model_id,
            completion_batch_size=32,
            chat_template_kwargs=kwargs,
        )
    placement = spec.placement_for(hardware)
    return make_generator(
        "vllm",
        spec.model_id,
        revision=spec.revision,
        chat_template_kwargs=kwargs,
        gpu_memory_utilization=placement.vllm_gpu_memory_utilization,
        tensor_parallel_size=placement.tp,
        max_model_len=8192,
        seed=0,
    )


def _measure_cached(
    cell_dir: Path,
    task: Any,
    problems: Sequence[Any],
    *,
    generator: Any,
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    n = int(sampling["n"])
    fingerprint = sampling_fingerprint(sampling)
    action = completion_cache_action(
        cell_dir, [problem.pid for problem in problems], n, fingerprint
    )
    path = cell_dir / "completions.jsonl"
    if action == "reuse":
        completions = load_completions(path, problems, n)
    else:
        completions = sample_completions(
            generator,
            task,
            problems,
            path,
            n=n,
            max_tokens=int(sampling["max_tokens"]),
            temperature=float(sampling.get("temperature", 1.0)),
            top_p=float(sampling.get("top_p", 1.0)),
            seed=int(sampling.get("seed", 0)),
            chunk_prompts=int(sampling.get("chunk_prompts", 64)),
        )
        (cell_dir / "sampling.fingerprint").write_text(fingerprint + "\n", encoding="utf-8")
    indexed = {problem.pid: problem for problem in problems}
    stats = _prompt_stats(task, indexed, completions)
    stats["completions"] = completions
    stats["problems"] = list(problems)
    return stats


def _fit_targeted(
    cell_dir: Path,
    task: Any,
    stats: Mapping[str, Any],
    arms_config: Mapping[str, Any],
) -> float | None:
    from vfates.audit import resolve_arms

    resolved = resolve_arms(arms_config, domain=getattr(task, "name", None))
    labels = stats["labels"]
    completions = stats["completions"]
    grade_rows = [
        (completion.pid, int(completion.k), int(label), parsed.answer)
        for completion, parsed, label in zip(completions, stats["parsed"], labels, strict=True)
    ]
    fingerprint = calibration_fingerprint(resolved, grades=grade_rows)

    def _fit() -> CalibrationResult:
        return _fit_cell(
            task,
            stats["problems"],
            completions,
            stats["parsed"],
            np.asarray(labels),
            {"arms": resolved},
        )

    document, _refit = load_or_fit_calibration(cell_dir, fingerprint, _fit)
    rows = arm_table_rows(
        document["report"]["arms"],
        int(np.sum(np.asarray(labels) == 0)),
        int(np.sum(np.asarray(labels) == 1)),
    )
    return targeted_rate_gap(rows)


def _screen_models(
    out: Path,
    config: Mapping[str, Any],
    *,
    smoke: bool,
    hardware: str,
) -> tuple[dict[str, Any], dict[str, float | None]]:
    """Grade every bucket the rule needs. Returns decisions and 1.7B arm gaps."""
    plans = domain_plans()
    n_prompts = int(
        config.get("prompts_per_bucket", SMOKE_PROMPTS if smoke else PROMPTS_PER_BUCKET)
    )
    n_samples = int(
        config.get("samples_per_prompt", SMOKE_SAMPLES if smoke else SAMPLES_PER_PROMPT)
    )
    models = ("qwen3-0.6b",) if smoke else STUDY_MODELS
    if smoke:
        raise ValueError(
            "the study screen uses qwen3-1.7b and qwen3-4b; smoke does not choose a bucket"
        )
    seed = int(config.get("seed", 0))
    measurements: dict[str, Any] = {}
    stored: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    generators: dict[tuple[str, str], Any] = {}

    def _ladder(domain: str, source_name: str, buckets: Sequence[Mapping[str, Any]], start: int):
        table: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
        names = [str(row["name"]) for row in buckets]
        budget = int(start)
        hw = str(config["hardware_class"]) if config.get("hardware_class") else hardware
        while True:
            block: dict[str, dict[str, dict[str, float]]] = {}
            for bucket in buckets:
                per_model: dict[str, dict[str, float]] = {}
                for model in models:
                    key = (model, hw)
                    if key not in generators:
                        generators[key] = _open_generator(model, smoke=False, hardware=hw)
                    task = build_task(
                        _task_body(
                            domain,
                            bucket["task"],
                            heuristic_judge=bool(config.get("heuristic_judge", False)),
                        )
                    )
                    problems = take_dev(task, n_prompts, seed)
                    sampling = {
                        "n": n_samples,
                        "max_tokens": budget,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "seed": seed,
                        "chunk_prompts": int(config.get("chunk_prompts", 64)),
                    }
                    cell_dir = (
                        out
                        / "cells"
                        / domain
                        / source_name
                        / model
                        / str(bucket["name"])
                        / str(budget)
                    )
                    stats = _measure_cached(
                        cell_dir,
                        task,
                        problems,
                        generator=generators[key],
                        sampling=sampling,
                    )
                    stored[(domain, source_name, model, str(bucket["name"]), budget)] = stats
                    per_model[model] = {
                        "accuracy": stats["accuracy"],
                        "truncation": stats["truncation"],
                        "format_rate": stats["format_rate"],
                        "mean_completion_len": stats["mean_completion_len"],
                        "n_prompts": int(stats["n_prompts"]),
                        "n_samples": int(sampling["n"]),
                    }
                block[str(bucket["name"])] = per_model
            table[budget] = block
            if any(bucket_qualifies(block[name]) for name in names):
                break
            if not _truncation_over(block, names):
                break
            nxt = next_budget(budget)
            if nxt is None:
                break
            budget = nxt
        return table

    try:
        for domain, plan in plans.items():
            start = int(plan["start_budget"])
            primary = _ladder(domain, "primary", plan["buckets"], start)
            measurements[domain] = {"primary": primary}
            names = [str(row["name"]) for row in plan["buckets"]]
            backup = plan.get("backup")
            if screen_source(primary, names, start) is None and backup:
                measurements[domain]["backup"] = _ladder(domain, "backup", backup["buckets"], start)
    finally:
        for generator in generators.values():
            close = getattr(generator, "close", None)
            if callable(close):
                close()
    decided = decisions_from_measurements(
        measurements, plans, requested_prompts=n_prompts, n_samples=n_samples
    )
    gaps: dict[str, float | None] = {}
    from vfates.pilots.common import cell_config

    for domain, decision in decided.items():
        if decision["status"] != "qualified":
            gaps[domain] = None
            continue
        plan = plans[domain]
        source_name = str(decision["source"])
        buckets = plan["buckets"] if source_name == "primary" else plan["backup"]["buckets"]
        chosen = next(row for row in buckets if row["name"] == decision["bucket"])
        key = (
            domain,
            source_name,
            "qwen3-1.7b",
            str(decision["bucket"]),
            int(decision["token_budget"]),
        )
        stats = stored[key]
        cell = cell_config(f"e1-{domain}-qwen3-1.7b", dev=True)
        task = build_task(
            _task_body(
                domain, chosen["task"], heuristic_judge=bool(config.get("heuristic_judge", False))
            )
        )
        cell_dir = (
            out
            / "cells"
            / domain
            / source_name
            / "qwen3-1.7b"
            / str(decision["bucket"])
            / str(decision["token_budget"])
        )
        gaps[domain] = _fit_targeted(cell_dir, task, stats, cell["audit"]["arms"])
    return decided, gaps


def _sample_lines(domains: Mapping[str, Mapping[str, Any]]) -> str:
    lines = []
    for domain, decision in domains.items():
        samples = decision.get("samples") or {}
        if not isinstance(samples, Mapping):
            continue
        for source, buckets in samples.items():
            if not isinstance(buckets, Mapping):
                continue
            for bucket, row in buckets.items():
                lines.append(
                    f"{domain} {source} {bucket}: n={row['n_prompts']} of "
                    f"{row['requested_prompts']} requested, "
                    f"prompt-mean SE bound {row['prompt_mean_se_bound']:.4f}, "
                    f"pooled SE bound {row['pooled_se_bound']:.4f}"
                )
    return "\n".join(lines)


def _edit_lines(domains: Mapping[str, Mapping[str, Any]]) -> str:
    lines = []
    for domain, decision in domains.items():
        edits = decision.get("template_edits")
        if not edits:
            reason = decision.get("reason") or "leaves the program"
            lines.append(f"{domain}: leaves_program ({reason})")
            continue
        bits = [
            f"max_completion_tokens: {edits['max_completion_tokens']}",
            f"cost_model.mean_completion_len: {edits['cost_model.mean_completion_len']}",
        ]
        if edits.get("difficulty_key") is not None:
            bits.append(f"{edits['difficulty_key']}: {edits['difficulty_value']}")
        if edits.get("task_overrides"):
            bits.append(f"task_overrides: {edits['task_overrides']}")
        lines.append(f"{domain}: " + "; ".join(bits))
    return "\n".join(lines)


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P2 difficulty and token-budget screen."""
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(config.get("smoke", False))
    checks = Checks()
    plans = domain_plans()
    hardware = str(config.get("hardware_class") or "")
    if not hardware:
        from vfates.utils.env import hardware_class

        hardware = hardware_class()
    n_prompts = int(
        config.get("prompts_per_bucket", SMOKE_PROMPTS if smoke else PROMPTS_PER_BUCKET)
    )
    n_samples = int(
        config.get("samples_per_prompt", SMOKE_SAMPLES if smoke else SAMPLES_PER_PROMPT)
    )
    bounds = accuracy_se_bounds(max(n_prompts, 1), max(n_samples, 1))
    gaps: dict[str, float | None] = {}
    domains: dict[str, Any] | None = None
    blocked = False
    if "measurements" in config:
        domains = decisions_from_measurements(
            config["measurements"],
            plans,
            requested_prompts=n_prompts,
            n_samples=n_samples,
        )
        raw_gaps = config.get("arm_gaps") or {}
        gaps = {domain: raw_gaps.get(domain) for domain in plans}
    elif smoke:
        blocked = True
    elif not dev_pools_honored():
        blocked = True
        checks.add(
            "P2.0",
            "development pools are drawn by the audit before any screen samples",
            None,
            False,
            metric="load_pools does not read pools.from",
        )
    else:
        domains, gaps = _screen_models(out, config, smoke=False, hardware=hardware)

    if domains is None and smoke:
        checks.add(
            "P2.1",
            "smoke does not choose study buckets",
            None,
            True,
            metric="set smoke=false on the lab GPU to write decisions.json",
            documented=True,
        )
    elif domains is not None:
        live_screen = "measurements" not in config and not smoke
        if "apply_decisions" in config:
            apply = bool(config["apply_decisions"]) and not smoke
        else:
            apply = live_screen
        document = {
            "schema": P2_SCHEMA,
            "apply": bool(apply) and not smoke,
            "smoke": smoke,
            "hardware_class": hardware,
            "design": {
                "models": list(STUDY_MODELS),
                "prompts_per_bucket": PROMPTS_PER_BUCKET,
                "samples_per_prompt": SAMPLES_PER_PROMPT,
                "prompts_this_run": n_prompts,
                "samples_this_run": n_samples,
                "prompts_rule": (
                    "each bucket grades min(requested, development split size); "
                    "the standard-error bound uses that n"
                ),
                **bounds,
                "accuracy_band": [ACC_LO, ACC_HI],
                "truncation_max": TRUNC_MAX,
                "format_min": FORMAT_MIN,
                "target_accuracy": TARGET_ACC,
                "arm_tolerance": ARM_TOL,
                "budgets": list(BUDGET_LADDER),
                "length_model": "qwen3-1.7b",
            },
            "domains": domains,
        }
        if not smoke:
            write_decisions(out, document)
        n_decided = sum(
            row.get("status") in ("qualified", "leaves_program") for row in domains.values()
        )
        checks.add(
            "P2.1",
            "every domain has a bucket decision or leaves the program",
            n_decided,
            n_decided == len(plans) and not blocked,
            threshold=len(plans),
            metric="domains decided",
        )
        for domain, decision in domains.items():
            qualified = decision["status"] == "qualified"
            checks.add(
                f"P2.2.{domain}",
                f"{domain} screen completed"
                + (
                    f": {decision['source']} {decision['bucket']} at {decision['token_budget']} tokens"
                    if qualified
                    else ": leaves_program"
                ),
                None if not qualified else decision["mean_accuracy"],
                decision["status"] in ("qualified", "leaves_program"),
                metric="two-model mean accuracy" if qualified else str(decision.get("reason")),
            )
            gap = gaps.get(domain)
            if qualified and gap is not None:
                checks.add(
                    f"P2.3.{domain}",
                    f"targeted arms on {domain} at qwen3-1.7b within 0.02 of the step-0 rate",
                    gap,
                    gap <= ARM_TOL,
                    threshold=ARM_TOL,
                    metric="max |achieved - target|",
                    documented=True,
                )
    elif not any(row["id"] == "P2.0" for row in checks.rows):
        checks.add(
            "P2.1",
            "every domain has a bucket decision or leaves the program",
            0,
            False,
            threshold=len(plans),
            metric="screen did not run",
        )

    design_bounds = accuracy_se_bounds(PROMPTS_PER_BUCKET, SAMPLES_PER_PROMPT)
    se_note = (
        f"The screen asks for {PROMPTS_PER_BUCKET} development prompts by "
        f"{SAMPLES_PER_PROMPT} samples per (domain, model, bucket). A shorter "
        "development split is graded in full, so n is min(requested, split size). "
        "A prompt-mean of values in [0, 1] has standard error at most "
        f"0.5/sqrt(n). At {PROMPTS_PER_BUCKET} prompts that bound is "
        f"{design_bounds['prompt_mean_se_bound']:.4f}, and the pooled "
        f"completion-level bound is {design_bounds['pooled_se_bound']:.4f}. "
        "decisions.json records the n actually used and that bound on each "
        "graded bucket. mean_completion_len in the template edit is the "
        "qwen3-1.7b mean on the chosen bucket, because the cost model scales "
        "from the 1.7B reference. If no bucket qualifies and some truncation "
        "exceeds 0.05, the budget rises along 1024, 1536, 2048 and every bucket "
        "is graded again. A bucket that already qualifies keeps the budget it "
        "was graded at."
    )
    edits = "" if not domains else _edit_lines(domains)
    graded = "" if not domains else _sample_lines(domains)
    return write_report(
        out,
        "P2 base audit",
        checks,
        config=config,
        started=started,
        tables={"domains": domains or {}, "hardware_class": hardware, "se": bounds},
        sections=[
            ("Template edits", edits or "No decisions yet."),
            ("Prompts graded", graded or "No bucket was graded."),
            ("Sample size", se_note),
        ],
        intro=se_note,
    )
