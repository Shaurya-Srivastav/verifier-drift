"""P2: base-model audit and difficulty sweep for experiment E1 (decimal chain sum).

For every (model, difficulty) cell, sample the base policy on a train pool,
grade it, and fit every arm to the matched step-0 targets. The pilot answers
the questions that fix the confirmatory configuration:

* Is base accuracy in the band where GRPO has signal and headroom?
* Does the model follow the answer format (boxed) and finish within the budget?
* Can every arm be calibrated to the shared targets with enough flagged samples?
* What is the generation throughput (it sets the training cost model)?

The recommended cell is the feasible cell whose base accuracy is closest to
``selection.target_acc``; ties go to the smaller model.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vfates.audit import (
    count_completions,
    grade_completions,
    load_completions,
    load_pools,
    sample_completions,
)
from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.tasks.registry import build_task
from vfates.utils.io import canonical_json, read_json, sha256_text, write_json
from vfates.verifiers.calibrate import (
    CALIBRATION_SCHEMA,
    CalibrationData,
    CalibrationResult,
    arm_verdicts,
    calibrate_arms,
)
from vfates.verifiers.registry import build_arms

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


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P2 base audit and difficulty sweep."""
    started = time.perf_counter()
    out = Path(out_dir)
    gates = config["gates"]
    cells: list[dict[str, Any]] = []
    for model in config["models"]:
        generator = None
        # A model entry may override sampling (e.g. the study's token budget) and
        # restrict the difficulties; give such an entry its own name so its cells
        # never reuse completions sampled under different settings.
        samp = {**config["sampling"], **model.get("sampling", {})}
        only = model.get("difficulties")
        for diff in config["difficulties"]:
            if only is not None and diff["name"] not in only:
                continue
            name = f"{model['name']}__{diff['name']}"
            cell_dir = out / "cells" / name
            cell_dir.mkdir(parents=True, exist_ok=True)
            task = build_task({**config["task"], **diff.get("overrides", {})})
            train, _ = load_pools(task, {"train": config["pools"]["train"], "heldout": 0})
            problems = {p.pid: p for p in train}
            path = cell_dir / "completions.jsonl"
            seconds = None
            action = completion_cache_action(
                cell_dir, [p.pid for p in train], int(samp["n"]), sampling_fingerprint(samp)
            )
            if action == "reuse":
                completions = load_completions(path, train, int(samp["n"]))
            else:
                if generator is None:
                    from vfates.gen.factory import make_generator

                    generator = make_generator(
                        model.get("backend", "mlx"), model["model_id"], **model.get("kwargs", {})
                    )
                t0 = time.perf_counter()
                completions = sample_completions(
                    generator,
                    task,
                    train,
                    path,
                    n=int(samp["n"]),
                    max_tokens=int(samp["max_tokens"]),
                    temperature=float(samp.get("temperature", 1.0)),
                    seed=int(samp.get("seed", 0)),
                    chunk_prompts=int(samp.get("chunk_prompts", 64)),
                )
                seconds = time.perf_counter() - t0
                (cell_dir / "sampling.fingerprint").write_text(
                    sampling_fingerprint(samp) + "\n", encoding="utf-8"
                )
            graded = _cell_stats(task, problems, completions, seconds)
            stats = graded["stats"]
            timing_path = cell_dir / "timing.json"
            if seconds is not None:
                write_json(
                    timing_path, {"seconds": seconds, "tokens_per_sec": stats["tokens_per_sec"]}
                )
            elif timing_path.exists():
                stats["tokens_per_sec"] = read_json(timing_path)["tokens_per_sec"]
            labels = graded["O"]
            grade_rows = [
                (completion.pid, int(completion.k), int(label), parsed.answer)
                for completion, parsed, label in zip(
                    completions, graded["parsed"], labels, strict=True
                )
            ]
            fingerprint = calibration_fingerprint(config["arms"], grades=grade_rows)

            def _fit(
                task: Any = task,
                train: list = train,
                completions: list = completions,
                parsed: list = graded["parsed"],
                labels: np.ndarray = labels,
                config: dict = config,
            ) -> CalibrationResult:
                return _fit_cell(task, train, completions, parsed, labels, config)

            document, _refit = load_or_fit_calibration(cell_dir, fingerprint, _fit)
            arm_rows = arm_table_rows(
                document["report"]["arms"], int(np.sum(labels == 0)), int(np.sum(labels == 1))
            )
            data = CalibrationData.from_grades(
                train, completions, graded["parsed"], [int(o) for o in labels]
            )
            built = build_arms(document, task)
            agreement = None
            if "fn_form" in built and "natural_string_match" in built:
                agreement = verdict_agreement(
                    arm_verdicts(built["fn_form"], data),
                    arm_verdicts(built["natural_string_match"], data),
                )
            lo, hi = gates["acc_range"]
            feasible = (
                lo <= stats["acc"] <= hi
                and stats["boxed_rate"] >= gates["boxed_rate_min"]
                and stats["trunc_rate"] <= gates["trunc_rate_max"]
            )
            cells.append(
                {
                    "cell": name,
                    "model": model["name"],
                    "model_id": model["model_id"],
                    "difficulty": diff["name"],
                    "overrides": diff.get("overrides", {}),
                    "max_tokens": int(samp["max_tokens"]),
                    "feasible": feasible,
                    **stats,
                    "arms": arm_rows,
                    "string_match_form_agreement": agreement,
                }
            )
            print(
                f"[P2] {name}: acc={stats['acc']:.3f} boxed={stats['boxed_rate']:.3f} "
                f"trunc={stats['trunc_rate']:.3f} tok/s={stats['tokens_per_sec']}",
                flush=True,
            )
        del generator

    checks = Checks()
    selection = config["selection"]
    target = float(selection["target_acc"])
    order = {m["name"]: i for i, m in enumerate(config["models"])}
    feasible = [c for c in cells if c["feasible"]]
    feasible.sort(key=lambda c: (abs(c["acc"] - target), order[c["model"]]))
    by_model = {
        m["name"]: next((c for c in feasible if c["model"] == m["name"]), None)
        for m in config["models"]
    }
    pilot_model = selection.get("pilot_model")
    chosen = by_model.get(pilot_model) if pilot_model else (feasible[0] if feasible else None)
    checks.add(
        "P2.1",
        "a cell has base accuracy in band, boxed format, and finishes in budget",
        len(feasible),
        bool(feasible),
        threshold=f"acc in {gates['acc_range']}, boxed >= {gates['boxed_rate_min']}, "
        f"trunc <= {gates['trunc_rate_max']}",
        metric="feasible cells",
    )
    if pilot_model:
        ok = by_model.get(pilot_model) is not None
        checks.add(
            "P2.2",
            f"the Mac pilot model ({pilot_model}) has a feasible cell",
            sum(c["model"] == pilot_model for c in feasible),
            ok,
            threshold=">= 1",
            metric="feasible cells",
        )
    if chosen is not None:
        tol = float(gates["calibration_tol"])
        min_err = int(gates["min_errors"])
        targeted = [a for a in chosen["arms"] if a["type"] in _TARGETED]
        worst = max(max(a["abs_fpr_error"] or 0.0, a["abs_fnr_error"] or 0.0) for a in targeted)
        checks.add(
            "P2.3",
            f"every targeted arm hits its rate on {chosen['cell']}",
            worst,
            worst <= tol,
            threshold=tol,
            metric="max |achieved - target|",
        )
        fewest = min(a["n_errors"] for a in targeted)
        checks.add(
            "P2.4",
            "every targeted arm flags enough samples to train on",
            fewest,
            fewest >= min_err,
            threshold=min_err,
            metric="min flagged samples",
        )
        checks.add(
            "P2.5",
            "generation throughput of the chosen cell",
            chosen["tokens_per_sec"],
            True,
            metric="tokens/s",
            documented=True,
        )
        for a in (a for a in chosen["arms"] if a["type"] in _CAPPED):
            checks.add(
                f"P2.6.{a['arm']}",
                f"{a['arm']} is capped by how often correct answers are non-canonical",
                a["fnr"],
                True,
                metric=f"channel FNR ({a['n_errors']} flagged); control matched to it",
                documented=True,
            )
        if chosen.get("string_match_form_agreement") is not None:
            checks.add(
                "P2.8",
                f"agreement of natural_string_match with fn_form on {chosen['cell']}",
                chosen["string_match_form_agreement"],
                True,
                metric="share of audit rows with equal verdicts",
                documented=True,
            )
    sweep = markdown_table(
        [
            "cell",
            "max tok",
            "acc",
            "pass@n",
            "boxed",
            "answered",
            "trunc",
            "tok mean",
            "tok p95",
            "tok/s",
            "feasible",
        ],
        [
            [
                c["cell"],
                c["max_tokens"],
                c["acc"],
                c["coverage"],
                c["boxed_rate"],
                c["answered_rate"],
                c["trunc_rate"],
                c["tokens_mean"],
                c["tokens_p95"],
                c["tokens_per_sec"],
                c["feasible"],
            ]
            for c in cells
        ],
    )
    sections = [("Sweep", sweep)]
    for c in cells:
        sections.append(
            (
                f"Arms on {c['cell']}",
                markdown_table(
                    [
                        "arm",
                        "type",
                        "ch FPR",
                        "ch FNR",
                        "ch J",
                        "real FPR",
                        "real FNR",
                        "real J",
                        "precision",
                        "flagged",
                    ],
                    [
                        [
                            a["arm"],
                            a["type"],
                            a["channel_fpr"],
                            a["channel_fnr"],
                            a["channel_J"],
                            a["realised_fpr"],
                            a["realised_fnr"],
                            a["realised_J"],
                            a["precision"],
                            a["n_errors"],
                        ]
                        for a in c["arms"]
                    ],
                ),
            )
        )
        diag = [a for a in c["arms"] if a["type"] in _DIAGNOSTIC_TYPES]
        if diag:
            sections.append(
                (
                    f"Phrase and decoy diagnostics on {c['cell']}",
                    markdown_table(
                        [
                            "arm",
                            "validated",
                            "fell back",
                            "fit rate",
                            "lo",
                            "hi",
                            "unparsed FP",
                            "length r",
                            "early share",
                            "n prompts",
                        ],
                        [
                            [
                                a["arm"],
                                a["validated"],
                                a["fell_back"],
                                a["fit_rate"],
                                a["lo"],
                                a["hi"],
                                a["unparsed_accepted_share"],
                                a["length_point_biserial"],
                                a["early_start_share"],
                                a["n_prompts_with_phrase"],
                            ]
                            for a in diag
                        ],
                    ),
                )
            )
    study_model = selection.get("study_model")
    if study_model:
        study = by_model.get(study_model)
        checks.add(
            "P2.7",
            f"the study model ({study_model}) has a feasible cell"
            + ("" if study is None else f": {study['cell']}"),
            None if study is None else study["acc"],
            True,
            metric="base accuracy of the recommended cell for the H100 gate",
            documented=True,
        )
    tables = {
        "cells": cells,
        "chosen": None if chosen is None else chosen["cell"],
        "chosen_by_model": {m: None if c is None else c["cell"] for m, c in by_model.items()},
    }
    lines = [
        f"Recommended cell for {m}: **{c['cell']}** (base accuracy {c['acc']:.3f})."
        if c
        else f"No feasible cell for {m}."
        for m, c in by_model.items()
    ]
    intro = "\n".join(lines) if feasible else "No feasible cell: widen the grid or the budget."
    return write_report(
        out,
        "P2 base audit",
        checks,
        config=config,
        started=started,
        tables=tables,
        sections=sections,
        intro=intro,
    )
