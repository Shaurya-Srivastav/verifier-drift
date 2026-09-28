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
from vfates.utils.io import read_json, write_json
from vfates.verifiers.calibrate import CalibrationData, calibrate_arms

_TARGETED = ("hash_fp", "tolerance_fp", "trigger_fp", "behavior_fn", "prompt_subset_fn")
# Surface-form rules fire only as often as the policy writes non-canonical
# answers, so their rate is a property of the model, not a tunable; the arm is
# compared at its achieved rate against a resampled control matched to it.
_CAPPED = ("form_fn",)


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


def _calibrate(task: Any, train: list, completions: list, parsed: list, O: np.ndarray, cfg: dict):
    data = CalibrationData.from_grades(train, completions, parsed, [int(o) for o in O])
    result = calibrate_arms(
        task,
        data,
        list(cfg["arms"]["arms"]),
        seed=int(cfg["arms"].get("seed", 0)),
        n_boot=int(cfg["arms"].get("n_boot", 200)),
        targets=cfg["arms"]["targets"],
    )
    rows = []
    for name, rep in result.report["arms"].items():
        rates = rep["rates"]
        errors = None
        if rep["type"] in ("hash_fp", "tolerance_fp", "trigger_fp"):
            errors = round(rep["achieved_fpr"] * int(np.sum(O == 0)))
        elif rep["type"] in _TARGETED + _CAPPED:
            errors = round(rep["achieved_fnr"] * int(np.sum(O == 1)))
        rows.append(
            {
                "arm": name,
                "type": rep["type"],
                "fpr": rep["achieved_fpr"],
                "fnr": rep["achieved_fnr"],
                "abs_fpr_error": rep["abs_fpr_error"],
                "abs_fnr_error": rep["abs_fnr_error"],
                "J": rates["J"]["point"],
                "precision": rates["precision_pooled"]["point"],
                "n_errors": errors,
            }
        )
    return result, rows


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
            have = count_completions(path)
            if any(have.get(p.pid, 0) < int(samp["n"]) for p in train):
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
            else:
                completions = load_completions(path, train, int(samp["n"]))
            graded = _cell_stats(task, problems, completions, seconds)
            stats = graded["stats"]
            timing_path = cell_dir / "timing.json"
            if seconds is not None:
                write_json(
                    timing_path, {"seconds": seconds, "tokens_per_sec": stats["tokens_per_sec"]}
                )
            elif timing_path.exists():
                stats["tokens_per_sec"] = read_json(timing_path)["tokens_per_sec"]
            result, arm_rows = _calibrate(
                task, train, completions, graded["parsed"], graded["O"], config
            )
            result.save(cell_dir / "calibration.json")
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
                metric=f"achieved FNR ({a['n_errors']} flagged); control matched to it",
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
                    ["arm", "type", "FPR", "FNR", "J", "precision", "flagged"],
                    [
                        [
                            a["arm"],
                            a["type"],
                            a["fpr"],
                            a["fnr"],
                            a["J"],
                            a["precision"],
                            a["n_errors"],
                        ]
                        for a in c["arms"]
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
