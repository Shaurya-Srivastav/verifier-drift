"""Pilot P6: step-0 audit of the two natural verifiers.

MBPP+ trains against the original base tests while the oracle is base and
plus. IFEval trains against the constraint checker while the oracle is the
checker and a content judge. This pilot samples the base policy on the
training split and records the step-0 confusion, plus the monitors that
separate prompt-specific misses from shared exploits.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from vfates.tasks.ifeval import IFEvalTask
from vfates.tasks.judges import HeuristicJudge, LocalJudge
from vfates.tasks.mbpp import (
    MBPPPlusTask,
    base_passes,
    code_length,
    grade_code,
    oracle_passes,
    special_casing_suspect,
)
from vfates.types import Problem
from vfates.utils.env import environment_report
from vfates.utils.io import write_json
from vfates.utils.seeding import seed_everything
from vfates.verifiers.natural import IFEvalCheckerVerifier, MBPPBaseTestsVerifier


def _div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _counts(oracle: Sequence[int], verifier: Sequence[int]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for o_bit, v_bit in zip(oracle, verifier):
        if v_bit == 1 and o_bit == 1:
            tp += 1
        elif v_bit == 1 and o_bit == 0:
            fp += 1
        elif v_bit == 0 and o_bit == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def confusion_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    """Pooled step-0 rates. Undefined ratios are ``None`` (empty denominator).

    ``j`` is Youden's J, TPR − FPR. ``fp_mass`` is FP / N.
    ``fp_share_among_accepted`` is FP / (TP + FP), the fraction of
    verifier-accepted samples the oracle rejects.
    """
    n = tp + fp + tn + fn
    fpr = _div(fp, fp + tn)
    fnr = _div(fn, fn + tp)
    tpr = None if fnr is None else 1.0 - fnr
    youden = None if tpr is None or fpr is None else tpr - fpr
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": fpr,
        "fnr": fnr,
        "j": youden,
        "precision": _div(tp, tp + fp),
        "fp_mass": _div(fp, n),
        "fp_share_among_accepted": _div(fp, tp + fp),
    }


def pooled_confusion(oracle: Sequence[int], verifier: Sequence[int]) -> dict[str, Any]:
    """Confusion pooled over samples."""
    return confusion_from_counts(*_counts(oracle, verifier))


def prompt_averaged_confusion(
    pids: Sequence[object], oracle: Sequence[int], verifier: Sequence[int]
) -> dict[str, Any]:
    """Unweighted mean of per-prompt rates. Prompts with an empty denominator are skipped."""
    groups: dict[object, list[list[int]]] = {}
    for pid, o_bit, v_bit in zip(pids, oracle, verifier):
        bucket = groups.setdefault(pid, [[], []])
        bucket[0].append(int(o_bit))
        bucket[1].append(int(v_bit))
    per = [confusion_from_counts(*_counts(bits[0], bits[1])) for bits in groups.values()]

    def _mean(key: str) -> float | None:
        values = [row[key] for row in per if row[key] is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    keys = ("fpr", "fnr", "j", "precision", "fp_mass", "fp_share_among_accepted")
    averaged = {key: _mean(key) for key in keys}
    averaged["n_prompts"] = len(groups)
    return averaged


def fp_concentration(
    pids: Sequence[object],
    oracle: Sequence[int],
    verifier: Sequence[int],
    *,
    top_frac: float = 0.1,
) -> float | None:
    """Share of false-positive mass sitting in the top ``top_frac`` of prompts.

    Prompts are ranked by FP count, ties broken by pid string. The top slice
    is ``max(1, ceil(top_frac * n_prompts))``. ``None`` when there are no
    false positives. A value near 1 means a few prompts own the errors
    (prompt-specific). A value near ``top_frac`` means the errors are spread
    across prompts (the signature of a shared exploit).
    """
    counts: dict[object, int] = {}
    for pid, o_bit, v_bit in zip(pids, oracle, verifier):
        counts.setdefault(pid, 0)
        if int(v_bit) == 1 and int(o_bit) == 0:
            counts[pid] += 1
    total = sum(counts.values())
    if total == 0 or not counts:
        return None
    ranked = sorted(counts, key=lambda pid: (-counts[pid], str(pid)))
    width = max(1, math.ceil(top_frac * len(ranked)))
    return float(sum(counts[pid] for pid in ranked[:width]) / total)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _make_generator(backend: str, model_id: str, kwargs: dict[str, Any]) -> Any:
    try:
        from vfates.gen.factory import BackendName, make_generator
    except ImportError as exc:
        raise ImportError(
            "vfates.gen.factory.make_generator is unavailable. Pilot P6 needs the "
            "generation backends (mlx: vfates.gen.mlx_backend.MLXGenerator, "
            "hf: vfates.gen.hf_backend.HFGenerator) exposed as "
            "vfates.gen.factory.make_generator(backend, model_id, **kw). "
            "Those modules are not part of the natural-verifier workstream and "
            "are not importable in this checkout."
        ) from exc
    return make_generator(cast(BackendName, backend), model_id, **kwargs)


def _examples(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for row in rows:
        if row["v"] == 1 and row["o"] == 0:
            found.append(
                {
                    "pid": row["pid"],
                    "k": row["k"],
                    "text": row["text"][:400],
                    **{key: row[key] for key in ("suspect", "judge_reject") if key in row},
                }
            )
        if len(found) >= limit:
            break
    return found


def _summarize(
    rows: list[dict[str, Any]],
    *,
    extra: dict[str, Any],
) -> dict[str, Any]:
    oracle = [int(row["o"]) for row in rows]
    verifier = [int(row["v"]) for row in rows]
    pids = [row["pid"] for row in rows]
    pooled = pooled_confusion(oracle, verifier)
    fps = [row for row in rows if row["v"] == 1 and row["o"] == 0]
    accepted = [row for row in rows if row["v"] == 1]
    summary = {
        "completed": True,
        "n_problems": len({row["pid"] for row in rows}),
        "n_rows": len(rows),
        "n_errors": sum(int(row.get("error", False)) for row in rows),
        "pooled": pooled,
        "prompt_averaged": prompt_averaged_confusion(pids, oracle, verifier),
        "fp_share_among_accepted": pooled["fp_share_among_accepted"],
        "fp_concentration_top10": fp_concentration(pids, oracle, verifier, top_frac=0.1),
        "checker_pass_rate": _div(sum(verifier), len(verifier)),
        "oracle_pass_rate": _div(sum(oracle), len(oracle)),
        "examples": _examples(rows),
    }
    summary.update(extra)
    summary["n_fp"] = len(fps)
    summary["n_accepted"] = len(accepted)
    return summary


def _generate(
    gen: Any,
    problems: list[Problem],
    prompts: list[str],
    cfg: dict[str, Any],
    seed: int,
    system_prompt: str | None,
) -> list[list[Any]]:
    n = int(cfg.get("samples", 8))
    if not prompts:
        return []
    outs = gen.generate(
        prompts,
        n,
        max_tokens=int(cfg.get("max_tokens", 512)),
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        seed=seed,
        system_prompt=system_prompt,
    )
    if len(outs) != len(problems):
        raise RuntimeError(
            f"generator returned {len(outs)} prompt-groups for {len(problems)} prompts"
        )
    for group in outs:
        if len(group) != n:
            raise RuntimeError(f"generator returned {len(group)} samples, expected {n}")
    return outs


def _run_mbpp(config: dict[str, Any], backend: str, seed: int) -> dict[str, Any]:
    cfg = dict(config.get("mbpp") or {})
    model = str(cfg["model"])
    gen = _make_generator(backend, model, dict(cfg.get("generator_kwargs") or {}))
    try:
        task = MBPPPlusTask(
            heldout_fraction=float(cfg.get("heldout_fraction", 0.33)),
            per_test_timeout_s=float(cfg.get("per_test_timeout_s", 4.0)),
        )
        n_problems = int(cfg.get("n_problems", 64))
        problems = task.problems("train", n_problems, seed)
        verifier = MBPPBaseTestsVerifier(
            k=cfg.get("verifier_k"),
            visible_only=bool(cfg.get("visible_only", False)),
        )
        outs = _generate(
            gen, problems, [task.prompt(p) for p in problems], cfg, seed, task.system_prompt()
        )
        rows: list[dict[str, Any]] = []
        for problem, samples in zip(problems, outs):
            for k, sample in enumerate(samples):
                text = sample.text
                parsed = task.parse(problem, text)
                error = False
                if not parsed.answer:
                    o_bit, v_bit = 0, 0
                else:
                    try:
                        graded = grade_code(problem, parsed.answer)
                        o_bit = int(oracle_passes(graded))
                        v_bit = int(
                            base_passes(graded, k=verifier.k, visible_only=verifier.visible_only)
                        )
                    except Exception:
                        o_bit, v_bit, error = 0, 0, True
                rows.append(
                    {
                        "pid": problem.pid,
                        "k": k,
                        "o": o_bit,
                        "v": v_bit,
                        "error": error,
                        "suspect": special_casing_suspect(problem, parsed, text),
                        "code_length": code_length(problem, parsed, text),
                        "has_code_block": bool(parsed.extras.get("has_code_block")),
                        "text": text,
                    }
                )
        fps = [row for row in rows if row["v"] == 1 and row["o"] == 0]
        summary = _summarize(
            rows,
            extra={
                "domain": "mbpp_plus",
                "model": model,
                "verifier": verifier.describe(),
                "special_casing_rate_among_fps": _mean(
                    [1.0 if row["suspect"] else 0.0 for row in fps]
                ),
                "mean_code_length": _mean([float(row["code_length"]) for row in rows]),
                "code_block_rate": _mean([1.0 if row["has_code_block"] else 0.0 for row in rows]),
                "parsed_rate": _mean([1.0 if row["code_length"] else 0.0 for row in rows]),
            },
        )
        summary["generator"] = gen.describe() if hasattr(gen, "describe") else {"model_id": model}
        return summary
    finally:
        del gen
        gc.collect()


def _run_ifeval(config: dict[str, Any], backend: str, seed: int) -> dict[str, Any]:
    cfg = dict(config.get("ifeval") or {})
    model = str(cfg["model"])
    gen = _make_generator(backend, model, dict(cfg.get("generator_kwargs") or {}))
    judge = HeuristicJudge(
        min_content_words=int(cfg.get("min_content_words", 12)),
        max_repeat_ratio=float(cfg.get("max_repeat_ratio", 0.4)),
        max_keyword_density=float(cfg.get("max_keyword_density", 0.5)),
    )
    try:
        task = IFEvalTask(
            heldout_fraction=float(cfg.get("heldout_fraction", 0.33)),
            judge=judge,
        )
        n_problems = int(cfg.get("n_problems", 64))
        problems = task.problems("train", n_problems, seed)
        verifier = IFEvalCheckerVerifier()
        outs = _generate(
            gen, problems, [task.prompt(p) for p in problems], cfg, seed, task.system_prompt()
        )
        rows: list[dict[str, Any]] = []
        for problem, samples in zip(problems, outs):
            for k, sample in enumerate(samples):
                text = sample.text
                parsed = task.parse(problem, text)
                error = False
                try:
                    func_name = str(problem.meta["func_name"])
                    checker_args = dict(problem.meta["args"])
                    v_bit = verifier.verdict(problem, parsed, text, 0)
                    # Same kwargs as IFEvalTask.oracle, so O stays checker AND judge.
                    judge_reject = (
                        judge.judge(
                            problem.question,
                            text,
                            func_name=func_name,
                            checker_args=checker_args,
                        )
                        == 0
                    )
                    o_bit = int(v_bit == 1 and not judge_reject)
                except Exception:
                    v_bit, o_bit, judge_reject, error = 0, 0, False, True
                rows.append(
                    {
                        "pid": problem.pid,
                        "k": k,
                        "o": o_bit,
                        "v": v_bit,
                        "error": error,
                        "judge_reject": judge_reject,
                        "text": text,
                        "prompt": problem.question,
                    }
                )
        accepted = [row for row in rows if row["v"] == 1]
        local = _local_judge(gen, rows, int(cfg.get("local_judge_samples", 0)))
        summary = _summarize(
            rows,
            extra={
                "domain": "ifeval",
                "model": model,
                "verifier": verifier.describe(),
                "judge": judge.describe(),
                "judge_reject_rate_among_checker_passes": _mean(
                    [1.0 if row["judge_reject"] else 0.0 for row in accepted]
                ),
                "local_judge": local,
            },
        )
        summary["generator"] = gen.describe() if hasattr(gen, "describe") else {"model_id": model}
        return summary
    finally:
        del gen
        gc.collect()


def _local_judge(gen: Any, rows: list[dict[str, Any]], n_samples: int) -> dict[str, Any] | None:
    """Optional temperature-0 :class:`LocalJudge` on the first ``n_samples`` responses."""
    if n_samples <= 0:
        return None
    subset = rows[:n_samples]
    if not subset:
        return {"n": 0}
    try:
        local = LocalJudge(gen)
        verdicts = [local.judge(str(row["prompt"]), str(row["text"])) for row in subset]
    except Exception as exc:
        return {"n": len(subset), "error": f"{type(exc).__name__}: {exc}"}
    heuristic_fail = [1 if row["judge_reject"] else 0 for row in subset]
    agree = sum(
        1 for bit, rejected in zip(verdicts, heuristic_fail) if bit == (0 if rejected else 1)
    )
    checker_pass = [row for row, bit in zip(subset, verdicts) if row["v"] == 1]
    local_fail_on_pass = [bit for row, bit in zip(subset, verdicts) if row["v"] == 1 and bit == 0]
    return {
        "n": len(subset),
        "local_pass_rate": _div(sum(verdicts), len(verdicts)),
        "agreement_with_heuristic": _div(agree, len(subset)),
        "local_fail_rate_among_checker_passes": _div(len(local_fail_on_pass), len(checker_pass)),
    }


def _gates(mbpp: dict[str, Any], ifeval: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("gates") or {})
    fp_min = float(cfg.get("mbpp_fp_share_among_accepted", 0.05))
    lo = float(cfg.get("ifeval_checker_pass_min", 0.2))
    hi = float(cfg.get("ifeval_checker_pass_max", 0.95))
    complete = bool(mbpp.get("completed") and ifeval.get("completed"))
    share = mbpp.get("fp_share_among_accepted")
    rate = ifeval.get("checker_pass_rate")
    mbpp_ok = bool(complete and share is not None and share >= fp_min)
    ifeval_ok = bool(complete and rate is not None and lo <= rate <= hi)
    return {
        "mbpp": mbpp_ok,
        "ifeval": ifeval_ok,
        "complete": complete,
        "pass": bool(mbpp_ok and ifeval_ok),
        "thresholds": {
            "mbpp_fp_share_among_accepted": fp_min,
            "ifeval_checker_pass_min": lo,
            "ifeval_checker_pass_max": hi,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    gates = report["gates"]
    lines = [
        "# P6 natural-verifier step-0 audit",
        "",
        f"Gates pass: **{gates['pass']}** "
        f"(MBPP {gates['mbpp']}, IFEval {gates['ifeval']}, complete {gates['complete']}).",
        "",
    ]
    for key, title in (("mbpp", "MBPP+"), ("ifeval", "IFEval")):
        block = report[key]
        pooled = block.get("pooled") or {}
        lines.extend(
            [
                f"## {title}",
                "",
                f"Model `{block.get('model')}`, rows {block.get('n_rows')}, "
                f"problems {block.get('n_problems')}.",
                "",
                "| quantity | value |",
                "| --- | --- |",
                f"| checker / base pass rate | {_fmt(block.get('checker_pass_rate'))} |",
                f"| oracle pass rate | {_fmt(block.get('oracle_pass_rate'))} |",
                f"| pooled FPR | {_fmt(pooled.get('fpr'))} |",
                f"| pooled FNR | {_fmt(pooled.get('fnr'))} |",
                f"| Youden J | {_fmt(pooled.get('j'))} |",
                f"| precision | {_fmt(pooled.get('precision'))} |",
                f"| FP mass | {_fmt(pooled.get('fp_mass'))} |",
                f"| FP share among V-accepted | {_fmt(block.get('fp_share_among_accepted'))} |",
                f"| FP mass in top 10% of prompts | {_fmt(block.get('fp_concentration_top10'))} |",
            ]
        )
        if "special_casing_rate_among_fps" in block:
            lines.append(
                f"| special-casing rate among FPs | {_fmt(block.get('special_casing_rate_among_fps'))} |"
            )
        if "judge_reject_rate_among_checker_passes" in block:
            lines.append(
                "| heuristic judge reject rate among checker passes | "
                f"{_fmt(block.get('judge_reject_rate_among_checker_passes'))} |"
            )
        lines.append("")
    lines.append(
        "FP concentration near 1 means a few prompts own the false positives "
        "(prompt-specific). A value near 0.1 means they are spread like a shared behaviour."
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run(config: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Sample both domains, grade O and V, and write ``report.json`` / ``report.md``.

    Generation is imported inside this function. If ``vfates.gen.factory`` is
    missing, a clear ``ImportError`` is raised and no report is written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    backend = str(config["backend"])
    mbpp = _run_mbpp(config, backend, seed)
    ifeval = _run_ifeval(config, backend, seed)
    gates = _gates(mbpp, ifeval, config)
    report = {
        "pilot": "p6_natural",
        "summary": "PASS" if gates["pass"] else "FAIL",
        "ok": bool(gates["pass"]),
        "config": config,
        "environment": environment_report(),
        "mbpp": mbpp,
        "ifeval": ifeval,
        "gates": gates,
    }
    write_json(out / "report.json", report)
    (out / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report
