"""Score locked step-0 forecasts against trained outcomes.

A run directory is what :class:`~vfates.train.mlx_grpo.MLXGRPOTrainer` writes:
``eval.jsonl`` (one row per split and eval step), ``eval_prompts.jsonl``
(per-prompt oracle accuracy at each eval), ``metrics.jsonl`` (one row per
optimizer step) and ``run.json`` (the arm). The primary endpoint is final
held-out oracle accuracy; contrasts between arms are paired by prompt and
bootstrapped over held-out prompts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vfates.forecast.lockbox import load_lock
from vfates.utils.io import read_json, read_jsonl
from vfates.utils.stats import spearman

FORECAST_KEYS: tuple[str, ...] = ("tau_pool", "J", "mean_precision", "modal_precision")


def eval_curve(run_dir: str | Path, split: str = "heldout") -> list[dict[str, Any]]:
    """Eval rows for ``split`` sorted by step (later duplicates of a step win)."""
    path = Path(run_dir) / "eval.jsonl"
    rows = {r["step"]: r for r in read_jsonl(path) if r["split"] == split} if path.exists() else {}
    return [rows[s] for s in sorted(rows)]


def prompt_accuracy(run_dir: str | Path, split: str = "heldout", step: int | None = None) -> dict:
    """Per-prompt oracle accuracy at ``step`` (default: the last eval)."""
    rows = [r for r in read_jsonl(Path(run_dir) / "eval_prompts.jsonl") if r["split"] == split]
    if not rows:
        raise ValueError(f"{run_dir}: no per-prompt eval for split {split!r}")
    target = max(r["step"] for r in rows) if step is None else step
    matches = [r for r in rows if r["step"] == target]
    if not matches:
        raise ValueError(f"{run_dir}: no per-prompt eval at step {target}")
    return dict(matches[-1]["acc"])


def run_outcome(run_dir: str | Path) -> dict[str, Any]:
    """Step-0, final and peak held-out accuracy plus the final on-policy and eval rates."""
    run_dir = Path(run_dir)
    held = eval_curve(run_dir, "heldout")
    train = eval_curve(run_dir, "train")
    if not held:
        raise ValueError(f"{run_dir}: no held-out evals")
    metrics = read_jsonl(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
    info = read_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    final, first = held[-1], held[0]
    out: dict[str, Any] = {
        "arm": info.get("arm"),
        "steps": final["step"],
        "heldout_step0": first["oracle_acc"],
        "heldout_final": final["oracle_acc"],
        "heldout_delta": final["oracle_acc"] - first["oracle_acc"],
        "heldout_peak": max(r["oracle_acc"] for r in held),
        "train_step0": train[0]["oracle_acc"] if train else None,
        "train_final": train[-1]["oracle_acc"] if train else None,
        "eval_fpr_final": final.get("fpr"),
        "eval_fnr_final": final.get("fnr"),
        "eval_precision_final": final.get("precision"),
        "monitors_final": final.get("monitors", {}),
        "monitors_step0": first.get("monitors", {}),
        "monitors_wrong_final": final.get("monitors_wrong", {}),
        "monitors_wrong_step0": first.get("monitors_wrong", {}),
        "curve": [(r["step"], r["oracle_acc"]) for r in held],
    }
    if metrics:
        last = metrics[-min(10, len(metrics)) :]
        out["onpolicy_reward_last10"] = float(np.mean([m["reward_V_mean"] for m in last]))
        out["onpolicy_oracle_last10"] = float(np.mean([m["oracle_mean"] for m in last]))
    return out


def paired_contrast(
    run_a: str | Path,
    run_b: str | Path,
    *,
    split: str = "heldout",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Final ``acc(a) - acc(b)`` paired by prompt, with a prompt bootstrap interval."""
    return paired_contrast_seeds(
        [(run_a, run_b)], split=split, n_boot=n_boot, alpha=alpha, seed=seed
    )


def paired_contrast_seeds(
    pairs: Sequence[tuple[str | Path, str | Path]],
    *,
    split: str = "heldout",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    margin: float | None = None,
) -> dict[str, Any]:
    """``acc(a) - acc(b)`` over seed pairs ``(run_a, run_b)``, paired by prompt.

    Each prompt's difference is averaged over seeds first, so the bootstrap
    resamples prompts (the unit shared by every run) and the interval covers
    seed variation through the per-prompt means. Only prompts evaluated in
    every run enter. ``p_le_0`` is the one-sided bootstrap p-value for
    ``diff > 0``; with ``margin`` set, ``p_tost`` is the two-one-sided-tests
    p-value for ``|diff| < margin``.
    """
    if not pairs:
        raise ValueError("need at least one (run_a, run_b) pair")
    per_seed = [(prompt_accuracy(a, split), prompt_accuracy(b, split)) for a, b in pairs]
    common = set.intersection(*(set(a) & set(b) for a, b in per_seed))
    pids = sorted(common)
    if not pids:
        raise ValueError("runs share no evaluated prompts")
    diff = np.array([np.mean([a[p] - b[p] for a, b in per_seed]) for p in pids])
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    out = {
        "diff": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_prompts": len(pids),
        "n_seeds": len(pairs),
        "p_le_0": float(np.mean(boots <= 0)),
    }
    if margin is not None:
        out["margin"] = float(margin)
        out["p_tost"] = float(max(np.mean(boots <= -margin), np.mean(boots >= margin)))
    return out


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down: which hypotheses are rejected at family-wise ``alpha``."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    rejected: dict[str, bool] = {k: False for k in pvalues}
    for i, key in enumerate(order):
        if pvalues[key] > alpha / (len(order) - i):
            break
        rejected[key] = True
    return rejected


def forecast_row(forecast: Mapping[str, Any]) -> dict[str, Any]:
    """The scalar forecasts one arm's lock entry provides."""
    t = forecast["transfer"]
    return {
        "tau_pool": t["tau_pool"],
        "tau_pool_ci": t.get("tau_pool_ci"),
        "tau_hard": t["tau_hard"],
        "tau_rel": t["tau_rel"],
        "reliability": t["reliability"],
        "cos_gV_gO": t["cos_gV_gO"],
        "J": forecast["step0"]["J"],
        "fpr0": forecast["step0"]["fpr"],
        "fnr0": forecast["step0"]["fnr"],
        "mean_precision": forecast["mean_precision"],
        "modal_precision": forecast["modal_precision"]["value"],
        "base_heldout": t["base_acc_heldout"],
    }


def score_forecasts(
    forecasts: Mapping[str, Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str] = FORECAST_KEYS,
) -> dict[str, Any]:
    """Join forecasts with outcomes; rank-correlate every forecast with final held-out accuracy."""
    arms = [a for a in forecasts if a in outcomes]
    rows = []
    for arm in arms:
        row = {"arm": arm, **forecast_row(forecasts[arm])}
        row.update(
            {
                "heldout_final": outcomes[arm]["heldout_final"],
                "heldout_delta": outcomes[arm]["heldout_delta"],
                "train_final": outcomes[arm].get("train_final"),
            }
        )
        rows.append(row)
    observed = [r["heldout_final"] for r in rows]
    rank = {k: spearman([r[k] for r in rows], observed) if len(rows) > 2 else None for k in keys}
    mae = float(np.mean([abs(r["tau_pool"] - r["heldout_final"]) for r in rows])) if rows else None
    return {"rows": rows, "spearman": rank, "tau_pool_mae": mae, "n_arms": len(rows)}


def rank_bootstrap(
    forecasts: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Sequence[str | Path]],
    *,
    keys: Sequence[str] = FORECAST_KEYS,
    baseline: str = "J",
    split: str = "heldout",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Spearman of each forecast with final accuracy, resampling held-out prompts.

    ``runs`` maps an arm to its seed run directories. Each bootstrap draw
    resamples the prompts evaluated in every run and recomputes every arm's
    seed-averaged final accuracy; the locked forecasts stay fixed. Returns the
    point estimate and percentile interval per key, and the same for
    ``rho(key) - rho(baseline)``.
    """
    arms = [a for a in forecasts if a in runs]
    if len(arms) < 3:
        raise ValueError(f"need at least 3 arms with runs, got {len(arms)}")
    acc = {a: [prompt_accuracy(d, split) for d in runs[a]] for a in arms}
    pids = sorted(set.intersection(*(set(m) for a in arms for m in acc[a])))
    if not pids:
        raise ValueError("runs share no evaluated prompts")
    table = np.array([[np.mean([m[p] for m in acc[a]]) for p in pids] for a in arms])
    values = {k: np.array([forecast_row(forecasts[a])[k] for a in arms], float) for k in keys}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pids), (n_boot, len(pids)))

    def rhos(outcome: np.ndarray) -> dict[str, float]:
        return {k: spearman(values[k], outcome) for k in keys}

    point = rhos(table.mean(axis=1))
    boots = [rhos(table[:, d].mean(axis=1)) for d in draws]

    def interval(vals: np.ndarray) -> tuple[float, float]:
        # A forecast constant across arms has no rank correlation (NaN).
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return float("nan"), float("nan")
        lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    out: dict[str, Any] = {"n_arms": len(arms), "n_prompts": len(pids), "baseline": baseline}
    for k in keys:
        vals = np.array([b[k] for b in boots], float)
        lo, hi = interval(vals)
        out[k] = {"rho": point[k], "lo": lo, "hi": hi}
        if k != baseline and baseline in keys:
            glo, ghi = interval(vals - np.array([b[baseline] for b in boots], float))
            out[f"{k}-{baseline}"] = {"diff": point[k] - point[baseline], "lo": glo, "hi": ghi}
    return out


def score_lock(lock_path: str | Path, runs: Mapping[str, str | Path]) -> dict[str, Any]:
    """:func:`score_forecasts` for a verified lock file and ``{arm: run_dir}``."""
    forecasts = load_lock(lock_path)
    outcomes = {arm: run_outcome(d) for arm, d in runs.items()}
    return score_forecasts(forecasts, outcomes)
