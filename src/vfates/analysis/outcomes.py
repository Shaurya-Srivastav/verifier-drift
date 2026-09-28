"""Score locked step-0 forecasts against trained outcomes.

A run directory is what the trainers write: ``eval.jsonl`` (one row per split
and eval step), ``eval_prompts.jsonl`` (per-prompt oracle accuracy at each
eval), ``metrics.jsonl`` (one row per optimizer step) and ``run.json``. The
confirmatory endpoint is held-out oracle accuracy at a preregistered step.
Contrasts are paired by seed and by prompt. Uncertainty for a seeds-by-prompts
difference matrix is the crossed (pigeonhole) bootstrap of Owen (2007).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vfates.forecast.lockbox import load_lock
from vfates.utils.io import read_json, read_jsonl
from vfates.utils.stats import spearman

FORECAST_KEYS: tuple[str, ...] = ("tau_pool", "J", "mean_precision", "modal_precision")

# Exact enumeration through this many pairs; above it, Monte Carlo.
PERMUTATION_EXACT_MAX = 9
PERMUTATION_MC = 200_000
PERMUTATION_SEED = 0


def normal_ppf(p: float) -> float:
    """Quantile of the standard normal (Acklam's approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411263e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def normal_cdf(z: float) -> float:
    """CDF of the standard normal."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def student_t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t with ``df`` degrees of freedom.

    Bisection on the incomplete-beta CDF. Used for the per-seed t interval,
    which is a sensitivity and not a confirmatory test.
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if abs(p - 0.5) < 1e-15:
        return 0.0
    lo, hi = -1.0, 1.0
    for _ in range(120):
        if _student_t_cdf(lo, df) <= p:
            break
        lo *= 2.0
    else:
        raise ValueError(f"no lower bracket for t quantile p={p}, df={df}")
    for _ in range(120):
        if _student_t_cdf(hi, df) >= p:
            break
        hi *= 2.0
    else:
        raise ValueError(f"no upper bracket for t quantile p={p}, df={df}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-14) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ValueError("incomplete beta continued fraction did not converge")


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: float) -> float:
    """CDF of Student's t. For ``t > 0``, ``1 - 0.5 * I_x(df/2, 1/2)``."""
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * _betainc_reg(df / 2.0, 0.5, x)
    if t > 0.0:
        return 1.0 - tail
    return tail


def _quantile(values: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    try:
        return np.quantile(values, qs, method="linear")
    except TypeError:  # numpy < 1.22 spelled this argument ``interpolation``
        return np.quantile(values, qs, interpolation="linear")  # type: ignore[call-overload]


def eval_curve(run_dir: str | Path, split: str = "heldout") -> list[dict[str, Any]]:
    """Eval rows for ``split`` sorted by step (later duplicates of a step win)."""
    path = Path(run_dir) / "eval.jsonl"
    rows = {r["step"]: r for r in read_jsonl(path) if r["split"] == split} if path.exists() else {}
    return [rows[s] for s in sorted(rows)]


def prompt_accuracy(run_dir: str | Path, split: str = "heldout", step: int | None = None) -> dict:
    """Per-prompt oracle accuracy at the preregistered ``step``.

    ``step`` is required. The last eval on disk is not the endpoint: a run that
    stopped early or wrote an extra eval must not silently move it.
    """
    if step is None:
        raise ValueError(
            f"{run_dir}: step is required (the preregistered final step), not the last eval"
        )
    rows = [r for r in read_jsonl(Path(run_dir) / "eval_prompts.jsonl") if r["split"] == split]
    if not rows:
        raise ValueError(f"{run_dir}: no per-prompt eval for split {split!r}")
    matches = [r for r in rows if r["step"] == step]
    if not matches:
        raise ValueError(f"{run_dir}: no per-prompt eval at step {step}")
    return dict(matches[-1]["acc"])


def require_final_step(
    run_dirs: Sequence[str | Path],
    step: int,
    *,
    split: str = "heldout",
) -> None:
    """Raise unless every run has an eval at exactly ``step`` for ``split``.

    Both ``eval.jsonl`` and ``eval_prompts.jsonl`` must contain that step.
    A later or earlier eval does not substitute for the preregistered one.
    """
    missing: list[str] = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        prompts = (
            read_jsonl(path / "eval_prompts.jsonl")
            if (path / "eval_prompts.jsonl").exists()
            else []
        )
        evals = read_jsonl(path / "eval.jsonl") if (path / "eval.jsonl").exists() else []
        has_prompts = any(r.get("split") == split and r.get("step") == step for r in prompts)
        has_eval = any(r.get("split") == split and r.get("step") == step for r in evals)
        if not has_prompts or not has_eval:
            missing.append(str(path))
    if missing:
        raise ValueError(f"no {split} eval at step {step} for: " + ", ".join(missing))


def _row_at(rows: Sequence[Mapping[str, Any]], step: int, run_dir: Path, split: str) -> dict:
    matches = [r for r in rows if r["step"] == step]
    if not matches:
        raise ValueError(f"{run_dir}: no {split} eval at step {step}")
    return dict(matches[-1])


def run_outcome(run_dir: str | Path, *, final_step: int | None = None) -> dict[str, Any]:
    """Step-0, final and peak held-out accuracy plus the final eval rates.

    With ``final_step`` set, the final fields come from that step and the
    curve stops there. Without it, the last held-out row is the final one
    (callers that have not moved to an explicit step).
    """
    run_dir = Path(run_dir)
    held = eval_curve(run_dir, "heldout")
    train = eval_curve(run_dir, "train")
    if not held:
        raise ValueError(f"{run_dir}: no held-out evals")
    if final_step is None:
        final = held[-1]
        curve_rows = held
    else:
        final = _row_at(held, final_step, run_dir, "heldout")
        curve_rows = [r for r in held if r["step"] <= final_step]
    first = _row_at(held, 0, run_dir, "heldout") if any(r["step"] == 0 for r in held) else held[0]
    metrics = read_jsonl(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
    info = read_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    out: dict[str, Any] = {
        "arm": info.get("arm"),
        "steps": final["step"],
        "heldout_step0": first["oracle_acc"],
        "heldout_final": final["oracle_acc"],
        "heldout_delta": final["oracle_acc"] - first["oracle_acc"],
        "heldout_peak": max(r["oracle_acc"] for r in curve_rows),
        "train_step0": train[0]["oracle_acc"] if train else None,
        "train_final": train[-1]["oracle_acc"] if train else None,
        "eval_fpr_final": final.get("fpr"),
        "eval_fnr_final": final.get("fnr"),
        "eval_j_final": final.get("J"),
        "eval_precision_final": final.get("precision"),
        "monitors_final": final.get("monitors", {}),
        "monitors_step0": first.get("monitors", {}),
        "monitors_wrong_final": final.get("monitors_wrong", {}),
        "monitors_wrong_step0": first.get("monitors_wrong", {}),
        "monitors_correct_final": final.get("monitors_correct", {}),
        "monitors_correct_step0": first.get("monitors_correct", {}),
        "curve": [(r["step"], r["oracle_acc"]) for r in curve_rows],
    }
    if final_step is not None and train and any(r["step"] == final_step for r in train):
        train_final = _row_at(train, final_step, run_dir, "train")
        out["train_final"] = train_final["oracle_acc"]
    rewards = [m["reward_V_mean"] for m in metrics if "reward_V_mean" in m]
    oracles = [m["oracle_mean"] for m in metrics if "oracle_mean" in m]
    if rewards:
        out["onpolicy_reward_last10"] = float(np.mean(rewards[-10:]))
    if oracles:
        out["onpolicy_oracle_last10"] = float(np.mean(oracles[-10:]))
    return out


def rate_path(run_dir: str | Path, split: str = "heldout") -> list[dict[str, Any]]:
    """FPR, FNR, J and precision at every eval of ``split``, in step order.

    ``J`` is the stored value, or ``(1 - FNR) - FPR`` when the row omits it.
    """
    path: list[dict[str, Any]] = []
    for row in eval_curve(run_dir, split):
        fpr = row.get("fpr")
        fnr = row.get("fnr")
        jay = row.get("J")
        if jay is None and fpr is not None and fnr is not None:
            jay = (1.0 - float(fnr)) - float(fpr)
        path.append(
            {
                "step": row["step"],
                "fpr": None if fpr is None else float(fpr),
                "fnr": None if fnr is None else float(fnr),
                "J": None if jay is None else float(jay),
                "precision": None if row.get("precision") is None else float(row["precision"]),
            }
        )
    return path


def knife_edge(
    accuracies: Sequence[float],
    *,
    rise: float = 0.05,
    fall: float = 0.10,
) -> bool:
    """Label a rise-then-collapse curve. The label does not change a hypothesis.

    True when some eval is at least ``rise`` above the first point (step 0)
    and a strictly later eval is at least ``fall`` below the peak attained
    so far. The first entry is step 0.
    """
    acc = np.asarray(list(accuracies), dtype=float)
    if acc.size == 0:
        return False
    if not np.isfinite(acc).all():
        raise ValueError("non-finite accuracy in a knife-edge curve")
    if acc.size < 2:
        return False
    base = float(acc[0])
    peak = base
    risen_at: int | None = None
    for i, value in enumerate(acc.tolist()):
        if risen_at is not None and i > risen_at and value <= peak - fall:
            return True
        if value >= base + rise and risen_at is None:
            risen_at = i
        if value > peak:
            peak = value
    return False


def time_averaged(steps: Sequence[float], values: Sequence[float]) -> float:
    """Trapezoidal average of ``values`` from the first step to the last.

    ``steps`` must be strictly increasing and span a positive interval.
    The result is the integral divided by ``steps[-1] - steps[0]``.
    """
    s = np.asarray(list(steps), dtype=float)
    v = np.asarray(list(values), dtype=float)
    if s.shape != v.shape or s.size < 2:
        raise ValueError("time average needs at least two paired steps and values")
    if not np.isfinite(s).all() or not np.isfinite(v).all():
        raise ValueError("non-finite step or value in a time average")
    if np.any(np.diff(s) <= 0):
        raise ValueError("steps must be strictly increasing")
    span = float(s[-1] - s[0])
    if span <= 0:
        raise ValueError("time average needs a positive span")
    integral = float(np.sum((v[1:] + v[:-1]) * np.diff(s) / 2.0))
    return integral / span


def ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """OLS slope of ``y`` on ``x`` with an intercept. NaN when ``x`` has no variance."""
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    if xx.shape != yy.shape or xx.size < 2:
        return float("nan")
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("non-finite values in an OLS slope")
    xc = xx - xx.mean()
    var = float(np.dot(xc, xc))
    if var == 0.0:
        return float("nan")
    return float(np.dot(xc, yy - yy.mean()) / var)


def adjusted_contrast(contrast: float, d_jbar: float, slope: float) -> float:
    """``contrast - slope * d_jbar`` (control minus arm, same sign as the contrast)."""
    if not (math.isfinite(contrast) and math.isfinite(d_jbar) and math.isfinite(slope)):
        raise ValueError("adjusted contrast needs finite contrast, dJbar and slope")
    return float(contrast - slope * d_jbar)


def _require_finite(diff: np.ndarray) -> None:
    if diff.size == 0 or not np.isfinite(diff).all():
        raise ValueError("non-finite prompt differences")


def _central_interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    alpha = (1.0 - level) / 2.0
    lo, hi = _quantile(draws, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _bootstrap_tail(draws: np.ndarray, *, level: float, margin: float) -> dict[str, float]:
    lo, hi = _central_interval(draws, level)
    return {
        "lo": lo,
        "hi": hi,
        "p_sup": float(np.mean(draws <= margin)),
        "p_tost": float(max(np.mean(draws <= -margin), np.mean(draws >= margin))),
        "p_le_0": float(np.mean(draws <= 0.0)),
    }


def _seed_t_interval(per_seed: np.ndarray, level: float) -> dict[str, Any]:
    """One-sample t interval of the per-seed contrasts. Undefined for one seed."""
    point = float(per_seed.mean())
    n = int(per_seed.size)
    if n < 2:
        return {"defined": False, "point": point, "lo": None, "hi": None, "df": 0, "se": None}
    se = float(per_seed.std(ddof=1) / math.sqrt(n))
    df = n - 1
    crit = student_t_ppf(0.5 + level / 2.0, df)
    return {
        "defined": True,
        "point": point,
        "lo": point - crit * se,
        "hi": point + crit * se,
        "df": df,
        "se": se,
        "level": level,
    }


def paired_contrast_crossed(
    diff: np.ndarray | Sequence[Sequence[float]],
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
    margin: float = 0.05,
) -> dict[str, Any]:
    """Crossed bootstrap of the mean of a seeds-by-prompts difference matrix.

    Rows are seeds and columns are prompts. Each replicate resamples seeds
    and prompts independently with replacement and recomputes the mean. That
    is the pigeonhole bootstrap (Owen, A. B. (2007). The pigeonhole bootstrap.
    Annals of Applied Statistics, 1(2), 386-411). One RNG stream, seeded by
    ``seed``, draws the prompt indices first and the seed indices second, so
    a single seed reduces to the prompt bootstrap on the same draws.

    ``p_sup`` is ``P*(stat <= margin)`` (superiority). ``p_tost`` is
    ``max(P*(stat <= -margin), P*(stat >= margin))``. The prompt-only
    bootstrap (seeds averaged first) and a per-seed t interval are
    sensitivities: they are reported and not used as tests. Non-finite
    entries raise.
    """
    matrix = np.asarray(diff, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"diff must have shape (n_seeds, n_prompts), got {matrix.shape}")
    _require_finite(matrix)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    n_seeds, n_prompts = matrix.shape
    point = float(matrix.mean())
    per_seed = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    prompt_idx = rng.integers(0, n_prompts, size=(n_boot, n_prompts))
    seed_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    prompt_means = np.stack([matrix[s, prompt_idx].mean(axis=1) for s in range(n_seeds)], axis=1)
    crossed = prompt_means[np.arange(n_boot)[:, None], seed_idx].mean(axis=1)
    averaged = matrix.mean(axis=0)
    prompt_only = averaged[prompt_idx].mean(axis=1)
    tails = _bootstrap_tail(crossed, level=level, margin=margin)
    prompt_tails = _bootstrap_tail(prompt_only, level=level, margin=margin)
    return {
        "point": point,
        "per_seed": [float(v) for v in per_seed],
        "lo": tails["lo"],
        "hi": tails["hi"],
        "level": level,
        "margin": float(margin),
        "p_sup": tails["p_sup"],
        "p_tost": tails["p_tost"],
        "p_le_0": tails["p_le_0"],
        "n_seeds": n_seeds,
        "n_prompts": n_prompts,
        "n_boot": n_boot,
        "prompt_only": {"point": point, **prompt_tails, "level": level, "margin": float(margin)},
        "seed_t": _seed_t_interval(per_seed, level),
    }


def difference_matrix(
    pairs: Sequence[tuple[str | Path, str | Path]],
    *,
    step: int,
    split: str = "heldout",
) -> tuple[np.ndarray, list[str]]:
    """``acc(a) - acc(b)`` at ``step``, shape ``(n_seeds, n_prompts)``.

    Prompt ids are the sorted intersection of prompts present in every run.
    A non-finite accuracy raises.
    """
    if not pairs:
        raise ValueError("need at least one (run_a, run_b) pair")
    per_seed = [
        (prompt_accuracy(a, split, step), prompt_accuracy(b, split, step)) for a, b in pairs
    ]
    common = set.intersection(*(set(a) & set(b) for a, b in per_seed))
    pids = sorted(common)
    if not pids:
        raise ValueError("runs share no evaluated prompts")
    diff = np.array([[a[p] - b[p] for p in pids] for a, b in per_seed], dtype=float)
    _require_finite(diff)
    return diff, pids


def paired_contrast(
    run_a: str | Path,
    run_b: str | Path,
    *,
    step: int,
    split: str = "heldout",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Final ``acc(a) - acc(b)`` at ``step``, paired by prompt, prompt bootstrap."""
    return paired_contrast_seeds(
        [(run_a, run_b)], step=step, split=split, n_boot=n_boot, alpha=alpha, seed=seed
    )


def paired_contrast_seeds(
    pairs: Sequence[tuple[str | Path, str | Path]],
    *,
    step: int,
    split: str = "heldout",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    margin: float | None = None,
) -> dict[str, Any]:
    """Prompt bootstrap of ``acc(a) - acc(b)`` after averaging seeds within each prompt.

    This is the sensitivity to the crossed bootstrap, and the P4 screen.
    ``p_le_0`` is ``P*(diff <= 0)``. With ``margin`` set, ``p_tost`` is the
    two one-sided tests p-value for ``|diff| < margin``. Non-finite prompt
    differences raise. ``step`` is the preregistered eval, not the last row.
    """
    diff, _pids = difference_matrix(pairs, step=step, split=split)
    averaged = diff.mean(axis=0)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    draws = averaged[rng.integers(0, averaged.size, size=(n_boot, averaged.size))].mean(axis=1)
    level = 1.0 - alpha
    lo, hi = _central_interval(draws, level)
    out: dict[str, Any] = {
        "diff": float(averaged.mean()),
        "lo": lo,
        "hi": hi,
        "n_prompts": int(averaged.size),
        "n_seeds": int(diff.shape[0]),
        "p_le_0": float(np.mean(draws <= 0.0)),
    }
    if margin is not None:
        out["margin"] = float(margin)
        out["p_tost"] = float(max(np.mean(draws <= -margin), np.mean(draws >= margin)))
    return out


def holm_adjusted(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Monotone Holm adjusted p-values. A non-finite p-value raises."""
    bad = [key for key, value in pvalues.items() if not math.isfinite(float(value))]
    if bad:
        raise ValueError(f"non-finite p-value for {bad}")
    m = len(pvalues)
    order = sorted(pvalues, key=lambda key: (float(pvalues[key]), key))
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, key in enumerate(order):
        raw = float(pvalues[key]) * (m - i)
        running = max(running, raw)
        adjusted[key] = min(running, 1.0)
    return adjusted


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down: which hypotheses are rejected at family-wise ``alpha``.

    A non-finite p-value raises. It is not treated as significant.
    """
    adjusted = holm_adjusted(pvalues)
    return {key: adjusted[key] <= alpha for key in pvalues}


def permutation_spearman_p(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    n_mc: int = PERMUTATION_MC,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """One-sided permutation p-value of Spearman(x, y), greater alternative.

    Y is permuted. Ties use average ranks (the same ranks as
    :func:`vfates.utils.stats.spearman`). For ``n <= 9`` every ordering is
    enumerated and ``p = #{rho* >= rho} / n!``. Above that, ``n_mc`` Monte
    Carlo permutations are drawn from a generator seeded by ``seed`` (the
    preregistered default is 200,000 and seed 0) and
    ``p = (1 + #{rho* >= rho}) / (n_mc + 1)``, counting the observed ordering
    once.
    """
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    if xx.shape != yy.shape or xx.size < 2:
        raise ValueError("permutation Spearman needs at least two paired points")
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("non-finite values in a permutation Spearman")
    observed = spearman(xx, yy)
    if not math.isfinite(observed):
        return {
            "rho": observed,
            "p": float("nan"),
            "n": int(xx.size),
            "method": "undefined",
            "n_perm": 0,
        }
    n = int(xx.size)
    # Float Spearman of a permutation of the same values matches the observed
    # bit-for-bit on the identity, so a tolerance is only for accumulated error.
    tol = 1e-9

    def ge(value: float) -> bool:
        return value >= observed - tol

    if n <= PERMUTATION_EXACT_MAX:
        count = 0
        total = 0
        for perm in itertools.permutations(yy.tolist()):
            total += 1
            if ge(spearman(xx, perm)):
                count += 1
        return {
            "rho": float(observed),
            "p": count / total,
            "n": n,
            "method": "exact",
            "n_perm": total,
        }
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1")
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_mc):
        if ge(spearman(xx, rng.permutation(yy))):
            count += 1
    return {
        "rho": float(observed),
        "p": (count + 1) / (n_mc + 1),
        "n": n,
        "method": "monte_carlo",
        "n_perm": n_mc,
    }


def _seed_groups(
    labels: Sequence[tuple[int, ...]],
) -> list[tuple[tuple[int, ...], list[int]]]:
    """Distinct seed lists in order of first appearance, with the members using each."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(labels):
        groups.setdefault(key, []).append(i)
    return list(groups.items())


def crossed_pair_dacc(
    arm_acc: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
    control_acc: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
    *,
    n_boot: int,
    seed: int = 0,
    seed_lists: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Point and crossed-bootstrap ``acc(arm) - acc(control)`` for each pair.

    ``arm_acc[i]`` and ``control_acc[i]`` have shape ``(n_seeds_i, n_prompts)``:
    seed ``s`` of the arm is paired with seed ``s`` of its control, and every
    pair is evaluated on the same prompts. A pair's point is the mean over its
    own seeds and the prompts. One prompt-index draw is shared by every pair.
    Pairs with the same seed list (``seed_lists``, by default the seed count)
    share one seed-index draw; a pair with another seed list gets its own
    draw. With a single seed list this is the fully crossed draw. Forecasts are
    not touched here.
    """
    arms = [np.asarray(a, dtype=float) for a in arm_acc]
    controls = [np.asarray(c, dtype=float) for c in control_acc]
    if len(arms) != len(controls) or not arms:
        raise ValueError("arm_acc and control_acc must list the same, non-zero number of pairs")
    for a, c in zip(arms, controls, strict=True):
        if a.shape != c.shape or a.ndim != 2 or a.shape[0] < 1:
            raise ValueError("each pair must share shape (n_seeds, n_prompts) with n_seeds >= 1")
        if not np.isfinite(a).all() or not np.isfinite(c).all():
            raise ValueError("non-finite accuracy in a crossed rank bootstrap")
    n_prompts = arms[0].shape[1]
    if any(a.shape[1] != n_prompts for a in arms):
        raise ValueError("every pair must be evaluated on the same prompts")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if seed_lists is None:
        labels = [tuple(range(a.shape[0])) for a in arms]
    else:
        labels = [tuple(int(s) for s in seeds) for seeds in seed_lists]
        if len(labels) != len(arms) or any(
            len(key) != a.shape[0] for key, a in zip(labels, arms, strict=True)
        ):
            raise ValueError("seed_lists must give one seed per row of each pair")
    point = np.array([a.mean() - c.mean() for a, c in zip(arms, controls, strict=True)])
    rng = np.random.default_rng(seed)
    prompt_idx = rng.integers(0, n_prompts, size=(n_boot, n_prompts))
    groups = []
    for key, members in _seed_groups(labels):
        draws = rng.integers(0, len(key), size=(n_boot, len(key)))
        arm_stack = np.stack([arms[i] for i in members])
        control_stack = np.stack([controls[i] for i in members])
        groups.append((members, draws, arm_stack, control_stack))
    boots = np.empty((n_boot, len(arms)), dtype=float)
    for b in range(n_boot):
        p_idx = prompt_idx[b]
        for members, draws, arm_stack, control_stack in groups:
            s_idx = draws[b]
            arm_mean = arm_stack[:, s_idx][:, :, p_idx].mean(axis=(1, 2))
            control_mean = control_stack[:, s_idx][:, :, p_idx].mean(axis=(1, 2))
            boots[b, members] = arm_mean - control_mean
    return point, boots


def forecast_row(forecast: Mapping[str, Any]) -> dict[str, Any]:
    """The scalar forecasts one arm's lock entry provides."""
    t = forecast["transfer"]
    return {
        "tau_pool": t["tau_pool"],
        "tau_pool_std": t.get("tau_pool_std"),
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
        "probe_heldout_auc": forecast.get("probe_heldout_auc", t.get("probe_heldout_auc")),
        "behaviour_alignment": forecast.get("behaviour_alignment"),
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
    step: int,
    keys: Sequence[str] = FORECAST_KEYS,
    baseline: str = "J",
    split: str = "heldout",
    n_boot: int = 2000,
    alpha: float = 0.05,
    quantiles: Sequence[float] | None = None,
    seed: int = 0,
    seed_lists: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Spearman of each forecast with final accuracy, crossed bootstrap.

    ``runs`` maps an arm to its seed run directories. Without ``seed_lists``
    every arm must have the same seeds in the same order. With ``seed_lists``
    (arm to the seed labels of its directories) arms may differ: arms with the
    same seed list share one seed-index draw and each other list gets its own.
    Each draw resamples prompts once for every arm. An arm's accuracy is the
    mean over its own seeds and the prompts. Locked forecasts stay fixed. With
    one seed the seed draw is a no-op and the interval is the prompt bootstrap.

    ``quantiles``, when set, are the reported percentiles (for example
    ``(0.05, 0.95)`` for a 5th-percentile lower bound). Otherwise the central
    interval uses ``alpha / 2`` and ``1 - alpha / 2``. The return always
    includes the quantile map so a 5th percentile is not confused with the
    2.5th percentile of a two-sided interval.
    """
    arms = [a for a in forecasts if a in runs]
    if len(arms) < 3:
        raise ValueError(f"need at least 3 arms with runs, got {len(arms)}")
    acc = {a: [prompt_accuracy(d, split, step) for d in runs[a]] for a in arms}
    if seed_lists is None:
        if len({len(v) for v in acc.values()}) != 1:
            raise ValueError(
                "crossed rank bootstrap needs the same number of seeds for every arm "
                "unless seed_lists is given"
            )
        labels = [tuple(range(len(acc[a]))) for a in arms]
    else:
        labels = [tuple(int(s) for s in seed_lists[a]) for a in arms]
        if any(len(key) != len(acc[a]) for key, a in zip(labels, arms, strict=True)):
            raise ValueError("seed_lists must give one seed label per run directory")
    if any(not maps for maps in acc.values()):
        raise ValueError("every arm needs at least one run directory")
    pids = sorted(set.intersection(*(set(m) for maps in acc.values() for m in maps)))
    if not pids:
        raise ValueError("runs share no evaluated prompts")
    tables = [np.array([[m[p] for p in pids] for m in acc[a]], dtype=float) for a in arms]
    if not all(np.isfinite(t).all() for t in tables):
        raise ValueError("non-finite prompt accuracy")
    values = {k: np.array([forecast_row(forecasts[a])[k] for a in arms], float) for k in keys}
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    n_p = len(pids)
    n_s = max(len(key) for key in labels)
    rng = np.random.default_rng(seed)
    prompt_idx = rng.integers(0, n_p, size=(n_boot, n_p))
    groups = []
    for key, members in _seed_groups(labels):
        draws = rng.integers(0, len(key), size=(n_boot, len(key)))
        groups.append((members, draws, np.stack([tables[i] for i in members])))

    def rhos(outcome: np.ndarray) -> dict[str, float]:
        return {k: spearman(values[k], outcome) for k in keys}

    point = rhos(np.array([t.mean() for t in tables]))
    boots: list[dict[str, float]] = []
    sampled = np.empty(len(arms), dtype=float)
    for b in range(n_boot):
        p_idx = prompt_idx[b]
        for members, draws, stack in groups:
            sampled[members] = stack[:, draws[b]][:, :, p_idx].mean(axis=(1, 2))
        boots.append(rhos(sampled))
    qs = (
        (alpha / 2.0, 1.0 - alpha / 2.0)
        if quantiles is None
        else tuple(float(q) for q in quantiles)
    )

    def qmap(vals: np.ndarray) -> dict[str, float]:
        finite = vals[np.isfinite(vals)]
        if not len(finite):
            return {str(q): float("nan") for q in qs}
        estimated = _quantile(finite, qs)
        return {str(q): float(v) for q, v in zip(qs, estimated, strict=True)}

    out: dict[str, Any] = {
        "n_arms": len(arms),
        "n_prompts": n_p,
        "n_seeds": n_s,
        "seed_lists": [list(key) for key, _members in _seed_groups(labels)],
        "baseline": baseline,
        "quantiles": list(qs),
    }
    for k in keys:
        vals = np.array([b[k] for b in boots], float)
        mapped = qmap(vals)
        ordered = [mapped[str(q)] for q in qs]
        finite_ends = [v for v in ordered if math.isfinite(v)]
        lo = finite_ends[0] if finite_ends else float("nan")
        hi = finite_ends[-1] if finite_ends else float("nan")
        out[k] = {"rho": point[k], "lo": lo, "hi": hi, "quantiles": mapped}
        if k != baseline and baseline in keys:
            base_vals = np.array([b[baseline] for b in boots], float)
            delta = vals - base_vals
            delta_map = qmap(delta)
            delta_ordered = [delta_map[str(q)] for q in qs]
            finite_delta = [v for v in delta_ordered if math.isfinite(v)]
            out[f"{k}-{baseline}"] = {
                "diff": point[k] - point[baseline],
                "lo": finite_delta[0] if finite_delta else float("nan"),
                "hi": finite_delta[-1] if finite_delta else float("nan"),
                "quantiles": delta_map,
            }
    return out


def score_lock(lock_path: str | Path, runs: Mapping[str, str | Path]) -> dict[str, Any]:
    """:func:`score_forecasts` for a verified lock file and ``{arm: run_dir}``."""
    forecasts = load_lock(lock_path)
    outcomes = {arm: run_outcome(d) for arm, d in runs.items()}
    return score_forecasts(forecasts, outcomes)
