"""Pooled and prompt-averaged confusion, plus the two step-0 precision forecasts.

Rates treat ``O = 1`` as correct and ``V = 1`` as accepted. Prompt-averaged
quantities give every prompt equal weight. ``fp_mass`` is Zhang's measure:
the mean over prompts of ``P(V = 1, O = 0 | x)``.

:func:`confusion_with_ci` resamples prompts with
:func:`vfates.utils.stats.cluster_bootstrap`. A prompt drawn twice must stay
two clusters; bootstrap indices are relabelled at block boundaries before the
metric is recomputed (``group_indices`` would otherwise merge the copies).
That relabeling assumes :func:`group_indices` returns increasing row indices,
which it does.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from vfates.utils.stats import GroupIds, cluster_bootstrap, group_indices

CONFUSION_KEYS: tuple[str, ...] = (
    "n",
    "n_prompts",
    "prevalence",
    "tpr",
    "fnr",
    "fpr",
    "tnr",
    "J",
    "precision_pooled",
    "precision_prompt_mean",
    "for_prompt_mean",
    "f1",
    "verifier_accuracy",
    "accept_rate",
    "fp_mass",
)


def confusion(
    O: Sequence[Any] | np.ndarray, V: Sequence[Any] | np.ndarray, pid: GroupIds
) -> dict[str, Any]:
    """Step-0 confusion of oracle labels ``O`` and verifier labels ``V``.

    ``precision_prompt_mean`` is the prompt-mean of ``P(O = 1 | V = 1, x)``.
    ``for_prompt_mean`` is the prompt-mean false omission rate
    ``P(O = 1 | V = 0, x)``. A prompt with no accepted sample contributes its
    base accuracy to ``precision_prompt_mean``; a prompt with no rejected
    sample contributes its base accuracy to ``for_prompt_mean``. That is the
    ``t`` to ``+infinity`` and ``t`` to ``-infinity`` endpoint in
    ``docs/theory.md``, the same substitution as :func:`mean_precision_forecast`
    on the accept side. Pooled rates with an empty class stay ``NaN``. ``n``
    and ``n_prompts`` are ints; every rate is a float.
    """
    pid_list = list(pid)
    n = len(pid_list)
    o_bits = _bits(O, "O", n)
    v_bits = _bits(V, "V", n)
    if n == 0:
        nan = float("nan")
        out: dict[str, Any] = {key: nan for key in CONFUSION_KEYS}
        out["n"] = 0
        out["n_prompts"] = 0
        return out
    groups = group_indices(pid_list)
    return _metrics(o_bits, v_bits, groups)


def confusion_with_ci(
    O: Sequence[Any] | np.ndarray,
    V: Sequence[Any] | np.ndarray,
    pid: GroupIds,
    n_boot: int = 1000,
    seed: int = 0,
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Percentile prompt-cluster bootstrap of :func:`confusion`.

    Each value is ``{"point", "lo", "hi"}`` from
    :func:`vfates.utils.stats.cluster_bootstrap` (default 95% interval).
    ``metrics`` restricts the keys; the default is every confusion key.
    ``n_boot`` must be at least 1.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    pid_list = list(pid)
    n = len(pid_list)
    wanted = CONFUSION_KEYS if metrics is None else tuple(metrics)
    unknown = [key for key in wanted if key not in CONFUSION_KEYS]
    if unknown:
        raise KeyError(f"unknown confusion metrics: {unknown}")
    if n == 0:
        base = confusion(O, V, pid)
        return {
            metric: {
                "point": float(base[metric]),
                "lo": float(base[metric]),
                "hi": float(base[metric]),
            }
            for metric in wanted
        }
    o_bits = _bits(O, "O", n)
    v_bits = _bits(V, "V", n)
    group_of_row = _group_ids(pid_list)

    def stat(idx: np.ndarray, metric: str = "") -> float:
        labels = _resample_labels(idx, group_of_row)
        sliced_o = o_bits[idx]
        sliced_v = v_bits[idx]
        groups = group_indices(labels.tolist())
        return float(_metrics(sliced_o, sliced_v, groups)[metric])

    def stat_for(m: str) -> Callable[[np.ndarray], float]:
        def inner(idx: np.ndarray) -> float:
            return stat(idx, m)

        return inner

    out: dict[str, dict[str, float]] = {}
    for metric in wanted:
        point, lo, hi = cluster_bootstrap(stat_for(metric), pid_list, n_boot, seed=seed)
        out[metric] = {"point": point, "lo": lo, "hi": hi}
    return out


def mean_precision_forecast(O: Sequence[Any], V: Sequence[Any], pid: GroupIds) -> float:
    """Mean over prompts of precision where ``V`` accepts, else base accuracy.

    This is the tilt-limit forecast (Lemma 1): a prompt with no accepted audit
    sample contributes its base accuracy ``P(O = 1 | x)``.
    """
    pid_list = list(pid)
    n = len(pid_list)
    o_bits = np.asarray(O, dtype=np.float64)
    v_bits = _bits(V, "V", n)
    if len(o_bits) != n:
        raise ValueError(f"O length {len(o_bits)} != {n}")
    groups = group_indices(pid_list)
    if not groups:
        return float("nan")
    vals = []
    for rows in groups:
        accepted = v_bits[rows] == 1
        if np.any(accepted):
            vals.append(float(o_bits[rows][accepted].mean()))
        else:
            vals.append(float(o_bits[rows].mean()))
    return float(np.mean(vals))


def modal_precision_forecast(
    O: Sequence[Any],
    V: Sequence[Any],
    pid: GroupIds,
    answer: Sequence[Any],
    logprob: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Mean over prompts of the oracle label of the modal accepted answer.

    Accepted samples are clustered by ``answer``. The largest cluster wins.
    Equal counts are broken by the highest mean ``logprob`` when ``logprob``
    is given; otherwise the tied clusters' mean labels are averaged (equal
    counts, so this matches the pooled mean). A prompt with no accept
    contributes its base accuracy.

    ``tie_fraction`` is the share of prompts whose winning count was tied,
    including prompts that ``logprob`` later broke. ``n_prompts_no_accept``
    counts prompts with no accepted sample.
    """
    pid_list = list(pid)
    n = len(pid_list)
    o_bits = np.asarray(O, dtype=np.float64)
    v_bits = _bits(V, "V", n)
    answers = list(answer)
    if len(o_bits) != n or len(answers) != n:
        raise ValueError("O, V, pid, and answer must have the same length")
    logs = None if logprob is None else np.asarray(logprob, dtype=np.float64)
    if logs is not None and len(logs) != n:
        raise ValueError(f"logprob length {len(logs)} != {n}")
    groups = group_indices(pid_list)
    n_prompts = len(groups)
    if n_prompts == 0:
        return {
            "value": float("nan"),
            "tie_fraction": float("nan"),
            "n_prompts": 0,
            "n_prompts_no_accept": 0,
        }
    vals: list[float] = []
    n_tie = 0
    n_no = 0
    for rows in groups:
        accepted = [int(i) for i in rows if v_bits[i] == 1]
        if not accepted:
            n_no += 1
            vals.append(float(o_bits[rows].mean()))
            continue
        clusters: dict[Any, list[int]] = {}
        for i in accepted:
            clusters.setdefault(answers[i], []).append(i)
        max_count = max(len(members) for members in clusters.values())
        top = [key for key, members in clusters.items() if len(members) == max_count]
        if len(top) > 1:
            n_tie += 1
        chosen = _break_ties(top, clusters, logs)
        cluster_means = [float(o_bits[clusters[key]].mean()) for key in chosen]
        vals.append(float(np.mean(cluster_means)))
    return {
        "value": float(np.mean(vals)),
        "tie_fraction": n_tie / n_prompts,
        "n_prompts": n_prompts,
        "n_prompts_no_accept": n_no,
    }


def _break_ties(
    top: list[Any],
    clusters: dict[Any, list[int]],
    logs: np.ndarray | None,
) -> list[Any]:
    if len(top) == 1 or logs is None:
        return top
    means: dict[Any, float] = {}
    for key in top:
        value = float(np.nanmean(logs[clusters[key]]))
        if value == value:
            means[key] = value
    if not means:
        return top
    best = max(means.values())
    return [key for key, value in means.items() if value == best]


def _bits(values: Sequence[Any] | np.ndarray, name: str, n: int) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} length {len(array)} != {n}")
    bits = np.asarray(array, dtype=np.int64)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError(f"{name} must be 0/1")
    return bits


def _metrics(O: np.ndarray, V: np.ndarray, groups: list[np.ndarray]) -> dict[str, Any]:
    n = len(O)
    n_prompts = len(groups)
    n_pos = int(np.sum(O == 1))
    n_neg = n - n_pos
    tp = int(np.sum((O == 1) & (V == 1)))
    fp = int(np.sum((O == 0) & (V == 1)))
    fn = n_pos - tp
    tn = n_neg - fp
    tpr = _div(tp, n_pos)
    fnr = _div(fn, n_pos)
    fpr = _div(fp, n_neg)
    tnr = _div(tn, n_neg)
    precision = _div(tp, tp + fp)
    J = tpr - fpr if tpr == tpr and fpr == fpr else float("nan")
    if precision != precision or tpr != tpr:
        f1 = float("nan")
    elif precision + tpr == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * tpr / (precision + tpr)
    prec_vals: list[float] = []
    for_vals: list[float] = []
    fp_vals: list[float] = []
    for rows in groups:
        o = O[rows]
        v = V[rows]
        fp_vals.append(float(np.mean((v == 1) & (o == 0))))
        accepted = v == 1
        rejected = v == 0
        # No accept (respectively no rejection) substitutes base accuracy, the
        # Lemma 1 endpoint when that conditional is undefined.
        if np.any(accepted):
            prec_vals.append(float(np.mean(o[accepted])))
        else:
            prec_vals.append(float(np.mean(o)))
        if np.any(rejected):
            for_vals.append(float(np.mean(o[rejected])))
        else:
            for_vals.append(float(np.mean(o)))
    return {
        "n": n,
        "n_prompts": n_prompts,
        "prevalence": _div(n_pos, n),
        "tpr": tpr,
        "fnr": fnr,
        "fpr": fpr,
        "tnr": tnr,
        "J": J,
        "precision_pooled": precision,
        "precision_prompt_mean": float(np.mean(prec_vals)) if prec_vals else float("nan"),
        "for_prompt_mean": float(np.mean(for_vals)) if for_vals else float("nan"),
        "f1": f1,
        "verifier_accuracy": _div(tp + tn, n),
        "accept_rate": _div(tp + fp, n),
        "fp_mass": float(np.mean(fp_vals)) if fp_vals else float("nan"),
    }


def _div(num: int, den: int) -> float:
    if den == 0:
        return float("nan")
    return num / den


def _group_ids(pid: GroupIds) -> np.ndarray:
    ids = np.empty(len(pid), dtype=np.int64)
    mapping: dict[object, int] = {}
    for i, key in enumerate(pid):
        if key not in mapping:
            mapping[key] = len(mapping)
        ids[i] = mapping[key]
    return ids


def _resample_labels(idx: np.ndarray, group_of_row: np.ndarray) -> np.ndarray:
    """Cluster ids that stay distinct when ``cluster_bootstrap`` repeats a prompt."""
    if len(idx) == 0:
        return np.zeros(0, dtype=np.int64)
    groups = group_of_row[idx]
    change = np.empty(len(idx), dtype=bool)
    change[0] = True
    if len(idx) > 1:
        change[1:] = (groups[1:] != groups[:-1]) | (idx[1:] <= idx[:-1])
    return np.cumsum(change) - 1
