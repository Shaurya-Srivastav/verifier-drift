r"""Transfer precision: the step-0 screen for which verifier errors spread.

Setting. A policy with shared parameters is trained by policy gradient against
a fixed verifier ``V``. Linearise the per-sample log-probability,
``log pi(y|x) ~ log pi_0(y|x) + <phi(x, y), theta>``, where ``phi`` is the
per-sample gradient of ``log pi_0(y|x)`` (or any feature map standing in for
it, such as pooled hidden states for a linear head).

The expected mean-centred policy-gradient step is

.. math::  g_V = \mathbb{E}_x\,\mathrm{Cov}_{\pi_0(\cdot|x)}(\phi, V)
               = g_O - g_{FN} + g_{FP},

exactly, because ``V = O - FN + FP`` for every sample. Resampled
class-conditional noise gives ``E[g_V] = J g_O``; fixed errors add a bias.

A held-out prompt ``x'`` is moved only through ``theta``. To first order the
log-probability of held-out sample ``y`` grows at rate ``s(y) = <phi(x', y), g_V>``,
so after a tilt of strength ``kappa`` along that direction the policy on ``x'``
is ``pi_0(y|x') exp(kappa s(y))`` (normalised). Because audit samples are drawn
from ``pi_0``, the tilted accuracy is estimated by self-normalised importance
weighting of the audit samples:

.. math::  \hat\tau(\kappa) = \mathbb{E}_{x'} \frac{\sum_i O_i e^{\kappa s_i}}{\sum_i e^{\kappa s_i}}.

``tau_hard`` (``kappa -> infinity``) is the oracle accuracy of each held-out
prompt's highest-velocity sample; ``kappa = 0`` recovers base accuracy. This is
a first-order screen: it can miss fates decided by later feature learning, and
the paper reports those cases.

With ``n`` audit samples per prompt, ``tau_hard`` cannot exceed the *coverage*
(the share of held-out prompts with at least one correct sample), and it is
inflated on prompts where the favoured behaviour was simply not sampled.
``tau_rel = tau_hard / tau_hard(oracle direction)`` cancels the ceiling but not
the inflation.

The primary finite-sample statistic is therefore the pooled tail
``tau_pool(q)``: the oracle accuracy of the fraction ``q`` of held-out samples,
pooled over prompts, whose log-probability rises fastest (largest prompt-centred
velocity). A behaviour carried by a shared feature is favoured on every prompt,
so pooling borrows strength across prompts instead of requiring each prompt to
have sampled it. ``q = 1/32`` is pre-registered (:data:`PRIMARY_Q`), chosen on
the toy worlds in pilot P0 before any LLM run; other values are sensitivity
checks.

All functions are numpy-only so the same code scores the exact toy worlds in
:mod:`vfates.theory` and real LLM audits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from vfates.utils.stats import group_indices

DEFAULT_KAPPAS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_QS: tuple[float, ...] = (1 / 64, 1 / 32, 1 / 16, 1 / 8)
PRIMARY_Q: float = 1 / 32


def q_label(q: float) -> str:
    """Stable dictionary key for a tail fraction, e.g. ``1/32``."""
    inv = 1.0 / q
    return f"1/{round(inv)}" if abs(inv - round(inv)) < 1e-9 else f"{q:g}"


def prompt_centered(values: np.ndarray, pid: Sequence[object]) -> np.ndarray:
    """``values`` minus their per-prompt mean (the mean-centred GRPO advantage)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.empty_like(v)
    for rows in group_indices(pid):
        out[rows] = v[rows] - v[rows].mean()
    return out


def velocity_direction(
    features: np.ndarray,
    pid: Sequence[object],
    reward: np.ndarray,
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    r"""Estimate ``g = E_x Cov_{pi_0(.|x)}(phi, reward)`` from audit samples.

    Each prompt contributes the mean of ``A_i phi_i`` over its samples, with
    ``A_i`` the prompt-centred reward; prompts are weighted equally, as in a
    GRPO batch. If ``lengths`` is given, ``phi_i`` is divided by the completion
    length, matching a per-sequence token-mean loss.
    """
    phi = np.asarray(features, dtype=np.float64)
    if lengths is not None:
        phi = phi / np.maximum(np.asarray(lengths, dtype=np.float64), 1.0)[:, None]
    adv = prompt_centered(reward, pid)
    groups = group_indices(pid)
    g = np.zeros(phi.shape[1])
    for rows in groups:
        g += (adv[rows, None] * phi[rows]).mean(axis=0)
    return g / max(len(groups), 1)


def velocities(features: np.ndarray, g: np.ndarray) -> np.ndarray:
    """First-order growth rate of each sample's log-probability along ``g``."""
    return np.asarray(features, dtype=np.float64) @ np.asarray(g, dtype=np.float64)


def _standardise(s: np.ndarray, pid: Sequence[object]) -> np.ndarray:
    """Divide by the pooled within-prompt standard deviation so ``kappa`` is unit-free."""
    centred = prompt_centered(s, pid)
    scale = float(np.sqrt(np.mean(centred**2)))
    return s / scale if scale > 0 else np.zeros_like(s)


def tau_hard(s: np.ndarray, pid: Sequence[object], O: np.ndarray, rtol: float = 1e-9) -> float:
    """Mean over prompts of the oracle label of the highest-velocity sample (ties averaged)."""
    O = np.asarray(O, dtype=np.float64)
    vals = []
    for rows in group_indices(pid):
        sr = s[rows]
        top = sr >= sr.max() - rtol * max(1.0, abs(sr.max()))
        vals.append(O[rows][top].mean())
    return float(np.mean(vals)) if vals else float("nan")


def tau_soft(
    s: np.ndarray, pid: Sequence[object], O: np.ndarray, kappa: float, standardise: bool = True
) -> float:
    """Self-normalised importance-weighted accuracy after a tilt of strength ``kappa``."""
    O = np.asarray(O, dtype=np.float64)
    z = _standardise(s, pid) if standardise else np.asarray(s, dtype=np.float64)
    vals = []
    for rows in group_indices(pid):
        w = kappa * z[rows]
        w = np.exp(w - w.max())
        vals.append(float((w * O[rows]).sum() / w.sum()))
    return float(np.mean(vals)) if vals else float("nan")


def tau_pool(s: np.ndarray, pid: Sequence[object], O: np.ndarray, q: float = PRIMARY_Q) -> float:
    """Oracle accuracy of the top ``q`` fraction of prompt-centred velocities, pooled.

    The tail holds ``k = max(1, round(q * N))`` samples. Samples tied with the
    ``k``-th largest value share the remaining slots equally, so the statistic
    does not depend on sample order.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError("q must lie in (0, 1]")
    O = np.asarray(O, dtype=np.float64)
    c = prompt_centered(s, pid)
    n = c.size
    if n == 0:
        return float("nan")
    k = max(1, round(q * n))
    threshold = np.partition(c, n - k)[n - k]
    above = c > threshold
    tied = c == threshold
    n_above = int(above.sum())
    share = (k - n_above) / int(tied.sum())
    return float((O[above].sum() + share * O[tied].sum()) / k)


def tau_pool_interval(
    train_features: np.ndarray,
    train_pid: Sequence[object],
    train_V: np.ndarray,
    held_features: np.ndarray,
    held_pid: Sequence[object],
    held_O: np.ndarray,
    q: float = PRIMARY_Q,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    train_lengths: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile interval of ``tau_pool`` resampling train and held-out prompts.

    Both levels are resampled because the velocity direction is itself an
    estimate from the training audit.
    """
    rng = np.random.default_rng(seed)
    t_groups, h_groups = group_indices(train_pid), group_indices(held_pid)
    t_feat, h_feat = np.asarray(train_features), np.asarray(held_features)
    t_v, h_o = np.asarray(train_V, dtype=np.float64), np.asarray(held_O, dtype=np.float64)
    stats = []
    for _ in range(n_boot):
        t_pick = rng.integers(0, len(t_groups), len(t_groups))
        h_pick = rng.integers(0, len(h_groups), len(h_groups))
        t_rows = np.concatenate([t_groups[i] for i in t_pick])
        h_rows = np.concatenate([h_groups[i] for i in h_pick])
        t_ids = np.concatenate([np.full(len(t_groups[i]), j) for j, i in enumerate(t_pick)])
        h_ids = np.concatenate([np.full(len(h_groups[i]), j) for j, i in enumerate(h_pick)])
        lengths = None if train_lengths is None else np.asarray(train_lengths)[t_rows]
        g = velocity_direction(t_feat[t_rows], t_ids, t_v[t_rows], lengths)
        stats.append(tau_pool(velocities(h_feat[h_rows], g), h_ids, h_o[h_rows], q))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def first_order_slope(
    s: np.ndarray, pid: Sequence[object], O: np.ndarray, standardise: bool = True
) -> float:
    """``d tau(kappa) / d kappa`` at ``kappa = 0``: the mean within-prompt covariance of s and O."""
    z = _standardise(s, pid) if standardise else np.asarray(s, dtype=np.float64)
    O = np.asarray(O, dtype=np.float64)
    vals = [float(np.mean((z[r] - z[r].mean()) * (O[r] - O[r].mean()))) for r in group_indices(pid)]
    return float(np.mean(vals)) if vals else float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def split_half_reliability(
    features: np.ndarray,
    pid: Sequence[object],
    reward: np.ndarray,
    n_splits: int = 20,
    seed: int = 0,
    lengths: np.ndarray | None = None,
) -> float:
    """Mean cosine between velocity directions estimated on random halves of the prompts."""
    groups = group_indices(pid)
    if len(groups) < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    pid_arr = np.asarray(pid, dtype=object)
    cos = []
    for _ in range(n_splits):
        perm = rng.permutation(len(groups))
        half = len(groups) // 2
        rows_a = np.concatenate([groups[i] for i in perm[:half]])
        rows_b = np.concatenate([groups[i] for i in perm[half:]])
        la = None if lengths is None else lengths[rows_a]
        lb = None if lengths is None else lengths[rows_b]
        ga = velocity_direction(features[rows_a], pid_arr[rows_a], reward[rows_a], la)
        gb = velocity_direction(features[rows_b], pid_arr[rows_b], reward[rows_b], lb)
        cos.append(cosine(ga, gb))
    return float(np.nanmean(cos))


@dataclass
class TransferForecast:
    """Everything the screen reports for one verifier arm."""

    tau_pool: float
    tau_pool_curve: dict[str, float]
    tau_pool_ci: tuple[float, float] | None
    tau_hard: float
    tau_soft: dict[str, float]
    slope: float
    cos_gV_gO: float
    norm_gV: float
    norm_gO: float
    reliability: float
    base_acc_heldout: float
    oracle_tau_hard: float
    coverage: float
    tau_rel: float
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def transfer_forecast(
    train_features: np.ndarray,
    train_pid: Sequence[object],
    train_V: np.ndarray,
    train_O: np.ndarray,
    held_features: np.ndarray,
    held_pid: Sequence[object],
    held_O: np.ndarray,
    kappas: Sequence[float] = DEFAULT_KAPPAS,
    qs: Sequence[float] = DEFAULT_QS,
    q: float = PRIMARY_Q,
    train_lengths: np.ndarray | None = None,
    reliability_splits: int = 20,
    n_boot: int = 0,
    seed: int = 0,
) -> TransferForecast:
    """Transfer precision of one verifier arm, from step-0 audit samples only.

    ``train_*`` are audit samples on the training pool labelled by the arm's
    verifier (``train_V``) and the oracle (``train_O``). ``held_*`` are audit
    samples on held-out prompts; only their oracle labels are used. ``n_boot > 0``
    adds a two-level prompt bootstrap interval for the primary ``tau_pool``.
    """
    gV = velocity_direction(train_features, train_pid, np.asarray(train_V, float), train_lengths)
    gO = velocity_direction(train_features, train_pid, np.asarray(train_O, float), train_lengths)
    s = velocities(held_features, gV)
    s_oracle = velocities(held_features, gO)
    held_O = np.asarray(held_O, dtype=np.float64)
    held_groups = group_indices(held_pid)
    base = float(np.mean([held_O[r].mean() for r in held_groups]))
    coverage = float(np.mean([held_O[r].max() for r in held_groups]))
    th, th_oracle = tau_hard(s, held_pid, held_O), tau_hard(s_oracle, held_pid, held_O)
    interval = None
    if n_boot > 0:
        interval = tau_pool_interval(
            train_features,
            train_pid,
            train_V,
            held_features,
            held_pid,
            held_O,
            q=q,
            n_boot=n_boot,
            seed=seed,
            train_lengths=train_lengths,
        )
    return TransferForecast(
        tau_pool=tau_pool(s, held_pid, held_O, q),
        tau_pool_curve={q_label(x): tau_pool(s, held_pid, held_O, x) for x in qs},
        tau_pool_ci=interval,
        tau_hard=th,
        tau_soft={f"{k:g}": tau_soft(s, held_pid, held_O, k) for k in kappas},
        slope=first_order_slope(s, held_pid, held_O),
        cos_gV_gO=cosine(gV, gO),
        norm_gV=float(np.linalg.norm(gV)),
        norm_gO=float(np.linalg.norm(gO)),
        reliability=split_half_reliability(
            train_features,
            train_pid,
            np.asarray(train_V, float),
            reliability_splits,
            seed,
            train_lengths,
        ),
        base_acc_heldout=base,
        oracle_tau_hard=th_oracle,
        coverage=coverage,
        tau_rel=th / th_oracle if th_oracle > 0 else float("nan"),
    )
