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

from vfates.utils.stats import GroupIds, group_indices

DEFAULT_KAPPAS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_QS: tuple[float, ...] = (1 / 64, 1 / 32, 1 / 16, 1 / 8)
PRIMARY_Q: float = 1 / 32
# Matches ``vfates.train.grpo_core.group_advantages`` (TRL ``scale_rewards="group"``).
_STD_EPS: float = 1e-4


def q_label(q: float) -> str:
    """Stable dictionary key for a tail fraction, e.g. ``1/32``."""
    inv = 1.0 / q
    return f"1/{round(inv)}" if abs(inv - round(inv)) < 1e-9 else f"{q:g}"


def length_division(normalize: str, loss_agg: str) -> bool:
    """Whether train rows of ``phi`` are divided by completion length.

    True only when features are summed completion-token log-probabilities
    (``normalize="sum"``) and the trainer reduces by the sequence mean
    (``loss_agg="seq_mean"``). In that case the parameter step is
    ``E[A * phi_sum / L]``, so the rows that form ``g`` are divided by ``L``.
    Held-out velocities stay undivided: ``s = <phi_sum, g>`` is the change in
    sequence log-probability.

    False for ``("token_mean", "seq_mean")`` (the feature is already a mean,
    and dividing again would reorder by length) and for ``("sum", "token_mean")``
    or ``("sum", "dr_constant")``. ``("token_mean", "token_mean")`` and
    ``("token_mean", "dr_constant")`` raise ``ValueError``: a token-mean feature
    is not the sum those reductions expect.
    """
    if normalize == "sum" and loss_agg == "seq_mean":
        return True
    if (normalize, loss_agg) in {
        ("token_mean", "seq_mean"),
        ("sum", "token_mean"),
        ("sum", "dr_constant"),
    }:
        return False
    if normalize == "token_mean" and loss_agg in ("token_mean", "dr_constant"):
        raise ValueError(
            f"token_mean features are incompatible with loss_agg {loss_agg!r}: "
            "the feature is already a per-token mean"
        )
    raise ValueError(f"unsupported (normalize, loss_agg) pair ({normalize!r}, {loss_agg!r})")


def prompt_centered(values: np.ndarray, pid: GroupIds) -> np.ndarray:
    """``values`` minus their per-prompt mean (the mean-centred GRPO advantage)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.empty_like(v)
    for rows in group_indices(pid):
        out[rows] = v[rows] - v[rows].mean()
    return out


def group_std_advantages(
    reward: np.ndarray,
    pid: GroupIds,
    eps: float = _STD_EPS,
) -> np.ndarray:
    """Prompt-centred reward divided by that prompt's unbiased standard deviation.

    This is ``vfates.train.grpo_core.group_advantages`` with ``mode="std"`` and
    ``std="unbiased"`` (divide by ``G - 1``, then add ``eps``, default
    ``1e-4``), on possibly unequal group sizes. A prompt whose standard
    deviation is ``np.isclose`` to 0, including a group of size 1, maps to
    advantage 0. Within one prompt the scale is a positive constant, so it
    does not change that prompt's ranking. Across prompts the scales differ,
    and the averaged direction can move. The primary screen stays the
    covariance; this vector is only the secondary direction behind
    ``tau_pool_std``.
    """
    values = np.asarray(reward, dtype=np.float64)
    out = np.empty_like(values)
    for rows in group_indices(pid):
        group = values[rows]
        centered = group - group.mean()
        g = int(group.size)
        if g <= 1:
            out[rows] = 0.0
            continue
        scale = float(np.sqrt(np.sum(centered * centered) / (g - 1)))
        # torch.isclose / np.isclose defaults: rtol=1e-05, atol=1e-08.
        if np.isclose(scale, 0.0):
            out[rows] = 0.0
        else:
            out[rows] = centered / (scale + eps)
    return out


def velocity_direction(
    features: np.ndarray,
    pid: GroupIds,
    reward: np.ndarray,
    lengths: np.ndarray | None = None,
    *,
    center: bool = True,
) -> np.ndarray:
    r"""Estimate ``g = E_x Cov_{pi_0(.|x)}(phi, reward)`` from audit samples.

    Each prompt contributes the mean of ``A_i phi_i`` over its samples, with
    ``A_i`` the prompt-centred reward (or ``reward`` itself when ``center`` is
    false, as for :func:`group_std_advantages`); prompts are weighted equally.

    For sum features under a sequence-mean loss (:func:`length_division` is
    true), pass ``lengths``: each training row is divided by its completion
    length, because the trainer's step is ``E[A * phi_sum / L]``. Do not pass
    ``lengths`` for token-mean features. Held-out velocities are not divided.
    :func:`velocities` scores the undivided features, so ``s = <phi, g>`` is
    the change in sequence log-probability.
    """
    phi = np.asarray(features, dtype=np.float64)
    if lengths is not None:
        phi = phi / np.maximum(np.asarray(lengths, dtype=np.float64), 1.0)[:, None]
    adv = prompt_centered(reward, pid) if center else np.asarray(reward, dtype=np.float64)
    groups = group_indices(pid)
    g = np.zeros(phi.shape[1])
    for rows in groups:
        g += (adv[rows, None] * phi[rows]).mean(axis=0)
    return g / max(len(groups), 1)


def velocities(features: np.ndarray, g: np.ndarray) -> np.ndarray:
    """First-order growth rate of each sample's log-probability along ``g``."""
    return np.asarray(features, dtype=np.float64) @ np.asarray(g, dtype=np.float64)


def _standardise(s: np.ndarray, pid: GroupIds) -> np.ndarray:
    """Divide by the pooled within-prompt standard deviation so ``kappa`` is unit-free."""
    centred = prompt_centered(s, pid)
    scale = float(np.sqrt(np.mean(centred**2)))
    return s / scale if scale > 0 else np.zeros_like(s)


def tau_hard(s: np.ndarray, pid: GroupIds, O: np.ndarray, rtol: float = 1e-9) -> float:
    """Mean over prompts of the oracle label of the highest-velocity sample (ties averaged)."""
    O = np.asarray(O, dtype=np.float64)
    vals = []
    for rows in group_indices(pid):
        sr = s[rows]
        top = sr >= sr.max() - rtol * max(1.0, abs(sr.max()))
        vals.append(O[rows][top].mean())
    return float(np.mean(vals)) if vals else float("nan")


def tau_soft(
    s: np.ndarray, pid: GroupIds, O: np.ndarray, kappa: float, standardise: bool = True
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


def tau_pool(s: np.ndarray, pid: GroupIds, O: np.ndarray, q: float = PRIMARY_Q) -> float:
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
    train_pid: GroupIds,
    train_V: np.ndarray,
    held_features: np.ndarray,
    held_pid: GroupIds,
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
    s: np.ndarray, pid: GroupIds, O: np.ndarray, standardise: bool = True
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
    pid: GroupIds,
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


def _prompt_contribution(phi: np.ndarray, reward: np.ndarray) -> np.ndarray:
    adv = reward - reward.mean()
    return np.asarray((adv[:, None] * phi).mean(axis=0))


def _jackknife_variance(phi: np.ndarray, reward: np.ndarray) -> float:
    """Jackknife estimate of ``tr Var(c_p)`` from one prompt's samples.

    ``c_p`` is the plug-in covariance of ``reward`` and ``phi``; each
    leave-one-out value is computed in closed form from the prompt's sums.
    """
    n = len(reward)
    m = n - 1
    v_loo = (reward.sum() - reward) / m
    phi_loo = (phi.sum(axis=0)[None, :] - phi) / m
    vphi_loo = ((reward[:, None] * phi).sum(axis=0)[None, :] - reward[:, None] * phi) / m
    loo = vphi_loo - v_loo[:, None] * phi_loo
    dev = loo - loo.mean(axis=0)
    return float((dev**2).sum() * m / n)


def _noise_at(budget: dict[str, float], samples: float) -> float:
    if samples <= 0:
        raise ValueError(f"samples per prompt must be positive, got {samples}")
    w = min(max(budget["within_share"], 0.0), 1.0)
    return budget["noise"] * ((1.0 - w) + w * budget["samples_per_prompt"] / samples)


def reliability_budget(
    features: np.ndarray,
    pid: GroupIds,
    reward: np.ndarray,
    lengths: np.ndarray | None = None,
) -> dict[str, float]:
    r"""Signal and noise of the velocity direction, for sizing an audit.

    Prompt ``p`` contributes ``c_p`` (the mean of ``A_i phi_i`` over its samples)
    to :func:`velocity_direction`. With ``signal = |E c_p|^2`` and
    ``noise = tr Cov(c_p)``, both estimated without bias across prompts, the
    split-half cosine on ``m`` prompts per half is close to
    ``signal / (signal + noise / m)`` in high dimension.

    ``within`` is the part of ``noise`` that is sampling noise inside a prompt,
    the mean over prompts of the jackknife variance of ``c_p`` (conservative in
    expectation; Efron and Stein 1981). It scales as one over the samples per
    prompt, while the between-prompt rest does not. ``within_share`` is
    ``within / noise`` and is not capped: above 1 means the between-prompt part
    is indistinguishable from zero. Every prompt needs four samples.
    """
    phi = np.asarray(features, dtype=np.float64)
    if lengths is not None:
        phi = phi / np.maximum(np.asarray(lengths, dtype=np.float64), 1.0)[:, None]
    v = np.asarray(reward, dtype=np.float64)
    groups = group_indices(pid)
    if len(groups) < 2 or min(len(r) for r in groups) < 4:
        raise ValueError("reliability_budget needs two prompts with at least four samples each")
    contrib = np.stack([_prompt_contribution(phi[r], v[r]) for r in groups])
    within = float(np.mean([_jackknife_variance(phi[r], v[r]) for r in groups]))
    n_prompts = len(groups)
    mean = contrib.mean(axis=0)
    noise = float(((contrib - mean) ** 2).sum() / (n_prompts - 1))
    signal = float(mean @ mean) - noise / n_prompts
    return {
        "signal": signal,
        "noise": noise,
        "within": within,
        "within_share": within / noise if noise > 0 else 0.0,
        "prompts": float(n_prompts),
        "samples_per_prompt": float(np.mean([len(r) for r in groups])),
    }


def predicted_reliability(budget: dict[str, float], prompts: float, samples: float) -> float:
    """Split-half cosine that :func:`reliability_budget` predicts for a pool of this shape.

    ``within_share`` is clipped to ``[0, 1]`` here.
    """
    if prompts <= 0:
        raise ValueError(f"prompts must be positive, got {prompts}")
    if budget["signal"] <= 0:
        return 0.0
    noise = _noise_at(budget, samples)
    return budget["signal"] / (budget["signal"] + noise / (prompts / 2.0))


def prompts_for_reliability(budget: dict[str, float], target: float, samples: float) -> int | None:
    """Train prompts at which the predicted split-half cosine reaches ``target``.

    At least 4, the fewest prompts :func:`split_half_reliability` accepts.
    """
    if budget["signal"] <= 0 or not 0.0 < target < 1.0:
        return None
    noise = _noise_at(budget, samples)
    return max(4, int(np.ceil(2.0 * noise / budget["signal"] * target / (1.0 - target))))


@dataclass
class TransferForecast:
    """Everything the screen reports for one verifier arm."""

    tau_pool: float
    tau_pool_std: float
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
    train_pid: GroupIds,
    train_V: np.ndarray,
    train_O: np.ndarray,
    held_features: np.ndarray,
    held_pid: GroupIds,
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

    Pass ``train_lengths`` exactly when :func:`length_division` is true (sum
    features under ``seq_mean``). Those training rows are divided by completion
    length. Held-out velocities stay undivided: ``s`` is the change in sequence
    log-probability along ``g``. The primary direction is the prompt-centred
    covariance. ``tau_pool_std`` is the same pooled tail along the
    group-std-weighted direction (:func:`group_std_advantages`); it is stored
    and is not the pre-registered statistic. Zero-variance prompts contribute 0
    to that secondary direction.
    """
    train_v = np.asarray(train_V, float)
    gV = velocity_direction(train_features, train_pid, train_v, train_lengths)
    gO = velocity_direction(train_features, train_pid, np.asarray(train_O, float), train_lengths)
    g_std = velocity_direction(
        train_features,
        train_pid,
        group_std_advantages(train_v, train_pid),
        train_lengths,
        center=False,
    )
    s = velocities(held_features, gV)
    s_std = velocities(held_features, g_std)
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
        tau_pool_std=tau_pool(s_std, held_pid, held_O, q),
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
            train_v,
            reliability_splits,
            seed,
            train_lengths,
        ),
        base_acc_heldout=base,
        oracle_tau_hard=th_oracle,
        coverage=coverage,
        tau_rel=th / th_oracle if th_oracle > 0 else float("nan"),
    )
