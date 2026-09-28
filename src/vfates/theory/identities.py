"""Finite-world checks of the tilt, decomposition and noise identities.

``V = O - FN + FP`` pathwise with ``FN = O (1-V)`` and ``FP = (1-O) V``, so
the shared update splits exactly. Class-conditional noise with Youden ``J``
has ``E[g_V] = J g_O``. Inside a GRPO group the mean-centred advantage of a
sample satisfies ``E[A | O] = (1 - 1/G) J (O - p)`` when each group-mate is
correct with probability ``p`` and the verifier realises Youden ``J``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vfates.forecast.transfer import cosine
from vfates.theory.dynamics import cov_phi_reward, shared_update, softmax, tilt
from vfates.theory.worlds import World


def check_tilt_invariance(
    world: World,
    v: np.ndarray,
    t: float | Sequence[float] = (1.0, 5.0, 20.0),
) -> float:
    """Max absolute error of ``pi_t(.|V=v, x)`` against ``pi_0(.|V=v, x)``.

    Prompts with no ``pi_0`` mass on a verifier value are skipped. ``t`` may
    be one horizon or several; the return is the worst case.
    """
    horizons = (float(t),) if isinstance(t, (int, float)) else tuple(float(x) for x in t)
    pi0 = softmax(world.base)
    vv = np.asarray(v)
    worst = 0.0
    for horizon in horizons:
        pi = tilt(world, vv, horizon)
        for level in np.unique(vv):
            mask = (vv == level).astype(np.float64)
            base = pi0 * mask
            moved = pi * mask
            mass = base.sum(axis=-1)
            ok = mass > 1e-15
            if not np.any(ok):
                continue
            cond0 = base[ok] / mass[ok, None]
            cond = moved[ok] / moved[ok].sum(axis=-1, keepdims=True)
            worst = max(worst, float(np.max(np.abs(cond - cond0))))
    return worst


def check_decomposition(
    world: World,
    v: np.ndarray,
    pi: np.ndarray | None = None,
) -> float:
    """Max absolute error of ``Cov(phi, V) - (Cov(phi, O) - Cov(phi, FN) + Cov(phi, FP))``.

    ``FN = O (1-V)`` and ``FP = (1-O) V``. The default policy is ``pi_0``.
    """
    if pi is None:
        pi = softmax(world.base)
    oo = np.asarray(world.O, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    fn = oo * (1.0 - vv)
    fp = (1.0 - oo) * vv
    got = cov_phi_reward(world.phi, pi, vv)
    expect = (
        cov_phi_reward(world.phi, pi, oo)
        - cov_phi_reward(world.phi, pi, fn)
        + cov_phi_reward(world.phi, pi, fp)
    )
    return float(np.max(np.abs(got - expect)))


def check_j_scaling(
    world: World,
    rho0: float,
    rho1: float,
    n_draws: int,
    seed: int,
) -> tuple[float, float]:
    """``(cosine, scale)`` of the Monte Carlo mean of ``g_V`` against ``J g_O``.

    Each draw samples an independent class-conditional verifier on every
    output and computes the exact shared update under ``pi_0``. ``scale`` is
    the least-squares multiplier ``mean(g_V) ≈ scale * J * g_O``.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    pi = softmax(world.base)
    oo = np.asarray(world.O, dtype=np.float64)
    youden = (1.0 - rho1) - rho0
    g_o = shared_update(world.phi, pi, oo)
    target = youden * g_o
    rng = np.random.default_rng(seed)
    acc = np.zeros(world.phi.shape[-1], dtype=np.float64)
    for _ in range(n_draws):
        coin = rng.random(oo.shape)
        sampled = np.where(oo == 1.0, coin >= rho1, coin < rho0).astype(np.float64)
        acc += shared_update(world.phi, pi, sampled)
    mean_g = acc / n_draws
    denom = float(np.dot(target, target))
    scale = float(np.dot(mean_g, target) / denom) if denom > 0.0 else float("nan")
    return cosine(mean_g, target), scale


def check_group_advantage_scaling(
    G_list: Sequence[int],
    p_list: Sequence[float],
    J_list: Sequence[float],
    n_groups: int,
    seed: int,
) -> list[dict[str, float]]:
    """Rows comparing simulated group advantages with ``(1 - 1/G) J (O - p)``.

    Each row draws ``n_groups`` groups. Members are correct with probability
    ``p``; the verifier is symmetric class-conditional noise with Youden ``J``
    (``FPR = FNR = (1-J)/2``). ``z_*`` is the clustered error of the
    mean-centred advantage in Monte Carlo standard errors. ``ratio_*`` is the
    empirical mean divided by the identity. ``std_sign_flip_rate`` is the
    fraction of non-zero mean-centred advantages whose sign changes under
    population-std normalisation (the ``1e-8`` floor does not flip it).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for group_size in G_list:
        for correct_p in p_list:
            for youden in J_list:
                if not 0.0 <= youden <= 1.0:
                    raise ValueError(f"J must lie in [0, 1], got {youden}")
                rate = 0.5 * (1.0 - youden)
                correct = rng.random((n_groups, group_size)) < correct_p
                coin = rng.random((n_groups, group_size))
                reward = np.where(correct, coin >= rate, coin < rate).astype(np.float64)
                mean = reward.mean(axis=1, keepdims=True)
                centred = reward - mean
                scale = reward.std(axis=1, keepdims=True)
                normalised = np.zeros_like(reward)
                ok = scale[:, 0] > 0.0
                normalised[ok] = centred[ok] / (scale[ok] + 1e-8)
                nonzero = centred != 0.0
                if np.any(nonzero):
                    flips = float(
                        np.mean(np.sign(normalised[nonzero]) != np.sign(centred[nonzero]))
                    )
                else:
                    flips = 0.0
                row: dict[str, float] = {
                    "G": float(group_size),
                    "p": float(correct_p),
                    "J": float(youden),
                    "std_sign_flip_rate": flips,
                }
                factor = (1.0 - 1.0 / group_size) * youden
                for label, mask in ((1.0, correct), (0.0, ~correct)):
                    theory = factor * (label - correct_p)
                    sums = (centred * mask).sum(axis=1)
                    counts = mask.sum(axis=1).astype(np.float64)
                    total = float(counts.sum())
                    emp = float(sums.sum() / total) if total > 0.0 else float("nan")
                    resid = sums - emp * counts
                    mean_count = total / n_groups
                    se = float(np.sqrt(np.var(resid, ddof=1) / n_groups) / mean_count)
                    z = (emp - theory) / se if se > 0.0 else float("nan")
                    ratio = emp / theory if theory != 0.0 else float("nan")
                    tag = "O1" if label == 1.0 else "O0"
                    row[f"theory_{tag}"] = float(theory)
                    row[f"emp_{tag}"] = emp
                    row[f"se_{tag}"] = se
                    row[f"z_{tag}"] = float(z)
                    row[f"ratio_{tag}"] = float(ratio)
                rows.append(row)
    return rows
