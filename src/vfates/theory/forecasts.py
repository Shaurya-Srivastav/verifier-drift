"""Exact and sampled step-0 forecasts on a softmax world.

Mean precision is the terminal accuracy of the exponential tilt (Lemma 1).
Modal precision is the terminal accuracy of Euclidean policy gradient at
``beta = 0`` (Proposition 2): the oracle label of the highest-logit accepted
output. Tied argmaxes are averaged with ``pi_0`` weights, the same rule as
``exact_tau``. Transfer precision ``tau`` is the oracle label of the
highest-velocity held-out output, with velocity ``s = phi @ g_V`` and
``g_V = E_x Cov_{pi_0}(phi, V)`` (Theorem 3). Ties are averaged with ``pi_0``
weights. Prompts with no accepted mass contribute their base accuracy to the
precision forecasts.
"""

from __future__ import annotations

import numpy as np

from vfates.forecast.transfer import TransferForecast, transfer_forecast
from vfates.theory.dynamics import (
    FlowResult,
    TrainResult,
    expected_reward,
    logits,
    oracle_accuracy,
    shared_update,
    softmax,
)
from vfates.theory.worlds import ResampledSpec, World

_TIE_RTOL = 1e-9


def mean_precision(pi0: np.ndarray, o: np.ndarray, v: np.ndarray) -> float:
    """Prompt-averaged precision of ``v`` under ``pi0``.

    A prompt whose accepted mass is zero contributes its base accuracy
    ``E_{pi0}[O]`` instead of a precision.
    """
    pi = np.asarray(pi0, dtype=np.float64)
    oo = np.asarray(o, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    weights = pi * vv
    mass = weights.sum(axis=-1)
    numer = (weights * oo).sum(axis=-1)
    base_acc = (pi * oo).sum(axis=-1)
    prec = np.where(mass > 0.0, numer / np.maximum(mass, 1e-30), base_acc)
    return float(np.mean(prec))


def modal_precision(base: np.ndarray, o: np.ndarray, v: np.ndarray) -> float:
    """Mean over prompts of ``O`` at the argmax base logit inside ``{V=1}``.

    Exact ties (relative tolerance ``1e-9``, the same rule as
    :func:`exact_tau`) are averaged with ``pi_0`` weights. Under ``beta = 0``
    the flow's derivative on a tie is zero, so the tie stays and the terminal
    accuracy is that weighted average, not the label of the lowest index. A
    prompt with nothing accepted contributes its base accuracy.
    """
    logits_b = np.asarray(base, dtype=np.float64)
    oo = np.asarray(o, dtype=np.float64)
    accepted = np.asarray(v, dtype=np.float64) > 0.5
    pi = softmax(logits_b)
    masked = np.where(accepted, logits_b, -np.inf)
    none = ~np.isfinite(masked).any(axis=-1)
    safe = np.where(none[:, None], 0.0, masked)
    maximum = np.max(safe, axis=-1, keepdims=True)
    tolerance = _TIE_RTOL * np.maximum(1.0, np.abs(maximum))
    tied = accepted & (logits_b >= maximum - tolerance)
    weights = pi * tied
    mass = weights.sum(axis=-1)
    numer = (weights * oo).sum(axis=-1)
    chosen = numer / np.maximum(mass, 1e-30)
    base_acc = (pi * oo).sum(axis=-1)
    return float(np.where(none, base_acc, chosen).mean())


def _as_reward(world: World, verifier: np.ndarray | ResampledSpec) -> np.ndarray:
    if isinstance(verifier, ResampledSpec):
        return expected_reward(world, verifier)
    return np.asarray(verifier, dtype=np.float64)


def exact_tau(
    train_world: World,
    v: np.ndarray | ResampledSpec,
    heldout_world: World,
) -> float:
    """Oracle accuracy of the top-velocity held-out output under exact ``g_V``.

    ``g_V = E_x Cov_{pi_0}(phi, V)`` on the training world. On each held-out
    prompt the outputs attaining the maximum of ``s = phi @ g_V`` (ties within
    ``1e-9`` relative tolerance, the same rule as ``tau_hard``) are averaged
    with ``pi_0`` weights.
    """
    reward = _as_reward(train_world, v)
    pi0 = softmax(train_world.base)
    g_v = shared_update(train_world.phi, pi0, reward)
    scores = np.einsum("pkd,d->pk", heldout_world.phi, g_v)
    held_pi = softmax(heldout_world.base)
    oo = np.asarray(heldout_world.O, dtype=np.float64)
    maximum = scores.max(axis=-1, keepdims=True)
    tolerance = _TIE_RTOL * np.maximum(1.0, np.abs(maximum))
    tied = scores >= maximum - tolerance
    weights = held_pi * tied
    mass = weights.sum(axis=-1)
    numer = (weights * oo).sum(axis=-1)
    return float((numer / np.maximum(mass, 1e-30)).mean())


def _audit(
    world: World,
    verifier: np.ndarray | ResampledSpec | None,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Draw ``n_samples`` actions per prompt from ``pi_0``.

    Returns features, prompt ids, oracle labels, and verifier labels (or
    ``None`` when ``verifier`` is ``None``, as on the held-out pool).
    """
    p_count, _k, _d = world.phi.shape
    pi = softmax(world.base)
    cdf = np.cumsum(pi, axis=1)
    cdf[:, -1] = 1.0
    draw = rng.random((p_count, n_samples, 1))
    actions = np.argmax(draw <= cdf[:, None, :], axis=-1)
    prompt = np.repeat(np.arange(p_count), n_samples)
    flat = actions.reshape(-1)
    features = world.phi[prompt, flat]
    labels = world.O[prompt, flat].astype(np.float64)
    if verifier is None:
        return features, prompt, labels, None
    if isinstance(verifier, ResampledSpec):
        correct = labels == 1.0
        coin = rng.random(labels.shape)
        reward = np.where(correct, coin >= verifier.rho1, coin < verifier.rho0).astype(np.float64)
    else:
        reward = np.asarray(verifier, dtype=np.float64)[prompt, flat]
    return features, prompt, labels, reward


def sampled_tau(
    train_world: World,
    v: np.ndarray | ResampledSpec,
    heldout_world: World,
    n_samples: int = 16,
    seed: int = 0,
    reliability_splits: int = 20,
) -> TransferForecast:
    """:func:`vfates.forecast.transfer.transfer_forecast` on ``pi_0`` audit samples.

    Training rows carry the sampled output's features and its ``V`` and ``O``
    labels. Held-out rows carry features and ``O`` only. ``seed`` feeds both
    the audit draws and the split-half reliability estimate.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    train_phi, train_pid, train_o, train_v = _audit(train_world, v, n_samples, rng)
    held_phi, held_pid, held_o, _held_v = _audit(heldout_world, None, n_samples, rng)
    if train_v is None:
        raise RuntimeError("training audit did not return verifier labels")
    return transfer_forecast(
        train_phi,
        train_pid,
        train_v,
        train_o,
        held_phi,
        held_pid,
        held_o,
        reliability_splits=reliability_splits,
        seed=seed,
    )


def observed_heldout_accuracy(result: TrainResult | FlowResult, heldout_world: World) -> float:
    """Held-out oracle accuracy of ``result.theta`` with the tabular head off."""
    pi = softmax(logits(heldout_world, theta=result.theta, tab=None))
    return oracle_accuracy(pi, heldout_world.O)
