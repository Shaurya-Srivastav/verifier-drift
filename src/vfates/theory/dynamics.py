"""Population and sampled policy-gradient dynamics on a :class:`World`.

The objective on each prompt is ``E_pi[V] - beta KL(pi || pi_0)`` with
``pi_0 = softmax(base)`` held fixed. Euclidean (vanilla) softmax policy
gradient updates the logits by ``eta * pi * (advantage)``; that flow is
mode-seeking inside a level set of ``V``. The replicator / natural gradient
stays on the exponential tilt ``pi_0 exp(t V)``, whose terminal accuracy is
mean precision. Any ``beta > 0`` equilibrium is the tilt at temperature
``t = 1/beta``.

Shared parameters read ``logits = base + phi @ theta + tab``. Held-out prompts
are scored with ``theta`` only: ``tab`` is prompt-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vfates.theory.worlds import ResampledSpec, World


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax on the last axis, stable for large logits."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    out = np.exp(z)
    out /= out.sum(axis=-1, keepdims=True)
    return out


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def logits(
    world: World,
    theta: np.ndarray | None = None,
    tab: np.ndarray | None = None,
) -> np.ndarray:
    """``base + phi @ theta + tab``. Missing ``theta`` or ``tab`` contribute nothing."""
    out = np.array(world.base, dtype=np.float64, copy=True)
    if theta is not None:
        out += np.einsum("pkd,d->pk", world.phi, np.asarray(theta, dtype=np.float64))
    if tab is not None:
        out += np.asarray(tab, dtype=np.float64)
    return out


def tilt(world: World, v: np.ndarray, t: float) -> np.ndarray:
    """Exact exponential tilt ``pi_0(y|x) exp(t V(x,y))``, as a softmax."""
    return softmax(world.base + float(t) * np.asarray(v, dtype=np.float64))


def expected_reward(world: World, verifier: np.ndarray | ResampledSpec) -> np.ndarray:
    """Per-output training target. A :class:`ResampledSpec` is replaced by ``E[V|O]``."""
    if isinstance(verifier, ResampledSpec):
        o = np.asarray(world.O, dtype=np.float64)
        return (1.0 - verifier.rho1) * o + verifier.rho0 * (1.0 - o)
    reward = np.asarray(verifier, dtype=np.float64)
    if reward.shape != world.O.shape:
        raise ValueError(f"verifier shape {reward.shape} != O shape {world.O.shape}")
    return reward


def oracle_accuracy(pi: np.ndarray, o: np.ndarray) -> float:
    """Mean over prompts of ``E_pi[O]``."""
    return float((np.asarray(pi) * np.asarray(o, dtype=np.float64)).sum(axis=-1).mean())


def cov_phi_reward(phi: np.ndarray, pi: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """Per-prompt ``Cov_pi(phi, reward)``, shape ``[P, d]``.

    The reward is centred under ``pi``. Because those weights sum to zero,
    this equals ``E_pi[(phi - E phi) reward]``.
    """
    mean_r = (pi * reward).sum(axis=-1, keepdims=True)
    centered = reward - mean_r
    return (pi[..., None] * centered[..., None] * phi).sum(axis=1)


def shared_update(phi: np.ndarray, pi: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """``g = E_x Cov_pi(phi, reward)``, prompts weighted equally."""
    return cov_phi_reward(phi, pi, reward).mean(axis=0)


@dataclass
class FlowResult:
    """Terminal state of :func:`population_flow`.

    ``history`` holds ``step``, ``train_acc`` and, when a held-out world was
    passed, ``heldout_acc`` after every Euler step including step 0.
    """

    pi: np.ndarray
    theta: np.ndarray
    tab: np.ndarray
    history: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Terminal parameters of :func:`sampled_grpo`.

    ``history`` holds ``step``, ``train_acc`` and optionally ``heldout_acc``
    at step 0, every ``eval_every`` steps, and the final step.
    """

    theta: np.ndarray
    tab: np.ndarray
    history: dict[str, Any] = field(default_factory=dict)


def _scores(world: World, theta: np.ndarray, tab: np.ndarray | None) -> float:
    return oracle_accuracy(softmax(logits(world, theta, tab)), world.O)


def population_flow(
    world: World,
    V_or_spec: np.ndarray | ResampledSpec,
    eta: float,
    steps: int,
    param: str = "tabular",
    beta: float = 0.0,
    dt: float = 1.0,
    heldout: World | None = None,
) -> FlowResult:
    """Euler discretisation of the Euclidean expected policy gradient.

    ``param='tabular'`` updates prompt-specific logits, ``'shared'`` updates
    ``theta``, ``'both'`` updates each from the same pre-step policy. The
    logit step is ``eta * dt * pi * ((V - beta log(pi/pi_0)) centred)``.
    """
    if param not in ("tabular", "shared", "both"):
        raise ValueError("param must be 'tabular', 'shared' or 'both'")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    reward = expected_reward(world, V_or_spec)
    p, _k, d = world.phi.shape
    theta = np.zeros(d, dtype=np.float64)
    tab = np.zeros((p, world.O.shape[1]), dtype=np.float64)
    log_pi0 = log_softmax(world.base)
    use_tab = param in ("tabular", "both")
    use_shared = param in ("shared", "both")
    history: dict[str, list[Any]] = {"step": [], "train_acc": []}
    if heldout is not None:
        history["heldout_acc"] = []
    step_scale = float(eta) * float(dt)

    def current_logits() -> np.ndarray:
        built = logits(world, theta if use_shared else None, tab if use_tab else None)
        return built

    def record(step: int, pi: np.ndarray) -> None:
        history["step"].append(int(step))
        history["train_acc"].append(oracle_accuracy(pi, world.O))
        if heldout is not None:
            history["heldout_acc"].append(_scores(heldout, theta, None))

    for step in range(steps + 1):
        pi = softmax(current_logits())
        record(step, pi)
        if step == steps:
            break
        mean_v = (pi * reward).sum(axis=-1, keepdims=True)
        grad = pi * (reward - mean_v)
        if beta != 0.0:
            log_ratio = log_softmax(current_logits()) - log_pi0
            centered = log_ratio - (pi * log_ratio).sum(axis=-1, keepdims=True)
            grad = grad - float(beta) * pi * centered
        if use_tab:
            tab = tab + step_scale * grad
        if use_shared:
            g = np.einsum("pk,pkd->d", grad, world.phi) / p
            theta = theta + step_scale * g
    return FlowResult(pi=pi, theta=theta, tab=tab, history=history)


def _advantages(rewards: np.ndarray, kind: str) -> np.ndarray:
    """Group advantages. ``std`` uses the population standard deviation.

    A group with zero reward variance gets advantage 0. The ``1e-8`` floor is
    only applied when the standard deviation is positive, matching the loop
    that produced the validated runs (a zero-variance group must not be
    rescaled by ``1e-8``).
    """
    mean = rewards.mean(axis=1, keepdims=True)
    if kind == "mean":
        return rewards - mean
    if kind != "std":
        raise ValueError("adv must be 'std' or 'mean'")
    scale = rewards.std(axis=1, keepdims=True)
    out = np.zeros_like(rewards)
    ok = scale[:, 0] > 0.0
    out[ok] = (rewards[ok] - mean[ok]) / (scale[ok] + 1e-8)
    return out


def _sample_actions(pi: np.ndarray, g: int, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(pi, axis=1)
    cdf[:, -1] = 1.0
    draw = rng.random((pi.shape[0], g, 1))
    return np.argmax(draw <= cdf[:, None, :], axis=-1)


def _rewards_for(
    world: World,
    verifier: np.ndarray | ResampledSpec,
    idx: np.ndarray,
    actions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(verifier, ResampledSpec):
        correct = world.O[idx[:, None], actions] == 1
        draw = rng.random(actions.shape)
        return np.where(correct, draw >= verifier.rho1, draw < verifier.rho0).astype(np.float64)
    return np.asarray(verifier, dtype=np.float64)[idx[:, None], actions]


def sampled_grpo(
    world: World,
    verifier: np.ndarray | ResampledSpec,
    steps: int = 800,
    batch: int = 64,
    G: int = 8,
    lr_shared: float = 1.0,
    lr_tab: float = 2.0,
    adv: str = "std",
    beta: float = 0.0,
    seed: int = 0,
    heldout: World | None = None,
    eval_every: int = 50,
) -> TrainResult:
    """One-pass sampled GRPO on tabular logits plus a shared linear head.

    Each step draws ``batch`` prompts without replacement and ``G`` actions
    from the current policy. Rewards come from ``verifier``; a
    :class:`ResampledSpec` redraws class-conditional flips per sample.
    The per-prompt logit gradient is the group mean of
    ``A * (onehot - pi)``, optionally minus the Euclidean gradient of
    ``beta KL(pi || pi_0)``. ``tab`` takes ``lr_tab`` times that gradient and
    ``theta`` takes ``lr_shared`` times the mean over the batch of
    ``g @ phi``.
    """
    if steps < 0 or batch < 1 or G < 1 or eval_every < 1:
        raise ValueError("steps >= 0, batch >= 1, G >= 1, eval_every >= 1 are required")
    if adv not in ("std", "mean"):
        raise ValueError("adv must be 'std' or 'mean'")
    p_count, k, d = world.phi.shape
    if batch > p_count:
        raise ValueError(f"batch {batch} exceeds P {p_count}")
    theta = np.zeros(d, dtype=np.float64)
    tab = np.zeros((p_count, k), dtype=np.float64)
    log_pi0 = log_softmax(world.base) if beta != 0.0 else None
    rng = np.random.default_rng(seed)
    history: dict[str, list[Any]] = {"step": [], "train_acc": []}
    if heldout is not None:
        history["heldout_acc"] = []
    batch_index = np.arange(batch)[:, None]

    def record(step: int) -> None:
        history["step"].append(int(step))
        history["train_acc"].append(_scores(world, theta, tab))
        if heldout is not None:
            history["heldout_acc"].append(_scores(heldout, theta, None))

    record(0)
    for step in range(steps):
        idx = rng.choice(p_count, size=batch, replace=False)
        current = world.base[idx] + tab[idx]
        current = current + np.einsum("bkd,d->bk", world.phi[idx], theta)
        pi = softmax(current)
        actions = _sample_actions(pi, G, rng)
        rewards = _rewards_for(world, verifier, idx, actions, rng)
        advantages = _advantages(rewards, adv)
        grad = np.zeros((batch, k), dtype=np.float64)
        np.add.at(grad, (batch_index, actions), advantages)
        grad /= G
        grad -= advantages.mean(axis=1, keepdims=True) * pi
        if beta != 0.0 and log_pi0 is not None:
            log_ratio = log_softmax(current) - log_pi0[idx]
            centered = log_ratio - (pi * log_ratio).sum(axis=-1, keepdims=True)
            grad = grad - float(beta) * pi * centered
        tab[idx] += lr_tab * grad
        theta += lr_shared * np.einsum("bk,bkd->d", grad, world.phi[idx]) / batch
        done = step + 1
        if done % eval_every == 0 or done == steps:
            record(done)
    return TrainResult(theta=theta, tab=tab, history=history)
