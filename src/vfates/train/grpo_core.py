"""Numpy reference for the GRPO objective shared by the MLX and TRL trainers.

TRL 0.25.1 (``scale_rewards="group"``, the default) computes advantages as

.. code-block:: text

    mean_g = mean(r)
    std_g  = std(r, unbiased=True)          # torch.std, divide by G-1
    A      = (r - mean_g) / (std_g + 1e-4)
    A      = 0  when std_g isclose 0        # torch.isclose atol=1e-8, rtol=1e-5

``mode="mean"`` is TRL ``scale_rewards="none"`` (Dr.GRPO): center only, no std scaling.
Zero-variance groups are exactly 0 in both modes.

The clipped surrogate matches ``GRPOTrainer._compute_loss`` at token level:

.. code-block:: text

    ratio = exp(new - old)                  # old = new when old is omitted
    loss  = -min(ratio * A, clip(ratio, 1-eps_low, 1+eps_high) * A)

When ``old`` is omitted (or equal to ``new``) every ratio is 1, so the value is ``-A``
on each token. That is the same point as plain REINFORCE: the gradient of
``-exp(logp - stopgrad(logp)) * A`` w.r.t. ``logp`` equals ``-A``, i.e. the gradient of
``-A * log π``. TRL skips the old-logprob forward in this case and detaches the current
log-probabilities, which is the same construction. ``num_iterations=1`` therefore does
not clip; clipping is active only when ``old`` differs from ``new``.

Aggregation (TRL ``loss_type``):

* ``seq_mean`` → ``"grpo"``: mean over sequences of (token-mean loss).
* ``token_mean`` → ``"dapo"``: mean over tokens in the whole step.
* ``dr_constant`` → ``"dr_grpo"``: sum of token losses / (N * max_len).

KL is the Schulman k3 estimator used by TRL, added per token and aggregated with the
same weights: ``exp(ref - new) - (ref - new) - 1``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TRL_LOSS_TYPE = {
    "seq_mean": "grpo",
    "token_mean": "dapo",
    "dr_constant": "dr_grpo",
}
TRL_SCALE_REWARDS = {"std": "group", "mean": "none"}

_AGGS = tuple(TRL_LOSS_TYPE)
_MODES = tuple(TRL_SCALE_REWARDS)
_STDS = ("unbiased", "population")


def group_advantages(
    rewards: np.ndarray,
    mode: str = "std",
    eps: float = 1e-4,
    std: str = "unbiased",
) -> tuple[np.ndarray, np.ndarray]:
    """Group-relative advantages.

    ``rewards`` has shape ``[B, G]`` (prompts, group size). Returns ``(A, informative)``
    where ``A`` has the same shape and ``informative`` is a length-``B`` bool mask,
    True when the group's standard deviation is not ``isclose`` to 0 (a constant
    reward gives a zero advantage and is not a learning signal).

    ``std="unbiased"`` matches ``torch.std(..., unbiased=True)`` (TRL). ``"population"``
    divides by ``G`` instead of ``G-1``.
    """
    if mode not in _MODES:
        raise ValueError(f"adv mode must be one of {_MODES}, got {mode!r}")
    if std not in _STDS:
        raise ValueError(f"std must be one of {_STDS}, got {std!r}")
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"rewards must have shape [B, G], got {r.shape}")
    if not np.isfinite(r).all():
        raise ValueError("rewards must be finite")
    _b, g = r.shape
    mean = r.mean(axis=1, keepdims=True)
    centered = r - mean
    ddof = 1 if std == "unbiased" else 0
    if g - ddof <= 0:
        scale = np.zeros((r.shape[0], 1), dtype=np.float64)
        zero = np.ones((r.shape[0], 1), dtype=bool)
    else:
        var = np.sum(centered * centered, axis=1, keepdims=True) / (g - ddof)
        scale = np.sqrt(var)
        # torch.isclose / np.isclose defaults: rtol=1e-05, atol=1e-08.
        zero = np.isclose(scale, 0.0)
    if mode == "std":
        adv = centered / (scale + eps)
        adv = np.where(zero, 0.0, adv)
    else:
        adv = centered
    informative = ~zero.reshape(-1)
    return adv, informative


def loss_weights(lengths: Sequence[float] | np.ndarray, agg: str, max_len: int) -> np.ndarray:
    """Per-sequence weights on the *sum* of token losses.

    ``sum_i weight_i * sum_t loss_{i,t}`` equals the TRL reduction for one optimizer
    step whose completion lengths are ``lengths`` (padding excluded). ``max_len`` is
    ``max_completion_length`` and is used only by ``dr_constant``.
    """
    if agg not in _AGGS:
        raise ValueError(f"agg must be one of {_AGGS}, got {agg!r}")
    lens = np.asarray(
        list(lengths) if not isinstance(lengths, np.ndarray) else lengths, dtype=np.float64
    )
    if lens.ndim != 1:
        raise ValueError(f"lengths must be rank-1, got shape {lens.shape}")
    n = int(lens.shape[0])
    if n == 0:
        return lens
    if np.any(lens < 0):
        raise ValueError("lengths must be non-negative")
    if agg == "seq_mean":
        # TRL: mean_i( sum_t loss / max(L_i, 1) ). Empty completions contribute 0.
        return 1.0 / (np.maximum(lens, 1.0) * n)
    if agg == "token_mean":
        total = float(lens.sum())
        if total <= 0.0:
            total = 1.0
        return np.full(n, 1.0 / total, dtype=np.float64)
    if int(max_len) < 1:
        raise ValueError(f"max_len must be >= 1 for dr_constant, got {max_len}")
    return np.full(n, 1.0 / (n * float(max_len)), dtype=np.float64)


def kl_k3(new: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Schulman k3 KL ``KL(π || π_ref) ≈ exp(ref-new) - (ref-new) - 1``, per token.

    ``new`` and ``ref`` are log-probabilities. The estimator is non-negative and zero
    iff the two log-probabilities are equal.
    """
    new_a = np.asarray(new, dtype=np.float64)
    ref_a = np.asarray(ref, dtype=np.float64)
    delta = ref_a - new_a
    return np.exp(delta) - delta - 1.0


def pg_loss_reference(
    token_logps: Sequence[np.ndarray],
    advantages: np.ndarray,
    agg: str,
    max_len: int,
    old_token_logps: Sequence[np.ndarray] | None = None,
    ref_token_logps: Sequence[np.ndarray] | None = None,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
) -> float:
    """Clipped GRPO surrogate plus optional k3 KL, reduced like TRL.

    ``token_logps[i]`` is the current-policy log-probability of each completion token.
    Sequence-level advantages are broadcast across tokens. See the module docstring
    for the TRL correspondence. Returns a Python float (the minimized loss).
    """
    adv = np.asarray(advantages, dtype=np.float64)
    n = len(token_logps)
    if adv.shape != (n,):
        raise ValueError(f"advantages must have shape ({n},), got {adv.shape}")
    if old_token_logps is not None and len(old_token_logps) != n:
        raise ValueError("old_token_logps must have one array per sequence")
    if ref_token_logps is not None and len(ref_token_logps) != n:
        raise ValueError("ref_token_logps must have one array per sequence")
    if eps_low < 0 or eps_high < 0:
        raise ValueError("clip epsilons must be non-negative")
    lengths = np.array([len(np.asarray(lp)) for lp in token_logps], dtype=np.float64)
    weights = loss_weights(lengths, agg, max_len)
    total = 0.0
    for i, raw in enumerate(token_logps):
        new = np.asarray(raw, dtype=np.float64)
        if old_token_logps is None:
            ratio = np.ones_like(new)
        else:
            old = np.asarray(old_token_logps[i], dtype=np.float64)
            if old.shape != new.shape:
                raise ValueError(f"old log-probs shape {old.shape} != {new.shape} at sequence {i}")
            ratio = np.exp(new - old)
        clipped = np.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
        per = -np.minimum(ratio * adv[i], clipped * adv[i])
        if ref_token_logps is not None and beta != 0.0:
            ref = np.asarray(ref_token_logps[i], dtype=np.float64)
            if ref.shape != new.shape:
                raise ValueError(f"ref log-probs shape {ref.shape} != {new.shape} at sequence {i}")
            per = per + float(beta) * kl_k3(new, ref)
        total += float(weights[i]) * float(per.sum())
    return float(total)
