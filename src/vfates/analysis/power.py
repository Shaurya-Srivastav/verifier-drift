"""Normal approximation to the crossed-design power model in review R4.

R4's eval noise is a shared latent accuracy ``p ~ Beta(mu * kappa, (1 - mu) * kappa)``
with an independent ``Binomial(n_samples, p)`` evaluation on each arm. Prompt
difficulty cancels in a paired difference, so the variance of one prompt's
difference is ``2 * E[p(1 - p)] / n_samples``. Averaging prompts and seeds
gives the eval-only standard error (R4's table: about 0.028 for the P4 cell
and about 0.0076 for E1 with 3 seeds).

The crossed design also resamples seeds. ``seed_sd`` is the standard deviation
of one arm's seed-level mean accuracy, the quantity P5 measures on its two
oracle seeds. Eval noise already inside that standard deviation is removed
before it is added back as the paired-difference eval term, so a ``seed_sd``
of 0 reproduces R4. Two arms are treated as independent, so the contrast's
training-noise variance is twice one arm's.

The superiority test rejects ``mean <= margin`` at one-sided level ``alpha``.
The minimum detectable difference is the true gap with power ``power`` under
a normal approximation to that test. This is not a Monte Carlo of the
bootstrap; R4's simulated TOST power (0.14 on P4) sits within about 0.02 of
the normal approximation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vfates.analysis.outcomes import normal_cdf, normal_ppf


def expected_bernoulli_variance(mu: float, var_p: float) -> float:
    """``E[p(1 - p)] = mu(1 - mu) - Var(p)`` for a latent accuracy."""
    if not 0.0 < mu < 1.0:
        raise ValueError(f"mu must lie in (0, 1), got {mu}")
    if var_p < 0:
        raise ValueError(f"var_p must be >= 0, got {var_p}")
    value = mu * (1.0 - mu) - var_p
    if value <= 0:
        raise ValueError("var_p leaves no Bernoulli variance")
    return value


def contrast_se(
    *,
    n_prompts: int,
    n_seeds: int,
    n_samples: int = 4,
    mu: float = 0.215,
    var_p: float = 0.037,
    seed_sd: float = 0.0,
) -> float:
    """Standard error of a paired contrast under the crossed design.

    ``seed_sd`` is the SD of one arm's seed-level mean accuracy. Zero recovers
    the prompt-bootstrap SE in R4 (eval noise only).
    """
    if n_prompts < 1 or n_seeds < 1 or n_samples < 1:
        raise ValueError("n_prompts, n_seeds and n_samples must be >= 1")
    if seed_sd < 0:
        raise ValueError(f"seed_sd must be >= 0, got {seed_sd}")
    e_p = expected_bernoulli_variance(mu, var_p)
    var_prompt_diff = 2.0 * e_p / n_samples
    var_eval_arm = e_p / (n_samples * n_prompts)
    var_training = max(seed_sd**2 - var_eval_arm, 0.0)
    var = (2.0 * var_training) / n_seeds + var_prompt_diff / (n_prompts * n_seeds)
    return math.sqrt(var)


def _z_sum(alpha: float, power: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must lie in (0, 1), got {power}")
    return normal_ppf(1.0 - alpha) + normal_ppf(power)


def superiority_power(true_gap: float, se: float, *, margin: float, alpha: float) -> float:
    """Power of a one-sided normal test of ``mean <= margin``."""
    if se <= 0:
        raise ValueError(f"se must be positive, got {se}")
    z_alpha = normal_ppf(1.0 - alpha)
    return normal_cdf((true_gap - margin) / se - z_alpha)


def tost_power(true_gap: float, se: float, *, margin: float, alpha: float) -> float:
    """Power of TOST at level ``alpha`` inside ``+/- margin`` (normal approximation).

    The central ``1 - 2 alpha`` interval must lie inside the margin, so each
    side uses the one-sided critical value ``z_{1 - alpha}``.
    """
    if se <= 0:
        raise ValueError(f"se must be positive, got {se}")
    z_alpha = normal_ppf(1.0 - alpha)
    half = z_alpha * se
    if margin <= half:
        return 0.0
    lo = -margin + half
    hi = margin - half
    return normal_cdf((hi - true_gap) / se) - normal_cdf((lo - true_gap) / se)


def minimum_detectable_difference(
    seed_sd: float,
    *,
    n_seeds: int,
    n_prompts: int = 384,
    n_samples: int = 4,
    mu: float = 0.215,
    var_p: float = 0.037,
    margin: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, Any]:
    """True gap with the requested power for superiority at ``margin``.

    Returns the crossed-design SE, the minimum detectable difference, and the
    normal-approximation power at a true gap equal to ``margin`` (about
    ``alpha``) and at 0.05. P5 calls this for 3 and 5 seeds at margin 0.05.
    """
    se = contrast_se(
        n_prompts=n_prompts,
        n_seeds=n_seeds,
        n_samples=n_samples,
        mu=mu,
        var_p=var_p,
        seed_sd=seed_sd,
    )
    mdd = margin + _z_sum(alpha, power) * se
    return {
        "se": se,
        "mdd": mdd,
        "margin": margin,
        "alpha": alpha,
        "power": power,
        "n_seeds": n_seeds,
        "n_prompts": n_prompts,
        "n_samples": n_samples,
        "mu": mu,
        "var_p": var_p,
        "seed_sd": seed_sd,
        "power_at_margin": superiority_power(margin, se, margin=margin, alpha=alpha),
        "power_at_0.05": superiority_power(0.05, se, margin=margin, alpha=alpha),
        "tost_power_at_0": tost_power(0.0, se, margin=margin, alpha=alpha),
        "rule": (
            "Normal approximation to the crossed bootstrap. "
            "seed_sd is the SD of one arm's seed-level mean accuracy. "
            "Superiority rejects mean <= margin at one-sided alpha. "
            "A seed_sd of 0 reproduces R4's eval-only SE."
        ),
    }


def token_projection(spec: dict[str, Any]) -> dict[str, Any]:
    """H100 hours from a token-rate model, with eval tokens and the audit included.

    This is the placeholder E1 cost. P5 replaces ``hours_per_arm_seed`` with a
    wall-clock measurement before E1 is launched. Early baselines count as
    ``early_steps / steps`` of a full arm-seed.
    """
    length = float(spec["mean_completion_len"])
    steps = int(spec["steps"])
    group = int(spec["prompts_per_step"]) * int(spec["group_size"])
    full_runs = int(spec["full_runs"])
    early_runs = int(spec.get("early_runs", 0))
    early_steps = int(spec.get("early_steps", 0))
    train_tokens = steps * group * length * full_runs
    if early_runs and early_steps:
        train_tokens += early_steps * group * length * early_runs
    eval_every = int(spec["eval_every"])
    eval_prompts = int(spec["eval_prompts"]) + int(spec.get("eval_train_prompts", 0))
    eval_samples = int(spec["eval_samples"])
    full_evals = (steps // eval_every + 1) * full_runs
    early_evals = (early_steps // eval_every + 1) * early_runs if early_runs else 0
    eval_tokens = (full_evals + early_evals) * eval_prompts * eval_samples * length
    audit_tokens = int(spec["audit_prompts"]) * int(spec["audit_samples"]) * length
    gen_tokens = train_tokens + eval_tokens + audit_tokens
    gen_rate = float(spec["h100_gen_tok_s"])
    train_rate = float(spec["h100_train_tok_s"])
    hours = (gen_tokens / gen_rate + train_tokens / train_rate) / 3600.0
    hours *= float(spec.get("overhead", 1.0))
    usd = hours * float(spec["usd_per_h100_hour"])
    return {
        "train_tokens": train_tokens,
        "eval_tokens": eval_tokens,
        "audit_tokens": audit_tokens,
        "generated_tokens": gen_tokens,
        "h100_hours": hours,
        "usd": usd,
        "eval_tokens_included": True,
        "source": "token_rates",
        "rule": "Placeholder token-rate model. Replace with P5's measured hours per arm-seed.",
    }


def pooled_h3_power(
    n_pairs: int,
    true_rho: float,
    *,
    n_cells: int = 1,
    n_sim: int = 200,
    n_perm: int = 199,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulation power of the within-cell Spearman permutation test.

    Pairs are split across ``n_cells`` as evenly as possible. Each cell is a
    Gaussian copula with Spearman correlation ``true_rho`` (Pearson
    ``2 sin(pi rho / 6)``). A simulation rejects when the within-cell
    permutation p-value is below ``alpha``. The bootstrap leg and the
    minimum-pair gate are not part of this power.
    """
    from vfates.analysis.pooled import within_cell_spearman_p

    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs < 2:
        raise ValueError("n_pairs must be an integer >= 2")
    if isinstance(n_cells, bool) or not isinstance(n_cells, int) or not 1 <= n_cells <= n_pairs:
        raise ValueError("n_cells must be an integer from 1 to n_pairs")
    if isinstance(n_sim, bool) or not isinstance(n_sim, int) or n_sim < 1:
        raise ValueError("n_sim must be an integer >= 1")
    if isinstance(n_perm, bool) or not isinstance(n_perm, int) or n_perm < 1:
        raise ValueError("n_perm must be an integer >= 1")
    if not -1.0 <= float(true_rho) <= 1.0:
        raise ValueError("true_rho must lie in [-1, 1]")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    sizes = [n_pairs // n_cells] * n_cells
    for i in range(n_pairs % n_cells):
        sizes[i] += 1
    rho = float(true_rho)
    if rho >= 1.0:
        pearson = 1.0
    elif rho <= -1.0:
        pearson = -1.0
    else:
        pearson = 2.0 * math.sin(rho * math.pi / 6.0)
    residual = math.sqrt(max(0.0, 1.0 - pearson * pearson))
    rng = np.random.default_rng(seed)
    rejects = 0
    for _ in range(n_sim):
        dtaus: list[np.ndarray] = []
        daccs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for c, n in enumerate(sizes):
            z = rng.standard_normal((n, 2))
            z[:, 1] = pearson * z[:, 0] + residual * z[:, 1]
            dtaus.append(z[:, 0])
            daccs.append(z[:, 1])
            labels.append(np.full(n, c, dtype=int))
        result = within_cell_spearman_p(
            np.concatenate(dtaus),
            np.concatenate(daccs),
            np.concatenate(labels),
            n_perm=n_perm,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        p_value = result.get("p")
        if isinstance(p_value, float) and math.isfinite(p_value) and p_value < alpha:
            rejects += 1
    return {
        "power": rejects / n_sim,
        "n_rejects": rejects,
        "n_sim": n_sim,
        "n_perm": n_perm,
        "n_pairs": n_pairs,
        "n_cells": n_cells,
        "cell_sizes": sizes,
        "true_rho": rho,
        "alpha": alpha,
        "seed": seed,
        "rule": (
            "Power of the within-cell permutation test of Spearman rho, "
            "simulated from a Gaussian copula. Reject when p < alpha. "
            "The crossed-bootstrap percentile and the minimum pair count "
            "are not part of this power."
        ),
    }
