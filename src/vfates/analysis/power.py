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


def project_measured_cost(
    *,
    train_seconds_per_arm_seed: float,
    reference_steps: int,
    spec: dict[str, Any],
    eval_seconds_per_arm_seed: float | None = None,
    gen_tok_s: float | None = None,
    mean_completion_len: float | None = None,
    audit_seconds: float | None = None,
    audit_completions: int | None = None,
) -> dict[str, Any]:
    """Project E1 hours from one measured arm-seed, with eval and audit included.

    Training seconds scale with optimiser steps, including early baselines as
    ``early_steps / reference`` of an arm-seed. Measured eval seconds, when
    present, scale with the number of evals. Otherwise eval tokens (held-out
    and train prompts at every eval) are priced at ``gen_tok_s``.

    The audit is not inside the arm-seed wall clock. It is priced, in order, from
    a measured audit wall clock scaled by completions, from the measured
    generation rate, or from the spec's placeholder ``h100_gen_tok_s`` and
    ``mean_completion_len``. ``audit_source`` names the one used. With none of
    them the projection raises rather than leave the audit out.
    """
    if reference_steps < 1:
        raise ValueError("reference_steps must be >= 1")
    if train_seconds_per_arm_seed < 0:
        raise ValueError("train_seconds_per_arm_seed must be >= 0")
    e1_steps = int(spec["steps"])
    full_runs = int(spec["full_runs"])
    early_runs = int(spec.get("early_runs", 0))
    early_steps = int(spec.get("early_steps", 0))
    train_seconds = (
        train_seconds_per_arm_seed
        / reference_steps
        * (full_runs * e1_steps + early_runs * early_steps)
    )
    eval_every = int(spec["eval_every"])

    def n_evals(steps: int, runs: int) -> int:
        if runs <= 0 or steps <= 0:
            return 0
        return (steps // eval_every + 1) * runs

    e1_evals = n_evals(e1_steps, full_runs) + n_evals(early_steps, early_runs)
    ref_evals = n_evals(reference_steps, 1)
    if eval_seconds_per_arm_seed is not None:
        if ref_evals < 1:
            raise ValueError("reference run has no evals to scale")
        eval_seconds = eval_seconds_per_arm_seed / ref_evals * e1_evals
        eval_source = "measured_eval_seconds"
    else:
        eval_seconds = _token_seconds(
            e1_evals
            * (int(spec["eval_prompts"]) + int(spec.get("eval_train_prompts", 0)))
            * int(spec["eval_samples"]),
            gen_tok_s,
            mean_completion_len,
        )
        eval_source = "eval_tokens"
    e1_completions = int(spec["audit_prompts"]) * int(spec["audit_samples"])
    if audit_seconds is not None:
        if audit_completions is None or audit_completions < 1 or audit_seconds < 0:
            raise ValueError("a measured audit clock needs audit_completions >= 1")
        audit_total = audit_seconds / audit_completions * e1_completions
        audit_source = "measured_audit_wall_clock"
    elif gen_tok_s is not None and mean_completion_len is not None:
        audit_total = _token_seconds(e1_completions, gen_tok_s, mean_completion_len)
        audit_source = "measured_gen_rate"
    elif "h100_gen_tok_s" in spec and "mean_completion_len" in spec:
        audit_total = _token_seconds(
            e1_completions, float(spec["h100_gen_tok_s"]), float(spec["mean_completion_len"])
        )
        audit_source = "placeholder_gen_rate"
    else:
        raise ValueError(
            "the audit cannot be priced: no audit clock, generation rate or placeholder rate"
        )
    total_seconds = train_seconds + eval_seconds + audit_total
    hours = total_seconds / 3600.0
    return {
        "train_hours": train_seconds / 3600.0,
        "eval_hours": eval_seconds / 3600.0,
        "audit_hours": audit_total / 3600.0,
        "h100_hours": hours,
        "usd": hours * float(spec["usd_per_h100_hour"]),
        "eval_tokens_included": True,
        "eval_source": eval_source,
        "audit_source": audit_source,
        "source": "measured_wall_clock",
        "rule": (
            "Training wall clock scaled by steps. Eval tokens are included, "
            "from measured eval seconds or from the generation rate. The audit "
            "is included, from its measured clock scaled by completions, the "
            "measured generation rate, or the placeholder rate (audit_source)."
        ),
    }


def _token_seconds(n_completions: int, gen_tok_s: float | None, length: float | None) -> float:
    if gen_tok_s is None or length is None:
        raise ValueError("token pricing needs gen_tok_s and mean_completion_len")
    if gen_tok_s <= 0 or length < 0:
        raise ValueError("gen_tok_s must be positive and length must be >= 0")
    return n_completions * length / gen_tok_s
