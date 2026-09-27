# Pre-registration — Verifier drift under RLVR

Frozen before the real (model-based) run. The mock pilot is a harness check only
and does not count as a confirmatory run. The verdict is computed mechanically in
`vdrift/pilot.py` against the thresholds below.

## Question

When a policy is trained with GRPO against a **fixed, unmodified** rule verifier,
does the verifier's error rate change **within answer-form strata** as the
policy's output-form distribution shifts over training, and does that drift flip
the sign of the group-relative advantage versus a ground-truth oracle — **holding
problem difficulty fixed**?

## Why it is not already answered

- Random, symmetric, resampled verifier noise is tolerated by RLVR up to ~15%
  ([arXiv:2604.07666](https://arxiv.org/abs/2604.07666)). That is a different
  object from systematic, policy-correlated drift.
- Systematic verifier error harms RLVR, but demonstrated with **injected**
  triggers ([arXiv:2605.02909](https://arxiv.org/abs/2605.02909)).
- Real verifier error has been measured on **frozen** rollouts, not over a
  training run, and difficulty was not separated from answer form
  ([arXiv:2609.06386](https://arxiv.org/abs/2609.06386)).

Our object: a real unmodified verifier, natural (not injected) error, measured
across a GRPO trajectory, within answer-form strata, difficulty-controlled, with
advantage-sign flips as the training-bias metric.

## Metrics

- `advantage_sign_flip_rate`: fraction of rollouts whose GRPO advantage sign
  under the verifier reward disagrees with the sign under the oracle reward.
- `fn_rate` / `fp_rate` per answer-form stratum, per checkpoint.
- `form_distribution`: policy answer-form mix per checkpoint (the driver).
- `difficulty_controlled_drift`: sign-flip drift computed **within** each
  difficulty bin.

## Thresholds (DRIFT requires all)

| Gate | Metric | Threshold |
|---|---|---|
| Drift | sign_flip_final − sign_flip_start | ≥ 0.05 |
| Magnitude | sign_flip_final | ≥ 0.10 |
| Control | sign-flip drift within ≥1 difficulty bin | ≥ 0.05 |

Report bootstrap 95% CIs (`metrics.bootstrap_ci`) on the final sign-flip rate
across seeds; a DRIFT claim requires the interval's lower bound above 0.

## Kill / honest negative

If, after the difficulty control, the verifier's stratified error and the
advantage-sign flips are flat over training (consistent with "an imperfect
verifier is good enough"), report the null: a fixed rule verifier's effective
error is stable under GRPO. This is a clean, reportable result (workshop-scale)
and stops the line before large spend.

## Real-run protocol (frozen)

- Models: Qwen2.5-1.5B-Instruct (primary), 3B (replication). 3 seeds each.
- RL: GRPO, group size 8, fixed prompt set across training, ~a few hundred steps.
- Verifier (unmodified): a real rule/numeric math checker. Oracle: independent
  second checker + symbolic (sympy) + a small judged subsample.
- Data: GSM8K + MATH; difficulty from released labels or a fixed proxy.
- No change to thresholds after seeing results. One pre-registered retry
  (more steps or the 3B model) is allowed if training fails to converge.

## Deviations

Logged here with date and rationale.
