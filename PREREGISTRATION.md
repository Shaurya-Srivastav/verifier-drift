# Preregistration: Same Rate, Different Fates

This document fixes the hypotheses, endpoint, estimator, design and analysis of the confirmatory experiments before any confirmatory training step. The pilots in `pilots/` are exploratory: they chose the configuration and may change the design (their decision log is in `pilots/README.md`); they do not count toward any confirmatory test. Changes after this document is frozen go in [Deviations](#deviations) with a date and reason, and the paper reports them.

**Frozen** at the commit that adds `runs/e1/audit/lock.json` for E1. Each experiment's step-0 forecasts are locked (`vfates.lock/1`: SHA-256 over the canonical forecasts and the full audit configuration) before its first training step, and the lock hash is reported in the paper.

## Question

For a fixed (non-adaptive) verifier used as the RLVR reward, is the step-0 error rate enough to predict what training does to held-out accuracy?

The theory (`docs/theory.md`) says no. The verifier-induced update splits as `g_V = g_O - g_FN + g_FP`. Resampled noise only rescales the clean update (`E[g_V] = J g_O`). A fixed error whose indicator is carried by a behaviour shared across prompts adds a component that transfers to unseen prompts; a prompt-specific error does not. The step-0 statistic `tau_pool` (the oracle accuracy of the held-out outputs the step-0 update favours most) forecasts which case holds.

## Hypotheses

Confirmatory (E1, tested at family-wise alpha = 0.05):

- **H1 (same rate, different fates).** At matched step-0 (FPR, FNR), each shared-behaviour arm ends with lower final held-out oracle accuracy than its resampled control: `noise_fp - fp_tolerance`, `noise_fp - fp_trigger` and `noise_fn - fn_behavior` are each positive.
- **H2 (prompt-specific errors stay local).** `noise_fp - fp_hash` and `noise_fn - fn_prompt_subset` are each equivalent to 0 within a margin of 0.05 (two one-sided tests).
- **H3 (the screen forecasts the fate).** Across the 12 E1 arms, the Spearman correlation between locked `tau_pool` and final held-out accuracy is positive, and exceeds the correlation achieved by step-0 Youden J.

Secondary (E1, reported with intervals, no family-wise control):

- **H4 (mechanism).** Shared errors are linearly decodable from step-0 features on unseen prompts (prompt-grouped AUC > 0.65); prompt-specific errors are not (AUC ≤ 0.60).
- **H5 (behaviour).** Under its own reward, a shared FP behaviour becomes more frequent among wrong held-out outputs than under the control; a shared FN behaviour (a conformance tax on a correct behaviour) becomes less frequent.
- **H6 (the forecast is calibrated).** Mean absolute error between locked `tau_pool` and final held-out accuracy across arms, and the Spearman correlations of mean precision and modal precision, are reported for comparison.

Replication and extension (E2 to E4) are described below with their own primary contrasts.

## Primary endpoint

**Final held-out oracle accuracy**: the mean over the 384 held-out prompts of each prompt's mean oracle bit over 4 samples at temperature 1.0 and top-p 1.0, from the policy after the last training step (step 300), evaluated with `vfates.train.evaluate.evaluate_policy` (MLX) or `vfates.train.checkpoint_eval.evaluate_checkpoints` (TRL adapters). The held-out pool is disjoint from the training pool (dataset seed + 10,000,000) and is never used for calibration, training or any design decision.

Secondary endpoints: held-out accuracy at every 25 steps (the curve), its peak, train-pool oracle accuracy, on-policy verifier rates, and rule-behaviour monitor rates.

## Primary estimator

`tau_pool` at `q = 1/32` (`vfates.forecast.transfer.PRIMARY_Q`, fixed by P0 before any language-model run):

- **Features:** LoRA-B gradients of the completion log-likelihood at initialisation (rank 16, alpha 32, all seven projections of every layer, `lora_seed = 0`), CountSketch to 8,192 dimensions (`sketch_seed = 0`), float32.
- **Direction:** `g_V`, the prompt-centred covariance of features with the arm's verdicts on the train-pool audit (16 samples per prompt).
- **Velocities:** `s = <phi, g_V>` on the held-out audit (16 samples per prompt), centred within each prompt.
- **Statistic:** the oracle accuracy of the pooled top `q` fraction of held-out velocities (ties shared).

Reported alongside, not used in any hypothesis test: `tau_hard`, `tau_rel`, `tau_soft`, the `tau_pool` curve over `q` in {1/64, 1/32, 1/16, 1/8}, `tau_pool` from pooled final-layer hidden states, step-0 J, mean precision, modal precision, and the split-half reliability of `g_V`.

## Design

### Arms (`configs/arms/chain_sum.yaml`)

| Arm | Error | Structure |
| --- | --- | --- |
| `oracle` | none | clean reward |
| `fp_hash` | FP 0.15 | prompt-specific: accepts a wrong answer when a hash of (prompt, answer) is below a threshold |
| `fp_tolerance` | FP 0.15 | shared: accepts a wrong answer within a fitted absolute tolerance of the gold value |
| `fp_trigger` | FP 0.15 | shared: accepts a wrong answer containing a fitted n-gram that is not enriched for correct answers |
| `noise_fp` | FP matched to `fp_tolerance` | resampled per completion, on answered wrong outputs (the support of the answer-based FP rules) |
| `fn_behavior` | FN 0.30 | shared: rejects a correct answer containing a fitted n-gram that predicts correctness |
| `fn_prompt_subset` | FN 0.30 | prompt-specific: rejects every correct answer on a hashed subset of prompts |
| `noise_fn` | FN matched to `fn_behavior` | resampled per completion |
| `fn_form` | FN at its achievable rate | shared: rejects a correct answer in a non-canonical surface form (capped: the policy rarely writes one) |
| `noise_fn_form` | FN matched to `fn_form` | resampled per completion |
| `combo_fnform_fphash` | FN of `fn_form`, FP of `fp_hash` | both rules |
| `natural_string_match` | natural | exact string match against the gold answer |

Every imperfect arm is fitted on the base model's step-0 samples of the train pool only (`vfates.verifiers.calibrate.calibrate_arms`), to FPR 0.15 and FNR 0.30. Resampled controls copy their matched arm's achieved pooled rates. Salts and candidate sets are fixed in the config before sampling.

### E1: decimal chain sum (`configs/experiments/e1_chain_sum.yaml`)

- **Task:** Reasoning Gym `decimal_chain_sum` with the Egashira et al. configuration (3 to 6 terms, 3 to 6 digits, 3 to 6 decimal places). The oracle is exact decimal equality of the boxed answer.
- **Pools:** 1,024 train prompts, 384 held-out prompts, fixed, reused across epochs.
- **Model:** Qwen2.5-1.5B-Instruct.
- **Audit:** vLLM, 16 samples per prompt, temperature 1.0, 1,024 tokens.
- **Training:** GRPO with LoRA (rank 16, alpha 32, all projections), 300 steps of 64 prompts × 8 samples, learning rate 2e-5 with 10 warmup steps, no KL, group-std advantages, sequence-mean loss, one update per batch (TRL 0.25 with colocated vLLM).
- **Seeds:** 3 seeds × 12 arms.
- **Replication:** `oracle`, `noise_fp`, `fp_tolerance` and `fp_hash` at Llama-3.2-3B-Instruct and Qwen2.5-7B-Instruct × 2 seeds, same protocol, each with its own audit and lock.

### E2: natural verifiers

- **MBPP+ (v0.2.0, 378 tasks).** The verifier runs the base tests; the oracle runs base plus the MBPP+ extra tests. Qwen2.5-Coder-1.5B-Instruct and 7B-Instruct, 2 seeds.
- **IFEval (allenai/RLVR-IFeval).** The verifier is the constraint checkers ported from open-instruct. The oracle is the stricter checkers plus an LLM judge on a subsample.

Primary contrast per domain: final held-out oracle accuracy under the natural verifier minus under the oracle, with the direction and size forecast by the locked `tau_pool` of the two arms.

### E3: scaling axes (exploratory)

Model size (0.5B to 7B), KL coefficient (0, 0.01, 0.05), and data regime. For the KL axis, the theory predicts that with a KL penalty repeated-prompt accuracy settles on a finite tilt between base accuracy and mean precision, and that without one Euclidean policy gradient drifts toward modal precision (`docs/theory.md`, Proposition 2). In single-pass streaming the theory predicts prompt-specific errors are harmless even on train prompts; with multiple epochs, prompt-specific FPs are memorised on train prompts without held-out harm.

### E4: audit-budget patching (exploratory)

At an equal budget of relabelled step-0 samples, fix the errors ranked by `tau`-attribution versus by rate or by precision, then retrain. Choose among candidate verifiers by `tau_pool` versus by static F1.

## Analysis plan

- **Contrasts (H1, H2)**, via `vfates.analysis.outcomes.paired_contrast_seeds`:
  - The contrast is paired by held-out prompt.
  - Each prompt's difference is averaged over seeds.
  - The interval is a percentile bootstrap over prompts with 10,000 resamples.
  - H1 is one-sided, with bootstrap p-value `P(diff <= 0)`. Holm-Bonferroni is applied across its three contrasts at alpha = 0.05.
  - H2 is two one-sided tests at alpha = 0.05 (the 90% interval inside ±0.05), with Holm-Bonferroni across its two contrasts.
- **Rank (H3)**, via `vfates.analysis.outcomes.rank_bootstrap`:
  - Spearman over the 12 arms of the seed-averaged final held-out accuracy against each locked forecast.
  - Intervals come from resampling held-out prompts, with the locked forecasts held fixed.
  - H3 holds if the lower 95% bound of `rho(tau_pool)` is above 0 and the lower 95% bound of `rho(tau_pool) - rho(J)` is above 0.
- **Probe (H4):** `vfates.forecast.probe.arm_error_probe`, 5 prompt-grouped folds, ridge penalty 1, on the train-pool audit.
- **Monitors (H5):** held-out rule-behaviour rates at steps 0 and 300 under each arm and its control.
- **Exclusions:**
  - A run whose loss becomes non-finite is restarted once from its last checkpoint with the same seed. If it fails again it is reported and excluded from contrasts, and that is stated.
  - No run is excluded for its outcome.
- **What a null means.** If H1 fails while the oracle arm improves held-out accuracy by at least 0.10, the conclusion is that the step-0 rate suffices at this scale for these error structures. The paper reports it as such, with the pilots' power estimate. The thresholds are not revisited.

## Go / no-go from the pilots

E1 starts only after P0, P1 and P3 pass, and P4 and the full P5 gate on one H100 pass:

- P4.1: the setup has signal.
- At least one P4.2: a shared error ends below its control.
- P4.3: a prompt-specific error does not.
- On the P5 gate, also P4.7: seed agreement.

`pilots/README.md` has the order and the commands.

## Deviations

Logged here with date and rationale once the document is frozen.
