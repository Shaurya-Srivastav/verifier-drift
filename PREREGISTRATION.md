# Preregistration: Same Rate, Different Fates

This document fixes the hypotheses, endpoint, estimator, design, exclusions and analysis of the confirmatory experiments before any confirmatory training step. The pilots in `pilots/` are exploratory. They chose the configuration and may still change the design (their decision log is in `pilots/README.md`), and they do not count toward any confirmatory test. Changes after this document is frozen go in [Deviations](#deviations) with a date and a reason, and the paper reports them.

**Frozen** at the commit that adds `runs/e1/audit/lock.json` for E1. Each experiment's step-0 forecasts are locked (`vfates.lock/2`) before its first training step. The lock is the SHA-256 of the canonical forecasts, the full audit configuration, the environment report, and the SHA-256 of every artifact the forecasts depend on: the problem pools, the step-0 completions, the oracle grades of those completions, the calibration, the arms file, the features, and the LoRA initialisation. Training refuses to start if any of them has changed, including when the current parser or oracle would grade the audited completions differently. The paper reports the lock hash.

## Question

For a fixed (non-adaptive) verifier used as the RLVR reward, is the step-0 error rate enough to predict what training does to held-out accuracy?

The theory (`docs/theory.md`) says it need not be. The verifier-induced update splits as `g_V = g_O - g_FN + g_FP`. Resampled noise only rescales the clean update (`E[g_V] = J g_O`). A fixed error whose indicator is carried by a behaviour shared across prompts adds a component that can move unseen prompts; a prompt-specific error does not. The step-0 statistic `tau_pool` (the oracle accuracy of the held-out outputs the step-0 update favours most) is a first-order screen for which case holds. It is computed at the base policy and makes no claim about where a long run ends beyond what the first-order direction implies; the knife-edge world in pilot P0 is the documented case where it does not.

## Hypotheses

Confirmatory (E1). H1, H2 and H3 are separate families, each tested at alpha = 0.05. There is no single family-wise claim across them. The abstract states only a hypothesis that passed its own test. `score_lock` and the confirmatory analysis run once, after every planned run has a completed `summary.json`. No seed is added after a confirmatory contrast has been computed.

- **H1 (same step-0 channel rate, different terminal accuracy).** Each of `noise_fp - fp_tolerance`, `noise_fp_trigger - fp_trigger` and `noise_fn - fn_behavior` is tested for superiority over a margin of 0.05: the null is `diff <= 0.05`. A contrast rejects if Holm-Bonferroni across the three rejects it and every completed seed's contrast is positive. H1 holds if at least one non-withdrawn contrast rejects. All three are reported. `noise_fp - fp_tolerance_decoy` is tested the same way and reported beside H1, outside its family.
- **H2 (prompt-specific errors stay local).** `noise_fp - fp_hash` and `noise_fn - fn_prompt_subset` are each equivalent to 0 within ±0.05 (two one-sided tests), with Holm-Bonferroni across the two.
- **H3 (the screen carries information beyond the rate).** Across the matched-rate pairs, the difference in locked `tau_pool` between an arm and its own control rank-correlates with the difference in final held-out accuracy. The rate is equal inside each pair by construction, so step-0 Youden J cannot explain this correlation.

Secondary (E1, reported with intervals, no family-wise control):

- **H4 (mechanism).** Shared errors are linearly decodable from step-0 features on unseen prompts (held-out AUC above 0.65 for a ridge probe fit on the train audit); prompt-specific errors are not (AUC at most 0.60).
- **H5 (behaviour).** Under its own reward, a shared false-positive behaviour becomes more frequent among wrong held-out outputs than under its control, and a shared false-negative behaviour becomes less frequent among correct held-out outputs.
- **H6 (calibration and baselines).** Mean absolute error between locked `tau_pool` and final held-out accuracy across arms, and the rank correlations of every baseline forecast, are reported.

Replication (a 3B model), a KL check, and the natural-verifier experiment E2 are described below with their own primary contrasts. E3 and E4 are exploratory.

## Primary endpoint

**Final held-out oracle accuracy**: the mean over the 384 held-out prompts of each prompt's mean oracle bit over 4 samples at temperature 1.0 and top-p 1.0, from the policy after exactly the last training step (step 300). It is computed by `vfates.train.checkpoint_eval.evaluate_checkpoints` (TRL adapters) through `vfates.train.evaluate.evaluate_policy`. A run without an evaluation at exactly step 300 is incomplete.

The oracle for decimal chain sum reads the last `\boxed{...}` (or `\fbox{...}`) in the completion. It scores 1 when the value inside equals the gold `Decimal`:

- Scale is ignored, so `12.500` equals `12.5`.
- Thousands commas in groups of three, spaces around the decimal point, and a unicode minus are accepted.
- A box with exactly one `=` is read as its right-hand side, when that side is a plain decimal and the left side is empty, one identifier (`x = 12.5`), or an arithmetic expression (`71.8 - 83.8 = -12.0`).
- A second bare number (`12.5 = 13`) or more than one `=` scores 0.

The exact rules and their differences from Reasoning Gym's `score_answer` are in the `vfates.tasks.chain_sum` module docstring.

The held-out pool is dataset seed + 10,000,000, disjoint from the train pool at the question level (checked when the pools are built), and is never used for calibration, training or any design decision. The GPU gate (P5) uses a separate development pool, so no pilot has trained or been scored on these 384 prompts.

Secondary endpoints: held-out accuracy at every 25 steps (the curve) and its peak, train-pool oracle accuracy, on-policy verifier rates (FPR, FNR, J, precision) of each arm at every evaluation, and conditional rule-behaviour rates.

## Primary estimator

`tau_pool` at `q = 1/32` (`vfates.forecast.transfer.PRIMARY_Q`, fixed by P0 before any language-model run):

- **Features:** LoRA-B gradients of the summed completion-token log-likelihood at initialisation, computed on the sampled token ids (including the terminating end-of-sequence token), not on re-tokenised text. Rank 16, alpha 32, all seven projections of every layer, `lora_seed = 0`. The audit saves the exact LoRA A matrices it used, and every training run of every seed loads them, so the forecast and the trained adapter share one parameterisation. CountSketch to 8,192 dimensions (`sketch_seed = 0`), float32.
- **Direction:** `g_V`, the prompt-centred covariance of the features with the arm's verdicts on the train-pool audit (16 samples per prompt). Each train row is divided by its completion length, which makes `g_V` the step-0 direction of the sequence-mean GRPO loss used in training.
- **Velocities:** `s = <phi, g_V>` on the held-out audit (16 samples per prompt), undivided (the change in sequence log-probability), centred within each prompt.
- **Statistic:** the oracle accuracy of the pooled top `q` fraction of held-out velocities, with ties at the boundary sharing the remaining slots.

Reported alongside and not used in any confirmatory test: `tau_pool_std` (the same statistic with each prompt's centred verdicts divided by their standard deviation, the group-std advantage of the trainer), `tau_hard`, `tau_rel`, `tau_soft`, the `tau_pool` curve over `q` in {1/64, 1/32, 1/16, 1/8}, `tau_pool` from final-layer hidden states, step-0 J, mean precision, modal precision, `cos(g_V, g_O)`, the held-out probe AUC, and the split-half reliability of `g_V`.

## Design

### Arms (`configs/arms/chain_sum.yaml`)

| Arm | Error | Structure |
| --- | --- | --- |
| `oracle` | none | clean reward |
| `fp_hash` | FP 0.15 | prompt-specific: accepts a wrong answer when a hash of (prompt, answer) is below a threshold |
| `fp_tolerance` | FP 0.15 | shared: accepts a wrong answer within a fitted absolute tolerance of the gold value |
| `fp_tolerance_decoy` | FP 0.15 | shared, no near misses: accepts a wrong answer whose error lies in a band just beyond `fp_tolerance`'s tolerance |
| `fp_trigger` | FP 0.15 | shared: accepts a wrong completion containing a fitted phrase that is not enriched for correct answers |
| `noise_fp` | FP matched to `fp_tolerance` | resampled per completion, on answered wrong outputs (the support of the answer-based rules) |
| `noise_fp_trigger` | FP matched to `fp_trigger` | resampled per completion, on every wrong output (the support of the phrase rule) |
| `fn_behavior` | FN 0.30 | shared: rejects a correct completion containing a fitted phrase that predicts correctness |
| `fn_prompt_subset` | FN 0.30 | prompt-specific: rejects every correct answer on a hashed subset of prompts |
| `noise_fn` | FN matched to `fn_behavior` | resampled per completion |
| `fn_form` | FN at its achievable rate | shared: rejects a correct answer written with fewer decimal places than the gold string (capped: the policy rarely does this) |
| `noise_fn_form` | FN matched to `fn_form` | resampled per completion |
| `combo_fnform_fphash` | FN of `fn_form`, FP of `fp_hash` | both rules |
| `natural_string_match` | natural | exact string match against the gold answer; audited and reported, not trained |

Every imperfect arm is fitted on the base model's step-0 samples of the train pool only (`vfates.verifiers.calibrate.calibrate_arms`) to FPR 0.15 and FNR 0.30. Resampled controls copy their matched arm's channel rates. Salts and candidate rules are fixed in the config before sampling.

The phrase arms (`fp_trigger`, `fn_behavior`) use these fixed rules:

- A phrase is a run of one to three word tokens, matched only on token boundaries. Candidates containing a digit are excluded, because numbers in chain sums are prompt-specific.
- The preference sign uses the within-prompt association between the phrase and correctness (prompt-centred covariance), so a phrase that only marks a kind of prompt does not count as a behaviour.
- The rate tolerance is 0.02, the withdrawal threshold, and it is applied to the phrase's rate on the full train pool, where every other arm is fitted and every channel rate is judged. Candidates are proposed, and the preference sign is estimated, on 7/8 of the train prompts. The sign is then checked on the fixed remaining 1/8 (prompts with `u01("phrase_validation", pid) < 1/8`). The trigger passes if its validation association is at most 0, or its 90% prompt-bootstrap interval covers 0. The behaviour phrase passes if its validation association is positive.

The decoy's lower edge is `fp_tolerance`'s fitted tolerance. Its upper edge is the smallest observed error whose pooled FPR is closest to 0.15. The decoy and `fp_tolerance` accept disjoint sets of wrong answers at the same rate, so a difference between their fates is not explained by the rate or by answer formatting.

The calibration report prints, for every arm, the channel rates (what training sees in expectation) and the rates of the one realised audit draw, each with its own Youden J.

### Withdrawals decided from the audit

These are applied after the audit and before the first training step, and written to `withdrawals.json` and to Deviations:

- A targeted arm (`fp_hash`, `fp_tolerance`, `fp_tolerance_decoy`, `fp_trigger`, `fn_behavior`, `fn_prompt_subset`) whose channel rate misses its target by more than 0.02, or cannot be computed, is withdrawn. `fp_tolerance_decoy` is also withdrawn if its band is capped. `fn_form` is capped by design, runs at its achievable rate, and is never withdrawn for its rate.
- `fp_trigger` or `fn_behavior` is withdrawn if no candidate met the preference inside the rate tolerance (`fell_back`), or if its sign check on the validation prompts fails.
- Withdrawing an arm withdraws every contrast and pair that uses it.

### E1: decimal chain sum (`configs/experiments/e1_chain_sum.yaml`)

- **Task:** Reasoning Gym `decimal_chain_sum` with the Egashira et al. configuration (3 to 6 terms, 3 to 6 digits, 3 to 6 decimal places).
- **Pools:** 1,024 train prompts (dataset seed 0) and 384 held-out prompts (dataset seed 10,000,000), fixed and reused across epochs.
- **Model:** Qwen2.5-1.5B-Instruct. Pilot P2 measured base held-out accuracy 0.215, boxed rate 0.967 and truncation 0.006 at 1,024 tokens.
- **Audit:** vLLM, 16 samples per prompt, temperature 1.0, top-p 1.0, 1,024 tokens.
- **Training:** GRPO with LoRA (rank 16, alpha 32, all projections, A loaded from the audit), 300 steps of 64 prompts × 8 samples, learning rate 2e-5 with a 10-step linear warmup, no KL penalty, group-std advantages, sequence-mean loss (the stop token is part of the loss), one update per batch, no importance-sampling correction. TRL 0.25.1 with colocated vLLM, versions pinned in `scripts/setup_gpu.sh` and recorded in every lock and run.
- **Trained arms and seeds:** the 13 trained arms above × seeds 0, 1 and 2. A seed fixes the data order and the sampling seeds of evaluation. The LoRA initialisation is the same for every seed. TRL's colocated vLLM rollouts take no per-request seed, so training runs are not bit-for-bit reproducible, and the seeds sample that variation.
- **Seed rule (fixed now, applied from P5):** if the sample standard deviation of the two P5 oracle seeds' final held-out accuracies exceeds 0.03, the H1 and H2 arms (`noise_fp`, `fp_tolerance`, `fp_tolerance_decoy`, `fp_trigger`, `noise_fp_trigger`, `fp_hash`, `noise_fn`, `fn_behavior`, `fn_prompt_subset`) also run seeds 3 and 4. The seed list is written into the E1 config before the E1 audit.
- **Early baseline:** every trained arm also runs once with seed 99 for 25 steps. Its held-out accuracy at step 25 is a competitor forecast for H3 (the cheap alternative a reviewer would propose). It is not a confirmatory run.

### KL check (`configs/experiments/e1_kl_check.yaml`, descriptive)

`beta_kl = 0.01` for `oracle` and `fp_tolerance`, seed 0, on the E1 audit and lock. This is the only language-model evidence offered for the KL-regularised case of Proposition 2. Without it, Proposition 2 remains a statement about the toy worlds.

### Replication (`configs/experiments/e1_replication_3b.yaml`)

- **Model rule:** `meta-llama/Llama-3.2-3B-Instruct` if its base held-out accuracy at 1,024 tokens on 128 prompts of the P5 development pool is in [0.10, 0.60] with truncation at most 0.05. Otherwise `Qwen/Qwen2.5-3B-Instruct`.
- **Arms:** `oracle`, `noise_fp`, `fp_hash` and `fp_tolerance`, plus every H1 arm that rejected in E1, with its control. Two seeds each.
- **Protocol:** E1's protocol, with its own audit and lock.
- **Primary contrast:** the E1 H1 and H2 tests applied to the contrasts present, Holm-Bonferroni within the replication. The replication is reported as such; it does not rescue a failed E1 hypothesis.
- **Not tested:** H3 (`analysis.h3.tested: false`). Its statistics are reported as descriptive with decision `not_tested`, and the early runs that only serve H3's baselines are not trained.
- **One model:** the audit generator, the gradient features and the training policy use the same `model_id`, and training uses the audit's token budget and temperature. The study refuses to run otherwise.

The 7B grid in the earlier draft is cut.

### E2: natural verifiers

E2 runs after E1 and has its own audit and lock. Its design is fixed here; its budget is spent only after E1.

Both domains use E1's protocol with their own audit and lock:

- 2 seeds, 300 steps;
- final held-out accuracy from 4 samples per prompt at temperature 1.0;
- seed-paired contrasts with the crossed bootstrap (B = 10,000);
- the same LoRA-gradient features and `seq_mean` loss.

In both, the natural verifier never rejects an output the oracle accepts, so its false-negative rate is 0 by construction and the resampled control only adds false positives.

- **MBPP+ (EvalPlus v0.2.0, 378 tasks; `configs/experiments/e2_mbpp.yaml`).**
  - **Policy:** Qwen2.5-Coder-1.5B-Instruct.
  - **Split:** a task is held out when `u01(0, task_id) < 0.33`, giving 241 train and 137 held-out tasks.
  - **Arms:**
    - `base_tests`, the natural verifier, runs the original MBPP tests.
    - `oracle` runs the base tests and the MBPP+ extra tests, with EvalPlus per-test time limits.
    - `resampled` copies `base_tests`' step-0 channel false-positive rate on the train split. It draws only on wrong outputs that define the entry-point function, because the base tests cannot pass otherwise.
- **IFEval (`allenai/RLVR-IFeval`; `configs/experiments/e2_ifeval.yaml`).**
  - **Policy:** Qwen2.5-1.5B-Instruct.
  - **Split:** a row is held out when `u01(0, row_id) < 0.33`. The pools are the first 1,024 train rows and the first 384 held-out rows by row id.
  - **Arms:**
    - `checker`, the natural verifier, is the constraint checker.
    - `oracle` is the checker conjoined with a content judge: Qwen2.5-7B-Instruct served by vLLM, temperature 0, 16 output tokens, with the rubric fixed by its SHA-256 in the config.
    - `resampled` copies the checker's step-0 channel false-positive rate, on every wrong output.
  - **Judge agreement check (reported, not tested):** `gpt-4.1` at temperature 0 with the same rubric grades a fixed random subsample of 500 held-out evaluation samples (seed 0).
- **Primary contrast per domain:** `resampled - natural`, the rate-matched comparison. Each domain is tested on its own, at alpha = 0.05, with the crossed bootstrap.
  - **Locality:** equivalence within ±0.05 (two one-sided tests). This is what the prompt-specific reading of natural false positives predicts.
  - **Screen direction:** whether the sign of the contrast matches the sign of `tau_pool(resampled) - tau_pool(natural)` from the lock. This counts only when the contrast's two-sided 95% interval excludes 0.
  - `oracle - natural` is reported for scale. It confounds rate and structure, so it is not the test.
  - E1's oracle-gain rule applies: a mean oracle held-out gain below 0.10 labels the E2 tests of that domain inconclusive. The statistics are still reported.
  - The tests are computed by `vfates.analysis.confirmatory` from each config's `analysis.natural` block.

### E3: scaling axes (exploratory)

Model size, KL coefficient (0, 0.01, 0.05) and data regime. For the KL axis, the theory predicts that with a KL penalty repeated-prompt accuracy settles on a finite tilt between base accuracy and mean precision, and that without one Euclidean policy gradient drifts toward modal precision (`docs/theory.md`, Proposition 2). In single-pass streaming, the theory predicts prompt-specific errors are harmless even on train prompts. With multiple epochs, prompt-specific false positives are memorised on train prompts without held-out harm.

### E4: audit-budget patching (exploratory)

At an equal budget of relabelled step-0 samples, fix the errors ranked by `tau`-attribution versus by rate or by precision, then retrain. Choose among candidate verifiers by `tau_pool` versus by static F1.

## Analysis plan

All confirmatory numbers come from `vfates.analysis.confirmatory` on the finished run directories.

- **Contrasts (H1, H2).** For a contrast `control - arm`, arm seed `s` is paired with control seed `s`. `D[s, p]` is control accuracy minus arm accuracy on held-out prompt `p` at step 300, and the estimate is the mean of `D` over seeds and prompts. The uncertainty comes from a crossed (pigeonhole) bootstrap with 10,000 resamples (Owen, 2007): seeds and prompts are resampled independently, with replacement. This includes training-seed variation, which a prompt-only bootstrap would not.
- **H1 test.** The p-value is the bootstrap proportion of resamples with `diff <= 0.05`. Holm-Bonferroni runs across the three H1 contrasts at alpha = 0.05. A contrast rejects only if, in addition, every completed seed's contrast (mean over prompts) is positive.
- **H2 test.** The p-value is `max(P*(diff <= -0.05), P*(diff >= 0.05))`, with Holm-Bonferroni across the two contrasts at alpha = 0.05.
- **H3 test.** The pairs are `fp_tolerance` vs `noise_fp`, `fp_tolerance_decoy` vs `noise_fp`, `fp_hash` vs `noise_fp`, `fp_trigger` vs `noise_fp_trigger`, `fn_behavior` vs `noise_fn`, `fn_prompt_subset` vs `noise_fn`, and `fn_form` vs `noise_fn_form`. For each non-withdrawn pair:
  - `dtau` is the locked `tau_pool` of the arm minus that of its control.
  - `dacc` is the seed-averaged final accuracy of the arm minus that of its control, over the pair's own planned seeds (five for the H1 and H2 arms if the seed rule adds seeds, three for `fn_form` and `noise_fn_form`). In the crossed bootstrap every pair shares one prompt draw, and pairs with the same seed list share one seed draw.
  - `rho` is the Spearman correlation of `dtau` with `dacc`, using average ranks.

  H3 holds if all three of these hold:
  1. The exact one-sided permutation p-value of `rho` (all orderings of `dacc`) is below 0.05.
  2. The 5th percentile of `rho` under the crossed bootstrap (forecasts fixed, accuracies resampled) is above 0.
  3. At least 5 pairs are not withdrawn.
- **Baselines for H3 (secondary).** The same within-pair `rho` is computed for each baseline, together with `rho(tau_pool) - rho(b)` and its crossed-bootstrap interval. The baselines are:
  - step-0 J, which is constant within a pair by construction;
  - mean precision and modal precision;
  - `cos(g_V, g_O)`;
  - the held-out probe AUC;
  - the step-25 accuracy of the seed-99 run;
  - `tau_pool_std`;
  - the behaviour alignment of Egashira et al., defined as follows.
    - Let `b` be the arm's behaviour predicate: the rule's acceptance test for a false-positive rule, its rejection test for a false-negative rule.
    - Let `f` be the frequency of `b` on the train audit.
    - Let `C` be the mean prompt-centred oracle advantage of the outputs with `b`.
    - The forecast is `f * C` for a false-positive rule and `-f * C` for a false-negative rule. It is 0 for resampled arms, which carry no behaviour the policy can express.

  Two more summaries are reported: the Spearman correlation across all trained arms for every forecast, and whether the sign of `dtau` agrees with the sign of `dacc` on each H1 pair.
- **Sensitivities (reported, not tested):** the prompt-only bootstrap, and a per-seed t interval with S - 1 degrees of freedom.
- **Probe (H4):** a ridge probe (penalty 1) of each arm's error indicator. It is fit on train-audit features and scored on held-out audit rows (`vfates.forecast.probe.heldout_error_probe`). An arm probed on both channels (a composite, or a resampled arm with both rates) reports both AUCs; H4 makes no call for it, and its H3 baseline is the mean of the two. A withdrawn H4 arm is reported as `withdrawn`.
- **Monitors (H5):** conditional behaviour rates, per seed and averaged, at step 0 and step 300, arm versus control. For false-positive rules this is `P(behaviour | wrong)`; for false-negative rules it is `P(behaviour | correct)`.
- **On-policy rates:** FPR, FNR, J and precision of each arm on its own evaluation samples at every evaluation. For each H1 pair, the paper also reports an adjusted contrast (secondary): the observed contrast minus `b * dJbar`, where:
  - `dJbar` is the control-minus-arm difference in on-policy J, averaged over the run (trapezoid rule over evaluation steps);
  - `b` is the least-squares slope of final accuracy on that time-averaged J, across all runs.

  If H1 rejects and the adjusted gap stays above 0.05, the fate is not explained by the rate path. If the adjusted gap collapses, the honest claim is narrower: fixed rules change the fate by changing the on-policy rate, and `tau_pool` is a step-0 predictor of that divergence.
- **Knife-edge label:** a run whose held-out accuracy rises at least 0.05 above step 0 at some evaluation and later falls at least 0.10 below its running peak is labelled knife-edge in every table. The endpoint stays step 300, so the label cannot change a hypothesis.
- **Exclusions during training:**
  - A run whose loss or gradient becomes non-finite restarts once from its last checkpoint with the same seed. A second failure withdraws that arm from H1 to H3.
  - An arm with fewer than its planned completed seeds is withdrawn, not analysed on the remainder.
  - No run is excluded for its outcome.
- **Inconclusive:** if the oracle arm's final held-out accuracy is less than 0.10 above its step-0 accuracy, E1 is inconclusive. H1 to H3 are reported and labelled inconclusive.
- **What a null means:** if the oracle gain is at least 0.10 and every non-withdrawn H1 contrast fails, these shared rules do not move terminal held-out accuracy by more than 0.05 relative to rate-matched resampled noise, on this model and task. The paper reports that as the finding. The thresholds are not revisited.

## Go / no-go from the pilots

The pilots decide feasibility, not the confirmatory question. E1 starts only after:

- P0, P1 and P3 pass. P3's gates are the estimator checks: the split-half reliability of `g_V` (P3.2, P3.3), the sketch fidelity (P3.4) and the lock (P3.8).
- P4.1 passes: on the Mac micro-GRPO pilot, the oracle arm raises held-out accuracy by at least 0.10.
- The full P5 gate on one H100 shows an oracle held-out gain of at least 0.10 on the development pool: 512 train prompts at dataset seed 50,000,000 and 384 held-out prompts at 60,000,000, disjoint from E1 at the question level.

The pilots' decodability probes and fate contrasts (P3.5, P3.6, P4.2 to P4.7, and P5's contrasts) are logged and do not decide whether E1 runs. With one seed and 128 held-out prompts, P4's hit rule (a shared arm at least 0.05 below its control with p < 0.05) has power of about 0.40 at a true gap of 0.05, so gating E1 on it would mostly stop the effects E1 is built to detect. P5 also fixes the E1 seed count by the rule above, and replaces the projected H100 cost with a measured cost per arm-seed.

`pilots/README.md` has the order and the commands.

## Software and compute

The H100 environment is built by `scripts/setup_gpu.sh`. The pinned versions of TRL, vLLM, PyTorch, Transformers, PEFT and Reasoning Gym are listed there, and the environment report inside every lock and run records what actually ran. The compute budget is about $5,000 of GPU time and $2,000 of API calls. The E1 cost in the config is a projection until P5 measures hours per arm-seed on the same hardware.

## Deviations

Logged here with date and rationale once the document is frozen.
