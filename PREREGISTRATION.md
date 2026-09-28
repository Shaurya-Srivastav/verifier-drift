# Preregistration: Same Rate, Different Fates

This document fixes the hypotheses, endpoint, estimator, design, exclusions and analysis of the confirmatory study before any confirmatory training step. The pilots in `pilots/` are exploratory. They check feasibility and fix the settings this document leaves to them by rules written here, and they do not count toward any confirmatory test. Changes after this document is frozen go in [Deviations](#deviations) with a date and a reason, and the paper reports them.

**Frozen** at the commit that adds the first confirmatory cell lock (`runs/<cell>/audit/lock.json` for any cell in [`configs/program.yaml`](configs/program.yaml)). Each cell's step-0 forecasts are locked (`vfates.lock/2`) before that cell's first training step. The lock is the SHA-256 of the canonical forecasts, the full cell configuration, the environment report (including the hardware class), and the SHA-256 of every artefact the forecasts depend on: the problem pools, the step-0 completions, the oracle grades of those completions, the calibration, the arms file, the features, the patch selections and the LoRA initialisation. Training refuses to start if any of them has changed, including when the current parser or oracle would grade the audited completions differently. The paper reports every lock hash.

## Question

For a fixed (non-adaptive) verifier used as the RLVR reward, is the step-0 error rate enough to predict what training does to held-out accuracy, across task domains, model sizes and model families? And can a statistic computed before training, from the base model and a few hundred oracle labels, predict which verifier errors will spread and which to fix first?

The theory (`docs/theory.md`) says the rate is not enough. The verifier-induced update splits as `g_V = g_O - g_FN + g_FP`. Resampled noise only rescales the clean update (`E[g_V] = J g_O`). A fixed error whose indicator is carried by a behaviour shared across prompts adds a component that can move unseen prompts; a prompt-specific error does not. The step-0 statistic `tau_pool`, the oracle accuracy of the held-out outputs the step-0 update favours most, is a first-order screen for which case holds. Its direction `g_V` is built from the verifier's own verdicts, so the oracle is needed only on the top `q = 1/32` of held-out samples. It is computed at the base policy and makes no claim about where a long run ends beyond what the first-order direction implies; the knife-edge world in pilot P0 is the documented case where it does not.

## The program

The confirmatory unit is a **cell**: one experiment, one domain, one model. A cell has one audit, one lock, one hardware class, an arm menu and a seed list. Every cell is listed in `configs/program.yaml`, and its full configuration is rendered from a domain template and a model spec into `configs/cells/<cell>.yaml` (`vfates program render`), which is committed before its audit.

| Experiment | Cells (domain × model) | Menu | Seeds | Question |
| --- | --- | --- | --- | --- |
| E1 fates | chain_sum, math, code, ifeval, qa × `qwen3-1.7b` | full | 0, 1, 2 | Same rate, different fates in every domain; the pooled screen |
| E2 scale | all five domains × `qwen3-4b`; math, code, qa × `qwen3-8b`; math, code × `qwen3-14b`; math × `qwen3-32b` | core | 0, 1 | Does the screen hold from 1.7B to 32B, and how does the effect of a shared error change with size? |
| E3 families | math, code × `olmo3-7b`, `llama3.1-8b` | core | 0, 1 | Does it hold outside the Qwen family, on an open-data model? |
| E4 patching | the `patch_tau` and `patch_rate` arms inside the E1 cells of math, code, ifeval and qa | | 0, 1, 2 | At an equal patch budget, does the screen choose better fixes than the error count? |

That is 20 confirmatory cells: 5 in E1, 11 in E2, 4 in E3. E4 shares the E1 audits and locks. The exploratory runs below are listed in `configs/program.yaml` as experiment X: X2, X3, X5 and X7 are named training variants of arms inside E1 cells (on those cells' audits), and X4 and X6 are cells with their own audits.

**Stages.** The confirmatory cells are analysed in two stages, each fixed now and computed once, when every planned run of that stage is complete (`configs/program.yaml`, `stages`):

| Stage | Cells | Hardware | Confirmatory tests |
| --- | --- | --- | --- |
| A, discovery | the 5 E1 cells (with E4) and the 5 E2 cells at `qwen3-4b` | lab GPUs | H1, H1-pooled, H2, H3, H7, H8, H10 |
| B, replication | E2 at `qwen3-8b`, `qwen3-14b` and `qwen3-32b`; all of E3 | rented H100s | H3-replication |

Stage B's test uses only stage B's pairs, which stage A's analysis never saw, so it is a prospective replication of the screen at larger scale and in two other families. Stage A's report does not wait for stage B, and no stage A result changes stage B's design; a change to either stage after this document is frozen is a deviation. Workers claim stage A jobs before stage B jobs, and exploratory jobs after every confirmatory job of their hardware class.

Exploratory, reported and not tested:

- **X1 cross-domain transfer.** Every final E1 adapter of seed 0 for `oracle`, `fp_trigger`, `noise_fp_trigger`, `fn_behavior`, `noise_fn`, `natural` and `noise_natural` is evaluated on the held-out pools of the other four domains, relative to that cell's oracle adapter. The question is whether a behaviour a leaky verifier rewards in one domain appears in another.
- **X2 KL.** `beta_kl = 0.01` for `oracle` and `fp_trigger` in the E1 math cell, seed 0, on the E1 audit and lock.
- **X3 streaming.** Single-pass data for `oracle`, `fp_hash` and `noise_fp` in the E1 chain_sum cell, seed 0. The theory predicts that prompt-specific errors are harmless even on train prompts when no prompt repeats.
- **X4 objective.** The math domain at `qwen3-1.7b` with the Dr. GRPO objective (mean-centred advantages, constant normaliser) and with the token-mean loss of DAPO, for `oracle`, `fp_trigger` and `noise_fp_trigger`, seed 0. Each variant has its own audit, because the forecast uses the trained loss's length weighting.
- **X5 full-parameter.** The same three arms in math at `qwen3-1.7b` with full-parameter training instead of LoRA, seed 0. The locked forecast is the LoRA one, so this checks the fate contrast, not the screen.
- **X6 base model.** The chain_sum domain with `Qwen/Qwen3-1.7B-Base`, the model and task of Egashira et al., with the kit at matched rates, seeds 0 and 1.
- **X7 horizon.** `oracle`, `fp_trigger` and `noise_fp_trigger` in math at `qwen3-1.7b` for 600 steps, seed 0, so that a gap at step 300 can be read against a longer run.

Two reported arms sit inside E1 cells outside every hypothesis: `random` in math (a reward independent of correctness, FPR = FNR = 0.5, J = 0; the spurious-reward reference of Shao et al.), and `natural_robust` in qa (the master-key-robust judge of Zhao et al., with its rate-matched control).

A correction that uses only the step-0 rates (the unbiased reward estimators of Cai et al. and Mansouri et al.) is not a separate arm. It treats an arm and its rate-matched control identically, so it cannot remove a difference between them; with group-standardised advantages an affine reward correction also leaves the GRPO update unchanged.

## Hypotheses

Confirmatory. H1, H1-pooled, H2, H3, H7, H8, H10 and H3-replication are separate families, each tested at alpha = 0.05. There is no family-wise claim across them. The abstract states only a hypothesis that passed its own test; the paper's abstract sentence on results is generated from the pooled report by that rule (`vfates.paper`). The per-cell reports are computed once, after every planned run of the cell has a completed `summary.json`, and each stage's pooled report (`vfates program analyze`) once, after every planned run of every cell of that stage has one. No seed is added after a confirmatory contrast has been computed.

- **H1 (same step-0 channel rate, different terminal accuracy).** In each E1 cell, the kit contrasts `noise_fp_trigger - fp_trigger` and `noise_fn - fn_behavior`, plus `noise_fp - fp_tolerance` in chain_sum and `noise_fp_tolerance - fp_tolerance` in math, are tested for superiority over a margin of 0.05: the null is `diff <= 0.05`. That is 12 contrasts. A contrast rejects if Holm-Bonferroni across the 12 rejects it and every completed seed's contrast is positive. H1 holds if at least one non-withdrawn contrast rejects; the paper reports which domains. `noise_fp - fp_tolerance_decoy` in chain_sum is tested the same way and reported beside H1, outside its family.
- **H1-pooled (the effect is general, not one domain).** For each kit contrast, the equal-weight mean over the E1 cells where that contrast is not withdrawn is greater than 0 (one-sided), with Holm-Bonferroni across the two kit contrasts. It requires at least three domains per contrast.
- **H2 (prompt-specific errors stay local).** In each E1 cell, `noise_fp - fp_hash` and `noise_fn - fn_prompt_subset` are equivalent to 0 within ±0.05 (two one-sided tests), with Holm-Bonferroni across the 10 contrasts.
- **H3 (the screen carries information beyond the rate).** Across every non-withdrawn matched pair of every conclusive stage A cell, the difference in locked `tau_pool` between an arm and its own control rank-correlates with the difference in final held-out accuracy. The rate is equal inside each pair by construction, so step-0 Youden J cannot explain the correlation. The null distribution permutes accuracy differences within cells, so a difference between cells in overall accuracy cannot create it.
- **H3-replication (the screen holds at scale and in other families).** The same test over the pairs of the conclusive stage B cells only.
- **H10 (the screen separates shared rules at the same rate).** Two E1 pairs hold the structure fixed and change only the direction of the shared behaviour. `fp_trigger_enriched` and `fp_trigger` both accept wrong completions containing a fitted phrase at FPR 0.15; the first phrase is enriched for correct answers, the second is not. `fn_behavior_depleted` and `fn_behavior` both reject correct completions containing a fitted phrase at FNR 0.30; the first phrase is depleted among correct answers, the second predicts them. For each pair `(a, b)`, the equal-weight mean over the conclusive E1 cells where neither arm is withdrawn of `sign(dtau) * (acc_a - acc_b)`, with the locked `dtau = tau_pool(a) - tau_pool(b)`, is greater than 0 (one-sided), with Holm-Bonferroni across the two pairs. A cell whose `dtau` is exactly 0 contributes 0. It requires at least three cells per pair. The theory's structure label is the same within each pair, so a reading of "shared errors transfer" without the statistic cannot pass H10.
- **H7 (natural verifiers).** In each E1 cell with a natural verifier (math has two), the rate-matched contrast `noise_natural - natural` differs from 0, with Holm-Bonferroni across the 5 natural pairs. For every pair that differs, the sign of the contrast is compared with the sign of the locked `tau_pool(noise_natural) - tau_pool(natural)`.
- **H8 (the screen picks better fixes).** Pooled over the E4 domains where the two selections differ, `patch_tau` ends with higher final held-out accuracy than `patch_rate` (one-sided). Both patch the same natural verifier in the same number of error categories; `patch_rate` fixes the categories with the most step-0 errors, and `patch_tau` fixes the pair with the highest locked `tau_pool`. It requires at least two domains.

Secondary (reported with intervals, no family-wise control):

- **H4 (mechanism).** Shared errors are linearly decodable from step-0 features on unseen prompts (held-out AUC above 0.65 for a ridge probe fit on the train audit); prompt-specific errors are not (AUC at most 0.60). Per cell.
- **H5 (behaviour).** Under its own reward, a shared false-positive behaviour becomes more frequent among wrong held-out outputs than under its control, and a shared false-negative behaviour becomes less frequent among correct held-out outputs. Per cell.
- **H6 (calibration and baselines).** Mean absolute error between locked `tau_pool` and final held-out accuracy across arms, and the rank correlations of every baseline forecast, per cell and pooled.
- **H9 (oracle-light screen).** H3's correlation (and, once stage B is complete, H3-replication's) recomputed with each arm's `tau_pool` estimated from only 32, 64, 128 or 192 oracle labels drawn without replacement from its top-`q` set (Horvitz-Thompson weighted, so unbiased for `tau_pool`). Each arm draws from its own random stream, and the audit locks 200 draws per arm and budget. Draw `d` gives one correlation over the H3 pairs; the report is the median and 5th percentile of those 200 correlations and the fraction above 0. A budget with a missing draw is reported as missing, never approximated.
- **Scale.** For each kit contrast, the slope of the per-cell contrast on log10 of the parameter count across the Qwen3 ladder of each domain, with a cell-level bootstrap interval: over 1.7B and 4B in stage A, and over the full ladder once stage B is complete. No direction is predicted.
- **H3 by stratum.** H3's correlation within each experiment, domain and model; within the pairs whose arm is a shared-behaviour rule or a natural verifier (the stratum where the structure label alone predicts nothing about direction); and over the union of stages A and B.

## Primary endpoint

**Final held-out oracle accuracy**: the mean over the 384 held-out prompts of each prompt's mean oracle bit over 4 samples at temperature 1.0 and top-p 1.0, from the policy after exactly the last training step (step 300). It is computed by `vfates.train.checkpoint_eval.evaluate_checkpoints` through `vfates.train.evaluate.evaluate_policy`. A run without an evaluation at exactly step 300 is incomplete.

Each domain's oracle is in [Domains](#domains). Held-out pools are disjoint from train pools at the question level (checked when the pools are built), and are never used for calibration, training, patch selection or any design decision. The pilots use development pools disjoint from both, so no pilot has trained or been scored on a confirmatory prompt.

Secondary endpoints: held-out accuracy at every 25 steps (the curve) and its peak, train-pool oracle accuracy, on-policy verifier rates (FPR, FNR, J, precision) of each arm at every evaluation, and conditional rule-behaviour rates.

## Primary estimator

`tau_pool` at `q = 1/32` (`vfates.forecast.transfer.PRIMARY_Q`, fixed by P0 before any language-model run):

- **Features:** LoRA-B gradients of the summed completion-token log-likelihood at initialisation, computed on the sampled token ids (including the terminating end-of-sequence token), not on re-tokenised text. Rank 16, alpha 32, all seven projections of every layer, `lora_seed = 0`. Model weights in bfloat16, gradients sketched and accumulated in float32. The audit saves the exact LoRA A matrices it used, and every training run of every seed loads them, so the forecast and the trained adapter share one parameterisation. CountSketch to 8,192 dimensions (`sketch_seed = 0`). Pilot P3 checks that bfloat16 features agree with float32 features (row cosine at least 0.99) before any lock.
- **Direction:** `g_V`, the prompt-centred covariance of the features with the arm's verdicts on the train-pool audit (16 samples per prompt). Each train row is divided by its completion length, which makes `g_V` the step-0 direction of the sequence-mean GRPO loss used in training.
- **Velocities:** `s = <phi, g_V>` on the held-out audit (16 samples per prompt), undivided (the change in sequence log-probability), centred within each prompt.
- **Statistic:** the oracle accuracy of the pooled top `q` fraction of held-out velocities, with ties at the boundary sharing the remaining slots. With 384 × 16 held-out samples, that is 192 oracle labels per verifier.

Reported alongside and not used in any confirmatory test: `tau_pool_std` (with each prompt's centred verdicts divided by their standard deviation, the group-std advantage of the trainer), `tau_hard`, `tau_rel`, `tau_soft`, the `tau_pool` curve over `q` in {1/64, 1/32, 1/16, 1/8}, `tau_pool` from final-layer hidden states, `tau_pool_budget` (the estimate from 32, 64, 128 and 192 labels), step-0 J, mean precision, modal precision, `cos(g_V, g_O)`, the held-out probe AUC, and the split-half reliability of `g_V`.

## Domains

Every domain uses pools of 1,024 train and 384 held-out prompts and a development pool for the pilots, all disjoint at the question level after deduplication. Splits are by a salted hash of the question (`vfates.utils.hashing.u01`), fixed before any sampling. Difficulty is chosen per domain by pilot P2 from the preregistered buckets below, by the rule in [Go / no-go](#go--no-go-from-the-pilots). It is a bucket-level choice; no question is kept or dropped for how the study model answers it. The same pools serve every model of a domain, so the scale and family cells see the same questions as E1.

| Domain | Source | Difficulty buckets | Oracle | Natural verifier | Tokens |
| --- | --- | --- | --- | --- | --- |
| chain_sum | Reasoning Gym `decimal_chain_sum`, Egashira et al. configuration (3 to 6 terms, 3 to 6 digits, 3 to 6 decimal places) | fixed | exact `Decimal` equality of the last `\boxed{}` (rules in `vfates.tasks.chain_sum`) | `natural_string_match` (audited, not trained) | 1,024 |
| math | `SynthLabsAI/Big-Math-RL-Verified`; proofs and multiple choice removed; statements matching MATH-500, GSM8K test or AIME 1983 to 2024 removed | `llama8b_solve_rate` in [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8) | last `\boxed{}`, math-verify symbolic equivalence at its defaults (floats compared after rounding to 6 decimal places, no relative tolerance), 5 s CPU limit, a timeout is incorrect | `natural`: lenient cascade (last box or last number, relative tolerance 1e-4); `natural_strict`: DeepSeek-Math strip-string exact match | 1,536 |
| code | `BAAI/TACO` train shards, Python, at least 5 hidden tests and a public example in the statement, deduplicated by normalised statement | `easy` = TACO `EASY`; `medium` = TACO `MEDIUM` and `MEDIUM_HARD` together (`MEDIUM` alone is 740 train and 250 held-out problems, short of the pools) | passes every hidden test (a fixed, hashed subset of at most 32 per problem), 2 s and 512 MB per test; the process that runs a submission never holds an expected output, and a guard installed in it refuses file, process and network access outside its working directory (with Landlock on Linux kernels that support it), so a program cannot read the answer it is graded against; refused operations are counted per sample and reported | `natural`: passes the public examples in the statement; an example whose input is also a hidden test is scored against that test's expected output | 1,024 |
| ifeval | `allenai/IF_multi_constraints_upto5` (IFEval and IFBench constraints) | 2 to 3 constraints, 4 to 5 constraints | every constraint checker passes AND a frozen content judge (Qwen3-14B, non-thinking, temperature 0, rubric fixed by its SHA-256) accepts | `natural`: the constraint checkers alone, so its errors are false positives only (FNR 0 by construction) | 768 |
| qa | `TIGER-Lab/WebInstruct-verified` (train split), answer types with a canonical form (integer, float, simple expression) | as available on the dataset card | strict rule: math-verify equivalence or normalised exact match, no tolerance | `natural`: `TIGER-Lab/general-verifier` (1.5B model judge, the reward General-Reasoner trains with) | 1,024 |

Backups, used only by the P2 and P4 rules: DeepScaleR without AIME (math) and `allenai/RLVR-IFeval` (ifeval). chain_sum, code and qa have none. For code, neither APPS introductory nor Codeforces problems from `deepmind/code_contests` at ratings up to 1,200 yield 1,024 train and 384 held-out problems under the same filters, so a code domain with no qualifying TACO bucket leaves the program. A backup is never selected implicitly: a source that fails to load stops the audit. The dataset revisions are pinned in `configs/tasks/`.

Every prompt ends with a fixed instruction naming the answer contract (a final `\boxed{}` in chain_sum, math and qa; one Python block in code), and its SHA-256 is part of the task description in the lock.

Before the E1 qa lock, one annotator labels about 200 held-out audit samples on which the rule oracle, Qwen3-32B and an API model disagree. The oracle's error rate on that sample is reported with the results. The same check is run for the ifeval content judge with `gpt-4.1` on 500 held-out evaluation samples after training (`analysis.agreement_check`). In pilot P6, one annotator also adjudicates 100 step-0 false positives of each natural verifier on the development pool, so that a natural error is a real error before it is called one. None of these checks changes a verdict.

## Design

### Arms

Every imperfect synthetic arm is fitted on the base model's step-0 samples of its cell's train pool only (`vfates.verifiers.calibrate.calibrate_arms`) to FPR 0.15 and FNR 0.30. Resampled controls copy their matched arm's channel rates. Salts and candidate rules are fixed in the arms files before sampling; each domain gets its own salts.

**The kit** (`configs/arms/kit.yaml`), identical in every domain:

| Arm | Error | Structure |
| --- | --- | --- |
| `oracle` | none | clean reward |
| `fp_hash` | FP 0.15 | prompt-specific: accepts a wrong answer when a hash of (prompt, answer) is below a threshold |
| `fp_trigger` | FP 0.15 | shared: accepts a wrong completion containing a fitted phrase that is not enriched for correct answers |
| `noise_fp` | FP matched to `fp_hash` (to `fp_tolerance` in chain_sum) | resampled per completion, on answered wrong outputs |
| `noise_fp_trigger` | FP matched to `fp_trigger` | resampled per completion, on every wrong output |
| `fn_behavior` | FN 0.30 | shared: rejects a correct completion containing a fitted phrase that predicts correctness |
| `fn_prompt_subset` | FN 0.30 | prompt-specific: rejects every correct answer on a hashed subset of prompts |
| `noise_fn` | FN matched to `fn_behavior` | resampled per completion |
| `fp_trigger_enriched` | FP 0.15 | shared: accepts a wrong completion containing a fitted phrase that is enriched for correct answers (full menu) |
| `noise_fp_trigger_enriched` | FP matched to `fp_trigger_enriched` | resampled per completion, on every wrong output (full menu) |
| `fn_behavior_depleted` | FN 0.30 | shared: rejects a correct completion containing a fitted phrase that is depleted among correct answers (full menu) |
| `noise_fn_depleted` | FN matched to `fn_behavior_depleted` | resampled per completion (full menu) |

**Natural verifiers and their controls.** `natural` is the domain's natural verifier; `noise_natural` is resampled noise at `natural`'s step-0 channel rates, on the same support. Math adds `natural_strict` and `noise_natural_strict`. Natural verifiers run at their own rates and are never withdrawn for their rate.

**Structural arms.** Chain sum keeps `fp_tolerance`, `fp_tolerance_decoy`, `fn_form`, `noise_fn_form`, `combo_fnform_fphash` and the audited `natural_string_match`, exactly as in `configs/arms/chain_sum.yaml`. Math adds `fp_tolerance` (a fitted absolute tolerance on numeric golds; it is withdrawn by the rate rule if it cannot reach 0.15) and its own control `noise_fp_tolerance` (resampled on answered wrong outputs at `fp_tolerance`'s channel rates), because the kit's `noise_fp` is matched to `fp_hash` there.

**Patch arms (E4).** Each domain with a natural verifier defines 6 to 8 error categories (`task.error_categories`), cheap deterministic predicates on an output, fixed in the task module before sampling. A patched verifier is `natural` except that outputs in its chosen categories are graded by the oracle. Both patch arms fix k = 2 categories:

- `patch_rate` takes the two categories with the most step-0 train-pool errors of `natural` (ties by category order).
- `patch_tau` takes the pair whose patched verifier has the highest `tau_pool` on the audit (ties: fewer errors fixed, then lexicographic).

The selection runs after the features and before the lock, and both selections are in the lock. If they are equal, `patch_rate` is withdrawn (`same_selection`) and the domain leaves H8.

The **full menu** (E1) is every arm above for the domain. The **core menu** (E2, E3) is the kit without the four full-menu arms, plus `natural` and `noise_natural` (the kit alone in chain_sum).

The phrase arms (`fp_trigger`, `fp_trigger_enriched`, `fn_behavior`, `fn_behavior_depleted`) use these fixed rules:

- A phrase is a run of one to three word tokens, matched only on token boundaries. Candidates containing a digit are excluded, because numbers are prompt-specific.
- The preference sign uses the within-prompt association between the phrase and correctness (prompt-centred covariance), so a phrase that only marks a kind of prompt does not count as a behaviour.
- The rate tolerance is 0.02, the withdrawal threshold, applied to the phrase's rate on the full train pool. Candidates are proposed, and the preference sign is estimated, on 7/8 of the train prompts. The sign is then checked on the fixed remaining 1/8 (prompts with `u01("phrase_validation", pid) < 1/8`). `fp_trigger` passes if its validation association is at most 0, or its 90% prompt-bootstrap interval covers 0. `fp_trigger_enriched` and `fn_behavior` pass if their validation association is positive, and `fn_behavior_depleted` if it is negative.

The chain-sum decoy's lower edge is `fp_tolerance`'s fitted tolerance. Its upper edge is the smallest observed error whose pooled FPR is closest to 0.15. The decoy and `fp_tolerance` accept disjoint sets of wrong answers at the same rate, so a difference between their fates is not explained by the rate or by answer formatting.

The calibration report prints, for every arm, the channel rates (what training sees in expectation) and the rates of the one realised audit draw, each with its own Youden J.

### Withdrawals decided from the audit

These are applied after the audit and before the first training step of a cell, and written to `withdrawals.json` and to Deviations:

- A targeted arm (`fp_hash`, `fp_tolerance`, `fp_tolerance_decoy`, `fp_trigger`, `fp_trigger_enriched`, `fn_behavior`, `fn_behavior_depleted`, `fn_prompt_subset`) whose channel rate misses its target by more than 0.02, or cannot be computed, is withdrawn. `fp_tolerance_decoy` is also withdrawn if its band is capped. `fn_form` is capped by design, runs at its achievable rate, and is never withdrawn for its rate.
- A phrase arm is withdrawn if no candidate met the preference inside the rate tolerance (`fell_back`), or if its sign check on the validation prompts fails.
- `patch_rate` is withdrawn when its selection equals `patch_tau`'s (`same_selection`). A patch arm is withdrawn if its base natural verifier has fewer than two categories with a step-0 error.
- Withdrawing an arm withdraws every contrast and pair that uses it.
- Only arms in the cell's menu can be withdrawn. The audit also fits the arms file's other arms (a core cell fits the full-menu arms it does not train); they are listed as outside the menu and are not withdrawals of that cell.

### Models

Model specs are in `configs/models/`, with the Hub revision of every checkpoint pinned.

| Spec | Checkpoint | Mode | Learning rate |
| --- | --- | --- | --- |
| `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b` | `Qwen/Qwen3-*` | non-thinking (`enable_thinking: false`, applied identically in sampling, features, training and evaluation) | 2e-5 up to 4B, 1e-5 from 8B |
| `olmo3-7b` | `allenai/Olmo-3-7B-Instruct-SFT`, the supervised checkpoint before OLMo 3's own preference and RLVR stages | chat (its shipped template, which adds its default system turn when a task has none) | 1e-5 |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | chat (its shipped template, which always writes a system header with the knowledge-cutoff date) | 1e-5 |
| `qwen3-1.7b-base` (X6 only) | `Qwen/Qwen3-1.7B-Base` | the tokenizer's chat template with no `enable_thinking` key, so no think block | 2e-5 |

In every cell the audit generator, the gradient features and the training policy use the same checkpoint and chat template, and training uses the audit's token budget and temperature. The study refuses to run otherwise. X5 trains every parameter (`full_finetune: true`, no adapter) and records `parameterisation: full` in each run; its forecast is still the audit's LoRA screen, so X5 asks whether the fates the screen predicts for an adapter hold for full-parameter training.

### Training protocol (every cell)

GRPO with LoRA (rank 16, alpha 32, all projections, A loaded from the audit), 300 steps of 64 prompts × 8 samples, the model spec's learning rate with a 10-step linear warmup, no KL penalty, group-std advantages, sequence-mean loss (the stop token is part of the loss), one update per batch, no importance-sampling correction. Rollouts are sampled at temperature 1.0 and top-p 1.0. Evaluation every 25 steps on the 384 held-out prompts (4 samples each) and 128 train prompts. TRL 0.25.1 with colocated vLLM; 14B trains on 2 GPUs and 32B on 4 GPUs, with vLLM tensor parallelism and one optimiser step always seeing exactly 512 completions. Versions are pinned in `scripts/setup_gpu.sh` and recorded in every lock and run.

- **Seeds.** A seed fixes the data order and the sampling seeds of evaluation. The LoRA initialisation is the same for every seed. Colocated vLLM rollouts take no per-request seed, so runs are not bit-for-bit reproducible, and the seeds sample that variation.
- **Seed rule (fixed now, applied from P5).** For each E1 domain, if the sample standard deviation of the two P5 oracle seeds' final development accuracies exceeds 0.03, that cell's H1 and H2 arms also run seeds 3 and 4. The seed list is written into the cell config before its audit.
- **Early baseline.** Every trained arm of every cell also runs once with seed 99 for 25 steps. Its held-out accuracy at step 25 is a competitor forecast for H3 (the cheap alternative a reviewer would propose). It is not a confirmatory run.
- **Hardware.** Every run of a cell, its audit and its early baselines run on the cell's hardware class (`configs/program.yaml`), with the same GPU count, micro-batch size and vLLM memory fraction. Training refuses to start on another class. Pooled analyses combine within-cell differences only, so no comparison crosses hardware.
- **Judges.** Judge models (the ifeval content judge, the qa natural verifier) are served by vLLM on their own GPU at temperature 0, and every verdict is cached by model, revision and prompt hash.

## Analysis plan

All confirmatory numbers come from `vfates.analysis.confirmatory` (per cell) and `vfates.analysis.pooled` (across cells) on the finished run directories.

### Per cell

- **Contrasts.** For a contrast `control - arm`, arm seed `s` is paired with control seed `s`. `D[s, p]` is control accuracy minus arm accuracy on held-out prompt `p` at step 300, and the estimate is the mean of `D` over seeds and prompts. The uncertainty comes from a crossed (pigeonhole) bootstrap with 10,000 resamples (Owen, 2007): seeds and prompts are resampled independently, with replacement. This includes training-seed variation, which a prompt-only bootstrap would not.
- **H1 test.** The p-value is the bootstrap proportion of resamples with `diff <= 0.05`. A contrast rejects only if, in addition, every completed seed's contrast (mean over prompts) is positive.
- **H2 test.** The p-value is `max(P*(diff <= -0.05), P*(diff >= 0.05))`.
- **Matched pairs.** For each non-withdrawn pair (arm, control), `dtau` is the locked `tau_pool` of the arm minus that of its control, and `dacc` is the seed-averaged final accuracy of the arm minus that of its control, over the pair's planned seeds. The pairs are listed in each cell config (`analysis.h3.pairs`): every kit arm with its control, every structural arm with its control, and each trained natural verifier with its control (the reported `natural_robust` and `random` arms are not pairs).
- **Baselines for H3 (secondary).** The same within-pair correlation is computed for each baseline, with `rho(tau_pool) - rho(b)` and its crossed-bootstrap interval:
  - step-0 J, which is constant within a pair by construction;
  - mean precision and modal precision;
  - `cos(g_V, g_O)`;
  - the held-out probe AUC;
  - the step-25 accuracy of the seed-99 run;
  - `tau_pool_std`;
  - the behaviour alignment of Egashira et al.: with `b` the arm's behaviour predicate, `f` its frequency on the train audit and `C` the mean prompt-centred oracle advantage of the outputs with `b`, the forecast is `f * C` for a false-positive rule, `-f * C` for a false-negative rule, and 0 for resampled arms;
  - the structure label: -1 for a shared-behaviour rule (a phrase, tolerance, band or form rule, from the arm's type) and 0 for everything else, which is the theory's qualitative reading without the statistic.
- **Cost of the screen and of the step-25 run.** Both are measured in GPU-hours (wall time times GPU count, recorded by the audit and by every run). The audit, including its features, is paid once per cell and serves every arm; the step-25 run is paid once per arm. The report gives, per cell and per stage, the audit's GPU-hours, the mean GPU-hours of a step-25 run, and the number of verifiers at which the audit becomes the cheaper forecast. A missing measurement is reported as missing.
- **H10 per cell.** For each pair `(a, b)`, `acc_a - acc_b` at step 300 with its crossed-bootstrap 95% interval, the locked `dtau`, and `sign(dtau) * (acc_a - acc_b)` with its interval.
- **Natural block (H7).** For each natural pair, the two-sided 95% crossed-bootstrap interval of `noise_natural - natural`, locality by TOST at ±0.05, the screen direction against the locked `dtau` (counted only when the interval excludes 0), and `oracle - natural` for scale (it confounds rate and structure, so it is not the test).
- **Probe (H4):** a ridge probe (penalty 1) of each arm's error indicator, fit on train-audit features and scored on held-out audit rows (`vfates.forecast.probe.heldout_error_probe`).
- **Monitors (H5):** conditional behaviour rates, per seed and averaged, at step 0 and step 300, arm versus control. For false-positive rules this is `P(behaviour | wrong)`; for false-negative rules it is `P(behaviour | correct)`.
- **On-policy rates.** FPR, FNR, J and precision of each arm on its own evaluation samples at every evaluation. For each H1 pair, the paper also reports an adjusted contrast (secondary): the observed contrast minus `b * dJbar`, where `dJbar` is the control-minus-arm difference in on-policy J averaged over the run (trapezoid rule over evaluation steps), and `b` is the least-squares slope of final accuracy on that time-averaged J across all runs of the cell. If H1 rejects and the adjusted gap stays above 0.05, the fate is not explained by the rate path. If it collapses, the honest claim is narrower: fixed rules change the fate by changing the on-policy rate, and `tau_pool` is a step-0 predictor of that divergence.
- **Knife-edge label.** A run whose held-out accuracy rises at least 0.05 above step 0 at some evaluation and later falls at least 0.10 below its running peak is labelled knife-edge in every table. The endpoint stays step 300, so the label cannot change a hypothesis.
- **Sensitivities (reported, not tested):** the prompt-only bootstrap, and a per-seed t interval with S - 1 degrees of freedom.

### Across cells

- **Conclusive cells.** A cell is conclusive when its oracle arm's final held-out accuracy is at least `oracle_gain_min` above its step-0 accuracy: 0.10 in E1, 0.05 in E2 and E3 (larger models start closer to their ceiling). An inconclusive cell's tests are reported and labelled inconclusive. It is excluded from H1-pooled, H3, H3-replication and H10, and a sensitivity analysis includes it.
- **H1.** Holm-Bonferroni over the 12 E1 contrasts, each with its per-cell p-value.
- **H1-pooled.** For each kit contrast, the equal-weight mean over conclusive E1 cells of the per-cell contrast. Its bootstrap resamples seeds and prompts within each cell independently (stratified crossed bootstrap, 10,000 draws); the p-value is the proportion of draws at or below 0. Holm-Bonferroni across the two kit contrasts.
- **H2.** Holm-Bonferroni over the 10 E1 TOST p-values.
- **H3 and H3-replication.** `rho` is the Spearman correlation (average ranks) of `dtau` with `dacc` over every non-withdrawn pair of every conclusive cell of the stage: stage A for H3, stage B for H3-replication. Each holds if all three hold:
  1. The permutation p-value, from 100,000 draws (seed 0) that permute `dacc` among the pairs of the same cell, is below 0.05 (one-sided).
  2. The 5th percentile of `rho` under the crossed bootstrap (each cell's seeds and prompts resampled, forecasts fixed) is above 0.
  3. At least 20 pairs are not withdrawn.

  Stage A plans 63 pairs in 10 cells and stage B 50 pairs in 10 cells. By simulation of the permutation leg (`vfates.analysis.power.pooled_h3_power`, 400 simulations), stage A has power 0.69 at a true within-cell Spearman correlation of 0.3 and 0.93 at 0.4; stage B has 0.60 and 0.83, or 0.42 and 0.64 if the budget removes three of its cells.
- **H7.** Holm-Bonferroni over the 5 E1 natural pairs of the two-sided test that the contrast differs from 0.
- **H8.** The equal-weight mean over eligible domains of `patch_tau - patch_rate` at step 300, with the stratified crossed bootstrap; the one-sided p-value is the proportion of draws at or below 0.
- **H10.** For each pair, the equal-weight mean of the per-cell `sign(dtau) * (acc_a - acc_b)` with the stratified crossed bootstrap (10,000 draws); the one-sided p-value is the proportion of draws at or below 0. Holm-Bonferroni across the two pairs.
- **Displays fixed now.** The paper's central figure is every stage A pair's `dtau` against its `dacc`, natural pairs marked; the baseline table reports each forecast's `rho`, its difference from `rho(tau_pool)` with interval, and its cost; the natural-verifier figure places each natural verifier's step-0 error rate against its locked `dtau` and its `dacc`. All are generated from the pooled report (`vfates.paper`).
- **H9, scale, H3 by stratum.** As defined under Hypotheses, with the same bootstraps.
- **Exclusions during training.** A run whose loss or gradient becomes non-finite restarts once from its last checkpoint with the same seed; a second failure withdraws that arm. A preempted run resumes from its last checkpoint and is not an exclusion. An arm with fewer than its planned completed seeds is withdrawn, not analysed on the remainder. No run is excluded for its outcome.
- **What a null means.** If E1's cells are conclusive and every non-withdrawn H1 contrast fails, these shared rules do not move terminal held-out accuracy by more than 0.05 relative to rate-matched resampled noise, on these domains at 1.7B. The paper reports that as the finding. If H3 fails, the screen does not order fates beyond the rate at this power. If H3 passes and H10 fails, the screen orders fates across structures but does not read the direction of a shared behaviour, and the paper says so. If H3 passes and H3-replication fails, the screen's reach is stated as the stage A models. The thresholds are not revisited.

## Go / no-go from the pilots

The pilots decide feasibility and the settings this document delegates to them, never the confirmatory question. Each pilot's rules are fixed in its README before it runs.

- **P0 (theory)** and **P1 (harness)** pass: the identities hold on exactly solvable worlds, and every stage of every domain, the patch selection and the program runner compute what they claim on a small model.
- **P2 (base audit)** chooses each domain's difficulty bucket on the development pool with `qwen3-1.7b` and `qwen3-4b`: the bucket whose held-out accuracy is inside [0.15, 0.55] for both, with truncation at most 0.05 at the domain's token budget and answer-format rate at least 0.75. Ties go to the bucket whose mean accuracy over the two models is closest to 0.35. If no bucket qualifies, the domain's backup source is screened by the same rule; if that fails too, or the domain has no backup, the domain leaves the program and the paper says so. P2 also confirms the token budget: if truncation exceeds 0.05, the budget rises to the next of 1,024, 1,536 and 2,048 and the screen repeats.
- **P3 (estimator)** passes in every domain at `qwen3-1.7b` on the development pool: split-half reliability of `g_V` at least 0.8 for every arm (P3.2, P3.3), sketch fidelity (P3.4), bfloat16 agreement with float32 features (P3.11), and the lock (P3.8).
- **P4 (micro-GRPO)** passes P4.1 in every domain: on the development pool, the oracle arm of `qwen3-1.7b` raises held-out accuracy by at least 0.10 in 150 steps. A domain that fails is rescreened once with its backup source, if it has one; a second failure, or a failure with no backup, removes it from the program.
- **P5 (GPU gate)** passes for every model and hardware class in the program: a 10-step training and evaluation run completes on each cell's placement, 14B and 32B with tensor parallelism and a reloaded adapter (P5.1); a run preempted by `SIGUSR1` resumes to the same final step on each hardware class (P5.2); and each model's oracle arm gains at least 0.05 held-out accuracy on the math development pool within 150 steps at the spec's learning rate (P5.3). P5 also runs oracle seeds 0 and 1 of every E1 domain with the full E1 protocol on the development pool, which applies the seed rule (P5.4), and measures throughput and hours per arm-seed for every model and hardware class into `configs/throughput.yaml` (P5.5).
- **P6 (natural verifiers)** documents each natural verifier's step-0 rates with intervals, the patch categories' error counts, whether the two patch selections differ, the judge agreement checks, and the human adjudication of 100 step-0 false positives per natural verifier. It gates nothing.

Every pilot runs on development pools (`audit.pools.from: dev`), whose locks record `pool_source: dev`; the study refuses such an audit for a confirmatory cell.

The pilots' decodability probes and fate contrasts are logged and do not decide whether the study runs. `pilots/README.md` has the order and the commands.

## Software and compute

The GPU environment is built by `scripts/setup_gpu.sh`. The pinned versions of TRL, vLLM, PyTorch, Transformers, PEFT, math-verify and Reasoning Gym are listed there, and the environment report inside every lock and run records what actually ran. `vfates program plan` prints the projected GPU hours of every job by hardware class and by cell, in H100-equivalent hours, in each class's own GPU-hours and in rented dollars. P5 measures generation and training throughput for every model and hardware class, writes them to `configs/throughput.yaml`, and the plan is recomputed from those rates before any cell is audited.

The budget is lab GPUs (RTX A6000, RTX 5000 Ada, A100) for E1 and the 4B cells, rented H100s for the 8B, 14B and 32B cells and E3 (about $8,000 to $10,000), and about $500 of API calls for the judge checks. If the plan from measured rates (`vfates program plan --budget-usd`) exceeds the budget, stage B cells leave the program in this order before any stage B audit, until it fits: `e2-math-qwen3-32b`, the E3 code cells, `e2-code-qwen3-14b`, `e2-qa-qwen3-8b`. No stage A cell can leave for budget. A cell that leaves is reported as not run; it is never partially analysed. Exploratory runs are scheduled after every confirmatory job of their class.

## Deviations

Logged here with date and rationale once the document is frozen.
