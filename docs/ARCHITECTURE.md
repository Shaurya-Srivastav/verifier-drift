# Architecture

The code map and the contracts between modules. Every public name below exists in `src/vfates` and is exercised by `tests/`.

## Data flow

```
configs/*.yaml
     │
     ▼
 tasks ──problems──► gen (MLX / HF / vLLM) ──completions──► audit ──────────────┐
  │ parse, oracle O                                          │ O, V per arm      │
  ▼                                                          ▼                   │
 verifiers ──calibrated on the train pool──────────► AuditTable (audit.npz)      │
                                                             │                   │
 features (LoRA-B gradients, CountSketch) ──────────────────►│                   │
                                                             ▼                   │
                     forecast: confusion, tau_pool, probe, baselines ──► lock.json (+ artifact hashes)
                                                                                 │
 study: withdrawals.json (from calibration validity, before any training)        │
                                                                                 │
 experiment.train_arm ──► train (MLX in-loop eval | TRL + checkpoint_eval)       │
                               │ eval.jsonl, eval_prompts.jsonl, metrics.jsonl   │
                               │ eval_samples.jsonl when the task has a judge    │
                               ▼                                                 ▼
          analysis.confirmatory (H1 to H6) on analysis.outcomes ──► report.json, report.md
```

A study is two stages: an **audit** (sample the base policy on frozen pools, grade, fit the arms on the train pool, extract features, forecast, lock) and **training** (one run per arm and seed on the audit's pools, then the preregistered analysis against the lock). `vfates study` runs both from one config: it audits, records withdrawals, trains every planned run that is not finished, and writes the confirmatory report once all planned runs are complete. The pilots in `pilots/` are staged versions of the same pipeline with gates.

## Package map

### Entry points

| Module | Public API | Role |
| --- | --- | --- |
| `vfates.cli` | `main(argv)`, `PILOTS` | `vfates {env, audit, train, score, study, pilot}` |
| `vfates.audit` | `run_audit(config, out_dir) -> AuditRun`, `load_audit(out_dir)`, `stage_fingerprints(config)`, `load_pools`, `sample_completions`, `grade_completions`, `apply_arms`, `build_table`, `confusion_report`, `extract_features`, `forecast_arms`, `behaviour_indicators`, `behaviour_alignment`, `resolve_arms`, `read_lock_record`, `artifact_sha256`, `grades_sha256`, `count_completions`, `load_completions` | The step-0 pipeline. Every stage is resumable, fingerprinted in `stages.json`, and writes its artefact before the next starts. A changed fingerprint recomputes that stage and everything downstream; under an existing lock it is an error. |
| `vfates.experiment` | `train_arm(audit_dir, arm, train_config, out_dir, *, backend, resume, max_train, max_heldout, eval_generator)` | One GRPO run of one arm on an audit's frozen pools. Refuses to train unless the lock verifies, its artifact hashes match the audit files, the current parser and oracle reproduce the locked grades, and the run starts from the audit's saved LoRA init. |
| `vfates.study` | `run_study(config, out_dir)`, `reanalyze(config, out_dir)`, `planned_seeds(study)` | The whole preregistered study: audit, lock, withdrawals, every planned run (plus the early-baseline runs), then the confirmatory report. When `analysis.agreement_check` is set and every planned run finished, it also writes the reported judge-agreement check. |

### Data and utilities

| Module | Public API |
| --- | --- |
| `vfates.types` | `Problem`, `GenOut`, `Completion` (both carry the sampled `token_ids`, stop token included), `Parsed`, `AuditTable` (columnar audit with per-arm verdicts; atomic `save` / `load` as `.npz` + `.meta.json`), `SPLITS` |
| `vfates.utils.io` | `write_json`, `read_json`, `write_jsonl`, `append_jsonl`, `read_jsonl`, `iter_jsonl`, `atomic_npz`, `read_yaml`, `write_yaml`, `canonical_json`, `sha256_text`, `sha256_file`; writes are atomic |
| `vfates.utils.hashing` | `u01(*parts)`, `stable_u64(*parts)`, `stable_seed(*parts)`: BLAKE2b over parts joined by U+001F (a part containing it is rejected), identical on every machine; every fixed pseudo-random decision goes through these |
| `vfates.utils.stats` | `GroupIds`, `rankdata`, `pearson`, `spearman`, `group_indices`, `cluster_bootstrap` (resamples prompts), `mean_ci`, `paired_diff_ci` |
| `vfates.utils.env` | `environment_report()`: chip, memory, accelerator, package versions, git commit |
| `vfates.utils.seeding` | `seed_everything(seed)` |

### Tasks (`vfates.tasks`)

| Module | Public API |
| --- | --- |
| `base` | `Task`: `problems(split, n, seed)`, `prompt(problem)`, `system_prompt()`, `parse(problem, text) -> Parsed`, `oracle(problem, parsed, text) -> int` |
| `registry` | `build_task(spec)` for `type` in `TASK_TYPES` (`chain_sum`, `mbpp_plus`, `ifeval`) |
| `chain_sum` | `ChainSumTask`, `ChainSumConfig`, `FORM_RULES`, `builtin_items`, `abs_error`. Reasoning Gym `decimal_chain_sum` (Egashira et al. configuration by default) with a bit-identical built-in generator; held-out uses dataset seed + 10,000,000. Oracle: exact `Decimal` equality of the last `\boxed{}` or `\fbox{}` with the gold answer, under the normalisation rules listed in `PREREGISTRATION.md`. |
| `answer` | `extract_boxed`, `parse_decimal`, `canonical_decimal`, `decimal_parts` |
| `mbpp` | `MBPPPlusTask`: MBPP+ v0.2.0; oracle = base and extra tests pass |
| `ifeval`, `if_checkers` | `IFEvalTask` on `allenai/RLVR-IFeval`; the constraint checkers ported from open-instruct |
| `judges` | `Judge`, `HeuristicJudge`, `LLMJudge`, `LocalJudge`, `CachedJudge`, `judge_result` (verdict and whether a YES/NO reply parsed), `parse_yes_no` (oracle side of IFEval) |
| `sandbox` | `run_python`, `run_many`: subprocess with time, memory, output and process-count limits (see `SECURITY.md` for what it does not isolate) |

### Verifiers (`vfates.verifiers`)

| Module | Public API |
| --- | --- |
| `base` | `Verifier.verdict(problem, parsed, text, o, key=None) -> int`, `describe()`; `kind`, `stochastic` |
| `arms` | `OracleVerifier`; FP rules `HashFP` (prompt-specific), `NumericToleranceFP`, `TriggerFP` (shared), `ToleranceBandFP` (the decoy: accepts errors in a band away from zero); FN rules `FormFN`, `BehaviorFN` (shared), `PromptSubsetFN` (prompt-specific); `ResampledNoise(rho0, rho1, seed, support)` with draws keyed by `(pid, k)`; `Composite`; `StringMatchVerifier`. Phrases match whole tokens (`WORD_RE`). |
| `calibrate` | `CalibrationData.from_grades`, `calibrate_arms(task, data, specs, seed, n_boot, targets) -> CalibrationResult`, `arm_validity(report)`: fits every arm on the base policy's train-pool samples to the target pooled rate and reports achieved rates with prompt-clustered intervals; resampled arms copy a matched arm's rates. Phrase arms are rate-matched on the full train pool, take their sign from the within-prompt association on 7/8 of the train prompts, and are validated on the fixed remaining 1/8. Schema `vfates.calibration/2`. |
| `registry` | `build_verifier(spec, task)`, `build_arms(specs, task)`; a calibrated arm's `describe()` rebuilds it exactly |
| `monitors` | `rule_monitors(arms)`: the behaviour indicator of every rule arm (plus `has_answer`), logged through training |
| `natural` | `MBPPBaseTestsVerifier`, `IFEvalCheckerVerifier` |

### Generation (`vfates.gen`)

| Module | Public API |
| --- | --- |
| `base` | `Generator.generate(prompts, n, *, max_tokens, temperature, top_p, seed, system_prompt) -> list[list[GenOut]]`, `describe()` |
| `chat` | `encode_prompt`, `encode_completion`, `encode_prompt_completion`, `prompt_completion_ids` (the chat template every backend and feature extractor shares; the last takes sampled ids so features see exactly the tokens the policy produced), `check_generate_args` |
| `factory` | `make_generator(backend, model_id, **kwargs)` for `mlx`, `hf`, `vllm` |
| `mlx_backend` | `MLXGenerator` (Apple silicon, batched; per-token log-probabilities opt-in with `logprobs=True`) |
| `hf_backend` | `HFGenerator` (MPS / CUDA / CPU); `set_adapter(path)` loads a PEFT checkpoint |
| `vllm_backend` | `VLLMGenerator` (CUDA); `set_adapter(path)` routes through a LoRA request |

### Features (`vfates.features`)

| Module | Public API |
| --- | --- |
| `lora_grad` | `lora_grad_features(model_or_id, tokenizer, prompts, completions, *, sketch_dim, lora_r, lora_alpha, target_modules, lora_seed, sketch_seed, completion_ids, save_init, ...) -> LoraGradFeatures`: per-sample gradient of the completion log-likelihood with respect to the LoRA B matrices at initialisation (the directions GRPO's first step moves), CountSketch-projected |
| `lora_init` | `LORA_INIT_FILENAME`, `lora_a_state`, `assert_lora_a_equal`, `save_lora_init`, `load_lora_init`: the A matrices the features used, saved once (`vfates.lora_init/1`, keys are base `nn.Linear` paths, shape `(r, in)`) and loaded by both trainers so every run starts from them |
| `sketch` | `CountSketch(dim, seed)`: unbiased inner-product sketch |
| `hidden` | `load_model(model_id, device, dtype)`, `hidden_features(..., completion_ids, pad_multiple)`: pooled final-layer states (ablation); `pad_multiple` bounds the number of batch shapes, which MPS caches graph memory for |

### Forecasts (`vfates.forecast`)

| Module | Public API |
| --- | --- |
| `confusion` | `confusion`, `confusion_with_ci`: pooled and prompt-averaged FPR, FNR, J, precision; `mean_precision_forecast` (the tilt limit), `modal_precision_forecast` (the zero-KL policy-gradient limit) |
| `transfer` | `length_division(normalize, loss_agg)`, `prompt_centered`, `group_std_advantages`, `velocity_direction`, `velocities`, `tau_pool` (primary, `PRIMARY_Q = 1/32`), `tau_pool_interval` (two-level prompt bootstrap), `tau_hard`, `tau_soft`, `first_order_slope`, `cosine`, `split_half_reliability`, `reliability_budget` (signal and noise of `g_V`, with the within-prompt part as a jackknife), `predicted_reliability`, `prompts_for_reliability`, `transfer_forecast -> TransferForecast` (includes the secondary `tau_pool_std`, which uses TRL's per-prompt standard-deviation scaling) |
| `probe` | `auc`, `grouped_folds`, `ridge_scores`, `probe_auc`, `arm_error_probe`, `heldout_error_probe`: is an arm's error decodable from step-0 features on unseen prompts? The locked value is `heldout_error_probe` (fit on train prompts, scored on held-out prompts) |
| `lockbox` | `lock_forecasts(payload, path, *, config, note, artifacts)`, `verify_lock(path)`, `load_lock(path)`, `load_lock_record(path)`, `assert_matches_lock(path, payload)`; writes schema `vfates.lock/2` (payload, artifact hashes, digest) and reads `/1` |

### Training (`vfates.train`)

| Module | Public API |
| --- | --- |
| `config` | `GRPOConfig` (`lora_seed`, `lora_init`, `lora_init_sha256`), `LoRAConfig` (`mlx_keys`, `peft_keys`, `scale`), `NonFiniteLossError`, `raise_if_nonfinite`: one config for both backends |
| `grpo_core` | numpy reference for advantages (`std`, `mean`), loss aggregation (`seq_mean`, `token_mean`, `dr_constant`), the clipped surrogate and k3 KL; both trainers are tested against it |
| `mlx_grpo` | `MLXGRPOTrainer(config, task, verifier, train_problems, heldout_problems, monitors).run(resume)`, `constant_with_warmup_lr`: LoRA GRPO on Apple silicon with in-loop held-out evaluation, checkpoints and exact resume; the warmup schedule matches Hugging Face `constant_with_warmup` (the first update has learning rate 0) |
| `trl_grpo` | `build_trl_trainer(...)`, `trl_grpo_kwargs`, `trl_run_metadata`, `peft_lora_matrices`: the same objective with TRL 0.25 + vLLM on CUDA; oracle, FP and FN are zero-weight reward heads; float32 master weights with bf16 autocast; vLLM rollouts are not seeded per request, which `run_metadata.json` records |
| `checkpoint_eval` | `peft_checkpoints(run_dir)`, `evaluate_checkpoints(...)`: the endpoint for TRL runs, from saved adapters; tasks with a content judge also append `eval_samples.jsonl` |
| `evaluate` | `evaluate_policy(generator, task, problems, *, n_samples, ...)`, `eval_log_rows`, `eval_sample_rows`, `append_eval_samples`, `EVAL_ROW_KEYS`, `EVAL_SAMPLES_FILENAME` |
| `rewards` | `RewardContext`, `onpolicy_rates`, `make_trl_reward_funcs` |

### Analysis (`vfates.analysis`)

| Module | Public API |
| --- | --- |
| `outcomes` | `eval_curve`, `prompt_accuracy`, `run_outcome`, `paired_contrast`, `paired_contrast_seeds` (prompt-paired, seed-averaged, bootstrap; one-sided and TOST p-values), `holm`, `forecast_row`, `score_forecasts`, `rank_bootstrap` (Spearman intervals from resampling held-out prompts), `score_lock` |
| `confirmatory` | `audit_withdrawal_reasons`, `decide_audit_withdrawals` (from `arm_validity`, before any training), `confirmatory_report` (H1 to H6 with the crossed prompt and seed bootstrap, Holm, TOST, the within-pair rank test, sensitivities and labels; with `analysis.natural`, E2's tests of a natural verifier against its matched control: the two-sided interval, TOST locality, the screen direction against the locked `tau_pool` difference, and `oracle - natural` for scale), `render_markdown` |
| `power` | `contrast_se`, `superiority_power`, `tost_power`, `minimum_detectable_difference`, `token_projection`, `project_measured_cost`: the seed rule and the compute budget from P5's measurements |
| `judge_agreement` | `sampling_frame`, `draw_subsample`, `agreement_statistics`, `load_heldout_frame`, `run_judge_agreement`, `render_agreement_markdown`: reported (not tested) agreement of `gpt-4.1` with the IFEval content judge on a fixed subsample of final-step held-out eval samples |

### Theory (`vfates.theory`)

`worlds` (`World`, `hand_featured_world`, `random_feature_world`, `knife_edge_world`, `verifier_matrix`, `step0_rates`, `matched_resampled`, `ResampledSpec`), `dynamics` (`population_flow`, `sampled_grpo`), `forecasts` (`exact_tau`, `sampled_tau`, `observed_heldout_accuracy`), `identities` (the `check_*` functions for the pathwise decomposition, the resampled-noise scaling, tilt invariance and the policy-gradient limits).

### Pilots (`vfates.pilots`)

`common` (`Checks`, `write_report`, `markdown_table`, `cli`, `apply_override`, `main_guard`) and one module per pilot, `p0_theory` through `p6_natural`, each exposing `run(config, out_dir) -> report`. `pilots/<name>/run.py` is a three-line entry point.

## Artefacts

| File | Written by | Contents |
| --- | --- | --- |
| `audit/config.json` | `run_audit` | the audit config and `environment_report()` |
| `audit/stages.json` | `run_audit` | the fingerprint of every completed stage (`vfates.stages/1`) and `grades`, the hash of how the current parser and oracle grade the completions |
| `audit/problems.jsonl`, `completions.jsonl` | `run_audit` | frozen pools; one row per sample (`pid`, `k`, text, sampled `token_ids`, token count, finish reason) |
| `audit/calibration.json` | `calibrate_arms` | every arm's fitted spec, achieved rates, intervals, validity |
| `audit/audit.npz` (+ `.meta.json`), `confusion.json` | `build_table`, `confusion_report` | `pid`, `split`, `k`, `O`, and `V` for every arm; step-0 confusion per arm and split |
| `audit/features.npz`, `features.meta.json` | `extract_features` | step-0 features, completion lengths, and whether rows used sampled token ids |
| `audit/lora_init.safetensors` | `lora_grad_features` | the LoRA A matrices of the features, which every run starts from |
| `audit/forecasts.json`, `lock.json` | `forecast_arms`, `lock_forecasts` | forecasts and baselines per arm; the sealed copy with the sha256 of `problems.jsonl`, `completions.jsonl`, `calibration.json`, `features.npz`, `lora_init.safetensors`, the arms file and the grades |
| `study/withdrawals.json` | `run_study` | arms withdrawn by the preregistered audit rules, decided before training |
| `study/{runs,early}/<arm>__s<seed>/` | `train_arm` | one directory per planned run and per early-baseline run |
| `run/metrics.jsonl` | trainers | one row per optimiser step |
| `run/eval.jsonl`, `eval_prompts.jsonl` | MLX trainer, `evaluate_checkpoints` | the endpoint: accuracy per split and per prompt at every evaluation |
| `run/eval_samples.jsonl` | `append_eval_samples` | one row per sample when the task records a content judge (`pid`, `k`, prompt, text, judge bit, whether the reply parsed, oracle bit). Absent for tasks with no judge. A rerun keeps the last row for each `(step, split, pid, k)` |
| `run/run.json` | `train_arm` | the arm, audit, lock sha256, LoRA init sha256, git commit and environment |
| `run/run_metadata.json` | `build_trl_trainer` | `lora_seed`, `lora_init_sha256`, `rollout_seeded` |
| `run/summary.json` | trainers | completion status (TRL runs copy `run_metadata.json`'s fields) |
| `study/report.json`, `report.md` | `run_study`, `reanalyze` | the confirmatory analysis, plus `judge_agreement` when that check ran |
| `study/judge_agreement.json`, `judge_agreement_cache.jsonl` | `run_judge_agreement` | the E2 judge-agreement record (reported, not tested) and the cached remote replies. A rerun reads the cache and does not call the API again |
| `pilots/*/results/report.{md,json}` | `write_report` | checks, tables, config, environment |

## Conventions

- Error rates are pooled over samples and measured on the base policy's own samples of the frozen train pool (the step-0 on-policy confusion matrix). Prompt-averaged variants are reported alongside.
- Held-out prompts are never used to fit, choose or tune anything.
- Stochastic verdicts are keyed by `(pid, k)` in the audit and by `(seed, "eval", pid, k)` in evaluation, so a draw is reproducible and never reused across contexts.
- Every artefact records the config and environment that produced it. Anything a preregistered comparison uses is locked before the corresponding training starts.
- Heavy dependencies (torch, MLX, vLLM, TRL) are imported inside the functions that need them, so the core package, the theory and the analysis install and test without them.
