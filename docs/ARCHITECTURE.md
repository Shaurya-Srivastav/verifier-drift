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
                                   forecast: confusion, tau_pool, probe ──► lock.json
                                                                                 │
 experiment.train_arm ──► train (MLX in-loop eval | TRL + checkpoint_eval)       │
                               │ eval.jsonl, eval_prompts.jsonl, metrics.jsonl   │
                               ▼                                                 ▼
                        analysis.outcomes: contrasts, rank bootstrap, score_lock
```

A study is two commands: an **audit** (sample the base policy on frozen pools, grade, fit the arms on the train pool, extract features, forecast, lock) and **training** (one run per arm and seed on the audit's pools, then scoring against the lock). `vfates study` does both from one config; the pilots in `pilots/` are staged versions of the same pipeline with gates.

## Package map

### Entry points

| Module | Public API | Role |
| --- | --- | --- |
| `vfates.cli` | `main(argv)`, `PILOTS` | `vfates {env, audit, train, score, study, pilot}` |
| `vfates.audit` | `run_audit(config, out_dir) -> AuditRun`, `load_audit(out_dir)`, `load_pools`, `sample_completions`, `grade_completions`, `apply_arms`, `build_table`, `confusion_report`, `extract_features`, `forecast_arms`, `resolve_arms`, `count_completions`, `load_completions` | The step-0 pipeline. Every stage is resumable and writes its artefact before the next starts. |
| `vfates.experiment` | `train_arm(audit_dir, arm, train_config, out_dir, *, backend, resume, max_train, max_heldout, eval_generator)` | One GRPO run of one arm on an audit's frozen pools. |

### Data and utilities

| Module | Public API |
| --- | --- |
| `vfates.types` | `Problem`, `GenOut`, `Completion`, `Parsed`, `DrawKey`, `AuditTable` (columnar audit with per-arm verdicts; `save` / `load` as `.npz` + `.meta.json`), `SPLITS` |
| `vfates.utils.io` | `write_json`, `read_json`, `write_jsonl`, `append_jsonl`, `read_jsonl`, `iter_jsonl`, `read_yaml`, `write_yaml`, `canonical_json`, `sha256_text`, `sha256_file`; writes are atomic |
| `vfates.utils.hashing` | `u01(*parts)`, `stable_u64(*parts)`, `stable_seed(*parts)`: BLAKE2b, identical on every machine; every fixed pseudo-random decision goes through these |
| `vfates.utils.stats` | `rankdata`, `pearson`, `spearman`, `group_indices`, `cluster_bootstrap` (resamples prompts), `mean_ci`, `paired_diff_ci` |
| `vfates.utils.env` | `environment_report()`: chip, memory, accelerator, package versions, git commit |
| `vfates.utils.seeding` | `seed_everything(seed)` |

### Tasks (`vfates.tasks`)

| Module | Public API |
| --- | --- |
| `base` | `Task`: `problems(split, n, seed)`, `prompt(problem)`, `system_prompt()`, `parse(problem, text) -> Parsed`, `oracle(problem, parsed, text) -> int` |
| `registry` | `build_task(spec)` for `type` in `TASK_TYPES` (`chain_sum`, `mbpp_plus`, `ifeval`) |
| `chain_sum` | `ChainSumTask`, `ChainSumConfig`, `FORM_RULES`, `builtin_items`, `abs_error`. Reasoning Gym `decimal_chain_sum` (Egashira et al. configuration by default) with a bit-identical built-in generator; held-out uses dataset seed + 10,000,000. Oracle: exact `Decimal` equality of the last `\boxed{}` with the gold answer. |
| `answer` | `extract_boxed`, `parse_decimal`, `canonical_decimal`, `decimal_parts` |
| `mbpp` | `MBPPPlusTask`: MBPP+ v0.2.0; oracle = base and extra tests pass |
| `ifeval`, `if_checkers` | `IFEvalTask` on `allenai/RLVR-IFeval`; the 24 constraint checkers ported from open-instruct |
| `judges` | `Judge`, `HeuristicJudge`, `LLMJudge`, `LocalJudge`, `CachedJudge` (oracle side of IFEval) |
| `sandbox` | `run_python`: isolated subprocess with time and memory limits |

### Verifiers (`vfates.verifiers`)

| Module | Public API |
| --- | --- |
| `base` | `Verifier.verdict(problem, parsed, text, o, key=None) -> int`, `describe()`; `kind`, `stochastic` |
| `arms` | `OracleVerifier`; FP rules `HashFP` (prompt-specific), `NumericToleranceFP`, `TriggerFP` (shared); FN rules `FormFN`, `BehaviorFN` (shared), `PromptSubsetFN` (prompt-specific); `ResampledNoise(rho0, rho1, seed, support)` with draws keyed by `(pid, k)`; `Composite`; `StringMatchVerifier` |
| `calibrate` | `CalibrationData.from_grades`, `calibrate_arms(task, data, specs, seed, n_boot, targets) -> CalibrationResult`: fits every arm on the base policy's train-pool samples to the target pooled rate and reports achieved rates with prompt-clustered intervals; resampled arms copy a matched arm's rates |
| `registry` | `build_verifier(spec, task)`, `build_arms(specs, task)`; a calibrated arm's `describe()` rebuilds it exactly |
| `monitors` | `rule_monitors(arms)`: the behaviour indicator of every rule arm (plus `has_answer`), logged through training |
| `natural` | `MBPPBaseTestsVerifier`, `IFEvalCheckerVerifier` |

### Generation (`vfates.gen`)

| Module | Public API |
| --- | --- |
| `base` | `Generator.generate(prompts, n, *, max_tokens, temperature, top_p, seed, system_prompt) -> list[list[GenOut]]`, `describe()` |
| `chat` | `encode_prompt`, `encode_completion`, `encode_prompt_completion` (the chat template every backend and feature extractor shares) |
| `factory` | `make_generator(backend, model_id, **kwargs)` for `mlx`, `hf`, `vllm` |
| `mlx_backend` | `MLXGenerator` (Apple silicon, batched; per-token log-probabilities opt-in with `logprobs=True`) |
| `hf_backend` | `HFGenerator` (MPS / CUDA / CPU); `set_adapter(path)` loads a PEFT checkpoint |
| `vllm_backend` | `VLLMGenerator` (CUDA); `set_adapter(path)` routes through a LoRA request |

### Features (`vfates.features`)

| Module | Public API |
| --- | --- |
| `lora_grad` | `lora_grad_features(model_or_id, tokenizer, prompts, completions, *, sketch_dim, lora_r, lora_alpha, target_modules, lora_seed, sketch_seed, ...) -> LoraGradFeatures`: per-sample gradient of the completion log-likelihood with respect to the LoRA B matrices at initialisation (the directions GRPO's first step moves), CountSketch-projected |
| `sketch` | `CountSketch(dim, seed)`: unbiased inner-product sketch |
| `hidden` | `load_model(model_id, device, dtype)`, `hidden_features(...)`: pooled final-layer states (ablation) |

### Forecasts (`vfates.forecast`)

| Module | Public API |
| --- | --- |
| `confusion` | `confusion`, `confusion_with_ci`: pooled and prompt-averaged FPR, FNR, J, precision; `mean_precision_forecast` (the tilt limit), `modal_precision_forecast` (the zero-KL policy-gradient limit) |
| `transfer` | `prompt_centered`, `velocity_direction`, `velocities`, `tau_pool` (primary, `PRIMARY_Q = 1/32`), `tau_pool_interval` (two-level prompt bootstrap), `tau_hard`, `tau_soft`, `first_order_slope`, `split_half_reliability`, `transfer_forecast -> TransferForecast` |
| `probe` | `auc`, `grouped_folds`, `ridge_scores`, `probe_auc`, `arm_error_probe`: is an arm's error decodable from step-0 features on unseen prompts? |
| `lockbox` | `lock_forecasts(payload, path, *, config, note)`, `verify_lock(path)`, `load_lock(path)`; schema `vfates.lock/1` |

### Training (`vfates.train`)

| Module | Public API |
| --- | --- |
| `config` | `GRPOConfig`, `LoRAConfig` (`mlx_keys`, `peft_keys`, `scale`): one config for both backends |
| `grpo_core` | numpy reference for advantages (`std`, `mean`), loss aggregation (`seq_mean`, `token_mean`, `dr_constant`), the clipped surrogate and k3 KL; both trainers are tested against it |
| `mlx_grpo` | `MLXGRPOTrainer(config, task, verifier, train_problems, heldout_problems, monitors).run(resume)`: LoRA GRPO on Apple silicon with in-loop held-out evaluation, checkpoints and exact resume |
| `trl_grpo` | `build_trl_trainer(...)`: the same objective with TRL 0.25 + vLLM on CUDA; oracle, FP and FN are zero-weight reward heads |
| `checkpoint_eval` | `peft_checkpoints(run_dir)`, `evaluate_checkpoints(...)`: the endpoint for TRL runs, from saved adapters |
| `evaluate` | `evaluate_policy(generator, task, problems, *, n_samples, ...)`, `eval_log_rows`, `EVAL_ROW_KEYS` |
| `rewards` | `RewardContext`, `onpolicy_rates`, `make_trl_reward_funcs` |

### Analysis (`vfates.analysis.outcomes`)

`eval_curve`, `prompt_accuracy`, `run_outcome`, `paired_contrast`, `paired_contrast_seeds` (prompt-paired, seed-averaged, bootstrap; one-sided and TOST p-values), `holm`, `forecast_row`, `score_forecasts`, `rank_bootstrap` (Spearman intervals from resampling held-out prompts), `score_lock`.

### Theory (`vfates.theory`)

`worlds` (`World`, `hand_featured_world`, `random_feature_world`, `knife_edge_world`, `verifier_matrix`, `step0_rates`, `matched_resampled`, `ResampledSpec`), `dynamics` (`population_flow`, `sampled_grpo`), `forecasts` (`exact_tau`, `sampled_tau`, `observed_heldout_accuracy`), `identities` (the `check_*` functions for the pathwise decomposition, the resampled-noise scaling, tilt invariance and the policy-gradient limits).

### Pilots (`vfates.pilots`)

`common` (`Checks`, `write_report`, `markdown_table`, `cli`, `apply_override`, `main_guard`) and one module per pilot, `p0_theory` through `p6_natural`, each exposing `run(config, out_dir) -> report`. `pilots/<name>/run.py` is a three-line entry point.

## Artefacts

| File | Written by | Contents |
| --- | --- | --- |
| `audit/config.json` | `run_audit` | the audit config and `environment_report()` |
| `audit/problems.jsonl`, `completions.jsonl` | `run_audit` | frozen pools; one row per sample (`pid`, `k`, text, tokens, finish reason) |
| `audit/calibration.json` | `calibrate_arms` | every arm's fitted spec, achieved rates, intervals |
| `audit/audit.npz` (+ `.meta.json`) | `build_table` | `pid`, `split`, `k`, `O`, and `V` for every arm |
| `audit/features.npz`, `features.meta.json` | `extract_features` | step-0 features and completion lengths |
| `audit/forecasts.json`, `lock.json` | `forecast_arms`, `lock_forecasts` | forecasts per arm; the sealed copy |
| `run/metrics.jsonl` | trainers | one row per optimiser step |
| `run/eval.jsonl`, `eval_prompts.jsonl` | MLX trainer, `evaluate_checkpoints` | the endpoint: accuracy per split and per prompt at every evaluation |
| `run/summary.json`, `run.json` | trainers, `train_arm` | completion status; the arm and audit |
| `pilots/*/results/report.{md,json}` | `write_report` | checks, tables, config, environment |

## Conventions

- Error rates are pooled over samples and measured on the base policy's own samples of the frozen train pool (the step-0 on-policy confusion matrix). Prompt-averaged variants are reported alongside.
- Held-out prompts are never used to fit, choose or tune anything.
- Stochastic verdicts are keyed by `(pid, k)` in the audit and by `(seed, "eval", pid, k)` in evaluation, so a draw is reproducible and never reused across contexts.
- Every artefact records the config and environment that produced it. Anything a preregistered comparison uses is locked before the corresponding training starts.
- Heavy dependencies (torch, MLX, vLLM, TRL) are imported inside the functions that need them, so the core package, the theory and the analysis install and test without them.
