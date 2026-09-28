# Architecture

The code map and the contracts between modules. Every public name below exists in `src/vfates` and is exercised by `tests/`.

## Data flow

```
configs/program.yaml ── configs/templates/<domain>.yaml + configs/models/<model>.yaml
        │                          │
        │               vfates program render
        ▼                          ▼
 vfates program plan ──► queue.jsonl ──► workers (local, SSH, Slurm) ──► vfates study (one job)
                                                                           │
 tasks ──problems──► gen (MLX / HF / vLLM) ──completions──► audit ─────────┤
  │ parse, oracle O, categorize                              │ O, V per arm │
  ▼                                                          ▼              │
 verifiers ──calibrated on the train pool──────────► AuditTable (audit.npz) │
                                                             │              │
 features (LoRA-B gradients, CountSketch) ──────────────────►│              │
                                                             ▼              │
            patch selection (tau or rate) ─► forecast: tau_pool, budgets, baselines ─► lock.json
                                                                            │
 study: withdrawals.json (from calibration validity and patch selection, before any training)
                                                                            │
 experiment.train_arm ──► train (MLX in-loop eval | TRL + vLLM + checkpoint_eval)
                               │ eval.jsonl, eval_prompts.jsonl, metrics.jsonl, summary.json
                               ▼
      analysis.confirmatory (per cell) ──► analysis.pooled (across cells) ──► report.json, report.md
```

A cell is one (experiment, domain, model) combination with its own audit, lock, hardware class, arm menu and seeds. `vfates program render` writes each cell's config from its domain template and model spec. `vfates program plan` expands the cells into jobs (one audit, one run per arm and seed, one early baseline per arm, variants, transfer evaluations) with their dependencies and cost. Workers claim jobs of their hardware class and GPU count from a file-locked queue and run each as `vfates study --only audit|train`. `vfates program analyze` writes the per-cell reports and the pooled report once every planned run is complete. The pilots in `pilots/` run the same code on development pools.

## Package map

### Entry points

| Module | Public API | Role |
| --- | --- | --- |
| `vfates.cli` | `main(argv)`, `build_parser()`, `PILOTS` | `vfates {env, audit, train, score, study, pilot, program}` |
| `vfates.program` | `load_program`, `validate_program`, `cell_stage`, `cell_with_experiment`, `load_template`, `render_program`, `render_cell_document`, `load_model_spec`, `plan_jobs`, `job_gpu_hours`, `load_throughput`, `throughput_rate`, `class_gpu_hours`, `apply_budget`, `format_budget(..., by_stage)`, `parse_gpu_counts`, `transfer_eval`, `missing_train_runs`, `analyze_program(..., budget_usd)` | The study program: cells and their stage (A discovery, B replication), rendering (each cell carries its experiment's `oracle_gain_min`), the job plan and its cost (H100-equivalent hours, class GPU-hours, rented dollars, per stage with `plan --by-stage`, the preregistered cut order that only removes stage B cells), cross-domain transfer evaluation, and the pooled analysis, which computes each stage once when its planned runs are all complete. |
| `vfates.schedule` | `enqueue`, `read_jobs`, `claim_next`, `touch_heartbeat`, `mark_done`, `mark_failed`, `record_preemption`, `reap_stale`, `apply_result`, `build_worker_command`, `accelerate_config`, `run_worker`, `format_status`, `submit_slurm` | The job queue (`queue.jsonl` with an `fcntl` lock beside it). States `pending`, `claimed`, `done`, `failed`, `preempted`; stale claims return to `pending`; claims go to stage A, then unstaged confirmatory jobs, then stage B, then exploratory jobs. Multi-GPU training jobs launch through `accelerate` with the configs in `configs/accelerate/`. |
| `vfates.study` | `run_study(config, out_dir, *, only, arm, seed, early)`, `reanalyze(config, out_dir)`, `planned_seeds(study)` | One cell: audit, lock, withdrawals, every planned run (plus early baselines and variants), then the cell report. Refuses a cell whose generator, feature and training models or chat-template settings differ, whose hardware class does not match, or whose audit was drawn from a development pool. |
| `vfates.audit` | `run_audit(config, out_dir) -> AuditRun`, `visible_gpu_count`, `load_audit`, `stage_fingerprints`, `load_pools`, `sample_completions`, `grade_completions`, `apply_arms`, `build_table`, `confusion_report`, `extract_features`, `forecast_arms`, `iter_forecast_arms`, `behaviour_indicators`, `behaviour_alignment`, `resolve_arms`, `read_lock_record`, `is_dev_audit`, `require_confirmatory_pools`, `artifact_sha256`, `grades_sha256`, `count_completions`, `load_completions` | The step-0 pipeline. Every stage is resumable, fingerprinted in `stages.json`, and writes its artefact before the next starts. A changed fingerprint recomputes that stage and everything downstream; under an existing lock it is an error. The feature stage loads the model once. The audit meta records `timing` (`wall_seconds`, `gpus`, `hardware_class`, `resumed`); a resumed audit records no wall time, so a partial time cannot understate its cost. |
| `vfates.experiment` | `train_arm(audit_dir, arm, train_config, out_dir, *, backend, resume, hardware_class, ...)` | One GRPO run of one arm on an audit's frozen pools. Refuses to train unless the lock verifies, its artifact hashes match the audit files, the current parser and oracle reproduce the locked grades, the run starts from the audit's saved LoRA init (or is a declared full-parameter run), and the hardware class matches the cell. |
| `vfates.models` | `ModelSpec`, `Placement`, `load_model_spec(name)`; `ModelSpec.placement_for(hardware_class)`, `placement_full_for(hardware_class)`, `peft_target_modules()` | Pinned model specs from `configs/models/`. |

### Data and utilities

| Module | Public API |
| --- | --- |
| `vfates.types` | `Problem`, `GenOut`, `Completion` (both carry the sampled `token_ids`, stop token included), `Parsed`, `AuditTable` (columnar audit with per-arm verdicts; atomic `save` / `load` as `.npz` + `.meta.json`), `SPLITS` |
| `vfates.utils.io` | `write_json`, `read_json`, `write_jsonl`, `append_jsonl`, `read_jsonl`, `iter_jsonl`, `atomic_npz`, `read_yaml`, `write_yaml`, `canonical_json`, `sha256_text`, `sha256_file`; writes are atomic |
| `vfates.utils.hashing` | `u01(*parts)`, `stable_u64(*parts)`, `stable_seed(*parts)`: BLAKE2b over parts joined by U+001F (a part containing it is rejected), identical on every machine; every fixed pseudo-random decision goes through these |
| `vfates.utils.stats` | `GroupIds`, `rankdata`, `pearson`, `spearman`, `group_indices`, `cluster_bootstrap` (resamples prompts), `mean_ci`, `paired_diff_ci` |
| `vfates.utils.env` | `environment_report()`, `hardware()`, `hardware_class()`, `accelerator()`, `versions()`: chip, memory, accelerator, GPU name, the study hardware class, package versions, git commit |
| `vfates.utils.seeding` | `seed_everything(seed)` |

### Tasks (`vfates.tasks`)

Every task implements `Task` (`base`): `problems(split, n, seed)` for `train`, `heldout` and `dev`; `prompt(problem)`; `system_prompt()`; `parse(problem, text) -> Parsed`; `oracle(problem, parsed, text) -> int`; and the hooks `error_categories`, `categorize(problem, parsed, text)`, `abs_error(problem, parsed)`, `support_answered(parsed)`, `verifier_types`, `build_verifier(spec)`. Splits are hashes of a problem identifier, so they do not depend on dataset order.

| Module | Public API |
| --- | --- |
| `registry` | `build_task(spec)` for `type` in `TASK_TYPES` (`chain_sum`, `math`, `code`, `ifeval`, `qa`) |
| `chain_sum` | `ChainSumTask`, `ChainSumConfig`, `FORM_RULES`, `builtin_items`, `abs_error`. Reasoning Gym `decimal_chain_sum` (Egashira et al. configuration) with a bit-identical built-in generator; held-out and dev use fixed dataset-seed offsets. Oracle: exact `Decimal` equality of the last `\boxed{}`. |
| `math` | `MathTask`, `math_verify_equal`, `finite_real`, `load_contamination_sets`. Big-Math-RL-Verified (gated; DeepScaleR as the backup), difficulty buckets on `llama8b_solve_rate`, decontaminated against MATH-500, GSM8K test and AIME. Oracle: math-verify equivalence of the last `\boxed{}` with a CPU time limit. |
| `code` | `CodeTask`, `load_taco_rows`, `extract_public_examples`, `CodePublicTestsVerifier`. TACO from pinned parquet shards (no backup: a failed load raises), buckets `easy` and `medium`, at least 5 hidden tests, at most 32 per problem by a fixed hash. A public example whose input is also a hidden test takes that test's output. Oracle: every hidden test passes, graded by a process that never holds an expected output. `grade_batch` grades a whole step on the worker pool. |
| `code_grader` | The standard-library grader that runs one program against its tests: stdio and call-based problems, output comparison, resource limits. The process that executes a program never holds an expected output. |
| `sandbox` | `GraderPool`, `grade_job`, `measure_throughput`: subprocess limits and the long-lived grading pool (see `SECURITY.md` for what they do and do not isolate) |
| `ifeval`, `if_checkers`, `ifbench_checkers` | `IFEvalTask` on `allenai/IF_multi_constraints_upto5` (`allenai/RLVR-IFeval` as the backup) with a word-budget filter; the IFEval checkers ported from open-instruct and the IFBench training checkers. Oracle: every checker passes and the content judge accepts. |
| `qa` | `QATask`, `judge_agreement_row`, `master_key_opener`. WebInstruct-verified, canonical answer types only. Oracle: math-verify equivalence or normalised exact match. |
| `judges` | `Judge`, `HeuristicJudge`, `LLMJudge` (chat completions with revision and chat-template settings sent to the server), `LocalJudge`, `CachedJudge`, `GeneralVerifierJudge`, `MasterRMJudge`, `HeuristicVerifierJudge`, `judge_from_description`, `parse_yes_no`, `general_verifier_bit`, `parse_master_rm`, `verifier_cache_key`; verdicts are cached by model, revision and prompt hash |
| `answer` | `extract_boxed`, `extract_all_boxed`, `parse_decimal`, `canonical_decimal`, `decimal_parts`, `normalise_statement`, `statement_id`, `hash_split` |

### Verifiers (`vfates.verifiers`)

| Module | Public API |
| --- | --- |
| `base` | `Verifier.verdict(problem, parsed, text, o, key=None) -> int`, `describe()`; `kind`, `stochastic` |
| `arms` | `OracleVerifier`; FP rules `HashFP` (prompt-specific), `NumericToleranceFP`, `TriggerFP` (shared), `ToleranceBandFP` (the decoy); FN rules `FormFN`, `BehaviorFN` (shared), `PromptSubsetFN` (prompt-specific); `ResampledNoise(rho0, rho1, seed, support)` with draws keyed by `(pid, k)`; `Composite`; `StringMatchVerifier`. Phrases match whole tokens (`WORD_RE`). Numeric rules use `task.abs_error`. |
| `natural` | `MathLenientVerifier`, `MathStripStringVerifier`, `CodePublicTestsVerifier` (in `tasks.code`), `IFEvalCheckerVerifier`, `ModelJudgeVerifier` |
| `patch` | `PatchedVerifier(base, categories, task)`: the base verdict, except that outputs in the chosen error categories get the oracle bit; `select_patched_arms(...)`: the `patch_select` stage (`by: tau` picks the k categories whose patched verifier has the highest locked `tau_pool`, `by: rate` the k with the most step-0 errors) |
| `calibrate` | `CalibrationData.from_grades`, `calibrate_arms(task, data, specs, seed, n_boot, targets) -> CalibrationResult`, `arm_validity(report)`: fits every arm on the base policy's train-pool samples to its target pooled rate and reports achieved rates with prompt-clustered intervals; resampled arms copy a matched arm's rates or take declared `rates`. Schema `vfates.calibration/2`. |
| `registry` | `build_verifier(spec, task)`, `build_arms(specs, task)`; a calibrated arm's `describe()` rebuilds it exactly; task-specific types go through `task.build_verifier` |
| `monitors` | `rule_monitors(arms)`: the behaviour indicator of every rule arm (plus `has_answer`), logged through training |

Arms files (`configs/arms/<domain>.yaml`) `include: kit.yaml` for the shared kit and expand `{domain}` in salts.

### Generation (`vfates.gen`)

| Module | Public API |
| --- | --- |
| `base` | `Generator.generate(prompts, n, *, max_tokens, temperature, top_p, seed, system_prompt) -> list[list[GenOut]]`, `describe()` (includes the chat-template settings) |
| `chat` | `encode_prompt`, `encode_completion`, `encode_prompt_completion`, `prompt_completion_ids`, `check_generate_args`: the chat template every backend, feature extractor and trainer shares, with the model spec's `chat_template_kwargs` |
| `factory` | `make_generator(backend, model_id, **kwargs)` for `mlx`, `hf`, `vllm` |
| `mlx_backend` | `MLXGenerator` (Apple silicon, batched) |
| `hf_backend` | `HFGenerator` (MPS / CUDA / CPU); `set_adapter(path)`, `load_full_checkpoint(path)` |
| `vllm_backend` | `VLLMGenerator` (CUDA, tensor parallel); `set_adapter(path)` through a LoRA request, `load_full_checkpoint(path)` |

### Features (`vfates.features`)

| Module | Public API |
| --- | --- |
| `lora_grad` | `prepare_lora_grad(model_or_id, *, sketch_dim, lora_r, lora_alpha, target_modules, lora_seed, sketch_seed, device, dtype, device_map, gradient_checkpointing, save_init, load_init) -> PreparedLoraGrad`, `featurize_prepared(prepared, tokenizer, prompts, completions, *, completion_ids, chat_template_kwargs, logprob_chunk, ...) -> LoraGradFeatures`, `lora_grad_features(...)` (both in one call), `completion_token_logprobs`: per-sample gradients of the completion log-likelihood with respect to the LoRA B matrices at initialisation, CountSketch-projected as each is accumulated |
| `equivalence` | `feature_agreement`, `direction_agreement`, `bf16_float32_gate`: bfloat16 against float32 features (row cosines and the cosine of `g_V`) |
| `lora_init` | `LORA_INIT_FILENAME`, `lora_a_state`, `assert_lora_a_equal`, `save_lora_init`, `load_lora_init`: the A matrices the features used (`vfates.lora_init/1`), loaded by both trainers |
| `sketch` | `CountSketch(dim, seed)`, `sketch_tensor`: unbiased inner-product sketch |
| `hidden` | `load_model`, `hidden_features`: pooled final-layer states (ablation) |

### Forecasts (`vfates.forecast`)

| Module | Public API |
| --- | --- |
| `confusion` | `confusion`, `confusion_with_ci`: pooled and prompt-averaged FPR, FNR, J, precision; `mean_precision_forecast` (the tilt limit), `modal_precision_forecast` (the zero-KL policy-gradient limit) |
| `transfer` | `length_division`, `prompt_centered`, `group_std_advantages`, `velocity_direction`, `velocities`, `tau_pool` (primary, `PRIMARY_Q = 1/32`), `tau_pool_budget` (the estimate from 32 to 192 oracle labels, `ORACLE_LIGHT_BUDGETS`, Horvitz-Thompson over draws without replacement), `tau_pool_interval`, `tau_hard`, `tau_soft`, `first_order_slope`, `cosine`, `split_half_reliability`, `reliability_budget`, `predicted_reliability`, `prompts_for_reliability`, `transfer_forecast -> TransferForecast` |
| `probe` | `auc`, `grouped_folds`, `ridge_scores`, `probe_auc`, `arm_error_probe`, `heldout_error_probe` |
| `lockbox` | `lock_forecasts(payload, path, *, config, note, artifacts)`, `verify_lock`, `load_lock`, `load_lock_record`, `assert_matches_lock`; schema `vfates.lock/2` (payload, artifact hashes, hardware class, pool source, digest) |

### Training (`vfates.train`)

| Module | Public API |
| --- | --- |
| `config` | `GRPOConfig` (LoRA or `full_finetune`, `vllm_tensor_parallel_size`, `vllm_enable_sleep_mode`, `chat_template_kwargs`, `hardware_class`, `model_revision`, `attn_implementation`), `LoRAConfig`, `NonFiniteLossError`, `raise_if_nonfinite`: one config for both backends |
| `grpo_core` | NumPy reference for advantages (`std`, `mean`), loss aggregation (`seq_mean`, `token_mean`, `dr_constant`), the clipped surrogate and k3 KL; both trainers are tested against it |
| `mlx_grpo` | `MLXGRPOTrainer(...).run(resume)`, `constant_with_warmup_lr`: LoRA GRPO on Apple silicon with in-loop evaluation, checkpoints, exact resume and preemption |
| `trl_grpo` | `build_trl_trainer(...)`, `trl_grpo_kwargs`, `trl_run_metadata`, `global_batch_layout(config, world_size)` (512 completions per optimiser step on any GPU count), `training_world_size`, `PreemptionCallback` (`SIGUSR1`/`SIGTERM`: checkpoint, summary status `preempted`, clean exit), `bind_vllm_chat_template`, `peft_lora_matrices`: TRL 0.25 with colocated vLLM, tensor parallelism and sleep mode |
| `checkpoint_eval` | `saved_checkpoints`, `peft_checkpoints`, `is_adapter_checkpoint`, `is_full_model_checkpoint`, `apply_eval_checkpoint`, `evaluate_checkpoints`, `make_eval_generator`, `release_colocated_engine`: the endpoint for TRL runs, from saved adapters or full checkpoints |
| `evaluate` | `evaluate_policy(generator, task, problems, *, n_samples, ...)`, `eval_log_rows`, `eval_sample_rows`, `append_eval_samples` |
| `rewards` | `RewardContext` (batch grading through `task.grade_batch`), `onpolicy_rates`, `monitor_conditionals`, `make_trl_reward_funcs` |

### Analysis (`vfates.analysis`)

| Module | Public API |
| --- | --- |
| `outcomes` | `eval_curve`, `prompt_accuracy`, `run_outcome`, `paired_contrast`, `paired_contrast_seeds` (crossed seed and prompt bootstrap; one-sided and TOST p-values), `holm`, `score_lock` |
| `confirmatory` | `decide_audit_withdrawals(..., menu)` (rate rules, phrase sign checks and `same_selection`, before training; arms outside the cell's menu are listed as `outside_menu` and not withdrawn), `confirmatory_report` (one cell: contrasts, the within-cell rank test, natural pairs, monitors, on-policy rates, knife-edge labels), `natural_pair_blocks`, `arm_error_channel`, `render_markdown` |
| `pooled` | `pooled_report(program, runs_root)`, `staged_report`, `leaf_arm_class`, `classify_arms`, `render_pooled_markdown`, `within_cell_spearman_p`: stage A (H1 with Holm across E1 cells, H1-pooled, H2, H3 with the class baseline and the within-shared stratum, H10, H7 with step-0 natural rates, H8, H9 from locked draws, audit and step-25 cost), stage B (H3-replication and its H9), the union secondary, the scale slope, the conclusive-cell rule, and the exploratory section |
| `transfer` | `transfer_report(runs_root)`, `render_transfer_markdown`: X1 cross-domain transfer |
| `power` | `contrast_se`, `superiority_power`, `tost_power`, `minimum_detectable_difference`, `pooled_h3_power`, `token_projection` |
| `judge_agreement` | `agreement_statistics`, `run_judge_agreement`, `render_agreement_markdown`: reported agreement checks of the model judges |

### Paper (`vfates.paper`)

`assets` (`build_assets`, `read_program_report`): `python -m vfates.paper assets` reads `runs/program_report.json` and the committed P0 report and writes `paper/generated/` (macros, tables, the abstract results sentence, which states only hypotheses that passed) and `paper/figures/`. A number the report does not have yet renders as pending.

### Theory (`vfates.theory`)

`worlds` (`World`, `hand_featured_world`, `random_feature_world`, `knife_edge_world`, `verifier_matrix`, `step0_rates`, `matched_resampled`), `dynamics` (`population_flow`, `sampled_grpo`), `forecasts` (`exact_tau`, `sampled_tau`, `observed_heldout_accuracy`), `identities` (the `check_*` functions for the pathwise decomposition, the resampled-noise scaling, tilt invariance and the policy-gradient limits).

### Pilots (`vfates.pilots`)

`common` (`Checks`, `write_report`, `markdown_table`, `cli`, `apply_override`, `cell_config`, `main_guard`) and one module per pilot, `p0_theory` through `p6_natural`, each exposing `run(config, out_dir) -> report`. `pilots/<name>/run.py` is a three-line entry point.

## Configuration

| Path | Contents |
| --- | --- |
| `configs/program.yaml` | experiments (with `oracle_gain_min`), cells (`id`, `experiment`, `domain`, `model`, `hardware_class`, `menu` or `arms`, `seeds`, `variants`), `stages` (the stage A and stage B cell lists), hardware policy (speed factors, dollars per GPU-hour), the cut order |
| `configs/templates/<domain>.yaml` | the cell template of one domain: task, pools, sampling, arms file, features, forecast, training, evaluation, the `full` and `core` menus, and each menu's analysis lists (H1, H2, H3 pairs, H4, natural pairs, H8, reported arms), with `${model.*}`, `${placement.*}` and `${cell.*}` placeholders |
| `configs/cells/<id>.yaml` | rendered cell configs (committed; regenerated by `vfates program render`) |
| `configs/models/<name>.yaml` | model specs |
| `configs/arms/` | `kit.yaml` (including the enriched and depleted phrase arms of H10, `prefer: enriched` or `depleted`) and one arms file per domain, each of which includes the kit |
| `configs/tasks/` | one task config per domain, with pinned dataset revisions |
| `configs/train/`, `configs/accelerate/` | training presets and `accelerate` launch configs (single GPU, FSDP on 2 or 4 GPUs, full-parameter FSDP) |
| `configs/throughput.yaml` | measured throughput per model and hardware class, written by pilot P5 and read by the planner |

## Artefacts

A cell's directory is `runs/<cell>/`.

| File | Written by | Contents |
| --- | --- | --- |
| `audit/config.json` | `run_audit` | the audit config and `environment_report()` |
| `audit/stages.json` | `run_audit` | the fingerprint of every completed stage (`vfates.stages/1`) and `grades`, the hash of how the current parser and oracle grade the completions |
| `audit/problems.jsonl`, `completions.jsonl` | `run_audit` | frozen pools; one row per sample (`pid`, `k`, text, sampled `token_ids`, token count, finish reason) |
| `audit/calibration.json` | `calibrate_arms` | every arm's fitted spec, achieved rates, intervals, validity |
| `audit/audit.npz` (+ `.meta.json`), `confusion.json` | `build_table`, `confusion_report` | `pid`, `split`, `k`, `O`, and `V` for every arm, including patched arms; step-0 confusion per arm and split |
| `audit/features.npz`, `features.meta.json` | `extract_features` | step-0 features, completion lengths, and whether rows used sampled token ids |
| `audit/lora_init.safetensors` | `prepare_lora_grad` | the LoRA A matrices of the features, which every run starts from |
| `audit/forecasts.json`, `lock.json` | `forecast_arms`, `lock_forecasts` | forecasts and baselines per arm (with `tau_pool_budget` and its draws), `patch_selection`; the sealed copy with the sha256 of every input artefact |
| `withdrawals.json` | `run_study` | arms withdrawn by the preregistered audit rules, decided before training |
| `runs/<arm>__s<seed>/`, `early/<arm>__s<seed>/`, `variants/<name>/<arm>__s<seed>/` | `train_arm` | one directory per planned run, early baseline and variant run |
| `transfer/<arm>__s<seed>/<domain>.json` | `transfer_eval` | X1 evaluations of a final adapter on another domain's held-out pool |
| `<run>/metrics.jsonl` | trainers | one row per optimiser step |
| `<run>/eval.jsonl`, `eval_prompts.jsonl` | MLX trainer, `evaluate_checkpoints` | the endpoint: accuracy per split and per prompt at every evaluation |
| `<run>/eval_samples.jsonl` | `append_eval_samples` | one row per sample when the task has a content judge |
| `<run>/run.json`, `run_metadata.json`, `summary.json` | `train_arm`, trainers | the arm, audit, lock sha256, LoRA init sha256 or `parameterisation: full`, placement, hardware class declared and detected, environment; completion status (`complete`, `preempted`, `interrupted`) |
| `report.json`, `report.md` | `run_study`, `reanalyze` | the cell report |
| `runs/program_report.json`, `program_report.md` | `vfates program analyze` | the pooled report, with one block per stage |
| `paper/generated/`, `paper/figures/`, `paper/main.pdf` | `make paper` | the manuscript's macros, tables, figures and PDF |
| `pilots/*/results/` | pilots | `report.{md,json}`, `decisions.json` where a pilot decides a setting |

## Conventions

- Error rates are pooled over samples and measured on the base policy's own samples of the frozen train pool (the step-0 on-policy confusion matrix). Prompt-averaged variants are reported alongside.
- Held-out prompts are never used to fit, choose or tune anything. Pilots use development pools, and a development-pool audit cannot be used by a confirmatory cell.
- Stochastic verdicts are keyed by `(pid, k)` in the audit and by `(seed, "eval", pid, k)` in evaluation, so a draw is reproducible and never reused across contexts.
- Every artefact records the config and environment that produced it. Anything a preregistered comparison uses is locked before the corresponding training starts.
- Heavy dependencies (torch, MLX, vLLM, TRL, math-verify) are imported inside the functions that need them, so the core package, the theory and the analysis install and test without them.
