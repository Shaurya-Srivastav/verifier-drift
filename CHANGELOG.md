# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First public release of the `vfates` package. An earlier scaffold used the import name `vdrift` (`src/vdrift`, `python -m vdrift.pilot`). That name is gone: the package, the console script and the tests are all `vfates`.

### Added

- **Tasks.** Reasoning Gym `decimal_chain_sum` with a bit-identical built-in generator, MBPP+ (EvalPlus) and IFEval (`allenai/RLVR-IFeval`), plus answer parsing, content judges and a subprocess sandbox for model-written Python.
- **Verifier arms at matched step-0 rates.** The oracle; prompt-specific false positives (`fp_hash`) and false negatives (`fn_prompt_subset`); shared false positives (`fp_tolerance`, `fp_trigger`) and false negatives (`fn_behavior`, `fn_form`); resampled class-conditional noise (`noise_fp`, `noise_fn`, `noise_fn_form`); a composite; and a natural string match. Calibration fits every rule to a pooled on-policy target on the train pool only. Resampled noise can be restricted to answered outputs (`support: answered`). Arm lists load from YAML and can be merged with overrides.
- **Generation backends.** MLX (Apple silicon), Hugging Face (MPS, CUDA or CPU) and vLLM (CUDA). The Hugging Face and vLLM backends can load a saved LoRA adapter (`set_adapter`).
- **Step-0 features.** The LoRA B-matrix gradient at standard initialisation, compressed with CountSketch, and pooled final-layer hidden states as an ablation.
- **Forecasts.** Pooled and prompt-averaged confusion rates, mean and modal precision, and transfer precision. The preregistered statistic is `tau_pool` at `q = 1/32`. A decodability probe tests whether an arm's errors are visible in step-0 features on unseen prompts. Forecasts are sealed with SHA-256 (`vfates.lock/1`) before training.
- **Training.** A NumPy GRPO reference (group-standardised and mean-centred advantages; sequence-mean, token-mean and Dr.GRPO reductions; clipped surrogate; k3 KL), an MLX LoRA trainer with in-loop evaluation and exact resume, and a TRL trainer. Oracle, false-positive and false-negative indicators are logged as zero-weight rewards. TRL runs are evaluated afterwards from their saved adapters (`vfates.train.checkpoint_eval`), with the same per-prompt schema as the MLX trainer.
- **Analysis.** Prompt-paired, seed-averaged bootstrap contrasts with one-sided and equivalence (TOST) p-values, Holm-Bonferroni correction, Spearman rank intervals from resampling held-out prompts, and scoring of trained runs against a lock.
- **Theory.** Exactly solvable softmax worlds (hand-featured, random-feature, knife-edge) with population and sampled dynamics, and checks for tilt invariance, the covariance split, J-scaling and the GRPO group advantage.
- **Command line.** `vfates env`, `audit`, `train`, `score`, `study` (audit, lock, train every arm and seed, and score from one config) and `pilot`.
- **Pilots P0 to P6.** Theory, harness, base audit, estimator, micro-GRPO, GPU gate and natural verifiers, each with gated checks and a committed report. The H100 launcher is `pilots/p5_gpu_gate/launch_h100.sh`.
- **Configs.** Arms, tasks, the MLX and H100 training presets, and the confirmatory experiment `configs/experiments/e1_chain_sum.yaml`.
- **Project files.** Preregistration, theory, architecture and related-work documents; contributing guide, code of conduct, security policy, citation metadata, issue and pull request templates, CI and pre-commit configuration.

The 0.1.0 tag has not been cut.
