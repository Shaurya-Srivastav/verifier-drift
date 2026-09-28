# P3: estimator pilot

Runs the E1 audit on the Mac pilot model, checks the step-0 estimator before any training, and locks the forecasts that P4 is scored against.

The committed run is [`results/report.md`](results/report.md) and the lock is [`results/audit/lock.json`](results/audit/lock.json).

## What each check is for

| Check | Claim |
| --- | --- |
| P3.1 | Documented: held-out base accuracy and pass@16 coverage of the audit cell. |
| P3.2 | The velocity direction `g_V` of the oracle arm is stable: mean cosine between `g` estimated on disjoint halves of the train prompts is at least `gates.reliability`. |
| P3.3 | The same for the least reliable arm, so no forecast rests on an unstable direction. |
| P3.10 | Documented: a noise model of `g_V` (`reliability_budget`). With `signal = \|E c_p\|^2` and `noise = tr Cov(c_p)` over prompt contributions `c_p`, the split-half cosine on `m` prompts per half is about `signal / (signal + noise / m)`. The report gives the largest gap between predicted and measured cosine (marked as passing at or below `gates.noise_model_gap`, 0.05), the share of the noise that is within-prompt sampling noise (the mean jackknife variance of `c_p`, not capped at 1), and the train pool that would reach `gates.reliability`. |
| P3.4 | CountSketch preserves inner products of exact LoRA-B gradients: on `fidelity.n` completions drawn from the train split only, the off-diagonal Gram entries of the sketched and exact gradients correlate at least `gates.sketch_corr`. Held-out rows are not in this subset. |
| P3.5 | Documented: the prompt-specific false positive (`fp_hash`) is not linearly decodable from step-0 features on unseen prompts (prompt-grouped cross-validated AUC at most `gates.hash_auc_max`). There is no shared feature for training to amplify. |
| P3.6 | Documented: at least one shared false positive (`fp_tolerance`, `fp_trigger`) is decodable on unseen prompts (AUC at least `gates.shared_auc_min`): the mechanism that makes a fixed error transfer is visible at step 0. |
| P3.7 | Documented: pooled hidden states rank the arms' `tau_pool` like LoRA-B gradients (Spearman across arms). The cheap feature map is an ablation, not the estimator. Batch widths are rounded up to a multiple of 64 (`pad_multiple`): at exact widths every new shape left about 0.2 GiB of MPS graph buffers behind, and the first 1,024-prompt run ran out of memory at part 10 of 44. The rounding changed no row by more than 4.3e-7 relative to the exact-width features. |
| P3.8 | The forecasts are locked (SHA-256 over the canonical payload, config, and artifact hashes) before P4 trains a step. |
| P3.9 | Documented, not a gate. Features are extracted again with `lora_seed: 1` into `lora_seed_1/` (a sibling of the audit). The report gives the Spearman of `tau_pool` across arms and the max absolute difference of `tau_pool` between seeds 0 and 1. That is evidence the screen does not hinge on one draw of LoRA A. The result is cached in `lora_seed_sensitivity.json`. |

The forecasted fates (which arm is predicted to generalise) are reported in the "Locked forecasts" table and are not gated: gating on the prediction the study tests would be circular. P4 scores them.

## How to run

```bash
python pilots/p3_estimator/run.py          # or: vfates pilot p3
```

Every stage is resumable: sampling appends to `completions.jsonl`, features are cached in `feature_parts/`, and the fidelity and probe results are cached next to the report. Delete `results/` to start over. Never delete or edit `results/audit/lock.json` after P4 has started.

## Configuration

The task cell is written out in `config.yaml` (P2's recommendation for the pilot model), so the locked configuration is the one on disk. The arms come from `configs/arms/chain_sum.yaml` and are fitted on the train pool only; held-out samples never touch calibration. Pools have E1's counts (not its model or task cell): 1,024 train and 384 held-out prompts with 16 samples each, which gives `tau_pool` at `q = 1/32` a 192-sample held-out tail. They are disjoint at the question level from E1's pools and from P5's H100 development pool. P1 and the P5 dry run use the same cell and seed, so their toy pools are the first 24 and 12 questions of these. A first audit with 256 train and 128 held-out prompts failed P3.2 and P3.3 (see the decision log in [`../README.md`](../README.md)); its questions are the first 256 and 128 of these pools. This is the last P3 size. P4 trains on the first 256 train prompts only (`max_train` in its config). Sampling stays at `max_tokens: 512`. The forecast block sets `loss_agg: seq_mean`, the same aggregation P4 trains with, so the locked screen divides sum-gradients by completion length.

## Runtime

About 8 hours on a Mac Studio with an M4 Max. This is an extrapolation from the stage logs of the 256-prompt audit (6,144 completions), not yet a measured run: sampling 22,528 completions with MLX (about 2.2 hours at 36 minutes per 6,144), calibration (about 1.2 hours), LoRA-B gradient features in float32 on MPS (about 1.4 hours), the same feature pass again at `lora_seed` 1 (about 1.4 hours), hidden-state features (about 25 minutes), the forecasts and their bootstrap, 48 exact train-split gradients for the fidelity check, and the probe. Every stage is cached, so an interrupted run resumes.

## Expected outcome

Exit code 0 and summary `PASS`: the estimator checks (P3.2, P3.3, P3.4, P3.8) gate. P3.5 and P3.6 are documented. A miss there is informative rather than fatal to the study: it says the step-0 features do not separate prompt-specific from shared errors on this model, and training decides whether the fates differ.
