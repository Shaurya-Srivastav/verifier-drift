# P3: estimator pilot

Runs the E1 audit on the Mac pilot model, checks the step-0 estimator before any training, and locks the forecasts that P4 is scored against.

The committed run is [`results/report.md`](results/report.md) and the lock is [`results/audit/lock.json`](results/audit/lock.json).

## What each check is for

| Check | Claim |
| --- | --- |
| P3.1 | Documented: held-out base accuracy and pass@16 coverage of the audit cell. |
| P3.2 | The velocity direction `g_V` of the oracle arm is stable: mean cosine between `g` estimated on disjoint halves of the train prompts is at least `gates.reliability`. |
| P3.3 | The same for the least reliable arm, so no forecast rests on an unstable direction. |
| P3.4 | CountSketch preserves inner products of exact LoRA-B gradients: on `fidelity.n` completions drawn from the train split only, the off-diagonal Gram entries of the sketched and exact gradients correlate at least `gates.sketch_corr`. Held-out rows are not in this subset. |
| P3.5 | The prompt-specific false positive (`fp_hash`) is not linearly decodable from step-0 features on unseen prompts (prompt-grouped cross-validated AUC at most `gates.hash_auc_max`). There is no shared feature for training to amplify. |
| P3.6 | At least one shared false positive (`fp_tolerance`, `fp_trigger`) is decodable on unseen prompts (AUC at least `gates.shared_auc_min`): the mechanism that makes a fixed error transfer is visible at step 0. |
| P3.7 | Documented: pooled hidden states rank the arms' `tau_pool` like LoRA-B gradients (Spearman across arms). The cheap feature map is an ablation, not the estimator. |
| P3.8 | The forecasts are locked (SHA-256 over the canonical payload, config, and artifact hashes) before P4 trains a step. |
| P3.9 | Documented, not a gate. Features are extracted again with `lora_seed: 1` into `lora_seed_1/` (a sibling of the audit). The report gives the Spearman of `tau_pool` across arms and the max absolute difference of `tau_pool` between seeds 0 and 1. That is evidence the screen does not hinge on one draw of LoRA A. The result is cached in `lora_seed_sensitivity.json`. |

The forecasted fates (which arm is predicted to generalise) are reported in the "Locked forecasts" table and are not gated: gating on the prediction the study tests would be circular. P4 scores them.

## How to run

```bash
python pilots/p3_estimator/run.py          # or: vfates pilot p3
```

Every stage is resumable: sampling appends to `completions.jsonl`, features are cached in `feature_parts/`, and the fidelity and probe results are cached next to the report. Delete `results/` to start over. Never delete or edit `results/audit/lock.json` after P4 has started.

## Configuration

The task cell is written out in `config.yaml` (P2's recommendation for the pilot model), so the locked configuration is the one on disk. The arms come from `configs/arms/chain_sum.yaml` and are fitted on the train pool only; held-out samples never touch calibration. Pools are 256 train and 128 held-out prompts with 16 samples each, which gives `tau_pool` at `q = 1/32` a 64-sample held-out tail. Sampling stays at `max_tokens: 512`. The forecast block sets `loss_agg: seq_mean`, the same aggregation P4 trains with, so the locked screen divides sum-gradients by completion length.

## Runtime

About 90 minutes on a Mac Studio with an M4 Max: sampling 6,144 completions with MLX (about 20 minutes), LoRA-B gradient features in float32 on MPS (about 25 minutes), the same feature pass again at `lora_seed` 1 (about 25 minutes, cached after the first successful run), hidden-state features (about 5 minutes), 48 exact train-split gradients for the fidelity check, and the probe.

## Expected outcome

Exit code 0 and summary `PASS`. A failure of P3.5 or P3.6 is informative rather than fatal to the study: it says the step-0 features do not separate prompt-specific from shared errors on this model, and the P4 contrasts decide whether training does.
