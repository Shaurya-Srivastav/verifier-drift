# P2: base audit and difficulty sweep

P2 fixes the configuration of experiment E1 before anything is trained. For every combination of model and task difficulty, it samples the base policy on a train pool, grades each sample, and fits every verifier arm to the shared step-0 targets (FPR 0.15, FNR 0.30). It answers four questions:

1. Is base accuracy in the band where GRPO has both signal and headroom?
2. Does the model follow the answer format (a boxed number) and finish within the token budget?
3. Can every arm be calibrated to the shared targets, with enough flagged samples to train on?
4. How fast is generation? The throughput sets the training cost model.

The committed run is [`results/report.md`](results/report.md).

## The grid

| Axis | Values |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (the Mac pilot model), `Qwen/Qwen2.5-1.5B-Instruct` (the study model) |
| Difficulty | `egashira` (the configuration of Egashira et al., used in the H100 study), `medium`, `easy` |
| Sampling | 128 train prompts, 8 samples each, up to 512 tokens, temperature 1.0, MLX |
| Study budget | The study model again at 1,024 tokens (the E1 budget) on `egashira` and `medium`, as the model entry `qwen2.5-1.5b-1024` |

The difficulty overrides are in `config.yaml`. A model entry can override `sampling` and restrict `difficulties`; such an entry has its own name, so its cells never reuse completions sampled under other settings. Only the train pool is sampled; P2 never looks at held-out prompts.

## What each check tests

| Check | Claim |
| --- | --- |
| P2.1 | At least one cell is feasible: base accuracy in `gates.acc_range` (0.15 to 0.50), boxed-answer rate at least `gates.boxed_rate_min` (0.75), truncation rate at most `gates.trunc_rate_max` (0.05). |
| P2.2 | The Mac pilot model (`selection.pilot_model`) has a feasible cell. |
| P2.3 | On the chosen cell, every targeted arm reaches its step-0 rate within `gates.calibration_tol` (0.03). |
| P2.4 | On the chosen cell, every targeted arm flags at least `gates.min_errors` (30) samples, so training has errors to amplify. |
| P2.5 | Documented: generation throughput of the chosen cell, in tokens per second. |
| P2.6.fn_form | Documented: the achieved FNR of the capped `fn_form` arm (see below). |
| P2.7 | Documented: the recommended cell for the study model, used by the P5 gate and E1. |

Among feasible cells, the recommended cell for each model is the one whose base accuracy is closest to `selection.target_acc` (0.30). At accuracy `p`, a group of 8 samples is all wrong with probability `(1 - p)^8`, so 0.30 leaves about 6% of groups uninformative and 70 points of headroom.

## Findings that changed the design

**`fn_form` is capped.** Surface-form false negatives (for example "reject answers with fewer decimal places than the gold string") can fire only as often as the model writes non-canonical correct answers. For Qwen2.5-0.5B that is 4% to 8% of correct answers, so no form rule can reach the 0.30 target. The arm is kept at its achieved rate and compared with its own resampled control, `noise_fn_form`. It is reported as P2.6 rather than gated by P2.3.

**The false-positive control shares the rules' support.** The first sweep found only 66% to 80% of 0.5B outputs boxed. Answer-based false positives (`fp_hash`, `fp_tolerance`) can accept only outputs that state an answer, but unrestricted resampled noise would also reward unformatted outputs, and that would weaken the formatting incentive in the control arm alone. `noise_fp` therefore draws false positives only on wrong outputs that state an answer (`support: answered`), with its rate rescaled so that its pooled FPR still equals `fp_tolerance`'s. The boxed-rate floor is set at 0.75 with this in mind.

**The study model is audited at the study's token budget.** At 512 tokens, Qwen2.5-1.5B truncates 15% of its outputs on `egashira` and 13% on `medium`, and on `easy` its accuracy (0.63) is above the band, so no 1.5B cell is feasible at that budget. E1 samples up to 1,024 tokens, so the 512-token cells cannot judge it. The `qwen2.5-1.5b-1024` entry repeats the two in-band difficulties at 1,024 tokens, and P2.7 reports the recommended study cell from those.

## How to run

```bash
python pilots/p2_base_audit/run.py         # or: vfates pilot p2
```

The defaults are `--config pilots/p2_base_audit/config.yaml` and `--out pilots/p2_base_audit/results`. Completions are cached per cell under `results/cells/`, so a rerun after changing an arm or a gate only recalibrates and rewrites the report.

## Runtime

The six 512-token cells took 38 minutes on a Mac Studio with an M4 Max (2,288 seconds), and the two 1,024-token cells took another 21 minutes. With every cell cached, a rerun that only recalibrates takes about 2.5 minutes.

## Results

The committed run passes: 4 checks passed, 0 failed, 3 documented. Two cells are feasible, one for each purpose. The full sweep and every cell's arm table are in [`results/report.md`](results/report.md).

**Sweep at 512 tokens.**

| Cell | Accuracy | pass@8 | Boxed | Truncated | Tokens/s | Feasible |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5B `egashira` | 0.060 | 0.242 | 0.662 | 0.231 | 2,878 | no |
| 0.5B `medium` | 0.251 | 0.813 | 0.767 | 0.088 | 1,559 | no |
| 0.5B `easy` | 0.398 | 0.898 | 0.802 | 0.043 | 1,561 | **yes** |
| 1.5B `egashira` | 0.196 | 0.617 | 0.827 | 0.153 | 840 | no |
| 1.5B `medium` | 0.469 | 0.938 | 0.842 | 0.126 | 457 | no |
| 1.5B `easy` | 0.626 | 0.977 | 0.926 | 0.059 | 493 | no |

**The Mac pilot cell is `qwen2.5-0.5b__easy`** (base accuracy 0.398), the configuration P3 and P4 use. On that cell every targeted arm lands within 0.005 of its target rate (P2.3), the fewest samples any targeted arm flags is 93 (P2.4), and the capped `fn_form` reaches an FNR of 0.084 with 34 flagged samples (P2.6).

**Study model at the E1 budget of 1,024 tokens.**

| Cell | Accuracy | Boxed | Truncated | Tokens/s | Feasible |
| --- | --- | --- | --- | --- | --- |
| 1.5B `egashira` | 0.215 | 0.967 | 0.006 | 671 | **yes** |
| 1.5B `medium` | 0.527 | 0.959 | 0.005 | 599 | no (accuracy above the band) |

**The study cell is `qwen2.5-1.5b-1024__egashira`** (P2.7), which is exactly the E1 and P5 configuration: Qwen2.5-1.5B-Instruct on the Egashira chain-sum setting with up to 1,024 tokens. Doubling the budget cut truncation from 15.3% to 0.6% and raised the boxed rate from 0.83 to 0.97, so the 512-token infeasibility was a budget artefact, not a property of the model or task.
