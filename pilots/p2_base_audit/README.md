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
| P2.3 | On the chosen cell, every targeted arm reaches its step-0 rate within `gates.calibration_tol` (0.02, the preregistered withdrawal threshold). |
| P2.4 | On the chosen cell, every targeted arm flags at least `gates.min_errors` (30) samples, so training has errors to amplify. |
| P2.5 | Documented: generation throughput of the chosen cell, in tokens per second. |
| P2.6.fn_form | Documented: the channel FNR of the capped `fn_form` arm (see below). |
| P2.7 | Documented: the recommended cell for the study model, used by the P5 gate and E1. |
| P2.8 | Documented: share of audit rows on which `natural_string_match` and `fn_form` agree. Not a gate. |

Among feasible cells, the recommended cell for each model is the one whose base accuracy is closest to `selection.target_acc` (0.30). At accuracy `p`, a group of 8 samples is all wrong with probability `(1 - p)^8`, so 0.30 leaves about 6% of groups uninformative and 70 points of headroom.

## Findings that changed the design

**`fn_form` is fixed in advance and capped.** The rule is `fewer_decimal_places_than_gold`, chosen before the audit rather than selected among surface rules. It rejects a correct answer written with fewer decimal places than the gold string. The other form rules stay in `FORM_RULES`. The arm can fire only as often as the policy writes that surface, so it cannot be aimed at the 0.30 target. On the 0.5B cells the channel FNR is 0.0492 (`egashira`, 3 flagged), 0.0311 (`medium`, 8 flagged) and 0.0465 (`easy`, 19 flagged). On the study cell it is 0.1227 (27 flagged). The arm is kept at that achieved rate and compared with `noise_fn_form`. P2.6 documents the easy-cell rate. It is not gated by P2.3.

**The false-positive control shares the rules' support.** The 0.5B cells box 0.6621 to 0.8018 of outputs. Answer-based false positives (`fp_hash`, `fp_tolerance`) can accept only outputs that state an answer, but unrestricted resampled noise would also reward unformatted outputs, and that would weaken the formatting incentive in the control arm alone. `noise_fp` therefore draws false positives only on wrong outputs that state an answer (`support: answered`), with its rate rescaled so that its pooled FPR still equals `fp_tolerance`'s. The boxed-rate floor is set at 0.75 with this in mind. `fp_trigger` fires on raw text, including completions with no parsed answer, so its control `noise_fp_trigger` uses `support: all` and copies the trigger's channel FPR directly.

**A decoy sits outside the tolerance ball.** `fp_tolerance_decoy` accepts a wrong parsed answer whose absolute error is greater than `fp_tolerance`'s fitted tolerance and at most a fitted `hi`. It is aimed at the same FPR. The arm table records `lo` and `hi`.

**Phrase arms take their sign from 7/8 of the train prompts.** `fp_trigger` (`prefer: not_enriched`) and `fn_behavior` (`prefer: enriched`) match whole tokens, drop automatic candidates that contain a digit, and check the within-prompt association on the remaining 1/8. The rate tolerance (0.02, the withdrawal threshold) is applied to the phrase rate on the full train pool, which is also the channel rate. The diagnostics table records validation, the fit-set rate, the share of accepted wrong rows with no parsed answer, the point-biserial correlation with completion length, the share of matches that start in the first 40 characters, and the number of prompts where the phrase occurs.

**The study model is audited at the study's token budget.** At 512 tokens, Qwen2.5-1.5B truncates 0.1533 of its outputs on `egashira` and 0.1260 on `medium`, and on `easy` its accuracy (0.6260) is above the band, so no 1.5B cell is feasible at that budget. E1 samples up to 1,024 tokens, so the 512-token cells cannot judge it. The `qwen2.5-1.5b-1024` entry repeats the two in-band difficulties at 1,024 tokens, and P2.7 reports the recommended study cell from those.

## How to run

```bash
python pilots/p2_base_audit/run.py         # or: vfates pilot p2
```

The defaults are `--config pilots/p2_base_audit/config.yaml` and `--out pilots/p2_base_audit/results`. Completions are cached per cell under `results/cells/`. A cell keeps `completions.jsonl` when `sampling.fingerprint` matches the sampling settings (a complete cache that has no fingerprint yet is stamped and reused). `calibration.json` is refit when `calibration.fingerprint` is missing or differs. That fingerprint covers the resolved arms block (targets, `n_boot`, seed, arm specs), the calibration schema, and a SHA-256 of the graded train rows `(pid, k, O, parsed answer)`. An arm, fitter, oracle or parser change recalibrates without resampling.

## Runtime

The six 512-token cells took 38 minutes on a Mac Studio with an M4 Max (2,288 seconds), and the two 1,024-token cells took another 21 minutes. A later rerun that reused every cached `completions.jsonl` and only regraded and refit took 827 seconds.

## Arm table

Each cell's arm table has separate channel and realised columns. Channel FPR, FNR and Youden J are the rates training sees in expectation (`achieved_fpr` / `achieved_fnr` are the channel). For a deterministic arm the realised block is the same train-audit labelling. For a resampled arm the channel is the copied rate and the realised block is one draw keyed by `(pid, k)`, so its J need not equal `1 - channel FPR - channel FNR`. Precision is the realised labelling. A second table gives phrase diagnostics for `fp_trigger` and `fn_behavior`, and `lo` / `hi` for `fp_tolerance_decoy`.

The arm menu, in order, is `oracle`, `fp_hash`, `fp_tolerance`, `fp_tolerance_decoy`, `fp_trigger`, `noise_fp`, `noise_fp_trigger`, `fn_behavior`, `fn_prompt_subset`, `noise_fn`, `fn_form`, `noise_fn_form`, `combo_fnform_fphash`, `natural_string_match`. `fn_form` candidates exactly `fewer_decimal_places_than_gold`. `raw_differs_from_gold_string` stays in `FORM_RULES` and is the string-match predicate.

## Results

The numbers below match [`results/report.md`](results/report.md) from the rerun that reused every cached completion file and refit calibration. Accuracy is the train-pool prompt-mean. P2 does not score held-out prompts. Channel and realised rates are in the report's arm tables.

The run passed: 4 checks passed, 0 failed, 4 documented. Two cells were feasible, one for each purpose.

**Sweep at 512 tokens.**

| Cell | Accuracy | pass@8 | Boxed | Truncated | Tokens/s | Feasible |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5B `egashira` | 0.0596 | 0.2422 | 0.6621 | 0.2314 | 2877.8053 | no |
| 0.5B `medium` | 0.2510 | 0.8125 | 0.7666 | 0.0879 | 1558.7672 | no |
| 0.5B `easy` | 0.3994 | 0.8984 | 0.8018 | 0.0430 | 1560.9637 | **yes** |
| 1.5B `egashira` | 0.1963 | 0.6172 | 0.8271 | 0.1533 | 839.6849 | no |
| 1.5B `medium` | 0.4688 | 0.9375 | 0.8418 | 0.1260 | 456.7063 | no |
| 1.5B `easy` | 0.6260 | 0.9766 | 0.9258 | 0.0586 | 493.3666 | no |

**The Mac pilot cell is `qwen2.5-0.5b__easy`** (base accuracy 0.3994), the configuration P3 and P4 use. On that cell the largest targeted rate miss is 0.0066 (P2.3, tolerance 0.0200), the fewest samples any targeted arm flags is 92 (P2.4), and `fn_form` reaches a channel FNR of 0.0465 with 19 flagged samples (P2.6). `natural_string_match` agrees with `fn_form` on 0.9834 of the audit rows (P2.8). The behaviour phrase chosen there (`Let's solve the`, train FNR 0.2934) fails its sign check on the validation prompts, so the preregistered rule withdraws `fn_behavior` from P4; `fp_trigger` (`and carry over`, train FPR 0.1496) validates.

**Study model at the E1 budget of 1,024 tokens.**

| Cell | Accuracy | pass@8 | Boxed | Truncated | Tokens/s | Feasible |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5B `egashira` | 0.2148 | 0.6562 | 0.9668 | 0.0059 | 670.7952 | **yes** |
| 1.5B `medium` | 0.5273 | 0.9844 | 0.9590 | 0.0049 | 599.2567 | no (accuracy above the band) |

**The study cell is `qwen2.5-1.5b-1024__egashira`** (P2.7, base accuracy 0.2148), which is exactly the E1 and P5 configuration: Qwen2.5-1.5B-Instruct on the Egashira chain-sum setting with up to 1,024 tokens. Doubling the budget cut truncation from 0.1533 to 0.0059 and raised the boxed rate from 0.8271 to 0.9668, so the 512-token infeasibility was a budget artefact, not a property of the model or task. On this cell every arm passes the preregistered audit withdrawal rules: `fp_trigger` is `place` (train FPR 0.1505) and `fn_behavior` is `to the result` (train FNR 0.2864), both validated, and `fn_prompt_subset` (0.2955) sits 0.0091 from its shared control `noise_fn`.
