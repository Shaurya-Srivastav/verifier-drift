# P4: micro-GRPO pilot

Exploratory Mac pilot. It trains every arm with LoRA GRPO on the frozen pools of the P3 audit and scores the P3 lock. It does not carry the E1 confirmatory rules.

The question it asks, at matched step-0 rates: does a shared-behaviour arm end below its resampled control on held-out prompts, and does a prompt-specific arm stay near its control?

The committed run is [`results/report.md`](results/report.md).

## What each check is for

| Check | Claim |
| --- | --- |
| P4.0 | Documented: the preregistered audit withdrawals, decided from P3's `calibration.json` before any training and written to `withdrawals.json`. A withdrawn arm is not trained, and its `P4.2.<arm>` or `P4.3.<arm>` row says so. Its resampled control still trains. |
| P4.1 | The setup has signal: the oracle reward raises held-out oracle accuracy by at least `gates.oracle_gain`. This is a go/no-go check. |
| P4.2 | At least one shared arm ends below its control. A hit is a point difference `control - arm` of at least `gates.fate_gap` (0.05) and a one-sided prompt-bootstrap p-value below 0.05. Individual `P4.2.<arm>` rows are documented. |
| P4.3 | Screen, not an equivalence test: every prompt-specific arm has absolute difference at most 0.05. TOST is not attainable here (R4: power about 0.14 at a true gap of 0). `p_tost` is reported and does not decide the screen. |
| P4.4 | Within-pair rank (exact permutation Spearman of `tau_pool` with `acc(arm) - acc(control)`) and the overall Spearman. Documented, not a gate. |
| P4.5 | Conditional monitor rates, averaged over seeds. FP rules use `monitors_wrong` (P(monitor given O = 0)). FN rules use `monitors_correct` (P(monitor given O = 1)). Documented, not a gate. |
| P4.6 | Documented: measured throughput and the projected H100 cost. Eval tokens are included when the cost model lists them. P5's wall clock replaces this projection. |
| P4.7 | With more than one seed, the per-seed sign of each shared contrast. Documented, not a gate. |

The go decision (`ok`) is P4.1, at least one P4.2, and every P4.3 screen.

## Power (R4, normal approximation, one seed, 128 prompts, 4 samples)

Eval-noise SE is about 0.028. An 80% minimum detectable difference for a one-sided test at margin 0 is about 0.069. At a true gap of 0.05 that one-sided test has power about 0.53. The pilot gate (point difference at least 0.05 and p < 0.05) has power about 0.40, and its 80% minimum detectable difference is about 0.078. TOST power at a true gap of 0 inside +/- 0.05 is about 0.14, which is why P4.3 is a screen.

## Families and controls

`noise_fp` with shared arms `fp_tolerance` and `fp_tolerance_decoy`, and local arm `fp_hash`. `noise_fp_trigger` with shared arm `fp_trigger` (not `noise_fp`). `noise_fn` with shared arm `fn_behavior` and local arm `fn_prompt_subset`.

Contrasts use the prompt bootstrap after averaging seeds (`paired_contrast_seeds`) at the preregistered final step.

`noise_fp` draws false positives only on wrong outputs that state an answer (`support: answered`), like `fp_tolerance` and `fp_hash`, and its rate is rescaled so its pooled FPR equals `fp_tolerance`'s. The contrast differs in whether the accepted errors share a behaviour, not in whether answer formatting is rewarded.

## How to run

```bash
python pilots/p4_micro_grpo/run.py         # or: vfates pilot p4
```

P3 must have finished (its lock is read, verified and never rewritten). Arms train in the order of `arms` and are skipped once `summary.json` records every step, so rerunning after an interruption resumes the unfinished arm from its last checkpoint. To report on a subset while training continues elsewhere, run with `--set arms=[oracle,noise_fp,fp_tolerance]`.

## Runtime

About 110 minutes per arm on a Mac Studio with an M4 Max: 150 steps of 8 prompts with 8 samples each (up to 512 tokens, matching the P3 audit), plus evaluations of 128 held-out and 64 train prompts with 4 samples each. The ten arms take about 18 hours, less about 110 minutes for each arm withdrawn at the audit.

## Expected outcome

P4.1 must pass for the pilot to say anything. The go decision is P4.1, at least one P4.2, and every P4.3 screen. Rank statistics are reported and do not gate. The confirmatory rank test is E1's H3, not this pilot.
