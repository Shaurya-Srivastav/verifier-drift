# P4: micro-GRPO pilot

Trains every arm with LoRA GRPO on the Mac, on the frozen pools of the P3 audit, and scores the P3 lock. This is the first test of the hypothesis on a language model: at matched step-0 rates, does the arm whose errors are carried by a shared behaviour end below its resampled control on held-out prompts, and does the arm whose errors are prompt-specific track the control?

The committed run is [`results/report.md`](results/report.md).

## What each check is for

| Check | Claim |
| --- | --- |
| P4.1 | The setup has signal: the oracle reward raises held-out oracle accuracy by at least `gates.oracle_gain` within the step budget. Without this no contrast is interpretable. |
| P4.2.\<arm\> | Same rate, different fates: `<control> - <shared arm>` in final held-out accuracy, paired by prompt, is at least `gates.fate_gap` with a bootstrap interval above 0. |
| P4.3.\<arm\> | Prompt-specific errors stay local: `<control> - <local arm>` is within `gates.local_tol`, or its interval covers 0. |
| P4.4 | `tau_pool` ranks final held-out accuracy across arms (Spearman at least `gates.spearman`) and ranks it better than step-0 J and mean precision. Documented when fewer than `min_arms_for_rank_gate` arms have finished. |
| P4.5.\<arm\> | The shared behaviour responds to its own reward more than under the control: for an FP arm the share of held-out samples that are wrong and carry the behaviour rises; for an FN arm the share that carry the behaviour falls (the conformance tax). |
| P4.6 | Documented: measured throughput and the projected H100 cost of the full study. |
| P4.7.\<arm\> | With more than one seed, the `<control> - <shared arm>` gap is positive in every seed. |

## Families and controls

The arms are grouped into families in `config.yaml`: `noise_fp` with `fp_tolerance`, `fp_trigger` (shared) and `fp_hash` (prompt-specific); `noise_fn` with `fn_behavior` (shared) and `fn_prompt_subset` (prompt-specific). Each contrast compares an arm with its family's control on the same held-out prompts. With several seeds, each prompt's difference is averaged over seeds before the prompt bootstrap (`paired_contrast_seeds`).

`noise_fp` draws false positives only on wrong outputs that state an answer (`support: answered`), like `fp_tolerance` and `fp_hash`, and its rate is rescaled so its pooled FPR equals `fp_tolerance`'s. The contrast therefore differs only in whether the accepted errors share a behaviour, not in whether answer formatting is rewarded.

## How to run

```bash
python pilots/p4_micro_grpo/run.py         # or: vfates pilot p4
```

P3 must have finished (its lock is read, verified and never rewritten). Arms train in the order of `arms` and are skipped once `summary.json` records every step, so rerunning after an interruption resumes the unfinished arm from its last checkpoint and reports on everything finished. To report on a subset while training continues elsewhere, run with `--set arms=[oracle,noise_fp,fp_tolerance]`.

## Runtime

About 80 minutes per arm on a Mac Studio with an M4 Max: 150 steps of 8 prompts with 8 samples each (up to 384 tokens), plus 7 evaluations of 128 held-out and 64 train prompts with 4 samples each. The eight arms take roughly 11 hours, so run it overnight.

## Expected outcome

P4.1 must pass for the pilot to say anything. The go decision for the H100 gate is P4.1, at least one P4.2, and P4.3. P4.4 at eight arms is a coarse rank test; the study's rank test has twelve arms and three seeds.
