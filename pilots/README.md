# Pilots

Seven pilots take the project from theory to a go/no-go decision on rented GPUs. Each pilot answers one question, has explicit pass/fail gates, and commits its report, so every design choice in the confirmatory study can be traced to evidence.

## The suite

| Pilot | Question it answers | Hardware | Approximate runtime |
| --- | --- | --- | --- |
| [P0 theory](p0_theory/README.md) | Do exactly solvable worlds reproduce the identities and the same-rate, different-fates split, and does step-0 `tau_pool` track trained held-out accuracy? | CPU | seconds |
| [P1 harness](p1_harness/README.md) | With the real model, does every stage (sampling, grading, calibration, features, lock, GRPO, resume, analysis) compute what it claims? | Mac | 2 min |
| [P2 base audit](p2_base_audit/README.md) | Which model and task difficulty has signal and headroom, follows the answer format, and lets every arm reach its step-0 error rate? | Mac | 1 h |
| [P3 estimator](p3_estimator/README.md) | Is the step-0 estimator reliable, is the sketch faithful, and are shared errors decodable while prompt-specific ones are not? Locks the forecasts. | Mac | 1 h |
| [P4 micro-GRPO](p4_micro_grpo/README.md) | On a real LLM at matched rates, do shared errors end below their resampled control on held-out prompts while prompt-specific ones stay close to it? How do the locked forecasts line up with the outcomes? (Exploratory.) | Mac | 18 h |
| [P5 GPU gate](p5_gpu_gate/README.md) | Does the PyTorch, TRL and vLLM path work (dry run on the Mac)? On one H100, does the study model gain at least 0.10 held-out accuracy under the oracle on a development pool disjoint from E1, how much do two seeds differ, and what does one arm-seed cost? | Mac, then 1 H100 | 4 min, then about a day |
| [P6 natural verifiers](p6_natural/README.md) | Do natural verifiers (MBPP base tests, IFEval checkers) have the step-0 error structure the study needs? | Mac | 30 min |

Runtimes are for a Mac Studio with an M4 Max and 36 GB of memory, with model weights already downloaded. They are estimates until each pilot's committed report records a measured run.

## Running the pilots

Run them in order. P3 writes the lock that P4 is scored against, and P4's measured completion length feeds P5's cost projection.

```bash
vfates pilot p0                          # or: python pilots/p0_theory/run.py
vfates pilot p1
vfates pilot p2
vfates pilot p3
vfates pilot p4                          # about 18 hours; resumable
vfates pilot p5                          # dry run on the Mac
bash pilots/p5_gpu_gate/launch_h100.sh   # the full gate, on the GPU machine
vfates pilot p6
```

Every pilot accepts three options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--config PATH` | `pilots/<name>/config.yaml` | The configuration to run |
| `--out DIR` | `pilots/<name>/results` | Where the report and cached artefacts go |
| `--set KEY=VALUE` | none | Override one config entry; dotted keys, YAML values, repeatable |

For example, `vfates pilot p4 --set 'arms=[oracle,noise_fp,fp_tolerance]'` trains three arms instead of ten. Rerunning a pilot reuses everything already on disk: cached completions, calibrations, features and training checkpoints.

## Reading a report

Each pilot folder holds a `config.yaml`, a `run.py`, a `README.md` that explains its checks, and a `results/` directory. Only `results/report.md` and `results/report.json` are committed, plus the audit's `calibration.json`, `forecasts.json` and `lock.json` where a pilot runs an audit.

- `report.md` opens with `PASS` or `FAIL`, followed by a table of checks.
- `report.json` holds the same checks plus the full config, the environment (package versions, hardware, git commit) and the raw tables.

Every check has an id (for example `P4.2.fp_tolerance`), a measured value, a threshold and one of three statuses:

| Status | Meaning |
| --- | --- |
| `pass` | The value met its threshold. |
| `fail` | The value missed its threshold. The pilot exits with code 1. |
| `documented` | Reported for the record and never gated, for example a known counterexample or a diagnostic. |

## What decides the study

The confirmatory study (`configs/experiments/e1_chain_sum.yaml`, then E2 to E4 in [`PREREGISTRATION.md`](../PREREGISTRATION.md)) runs only if all three hold:

1. P0, P1 and P3 pass: the theory, the harness and the estimator are sound.
2. P4 passes three checks:
   - P4.1: the setup has signal.
   - At least one P4.2: a shared error ends at least 0.05 below its resampled control, with a one-sided p-value below 0.05.
   - Every P4.3 screen: no prompt-specific error ends more than 0.05 away from its control.
3. The full P5 gate shows an oracle held-out gain of at least 0.10 on its development pool. P5's fate contrasts are logged and do not decide.

P5 also fixes two E1 settings by rules written before it runs:

- **Seeds.** If the two oracle seeds differ by a sample standard deviation above 0.03, the H1 and H2 arms get five seeds instead of three.
- **Budget.** The measured hours per arm-seed replace the projected cost.

The pilots are exploratory. None of their outcomes is a confirmatory result, and E1 never trains or scores on a pilot's prompts. If P4.1 passes and every P4.2 fails, that is itself a finding at pilot scale. It is reported as such, not tuned away. E1 then still runs as preregistered, because a pilot cannot answer the confirmatory question.

## Decision log

Design changes, in the order they were made. Each is visible in the code and configs, and none used a held-out training outcome.

### Pilot evidence

1. **P0: the pooled `tau_pool` replaces the per-prompt `tau_rel` as the primary estimator.** On the 28-cell theory grid, the per-prompt argmax is inflated on prompts whose 16 samples miss the favoured behaviour (Spearman 0.777). The pooled top-`q` tail of prompt-centred velocities reaches 0.862 and lands within 0.10 of the trained outcome in 28 of 28 cells. The fraction `q = 1/32` was fixed here, before any language-model run.
2. **P2: `fn_form` is a capped arm.** Surface-form rules can fire only as often as the policy writes non-canonical answers, which is 4% to 8% of correct answers for Qwen2.5-0.5B, so no form rule reaches the 0.30 false-negative target. The arm is compared at its achieved rate against its own resampled control, `noise_fn_form`, and `noise_fn` is matched to `fn_behavior` instead.
3. **P2: the false-positive control has the same support as the false-positive rules.** Qwen2.5-0.5B leaves 20% to 34% of its outputs without a boxed answer. Answer-based false positives (hash, tolerance) can only accept outputs that contain an answer, while unrestricted resampled noise would also reward unformatted ones, weakening the formatting incentive in the control alone. `noise_fp` therefore uses `support: answered`, with its rate rescaled so that its pooled false-positive rate still equals `fp_tolerance`'s. The P2 boxed-format floor is 0.75.
4. **P2: the study model is audited at the study's token budget.** At 512 tokens, Qwen2.5-1.5B truncates 13% to 15% of outputs on the in-band difficulties, so no 1.5B cell is feasible at that budget. E1 samples up to 1,024 tokens, so P2 repeats those difficulties at 1,024 tokens (`qwen2.5-1.5b-1024`) and recommends the study cell from them. At 1,024 tokens the Egashira cell is feasible (accuracy 0.215, 0.6% truncated), which confirms the E1 configuration.
5. **P5: the endpoint is evaluated from saved adapters.** TRL's built-in evaluation logs pooled reward means under the training sampler, which is not the preregistered endpoint. Every backend now writes the same per-prompt held-out oracle accuracy, from `evaluate_policy` at step 0 and at each saved checkpoint.

### Pre-freeze review

Six independent reviews of the code and the design ran before the preregistration was frozen, with no language-model training yet. These changes came from them. None used a training outcome.

1. **The screen and the trainer share one parameterisation.** The audit saves the LoRA A matrices it featurised with, and every training run loads them. Features are computed on the sampled token ids, including the stop token, instead of re-tokenised text. With the sequence-mean loss, train rows are divided by their length when `g_V` is built, so `g_V` is the step-0 direction of the loss that is actually trained.
2. **The lock covers the inputs.** `vfates.lock/2` also hashes the pools, completions, calibration, arms file, features and LoRA initialisation, and training refuses to start when any of them has changed since the lock.
3. **Two arms were added and one was reduced to reporting.**
   - `fp_tolerance_decoy` accepts wrong answers just beyond `fp_tolerance`'s band, at the same rate. It separates a shared error from partial credit for near misses.
   - `noise_fp_trigger` is a control with the trigger's support: every wrong output, including those without an answer.
   - `natural_string_match` duplicated `fn_form` at step 0 on every P2 cell, so it is audited and not trained.
   - `fn_form` is fixed in advance as the decimal-places rule.
4. **The phrase arms are validated.** Phrases match on word boundaries and contain no digits. The preference sign uses the within-prompt association with correctness, and the chosen phrase must keep its sign on 1/8 of the train prompts held out from fitting.
5. **The analysis counts training seeds.** Contrasts pair seed with seed and prompt with prompt, and the bootstrap resamples both. H1 is a superiority test over a 0.05 margin, and H3 is a rank test within rate-matched pairs, where step-0 J is constant by construction.
6. **P5 moved to its own pool.** P5 used E1's held-out prompts, so it now uses a development pool disjoint from E1. Only its oracle gain gates the study.
7. **Seeds and cost are set by rules.** P5's oracle seeds decide between three and five E1 seeds, and P5's measured cost replaces the projection.
