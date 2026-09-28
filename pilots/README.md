# Pilots

Seven pilots take the project from theory to a go/no-go decision on rented GPUs. Each pilot answers one question, has explicit pass/fail gates, and commits its report, so every design choice in the confirmatory study can be traced to evidence.

## The suite

| Pilot | Question it answers | Hardware | Approximate runtime |
| --- | --- | --- | --- |
| [P0 theory](p0_theory/README.md) | Do exactly solvable worlds reproduce the identities and the same-rate, different-fates split, and does step-0 `tau_pool` track trained held-out accuracy? | CPU | seconds |
| [P1 harness](p1_harness/README.md) | With the real model, does every stage (sampling, grading, calibration, features, lock, GRPO, resume, analysis) compute what it claims? | Mac | 10 min |
| [P2 base audit](p2_base_audit/README.md) | Which model and task difficulty has signal and headroom, follows the answer format, and lets every arm reach its step-0 error rate? | Mac | 1 h |
| [P3 estimator](p3_estimator/README.md) | Is the step-0 estimator reliable, is the sketch faithful, and are shared errors decodable while prompt-specific ones are not? Locks the forecasts. | Mac | 1 h |
| [P4 micro-GRPO](p4_micro_grpo/README.md) | On a real LLM at matched rates, do shared errors end below their resampled control on held-out prompts while prompt-specific ones match it? Does `tau_pool` rank the outcomes? | Mac | 11 h |
| [P5 GPU gate](p5_gpu_gate/README.md) | Does the PyTorch, TRL and vLLM path work (dry run on the Mac), and does the study model pass the P4 gates with two seeds (one H100)? | Mac, then 1 H100 | 10 min, then about a day |
| [P6 natural verifiers](p6_natural/README.md) | Do natural verifiers (MBPP base tests, IFEval checkers) have the step-0 error structure the study needs? | Mac | 30 min |

Runtimes are for a Mac Studio with an M4 Max and 36 GB of memory, with model weights already downloaded. They are estimates until each pilot's committed report records a measured run.

## Running the pilots

Run them in order. P3 writes the lock that P4 is scored against, and P4's measured completion length feeds P5's cost projection.

```bash
vfates pilot p0                          # or: python pilots/p0_theory/run.py
vfates pilot p1
vfates pilot p2
vfates pilot p3
vfates pilot p4                          # overnight; resumable
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

For example, `vfates pilot p4 --set 'arms=[oracle,noise_fp,fp_tolerance]'` trains three arms instead of eight. Rerunning a pilot reuses everything already on disk: cached completions, calibrations, features and training checkpoints.

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
2. P4 passes P4.1 (the setup has signal), at least one P4.2 (a shared error ends below its resampled control) and P4.3 (a prompt-specific error does not).
3. The full P5 gate passes the same checks on the study model, with the sign of every contrast agreeing across seeds (P4.7).

If P4.1 passes and every P4.2 fails, that is itself a result: at this scale the error rate would be sufficient after all. It is reported as such, not tuned away.

## Decision log

Design changes forced by pilot evidence, in the order they were made. Each is visible in the code and configs, and none used a held-out training outcome.

1. **P0: the pooled `tau_pool` replaces the per-prompt `tau_rel` as the primary estimator.** On the 28-cell theory grid, the per-prompt argmax is inflated on prompts whose 16 samples miss the favoured behaviour (Spearman 0.777). The pooled top-`q` tail of prompt-centred velocities reaches 0.862 and lands within 0.10 of the trained outcome in 28 of 28 cells. The fraction `q = 1/32` was fixed here, before any language-model run.
2. **P2: `fn_form` is a capped arm.** Surface-form rules can fire only as often as the policy writes non-canonical answers, which is 4% to 8% of correct answers for Qwen2.5-0.5B, so no form rule reaches the 0.30 false-negative target. The arm is compared at its achieved rate against its own resampled control, `noise_fn_form`, and `noise_fn` is matched to `fn_behavior` instead.
3. **P2: the false-positive control has the same support as the false-positive rules.** Qwen2.5-0.5B leaves 20% to 34% of its outputs without a boxed answer. Answer-based false positives (hash, tolerance) can only accept outputs that contain an answer, while unrestricted resampled noise would also reward unformatted ones, weakening the formatting incentive in the control alone. `noise_fp` therefore uses `support: answered`, with its rate rescaled so that its pooled false-positive rate still equals `fp_tolerance`'s. The P2 boxed-format floor is 0.75.
4. **P2: the study model is audited at the study's token budget.** At 512 tokens, Qwen2.5-1.5B truncates 13% to 15% of outputs on the in-band difficulties, so no 1.5B cell is feasible at that budget. E1 samples up to 1,024 tokens, so P2 repeats those difficulties at 1,024 tokens (`qwen2.5-1.5b-1024`) and recommends the study cell from them. At 1,024 tokens the Egashira cell is feasible (accuracy 0.215, 0.6% truncated), which confirms the E1 configuration.
5. **P5: the endpoint is evaluated from saved adapters.** TRL's built-in evaluation logs pooled reward means under the training sampler, which is not the preregistered endpoint. Every backend now writes the same per-prompt held-out oracle accuracy, from `evaluate_policy` at step 0 and at each saved checkpoint.
