# P3 estimator

Question: on the development pool, with the E1 cell's arms, is `g_V` reliable, does the sketch preserve inner products, do bf16 features agree with float32, and does the lock verify?

## Rule

Per domain, on the dev split, with the E1 arms and the cell's pool sizes where the dev split allows. A smaller pool than the E1 train pool is conservative for split-half reliability.

Gated. A domain that fails any of these fails the pilot.

| Id | Check | Threshold |
| --- | --- | --- |
| P3.2 | Split-half cosine of `g_V` for oracle and natural arms (worst arm). Empty oracle/natural group fails. | >= 0.8 |
| P3.3 | Same cosine for every other forecast arm: targeted, resampled, patch, random | >= 0.8 |
| P3.4 | CountSketch off-diagonal Gram correlation against exact LoRA-B gradients, train rows | >= 0.95 |
| P3.11 | `bf16_float32_gate` and `direction_agreement` on at least 256 rows | both cosines > 0.99 |
| P3.8 | The lock verifies | verify |

Oracle and natural types are `oracle`, `math_lenient`, `math_strip_string`, `code_public_tests`, `ifeval_checker`, `model_judge`, `master_rm`, `natural_string_match`, and `string_match`, plus a name `natural` or `natural_*`. Resampled arms and names that start with `noise_` are P3.3.

Documented, not gates: P3.5 prompt-specific FP AUC at most 0.60, P3.6 shared FP AUC at least 0.65, P3.9 LoRA seed sensitivity, P3.10 noise-model gap at most 0.05. A hidden-state ablation, if a config asks for one, does not divide by length. It is not a go/no-go check.

Smoke does not score the numbers and does not write a decision. The thresholds are not loosened.

## Hardware and commands

Mac smoke:

```bash
python pilots/p3_estimator/run.py --set smoke=true
```

Study gates, qwen3-1.7b, development pool. The execute path rescores the first 256 feature rows in bf16 with the audit's LoRA A and compares them to the float32 rows. chain_sum and math use `rtx-a6000-48gb`. code, ifeval, and qa use `rtx-5000-ada-32gb`.

```bash
python pilots/p3_estimator/run.py --set smoke=false --set execute=true --set hardware_class=rtx-a6000-48gb
```

## Runtime

About 10 GPU-hours. Basis, not a measurement: 5 domains, 1408 prompts, 16 samples, about 400 tokens, about 3500 tokens/s is about 3.6 hours of generation, plus a feature pass and a second LoRA-seed pass. A short dev split reduces this. The estimate assumes the E1 pool fits in the dev split.

## Files

`results/report.json`, `results/report.md`, and on a real screen `results/decisions.json` (schema `vfates.p3_decisions/1`). Each domain is `pass` or `fail` with the failed check ids. `apply` is false on smoke and on a report that did not execute. A failing domain fails the pilot. The lead does not edit a template from a smoke file.

## How this feeds the study

A domain that fails P3.2, P3.3, P3.4, P3.11, or P3.8 does not enter the GPU study. Documented checks are reported beside the decision and do not by themselves drop the domain.
