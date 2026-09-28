# P6: natural-verifier pilot

P6 asks whether two natural verifiers, the MBPP base tests and the IFEval constraint checkers, are imperfect in a way the step-0 theory can see, before any GRPO update. It samples the frozen base policy and grades it; it does not train.

Both domains sample the **training** split only (`vfates.gen.factory.make_generator`). Held-out prompts are never used.

The committed run is [`results/report.md`](results/report.md).

## MBPP+ (EvalPlus)

- **Verifier `V`:** the original MBPP base tests (optionally only the first `k`).
- **Oracle `O`:** the base tests and the EvalPlus plus tests, compared the way EvalPlus compares them.
- **False positive:** a program that passes the tests the policy can see and fails the hidden plus tests.

| Quantity | Role |
| --- | --- |
| Pooled FPR, FNR, Youden J, precision and FP mass | The step-0 confusion of `V` against `O`, with prompt-averaged rates beside them. Because `O` includes the base tests, a full-base `V` has FNR exactly 0 and `J = 1 - FPR`, so all the information is on the false-positive side. |
| FP share among `V`-accepted samples | **Gate** (at least 0.05 by default). Checks that this policy actually produces the natural false positives that Zhang (arXiv:2607.11022) describes as mostly prompt-specific. If almost nothing passes `V`, the share is undefined and the gate fails: the screen is not estimable and needs a stronger coder or more samples. |
| Special-casing rate among FPs | The shared-exploit monitor. Flags code that returns or prints the visible assert's literal, or branches on an argument equal to the visible input (the hardcoding pattern of Countdown-Code, arXiv:2603.07084). |
| FP mass in the top 10% of prompts | Separates the two stories. Near 1, a few prompts own the errors (prompt-specific, so selecting on them should not transfer). Near 0.1, the errors are spread the way a shared behaviour is. |

## RLVR-IFeval

- **Verifier `V`:** the open-instruct checker for the row's `ground_truth` constraint.
- **Oracle `O`:** that checker and a content judge. The default judge is a heuristic that flags keyword stuffing, padding and degenerate repetition, including the Tulu 3 pattern (`timberlike, unconnectedly Conversation N User :`).

| Quantity | Role |
| --- | --- |
| Checker pass rate | **Gate:** must lie in `[0.2, 0.95]`. Outside that band the verifier is saturated or dead, and a step-0 false-positive rate is not a useful screen. |
| Stuffing rate among checker-passing responses | How often `V = 1` is not meaningful content. At step 0 this is the baseline to compare against training-time over-optimisation. With the default heuristic it equals the FP share of `V` against `O`, and FNR is 0 because the oracle includes the checker. |
| Local-judge agreement (optional) | `ifeval.local_judge_samples` scores that many responses with `LocalJudge` at temperature 0 and records agreement with the heuristic. The committed config sets it to 0. |

## How to run

```bash
python pilots/p6_natural/run.py            # or: vfates pilot p6
```

The defaults are `--config pilots/p6_natural/config.yaml` and `--out pilots/p6_natural/results`. The pilot writes `report.md` and `report.json` (with `environment_report()`) and exits with code 0 when both domain gates pass.

## Configuration and runtime

| Domain | Model | Problems and samples | Max tokens |
| --- | --- | --- | --- |
| MBPP+ | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 64 problems, 8 samples each | 512 |
| IFEval | `Qwen/Qwen2.5-1.5B-Instruct` | 64 prompts, 8 samples each | 384 |

Both use MLX at temperature 1.0 with completion, prefill and per-call batch sizes of 8, 4 and 16. The MLX backend's default completion batch of 128 aborts with a Metal out-of-memory error on a 36 GB M4 Max when a whole domain is queued as one call.

A smoke run with 32 problems and 4 samples per domain, weights already downloaded, finished in about 2.7 minutes and passed both gates. The committed configuration generates four times as many sequences.

## Expected outcome

P6 is a go/no-go on the natural domains, not a claim about transfer after training.

- A 1.5B coder fails many MBPP items; the MBPP gate passes only when at least 5% of base-test passes fail the plus tests.
- IFEval checkers are easy to satisfy by accident, so the pass rate should land inside the band, and the stuffing rate at step 0 should be low compared with a later RLVR run.
- High FP concentration on MBPP supports the prompt-specific reading of missing tests. A high special-casing rate would mean the shared hardcoding exploit is already present in the base policy.

## Sandbox

MBPP grading executes model-written code in a subprocess (`python -I`, a fresh temporary directory, resource limits, process-group kill). That is not a container. Run any training job that executes completions inside a container or VM. See `vfates.tasks.sandbox` and [`SECURITY.md`](../../SECURITY.md).
