# P1 harness

Question: before any study GPU time, does every stage of every domain compute what it claims, and do the training, program, and analysis entry points refuse the cases the study refuses?

## What this run checks

Structural command (no model). These checks are gated unless marked documented.

| Id | Check | Threshold |
| --- | --- | --- |
| P1.0 | Model path was not run | documented |
| P1.1 | Unit suite, only when `pytest: true` | exit 0 |
| P1.2 | Built-in chain-sum items equal Reasoning Gym, when that package imports | identical |
| P1.22 | SIGUSR1 sets `PreemptionCallback.preempted`. This checks the callback flag. It does not signal a running trainer. | flag is true |
| P1.23 | Study refuses a hardware-class mismatch | `ValueError` before the audit |
| P1.24 | Study refuses a `chat_template_kwargs` mismatch | `ValueError` before the audit |
| P1.25 | Render, plan, enqueue, local worker, `program status` | every enqueued job done |
| P1.26 | `pooled_report` reads base, early, and variant runs | schema `vfates.pooled/1`, oracle gain present, variant `streaming` |
| P1.27 | The plan contains a transfer-eval job | at least one |
| P1.28 | Confirmatory report is well formed | schema `vfates.confirmatory/1`; H3 may be `not_tested` |
| P1.29 | Confirmatory analysis refuses an incomplete cell | error contains `incomplete planned runs` |

`live: true` adds, per domain, on the development pool with `Qwen/Qwen3-0.6B`:

| Id | Check | Threshold |
| --- | --- | --- |
| P1.3 | Audit artifacts, including `patches.json` exactly when the menu has a patch arm | none missing |
| P1.5 | A second `run_audit` changes no artifact digest | 0 changed |
| P1.11 | Lock payload equals `forecasts.json` | canonical JSON |
| P1.12 | Features record `token_source: token_ids` | exact |
| P1.13 | `stages.json` fingerprints, `patches` exactly when patch arms exist | 64 hex chars |
| P1.15 | Lock `pool_source` is `dev` | exact |
| P1.16 | Lock `hardware_class` equals this machine | exact |
| P1.17 | Every arm has `tau_pool_budget` at 32, 64, 128, 192, with draws unless the budget is exact | all four |
| P1.18 | `patch_selection` is present exactly when patch arms exist | match |
| P1.14 | LoRA init file loads and its sha256 is the lock's | match |
| P1.4 | chain_sum lock verifies, and a one-number copy does not | both |
| P1.6 | MLX and PyTorch log-probs on chain_sum development prompts | median absolute difference at most 0.05 nats/token |
| P1.7 | Decoded completions re-encode to the sampled token ids | rate at least 0.95 |
| P1.8 | MLX and PEFT LoRA adapt the same modules at the same scale | same set and scales |
| P1.21 | chain_sum GRPO, 2 steps, MLX, audit LoRA A, held-out eval at step 0 and step 2 | both steps |
| P1.9 | That GRPO run logs, checkpoints, and resumes one step further | steps completed equals 3 |
| P1.10 | Analysis joins the chain_sum lock with the oracle run | 1 arm scored |

P1.4, P1.6, P1.7, P1.8, P1.9, and P1.10 use the chain_sum cell from `cell_config(..., dev=True)` and `Qwen/Qwen3-0.6B`. P1.6 samples the development split. The program, refusal, and confirmatory checks use no model.

Math uses DeepScaleR when `HF_TOKEN` is unset, and Big-Math otherwise. QA uses the heuristic judge on the live path. IFEval uses a heuristic `judge_config` on the live path. Feature extraction on the live path sets `load.device` to `cpu`.

## Hardware and commands

Structural, Apple silicon or CPU, no download:

```bash
python pilots/p1_harness/run.py
```

Live model path, Apple silicon, `Qwen/Qwen3-0.6B`:

```bash
python pilots/p1_harness/run.py --set live=true
```

## Runtime

Structural is about 10 seconds. That is the five fake worker jobs plus the confirmatory fixture, measured by the unit test of `program_queue_smoke`.

Live is an estimate of about 40 minutes on Apple silicon, under the 45 minute ceiling. Five domains at 6 prompts, 2 samples, and 32 tokens, plus feature passes, an 8-prompt log-probability rescore at 128 tokens, a LoRA parity load, and a GRPO run that stops at step 2 and resumes through step 3. The feature pass dominates. This is not a measurement of that run.

## Files

`results/report.json`, `results/report.md`. The live path also writes `results/domains/<domain>/` (an audit directory) and `results/train/`. The structural path writes `results/program/` and `results/confirmatory/`. Nothing here edits a study template.

## How this feeds the study

P1 does not choose a bucket or drop a domain. A failed gated check means that stage is not ready to spend GPU time. P1.0 is documented on the structural command, so a structural PASS is not a claim that the 0.6B audits ran.
