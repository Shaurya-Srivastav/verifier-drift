# P1: harness pilot

Checks, with the real pilot model, that every stage the study depends on computes what it claims: sampling, grading, calibration, LoRA-B features, forecasts, the lock, GRPO training, resumption, and the analysis join. Nothing here tests the hypothesis. A failure here means a later result could be an artefact of the code.

The committed run is [`results/report.md`](results/report.md).

## What each check is for

| Check | Claim |
| --- | --- |
| P1.1 | The unit and integration suite passes (`pytest -m "not slow and not network"`). |
| P1.2 | The built-in chain-sum generator is bit-identical to Reasoning Gym's `decimal_chain_sum` on `identity_items` items (question, answer, metadata). |
| P1.3 | A tiny audit writes every artefact: problems, completions, calibration, table, confusion, features, feature metadata, forecasts, the fingerprint manifest, the LoRA init file, and the lock. |
| P1.4 | The lock verifies, and editing one forecast value makes verification fail (tamper detection). |
| P1.5 | Rerunning the audit on its own directory changes no artifact hash: problems, completions, calibration, verdicts, features, forecasts, `stages.json`, `lora_init.safetensors`, and the lock (resume is idempotent: no resampling, no refitting). |
| P1.11 | The locked payload is the canonical JSON of `forecasts.json`. |
| P1.12 | `features.meta.json` records `token_source: token_ids`, so the screen was computed from the sampled token ids. |
| P1.13 | `stages.json` has a fingerprint for problems, completions, calibration, the verdict table, features, and forecasts. |
| P1.14 | `lora_init.safetensors` loads and its SHA-256 equals `lock.json` `artifacts.lora_init`. |
| P1.6 | Per-token log-probabilities from MLX sampling agree with a float32 PyTorch rescoring of the same token ids (median absolute difference per token at most `gates.logprob_per_token`). A tokenisation or alignment bug is off by more than a nat per token; bf16 rounding is a few hundredths. |
| P1.7 | Decoded completions re-encode to the sampled token ids (`encode_completion`) in at least `gates.roundtrip_rate` of samples, so features and training see the tokens that were sampled. |
| P1.8 | MLX `LoRALinear` and PEFT `LoraConfig` built from the same `LoRAConfig` wrap the same modules with the same scale (`alpha / rank`). |
| P1.9 | Two GRPO steps with the `train_arm` arm write metrics, evals, per-prompt evals and checkpoints; resuming to `steps + 1` continues from the checkpoint. |
| P1.10 | The analysis layer joins the run to the lock (`score_forecasts`) and computes a prompt-paired contrast. |

## How to run

```bash
python pilots/p1_harness/run.py            # or: vfates pilot p1
```

Defaults: `--config pilots/p1_harness/config.yaml`, `--out pilots/p1_harness/results`. `--set key=value` overrides a config entry (for example `--set pytest=false` to skip the suite when iterating). With `fresh: true` the tiny audit and run directories are rebuilt on every invocation.

## Runtime

About 10 minutes on a Mac Studio with an M4 Max: the test suite (about 2 minutes), the audit (36 prompts with 4 samples each, on MLX), LoRA-B features on MPS, the log-probability rescoring, and three GRPO steps (two, then one more after resuming).

## Expected outcome

Exit code 0 and summary `PASS`, with no documented checks.
