# P1 harness: PASS

14 passed, 0 failed, 0 documented. Runtime 123s.

## Checks

| id | name | metric | value | threshold | status |
| --- | --- | --- | --- | --- | --- |
| P1.1 | unit suite passes on this machine | pytest exit (SKIPPED [1] tests/test_natural_verifiers.py:203: vfates.gen.factory is already installed) | 0.0000 | 0 | pass |
| P1.2 | built-in chain-sum generator equals Reasoning Gym | items compared | 400.0000 | identical | pass |
| P1.3 | audit writes every artifact | missing artifacts | 0.0000 | 0 | pass |
| P1.4 | lock verifies and detects a one-number edit | verify(original) and not verify(tampered) | n/a | n/a | pass |
| P1.5 | resuming an audit resamples nothing and changes no artifact | changed digests | 0.0000 | 0 | pass |
| P1.11 | lock payload equals forecasts.json | canonical JSON | n/a | n/a | pass |
| P1.12 | features were computed from sampled token ids | features.meta.json token_source | n/a | n/a | pass |
| P1.13 | fingerprint manifest lists every stage | stages.json | n/a | n/a | pass |
| P1.14 | LoRA init loads and its sha256 matches the lock | lora_init.safetensors | n/a | n/a | pass |
| P1.6 | MLX and PyTorch agree on sampled-token log-probabilities | median \|diff\| nats/token | 0.0032 | 0.0500 | pass |
| P1.7 | decoded completions re-encode to the sampled token ids | rate | 1.0000 | 0.9500 | pass |
| P1.8 | MLX and PEFT LoRA adapt the same modules at the same scale | adapted modules (both) | 168.0000 | n/a | pass |
| P1.9 | a real-model GRPO run logs, checkpoints and resumes | steps after resume | 3.0000 | 3 | pass |
| P1.10 | analysis joins the lock with the run | arms scored | 1.0000 | 1 | pass |

## Log-probability agreement

| n | per_token_abs_diff_median | per_token_abs_diff_p95 | roundtrip_rate |
| --- | --- | --- | --- |
| 16 | 0.0032 | 0.0076 | 1.0000 |

## LoRA parity

| mlx_modules | peft_modules | same_set | mlx_scales | peft_scales |
| --- | --- | --- | --- | --- |
| 168 | 168 | yes | [2.0000] | [2.0000] |

## Training smoke

| arm | steps | held-out step 0 | held-out final | train final |
| --- | --- | --- | --- | --- |
| noise_fp | 3 | 0.1250 | 0.2500 | 0.2500 |
