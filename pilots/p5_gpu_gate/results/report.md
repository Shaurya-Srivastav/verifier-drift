# P5 GPU gate (dry run): PASS

4 passed, 0 failed, 2 documented. Runtime 254s.

## Checks

| id | name | metric | value | threshold | status |
| --- | --- | --- | --- | --- | --- |
| P5.1 | accelerator environment | gpu=none, mode=dry_run, vllm=None | n/a | n/a | documented |
| P5.6 | projected token-rate cost of the full gate (eval tokens included) | USD (31.3 H100 h), audit_prompts in the cost model | 78.3067 | n/a | documented |
| P5.2 | torch audit (HF sampling, LoRA-B features, lock) completes | lock.json written | n/a | n/a | pass |
| P5.3 | TRL GRPO + PEFT trains and saves every adapter | metric rows (checkpoints [1, 2]) | 2.0000 | n/a | pass |
| P5.4 | checkpoint evaluator writes the shared per-prompt schema | eval rows (steps 0 and every checkpoint) | 6.0000 | n/a | pass |
| P5.5 | re-evaluation is idempotent (resumable) | rows rewritten | 0.0000 | n/a | pass |

## Cost projection

| mean_completion_len | length_source | runs | train_tokens | eval_tokens | audit_tokens | generated_tokens | h100_hours | usd | eval_tokens_included | source | rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 350.0000 | config | 20 | 1.08e+09 | 1.86e+08 | 5.02e+06 | 1.27e+09 | 31.3227 | 78.3067 | yes | token_rates | Placeholder token-rate model. Replace with P5's measured hours per arm-seed. |

## Dry-run eval curve

| step | split | oracle_acc |
| --- | --- | --- |
| 0 | heldout | 0.2500 |
| 0 | train | 0.2500 |
| 1 | heldout | 0.0000 |
| 1 | train | 0.1250 |
| 2 | heldout | 0.0000 |
| 2 | train | 0.0000 |
