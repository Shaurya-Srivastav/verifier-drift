# P5: GPU gate

The go/no-go decision for renting the E1 compute. It has three modes:

- **Dry run** (`mode: dry_run`, the default without CUDA) on the Mac: the PyTorch path at toy size.
- **Full gate** (`mode: full`, the default with CUDA) on one H100: the development pool. Its only scientific go/no-go is the oracle held-out gain. Fate contrasts are logged and do not decide whether E1 runs.
- **Replication screen** (`pilots/p5_gpu_gate/replication_screen.yaml`): base accuracy and truncation of Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct at 1,024 tokens on 128 development prompts. Documented. It is not part of the full-gate decision, because scoring two 3B models is a separate job.

## Dry run (Mac, `mode: dry_run`)

| Check | Claim |
| --- | --- |
| P5.1 | Documented: accelerator, mode, vLLM version. |
| P5.2 | A torch audit (Hugging Face sampling, LoRA-B features, calibration, forecasts, lock) completes. |
| P5.3 | TRL `GRPOTrainer` with PEFT trains `train.steps` steps, logs one metrics row per step, and saves an adapter at every step. |
| P5.4 | The checkpoint evaluator writes `eval.jsonl` and `eval_prompts.jsonl` at step 0 and every adapter. |
| P5.5 | Re-evaluation writes nothing new. |
| P5.6 | Documented: token-rate cost of the full gate. Eval tokens are included. `audit_prompts` is 896 (512 train + 384 held-out). |

Training and evaluation use the audit's 96-token budget and temperature: `train_arm` refuses a run whose model, `max_tokens`, `eval_max_tokens`, `temperature` or `eval_temperature` differs from the audit that made the lock.

TRL's own evaluation reports pooled reward means only, so `train_arm` turns it off. The endpoint (per-prompt held-out oracle accuracy) comes from evaluating the saved adapters afterwards.

```bash
python pilots/p5_gpu_gate/run.py           # or: vfates pilot p5
```

The dry run takes about 10 minutes on a Mac Studio with an M4 Max.

## Full gate (H100, `mode: full`)

Development pool, disjoint from E1 at the question level: 512 train prompts at dataset seed 50,000,000 and 384 held-out prompts at 60,000,000. vLLM audit of Qwen2.5-1.5B-Instruct (16 samples, 1,024 tokens), then TRL GRPO on ten arms with two seeds (300 steps). The report is `pilots/p5_gpu_gate/results/gate/report.md`.

| Check | Claim |
| --- | --- |
| P5.0 | Every planned arm-seed finished. Execution, not a scientific claim. |
| P5.7 | Oracle held-out gain (final minus step 0, mean over seeds) is at least 0.10. This is the only scientific go/no-go. |
| P5.8 | Fate contrasts, prompt bootstrap at the final step. Logged, not a gate. |
| P5.9 | Sample SD (ddof 1) of the two oracle seeds' final held-out accuracies, and the E1 seed rule: SD > 0.03 means 5 seeds for the H1/H2 arms, otherwise 3. Documented. |
| P5.10 | Measured H100 hours per arm-seed and the projected E1 cost, with eval and audit included. Documented. |

The P5.10 clocks:

- **Training:** the per-step `wall_s`, which TRL and MLX both log and which excludes evaluation. The fallback is `gen_seconds` + `train_seconds`, then the whole `train_arm` clock.
- **Evaluation:** the gate writes the whole `train_arm` clock of every fresh run to `timing.json`. `train_arm` turns TRL's in-loop eval off and evaluates the saved adapters afterwards, so eval seconds are that clock minus the summed `wall_s` (checkpoint evaluation and model loading), scaled by the number of evals. TRL's `eval_seconds` on `trl_eval.jsonl` is used instead when it exists. Eval tokens are priced at the generation rate only when no eval clock exists. `eval_sources` in the timing table names the clocks used.
- **Audit:** the gate times its own audit when it starts fresh and writes `audit_timing.json`. E1's audit is that clock scaled by completions (1,408 x 16 against 896 x 16). Without it, the audit is priced at the measured generation rate, else at the placeholder `h100_gen_tok_s` and `mean_completion_len` in `e1_cost`. `audit_source` in the projection names the one used.

P5.9 also reports the E1 minimum detectable difference at margin 0.05 for 3 seeds and for 5 seeds, from `vfates.analysis.power` with the measured oracle seed SD. That function is a normal approximation to the crossed design (R4). A seed SD of 0 reproduces R4's eval-only standard error (about 0.0076 at 384 prompts and 3 seeds).

The projected E1 cost scales the measured arm-seed by steps. Three-seed E1 is 39 full runs (13 arms x 3) plus 13 early baselines of 25 steps. If the seed rule fires, full runs become 57 (9 arms x 5 + 4 arms x 3). The token-rate model in the dry run is only a comparison.

```bash
git clone https://github.com/Shaurya-Srivastav/verifier-drift && cd verifier-drift
bash pilots/p5_gpu_gate/launch_h100.sh
tail -f pilots/p5_gpu_gate/results/run.log
```

The launcher installs `.[torch,gpu,tasks,analysis]` into `.venv`, prints the environment, and starts the full gate in the background. Every stage resumes. Copy `pilots/p5_gpu_gate/results/` back before releasing the machine.

## Replication screen

```bash
python pilots/p5_gpu_gate/run.py \
  --config pilots/p5_gpu_gate/replication_screen.yaml \
  --out pilots/p5_gpu_gate/results/replication_screen
```

Use Llama-3.2-3B-Instruct when its base held-out accuracy is in [0.10, 0.60] and truncation (`finish_reason == length`) is at most 0.05. Otherwise use Qwen2.5-3B-Instruct. Write that choice into `configs/experiments/e1_replication_3b.yaml` before its audit is locked.
