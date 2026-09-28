# P5: GPU gate

The go/no-go decision for renting compute. It has two modes, chosen automatically unless `mode` is set:

- **Dry run** (`mode: dry_run`, the default without CUDA) on the Mac: exercises the entire PyTorch path end to end at toy size.
- **Full gate** (`mode: full`, the default with CUDA) on one H100: the E1 gate on the study model with two seeds.

## Dry run (Mac, `mode: dry_run`)

| Check | Claim |
| --- | --- |
| P5.1 | Documented: accelerator, mode, vLLM version. |
| P5.2 | A torch audit (Hugging Face sampling on MPS, LoRA-B features, calibration, forecasts, lock) completes. |
| P5.3 | TRL `GRPOTrainer` with PEFT trains `train.steps` steps on the audit's arm, logs one metrics row per step, and saves an adapter at every step. |
| P5.4 | The checkpoint evaluator (`vfates.train.checkpoint_eval`) evaluates step 0 and every adapter with the same protocol and schema as the MLX trainer (`eval.jsonl`, `eval_prompts.jsonl`). |
| P5.5 | Re-evaluation writes nothing new: the evaluator is resumable. |
| P5.6 | Documented: projected cost of the full gate, using the mean completion length measured in P4 when that report exists. |

TRL's own evaluation reports pooled reward means only, so it is turned off. The primary endpoint (per-prompt held-out oracle accuracy) comes from evaluating the saved adapters afterwards with vLLM (`enable_lora`) on the GPU or Hugging Face on the Mac.

```bash
python pilots/p5_gpu_gate/run.py           # or: vfates pilot p5
```

The dry run takes about 10 minutes on a Mac Studio with an M4 Max.

## Full gate (H100, `mode: full`)

A vLLM audit of Qwen2.5-1.5B-Instruct on the Egashira chain-sum configuration (512 train, 384 held-out prompts, 16 samples, 1,024 tokens), the lock, then TRL GRPO on eight arms with two seeds each (300 steps of 64 prompts with 8 samples), each evaluated on all held-out prompts every 25 steps. The report applies the P4 checks (`pilots/p5_gpu_gate/results/gate/report.md`) plus seed agreement (P4.7).

```bash
git clone https://github.com/Shaurya-Srivastav/verifier-drift && cd verifier-drift
bash pilots/p5_gpu_gate/launch_h100.sh
tail -f pilots/p5_gpu_gate/results/run.log
```

The launcher installs `.[torch,gpu,tasks,analysis]` into `.venv`, prints the environment, and starts the run in the background. Every stage resumes, so a preempted instance can rerun the launcher. Copy `pilots/p5_gpu_gate/results/` back before releasing the machine.

The go decision for the full study (E1 at 12 arms with 3 seeds, then E2 to E4) is P4.1, at least one P4.2, P4.3 and P4.7 on this run. The projected cost is in the dry-run report (P5.6).
