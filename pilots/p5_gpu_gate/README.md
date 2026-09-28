# P5 GPU gate

Question: does every model and hardware class in `configs/program.yaml` train, resume, and move oracle accuracy, and which E1 arms need seeds 3 and 4?

Each stage is a separate launch. Copy `<out>/stages/<stage>.json` back and run `collect` to merge them. Filter a launch with `--set model=qwen3-8b` and `--set hardware_class=h100-80gb`.

One GPU is `python -m vfates.cli study`. Two GPUs use `accelerate launch --config_file configs/accelerate/fsdp_2gpu.yaml --num_processes 2`. Four GPUs use `configs/accelerate/fsdp_4gpu.yaml`. The pilot builds that argv (`vfates.schedule.build_worker_command`). Do not wrap the pilot itself in `accelerate launch`. `configs/accelerate/single_gpu.yaml` is the one-GPU file and is not launched. 14B is tensor-parallel 2. 32B is tensor-parallel 4.

Run a cell on the machine whose `hardware_class()` matches it. The study refuses a mismatch unless `VFATES_ALLOW_HARDWARE_MISMATCH=1`.

## Stages

| id | stage | gated | rule |
| --- | --- | --- | --- |
| P5.1 | `path` | gated | Every (model, hardware class) that has a cell. 10-step oracle train plus checkpoint eval through TRL with colocated vLLM, adapter save and reload, held-out oracle accuracy at step 0 and the last step (recorded, not a gain gate), peak memory per GPU. MLX writes `peak_memory_gb`. TRL writes `peak_memory_gb` and `peak_memory_gb_per_gpu` (GiB, three decimals). A CUDA path with neither field fails closed. |
| P5.2 | `preempt` | gated | One run per hardware class. SIGUSR1 at a mid step (`status: preempted`), then resume to the same planned step with `steps_completed == steps` and `status: complete`. |
| P5.3 | `gain` | gated | Every model. Oracle on the math dev pool, the model spec's learning rate, 150 steps, held-out oracle gain at least **0.05** over step 0. `qwen3-1.7b-base` has no math cell; it retargets `e1-math-qwen3-1.7b` with that spec's overrides. |
| P5.4 | `seeds` | documented | Each E1 domain, oracle seeds 0 and 1, full E1 protocol, 300 steps, dev pool. If the sample SD (ddof=1) of the two final dev accuracies exceeds **0.03**, that cell's H1 and H2 arms (`study.seed_rule.arms`) also run seeds 3 and 4. The seed list still includes the E1 base seed 2. |
| P5.5 | `throughput` | documented | From stage logs: gen and train tokens per second per GPU, and wall hours per arm-seed scaled to 300 steps (`wall_s / measured_steps * 300 / 3600`). Writes `configs/throughput.yaml` only when a non-smoke pair is complete. |
| | `collect` | | Merges stage files into the report and `results/decisions.json`. |

Mac dry run (`--set smoke=true`) runs `path` and `preempt` on Qwen3-0.6B through the chain_sum cell (MLX or HF, no dataset download), then `collect`. Gain and seeds are not failed; they are documented as not part of the smoke.

The dev pool is never the confirmatory held-out split. If dev is shorter than 1024+384, cap `audit.pools` to the largest it allows. A smaller pool can only lower reliability.

## Commands

Mac:

```bash
.venv/bin/python -m vfates.cli pilot p5 --set smoke=true --out pilots/p5_gpu_gate/results
```

Lab RTX A6000 48GB (`rtx-a6000-48gb`): `qwen3-1.7b` and `qwen3-1.7b-base`.

```bash
.venv/bin/python -m vfates.cli pilot p5 --set stage=path --set hardware_class=rtx-a6000-48gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=preempt --set hardware_class=rtx-a6000-48gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=gain --set model=qwen3-1.7b
.venv/bin/python -m vfates.cli pilot p5 --set stage=gain --set model=qwen3-1.7b-base
.venv/bin/python -m vfates.cli pilot p5 --set stage=seeds --set cell=e1-chain_sum-qwen3-1.7b
.venv/bin/python -m vfates.cli pilot p5 --set stage=seeds --set cell=e1-math-qwen3-1.7b
```

Lab RTX 5000 Ada 32GB (`rtx-5000-ada-32gb`): `qwen3-1.7b` code, ifeval, qa. Seeds for those three E1 cells.

```bash
.venv/bin/python -m vfates.cli pilot p5 --set stage=path --set hardware_class=rtx-5000-ada-32gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=preempt --set hardware_class=rtx-5000-ada-32gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=seeds --set cell=e1-code-qwen3-1.7b
.venv/bin/python -m vfates.cli pilot p5 --set stage=seeds --set cell=e1-ifeval-qwen3-1.7b
.venv/bin/python -m vfates.cli pilot p5 --set stage=seeds --set cell=e1-qa-qwen3-1.7b
```

Lab A100 80GB (`a100-80gb`): `qwen3-4b`. Preempt uses the chain_sum cell on this class.

```bash
.venv/bin/python -m vfates.cli pilot p5 --set stage=path --set hardware_class=a100-80gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=preempt --set hardware_class=a100-80gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=gain --set model=qwen3-4b
```

Rented H100 80GB (`h100-80gb`): `qwen3-8b`, `qwen3-14b` (2 GPU), `qwen3-32b` (4 GPU), `olmo3-7b`, `llama3.1-8b`. Preempt on this class is `e2-math-qwen3-14b`.

```bash
.venv/bin/python -m vfates.cli pilot p5 --set stage=path --set model=qwen3-8b --set hardware_class=h100-80gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=path --set model=qwen3-14b --set hardware_class=h100-80gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=path --set model=qwen3-32b --set hardware_class=h100-80gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=path --set model=olmo3-7b --set hardware_class=h100-80gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=path --set model=llama3.1-8b --set hardware_class=h100-80gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=preempt --set hardware_class=h100-80gb
.venv/bin/python -m vfates.cli pilot p5 --set stage=gain --set model=qwen3-8b
.venv/bin/python -m vfates.cli pilot p5 --set stage=gain --set model=qwen3-14b
.venv/bin/python -m vfates.cli pilot p5 --set stage=gain --set model=qwen3-32b
.venv/bin/python -m vfates.cli pilot p5 --set stage=gain --set model=olmo3-7b
.venv/bin/python -m vfates.cli pilot p5 --set stage=gain --set model=llama3.1-8b
```

The 14B train command the pilot launches:

```bash
accelerate launch --config_file configs/accelerate/fsdp_2gpu.yaml --num_processes 2 \
  -m vfates.cli study --config <cell.yaml> --out <out> --only train --arm oracle --seed 0
```

32B uses `configs/accelerate/fsdp_4gpu.yaml` and `--num_processes 4`.

Collect, after the stage files are in one `--out` directory:

```bash
.venv/bin/python -m vfates.cli pilot p5 --set stage=collect --out pilots/p5_gpu_gate/results
.venv/bin/python -m vfates.cli program plan --by-cell --budget-usd 9000
```

`collect` also prints that plan in the report (skipped when `smoke` or `--set plan=false`).

Slurm, so a preemption reaches the study (`--requeue`, `--signal=B:USR1@180`). The batch script execs the worker; USR1 is forwarded to the study child. The preempt stage does not need this: it sends SIGUSR1 itself and then resumes.

```bash
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --gres=gpu:4
export VFATES_QUEUE="$PWD/queue.json"
export VFATES_HARDWARE_CLASS=h100-80gb
export VFATES_GPUS=4
sbatch scripts/slurm/job.sbatch
```

For an interactive pilot stage on the same node, run the `pilot p5` command under `srun` with the same two `#SBATCH` lines. One-GPU stages run the study in process, so USR1 hits the trainer. Multi-GPU stages spawn `accelerate launch` in the same process group.

## Runtime and cost

Same placeholder as `vfates program plan`: 10,000 gen tok/s and 30,000 train tok/s on one H100, overhead 1.25, 64 prompts times group size 8, domain lengths, times `params_b / 1.7`. Class GPU-hours divide by `speed_vs_h100`. A multi-GPU placement is not multiplied again, because the planner assumes tensor-parallel throughput scales with the GPU count. Hours below include one dev audit plus the short train. They are not measurements. Lab USD is 0. H100 USD is the H100-equivalent hours times $2.50.

Path, 10 steps: A6000 math 1.7B about 1.7 GPU-h; A6000 base about 1.2; RTX 5000 Ada code about 1.6; A100 4B about 2.5; H100 8B and Llama about 2.7 GPU-h ($6.80) each; OLMo about 2.4 GPU-h ($6.00); 14B about 4.8 GPU-h ($12); 32B about 10.9 GPU-h ($27).

Preempt, one short run per class: A6000 chain_sum about 1.2 GPU-h; RTX 5000 Ada code about 1.6; A100 4B chain_sum about 1.7; H100 14B about 4.8 GPU-h ($12).

Gain, 150 steps on math: 1.7B and 1.7B-base on the A6000 about 6.9 GPU-h each; 4B on the A100 about 10 GPU-h; H100 8B and Llama about 11 GPU-h ($28) each; OLMo about 10 GPU-h ($25); 14B about 20 GPU-h ($50); 32B about 46 GPU-h ($114).

Seeds, two oracle runs of 300 steps: chain_sum about 17 GPU-h, math about 24, code about 23, ifeval about 14, qa about 14, all lab, $0. If the SD exceeds 0.03, H1 and H2 add seeds 3 and 4 (another four arm-seeds per such arm). That is the expensive branch; the base list is seeds 0, 1 and 2, and seed 2 of oracle is not trained by this stage.

Mac smoke is minutes, not these hours.

## Files

`results/decisions.json` schema `vfates.p5_decisions/1`: `seeds` per E1 cell, `gain` pass per model, `path` pass per `model/hardware_class`, `preempt` pass per class.

`configs/throughput.yaml` schema `vfates.throughput/1`:

```yaml
schema: vfates.throughput/1
rates:
  <model>:
    <hardware_class>:
      gen_tok_s_per_gpu: ...
      train_tok_s_per_gpu: ...
      hours_per_arm_seed: ...
      measured_at: ...
      commit: ...
```

Smoke rows are not written. The program planner reads this file. Do not commit a hand-filled copy.

## What the study does with it

A failed path or preempt blocks that placement. A failed gain blocks that model. P5.4 writes the seed list; the report says which cells should set `study.seed_rule.apply` so the rule arms take seeds 3 and 4. P5.5's hours replace the placeholder in `vfates program plan`.
