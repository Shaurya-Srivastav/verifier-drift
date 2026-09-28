# P4 micro-GRPO

Question: does each domain have training signal at 1.7B?

Hardware: the E1 cell's class. `chain_sum` and `math` on a lab RTX A6000 48GB (`rtx-a6000-48gb`). `code`, `ifeval` and `qa` on a lab RTX 5000 Ada 32GB (`rtx-5000-ada-32gb`). Mac smoke uses Qwen3-0.6B on MLX when it imports, otherwise HF, and sets `study.hardware_class` to `hardware_class()` so the study accepts the machine.

## Rule

P4.1 is the only gate, once per domain. On the dev pool, the oracle arm of `qwen3-1.7b` must raise held-out oracle accuracy by at least **0.10** over its step-0 accuracy within **150** steps. Held-out here is the pseudo-held-out slice the audit draws from `dev` (`audit.pools.from: dev`). Everything else stays the E1 protocol: group size, prompts per step, learning rate, LoRA A from the pilot audit, and the token budget. The gain is the point estimate at step 150, not the peak.

A domain that misses is rescreened once, and only after P2 has screened that backup (`backup_screened` in the config):

| domain | backup |
| --- | --- |
| math | DeepScaleR (`agentica-org/DeepScaleR-Preview-Dataset`, source `deepscaler`) |
| ifeval | RLVR-IFeval (`allenai/RLVR-IFeval`, source `rlvr_ifeval`) |
| chain_sum, code, qa | none |

A second miss removes the domain (`leaves_program: true`). A first miss whose backup is not in `backup_screened` does not remove it and does not set `backup_used`. A run that never records a gain does not remove the domain.

Documented, never gated: one-seed fate contrasts, arm minus control, read against the pilot audit's locked `dtau` (sign agreement; a zero gap is undefined). Pairs: `fp_trigger` vs `noise_fp_trigger`, `fn_behavior` vs `noise_fn`, `fp_hash` vs `noise_fp`, and `natural` vs `noise_natural` (skipped on `chain_sum`; math also `natural_strict` vs `noise_natural_strict`). Each contrast carries the one-seed power statement from `vfates.analysis.power.minimum_detectable_difference` (`n_seeds=1`).

## How it runs

Training goes through the study: `only=audit` when `lock.json` and `forecasts.json` are both missing, then `vfates study --only train` on the pilot cell with `audit.pools.from: dev` and `study.train.steps: 150`. The pilot writes that cell to `<out>/<domain>/primary/cell.yaml`.

If the dev split is shorter than the E1 pools (1024 train + 384 held-out), the audit raises. Cap both pools to the largest the split allows and record that in the report. A smaller pool can only lower reliability, so the cap is conservative:

```yaml
cell_overrides:
  audit.pools.train: 256
  audit.pools.heldout: 128
```

Do not point this pilot at the confirmatory held-out split.

## Commands

Mac smoke (P4.1 documented, a few steps, not the 0.10 gate):

```bash
.venv/bin/python -m vfates.cli pilot p4 --set smoke=true --out pilots/p4_micro_grpo/results
```

Lab A6000, chain_sum and math:

```bash
.venv/bin/python -m vfates.cli pilot p4 \
  --set 'domains=[chain_sum, math]' \
  --out pilots/p4_micro_grpo/results
```

Lab RTX 5000 Ada, code, ifeval and qa:

```bash
.venv/bin/python -m vfates.cli pilot p4 \
  --set 'domains=[code, ifeval, qa]' \
  --out pilots/p4_micro_grpo/results
```

Rescreen one backup after P2 has screened it:

```bash
.venv/bin/python -m vfates.cli pilot p4 \
  --set 'domains=[math]' \
  --set 'backup_screened=[math]' \
  --out pilots/p4_micro_grpo/results
```

## Runtime

Placeholder token-rate model in `configs/program.yaml` (10,000 gen tok/s and 30,000 train tok/s on an H100, overhead 1.25, 64 prompts times group size 8, domain mean lengths, then times `params_b / 1.7`). Class hours divide by `speed_vs_h100`. At 1.7B that parameter factor is 1. P5.5 replaces the placeholder. Lab classes are $0/GPU-hour.

Oracle only, 150 steps, one dev audit included: about 4.8 GPU-h chain_sum and 6.9 GPU-h math on the A6000; 6.5 GPU-h code, 4.0 GPU-h ifeval and 4.0 GPU-h qa on the RTX 5000 Ada. About 26 GPU-hours, $0.

With the documented fate arms (7 on chain_sum, 11 on math, 9 on code, ifeval and qa) the same model gives about 206 GPU-hours, still $0 on lab machines. A backup rescreen adds one more oracle arm on that domain.

Mac smoke is a few minutes on Qwen3-0.6B.

## Files

`results/decisions.json` (schema `vfates.p4_decisions/1`): per domain `pass`, `gain`, `backup_used`, `leaves_program`. Also `report.md` and `report.json`. The report lists the config edits for the lead (remove a domain, or point its template at the backup). This pilot does not edit `configs/program.yaml`.

## What the study does with it

`leaves_program: true` drops that domain from the program. `backup_used` with `pass: true` switches the template task to the backup source. Fate rows stay in the appendix; they do not change arms.
