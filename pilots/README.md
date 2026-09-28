# Pilots

Seven pilots take the program from theory to a go/no-go decision. Each answers one question, and its gates and rules are fixed in its README before it runs. Every pilot writes `results/report.md` and `results/report.json`, and the pilots that set a study parameter also write `results/decisions.json`.

Every pilot after P0 runs on development pools. `cell_config(cell_id, dev=True)` in `vfates.pilots.common` loads a rendered cell from `configs/cells/`, sets `audit.pools.from: dev` and `pilot: true`, and the audit's lock records `pool_source: dev`. The study refuses a development-pool audit for any cell without `pilot: true`, so no pilot prompt is ever a confirmatory prompt.

## The suite

| Pilot | Question it answers | Hardware | Runtime (estimate) |
| --- | --- | --- | --- |
| [P0 theory](p0_theory/README.md) | Do exactly solvable worlds reproduce the identities and the same-rate, different-fates split, and does step-0 `tau_pool` track trained held-out accuracy? | CPU | seconds |
| [P1 harness](p1_harness/README.md) | Does every stage of every domain, the patch selection, the program runner and the analysis compute what it claims? | Apple silicon or CPU | 10 s structural; about 35 min with `live=true` (Qwen3-0.6B, five domains) |
| [P2 base audit](p2_base_audit/README.md) | Which difficulty bucket and token budget put both `qwen3-1.7b` and `qwen3-4b` inside the accuracy band in each domain? | Lab GPU of the domain's E1 class | 8 to 24 GPU-hours |
| [P3 estimator](p3_estimator/README.md) | Is `g_V` reliable for every arm, is the sketch faithful, do bfloat16 features agree with float32, and does the lock verify? | `qwen3-1.7b` on the E1 class | about 10 GPU-hours |
| [P4 micro-GRPO](p4_micro_grpo/README.md) | Does the oracle arm of `qwen3-1.7b` gain at least 0.10 held-out accuracy in 150 steps in each domain? | E1 class | about 26 GPU-hours oracle only, about 206 with the one-seed fate arms |
| [P5 GPU gate](p5_gpu_gate/README.md) | Does every placement train, resume after preemption and move oracle accuracy, which E1 arms need seeds 3 and 4, and what does each arm-seed cost? | Every hardware class in `configs/program.yaml` | per stage, in its README; about $320 of rented H100 time at placeholder rates |
| [P6 natural verifiers](p6_natural/README.md) | What are each natural verifier's step-0 error rates, and are its false positives real errors by human adjudication? | CPU, reading P3's audits; judges on vLLM | minutes, plus about $20 of API calls if `OPENAI_API_KEY` is set, plus the annotator's time |

E1 classes: `rtx-a6000-48gb` for chain_sum and math, `rtx-5000-ada-32gb` for code, ifeval and qa. Runtimes are placeholder-rate estimates until each committed report records a measured run; P5.5 replaces them.

## Order and commands

P2 fixes the buckets and token budgets that every later pilot's cells use, so it runs first on GPUs. After P2, the lead copies `template_edits` from its `decisions.json` onto `configs/templates/` and re-renders the cells (`vfates program render`). P3, P4 and P6 then run on those cells. P5's path and preempt stages need only the placements; its gain and seed stages use the rendered E1 cells.

```bash
vfates pilot p0
vfates pilot p1                                   # structural, no model
vfates pilot p1 --set live=true                   # Qwen3-0.6B on the Mac

vfates pilot p2 --set smoke=false --set hardware_class=rtx-a6000-48gb
vfates program render                             # after copying P2's template edits
vfates pilot p3 --set smoke=false --set execute=true --set hardware_class=rtx-a6000-48gb
vfates pilot p4 --set 'domains=[chain_sum,math]'
vfates pilot p5 --set stage=path --set hardware_class=rtx-a6000-48gb
vfates pilot p5 --set stage=collect               # writes configs/throughput.yaml
vfates program plan --by-cell --budget-usd 9000
vfates pilot p6
```

Run each GPU pilot once per hardware class with that class's domains. Every pilot has `--set smoke=true`, which runs a few prompts and steps with Qwen3-0.6B on the Mac to check the path, scores nothing and writes no decisions. Each README has the full commands, including P5's stages and P6's adjudication export and import.

Every pilot accepts three options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--config PATH` | `pilots/<name>/config.yaml` | The configuration to run |
| `--out DIR` | `pilots/<name>/results` | Where the report and cached artefacts go |
| `--set KEY=VALUE` | none | Override one config entry; dotted keys, YAML values, repeatable |

Rerunning a pilot reuses what is already on disk: cached completions, calibrations, features and checkpoints. Delete a pilot's `results/` before a run whose configuration has changed.

## Reading a report

`report.md` opens with `PASS` or `FAIL`, then a table of checks. `report.json` holds the same checks, the full config, the environment (package versions, hardware, git commit) and the raw tables. Only `report.md`, `report.json`, `decisions.json` and an audit's `calibration.json`, `forecasts.json` and `lock.json` are committed.

Every check has an id (for example `P4.1.math`), a measured value, a threshold and one of three statuses:

| Status | Meaning |
| --- | --- |
| `pass` | The value met its threshold. |
| `fail` | The value missed its threshold. The pilot exits with code 1. |
| `documented` | Reported for the record and never gated, for example a diagnostic or a smoke run. |

## What decides the study

The pilots decide feasibility and the settings the preregistration delegates to them, never the confirmatory question. The rules are in [Go / no-go from the pilots](../PREREGISTRATION.md#go--no-go-from-the-pilots):

1. P0 and P1 pass.
2. P2 picks each domain's bucket and token budget; a domain with no qualifying bucket on its primary or backup source leaves the program.
3. P3 passes P3.2, P3.3, P3.4, P3.11 and P3.8 in every domain that stays.
4. P4.1 passes in every domain, after at most one rescreen on the backup source.
5. P5 passes for every model and hardware class in the program. P5.4 applies the seed rule, and P5.5's measured hours replace the placeholder budget.
6. P6 gates nothing. Its rates, intervals and adjudication go in the paper.

The decodability probes (P3.5, P3.6) and P4's one-seed fate contrasts are logged and do not decide. With one seed and a development held-out pool, a gate on a fate contrast would stop a real effect about one time in four. A pilot outcome is never a confirmatory result.

## Design evidence

These design choices rest on pilot runs with Qwen2.5-0.5B and Qwen2.5-1.5B on chain-sum arithmetic, made before the program moved to Qwen3 and five domains. Those reports are in the git history at the commits named. None used a held-out training outcome.

1. **The pooled tail `tau_pool` at `q = 1/32` is the primary estimator (P0).** On the 28-cell theory grid, the per-prompt `tau_rel` is inflated on prompts whose 16 samples miss the favoured behaviour (Spearman 0.777 with trained accuracy). `tau_pool` reaches 0.862 and lands within 0.10 of the trained outcome in 28 of 28 cells (`5c24601`).
2. **`fn_form` is a capped arm with its own control (P2).** A surface-form rule fires only as often as the policy writes a non-canonical answer, 4% to 8% of correct answers for Qwen2.5-0.5B, so no form rule reaches the 0.30 target. `fn_form` is compared at its achieved rate with `noise_fn_form`, and `noise_fn` is matched to `fn_behavior` (`5c24601`).
3. **Answer-based false positives and their controls share support (P2).** Qwen2.5-0.5B left 20% to 34% of outputs without a boxed answer. Unrestricted resampled noise would reward those outputs while the answer-based arms cannot, so `noise_fp` uses `support: answered` with its rate rescaled to match (`5c24601`).
4. **The audit uses the E1 pool size (P3).** At 256 train and 128 held-out prompts, the split-half cosine of `g_V` was 0.719 for the oracle and 0.587 for `noise_fp_trigger`, below the 0.8 gate. A noise model of `g_V` put the need at 281 to 719 train prompts (`6165ac8`). At 1,024 train and 384 held-out prompts, P3 passed with the oracle at 0.919 and the least reliable arm at 0.868, and the noise model's prediction was within 0.0062 of every measured cosine (`ccf19fa`). Every domain now audits 1,024 and 384.
5. **The endpoint is scored from saved adapters.** TRL's built-in evaluation logs pooled reward under the training sampler, which is not the preregistered endpoint. Every backend writes per-prompt held-out oracle accuracy from `evaluate_policy` at step 0 and at each saved checkpoint (`5c24601`).
6. **The go rule is feasibility only.** Pilots establish feasibility and do not test efficacy (Leon, Davis and Kraemer, 2011, *J Psychiatr Res* 45(5):626-629). P4's one-seed hit rule has power of about 0.75 at E1's 80% detectable gap, so fate contrasts are documented (`37f00b7`, `8481622`).
7. **Pre-freeze review.** Six independent reviews before any language-model training produced: one LoRA parameterisation shared by the screen and the trainer, features on the sampled token ids, a lock over every input, the `fp_tolerance_decoy` and `noise_fp_trigger` arms, word-boundary phrase arms validated on held-out train prompts, seed-paired contrasts, and development pools for every pilot (`b6087c5`).
