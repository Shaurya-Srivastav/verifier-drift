# P2 base audit

Question: for each domain, which difficulty bucket and token budget put both qwen3-1.7b and qwen3-4b in the preregistered band on the development pool?

## Rule

Copied from the preregistration. Not loosened.

For each domain, on the dev split, grade every difficulty bucket at the domain token budget with qwen3-1.7b and qwen3-4b. A bucket qualifies when both models have oracle accuracy in [0.15, 0.55], truncation at most 0.05, and answer-format rate (`task.support_answered`) at least 0.75. Ties go to the bucket whose two-model mean accuracy is closest to 0.35. Equal distance keeps the earlier bucket in the preregistered list.

If no bucket qualifies and some model truncates past 0.05, the budget rises to the next of 1024, 1536, 2048 and every bucket is graded again. A bucket that already qualifies keeps the budget it was graded at, even if another bucket still truncates.

If no bucket qualifies, the backup source is screened the same way. Math's backup is DeepScaleR (one bucket, no solve-rate filter). IFEval's backup is RLVR-IFeval, which has a development split. Code, QA and chain_sum have no backup: if no primary bucket qualifies, that domain leaves the program. Leaving is a completed decision, not a failed pilot. The pilot fails only when the screen cannot be measured.

Buckets: math `llama8b_solve_rate` intervals [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8); code `easy` (TACO EASY) and `medium` (TACO MEDIUM and MEDIUM_HARD together, because MEDIUM alone is below the 1,024 and 384 pools); IFEval constraint counts [2, 3] and [4, 5]; QA the dataset card's five difficulty values; chain_sum the single fixed bucket.

Sample size: the screen asks for 625 prompts and 4 samples per prompt, per domain, model, and bucket. A shorter development split is used in full, so the graded count is min(625, that split). Code EASY is about 295 development problems, so that bucket's prompt-mean standard error bound is about 0.5/sqrt(295) = 0.029 rather than 0.02. A mean of values in [0, 1] has standard deviation at most 0.5, so the bound is 0.5/sqrt(n) for the n that was graded, and the pooled completion-level bound is 0.5/sqrt(n * 4). `decisions.json` records that n and both bounds on each graded bucket. `cost_model.mean_completion_len` is the qwen3-1.7b mean completion length on the chosen bucket, rounded, because the cost model scales from the 1.7B reference.

On the qualifying bucket at 1.7B, the largest |achieved minus target| over targeted arms (not `form_fn`) is reported. The tolerance is 0.02. That comparison is documented, not a gate.

## Checks

| Id | Gated | Threshold |
| --- | --- | --- |
| P2.1 | gated on a real screen; documented on smoke | a decision for every domain in the run |
| P2.2.`domain` | gated | `qualified` or `leaves_program` |
| P2.3.`domain` | documented | targeted arms within 0.02 of the step-0 rate, 1.7B, qualifying bucket |
| P2.0 | gated | only when the audit cannot draw a development pool; the screen does not sample |

Smoke does not choose a bucket and does not write `decisions.json`.

## Hardware and commands

Mac smoke (no download):

```bash
python pilots/p2_base_audit/run.py --set smoke=true
```

Study screen. Pass the machine's class. chain_sum and math are assigned `rtx-a6000-48gb`. code, ifeval, and qa are assigned `rtx-5000-ada-32gb`. One process can grade every domain; vLLM uses the class you pass.

```bash
python pilots/p2_base_audit/run.py --set smoke=false --set hardware_class=rtx-a6000-48gb
```

Caches land in `results/cells/<domain>/<source>/<model>/<bucket>/<budget>/` and resume per domain, model, bucket, and budget.

## Runtime

About 8 GPU-hours on one A6000-class GPU if the first budget qualifies, and up to about 24 if every domain climbs the ladder and screens a backup. Basis, not a measurement: 14 primary buckets, 2 models, up to 625 prompts, 4 samples, about 400 tokens each, about 3500 tokens/s (the program's 10000 H100 tokens/s times `speed_vs_h100` 0.35). Code easy is shorter than 625, so this is an upper bound. A second budget multiplies the domains that truncated. Backup screens add one bucket each for math and ifeval.

## Files

`results/decisions.json` (schema `vfates.p2_decisions/1`) and `results/report.md`. Each graded domain records `samples.<source>.<bucket>` with `n_prompts`, `n_samples`, `requested_prompts`, `prompt_mean_se_bound`, and `pooled_se_bound`. The report lists the template edits: difficulty key and value, `max_completion_tokens`, `cost_model.mean_completion_len`, and `task_overrides` when the backup source wins. `apply` is true only for a non-smoke screen. Smoke never sets `apply`.

## How this feeds the study

The lead copies the edits onto the domain templates. A domain with `leaves_program` is removed from the program. A smoke report is not an edit list.
