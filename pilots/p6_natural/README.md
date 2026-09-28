# P6 natural verifiers

Question: what do the natural verifiers do at step 0 on the dev pool?

Hardware: sampling is a 1.7B dev audit (vLLM). Judges are a vLLM server (`VFATES_JUDGE_BASE_URL` or `VFATES_QWEN32_BASE_URL` for Qwen3-32B). The API model reads `OPENAI_API_KEY` and defaults to gpt-4.1. Mac smoke (`--set smoke=true`) uses synthetic rows, no network, and does not apply a gate.

This pilot gates nothing. Every check is documented.

## What it records

For every natural verifier, step-0 FPR, FNR, Youden's J and precision on the dev pool, with prompt-cluster bootstrap intervals (`vfates.utils.stats.cluster_bootstrap`):

| domain | verifier |
| --- | --- |
| math | `natural`, `natural_strict` |
| code | `natural` |
| ifeval | `natural` |
| qa | `natural`, `natural_robust` |

Patch-category step-0 error counts, and whether `patch_tau` and `patch_rate` chose the same categories (`same_selection` per domain). When `forecasts.json` has arm records, this uses `vfates.verifiers.patch._same_selection` (tuple equality in menu order).

QA judge agreement: general-verifier vs the rule oracle vs an API model, on rows where the three disagree (`vfates.analysis.judge_agreement.agreement_statistics` and `vfates.tasks.qa.judge_agreement_row`). IFEval content judge vs gpt-4.1 on a dev sample (the study check uses 500; this pilot's `ifeval_n` defaults to 500, and a shorter dev pool only widens the interval).

Human adjudication: 100 step-0 false positives per natural verifier, deterministic in `(seed, item_id, pid)`, written to `<out>/adjudication/<domain>__<verifier>.csv`. Columns: `item_id`, `pid`, `prompt`, `response`, `gold`, `parsed`, `verifier_verdict`, `oracle_verdict`, `label` (blank; the annotator writes `real_error`, `oracle_wrong`, or `unclear`), `note`. Import reads the labelled CSVs and reports the fraction of `real_error` with a Wilson interval. An unknown label raises.

`export_oracle_check(audit_dir, out_csv, *, n=200)` writes the preregistered E1 QA check: held-out audit rows where the rule oracle, Qwen3-32B and an API model disagree. Columns add `rule_oracle`, `qwen3_32b`, `api_model`, plus `label` and `note`. Without `OPENAI_API_KEY` or a 32B endpoint the API stages are skipped and the check says so. Verdicts are never filled in locally.

The rates and the adjudication export read P3's dev-pool audits (`audit_dirs` in `config.yaml` points at `pilots/p3_estimator/results/<domain>/audit`), so P6 samples nothing of its own. With the default `--out`, the CSVs are `pilots/p6_natural/results/adjudication/<domain>__<verifier>.csv`, and the labelled CSVs are committed.

The dev pool is not the confirmatory held-out split. P3 audits with the largest pools the dev split allows, and a smaller pool can only widen the intervals.

## Commands

Mac smoke:

```bash
vfates pilot p6 --set smoke=true
```

After P3 has audited math, code, ifeval and qa on the development pool:

```bash
vfates pilot p6
```

Import after the annotator has filled `label`:

```bash
vfates pilot p6 --set stage=import
```

QA oracle check, on an audit's held-out rows:

```bash
vfates pilot p6 --set stage=oracle_check --set audit_dir=pilots/p3_estimator/results/qa/audit
```

## Runtime and cost

P6 adds no sampling: it reads P3's audits. Scoring them is minutes, on CPU.

API spend is not the study's judge budget. A few hundred gpt-4.1 calls is on the order of $20 if the key is set, and $0 when the key is absent and the check is skipped. Qwen3-32B is the server you already host; this pilot does not download it.

Mac smoke is seconds.

## Files

`report.md`, `report.json`, `tables.json`, and `<out>/adjudication/*.csv`. Nothing is deleted from the program on the basis of these numbers.

## What the study does with it

The fractions and intervals go in the paper as a step-0 description. The labelled CSVs are the human adjudication. The oracle-check CSV is the preregistered E1 QA disagreement sample. `same_selection` is the same fact the confirmatory analysis reads from `forecasts.json`.
