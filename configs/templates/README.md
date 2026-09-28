# Domain templates

`vfates program render` reads `configs/templates/<domain>.yaml`, the model spec, and one cell from `configs/program.yaml`, then writes `configs/cells/<id>.yaml`.

Cell id: `<experiment>-<domain>-<model>` in lower case, for example `e1-math-qwen3-1.7b`.

Every template has the same shape. A new domain copies it; there is one menu schema.

## Placeholders

A placeholder is a whole YAML scalar, quoted, of the form `"${a.b}"`. Render replaces it with the object (string, number, mapping, or list). A string that only contains a placeholder in passing is an error.

| Placeholder | Source |
| --- | --- |
| `${model.model_id}` | model spec `model_id` |
| `${model.revision}` | model spec `revision` (Hub commit) |
| `${model.chat_template_kwargs}` | model spec `chat_template_kwargs` |
| `${model.lr}` | model spec `lr` |
| `${model.lora_keys}` | model spec `lora_keys` (short names, for `study.train.lora.keys`) |
| `${model.lora_target_modules}` | those keys mapped to PEFT suffixes (`q` to `q_proj`) |
| `${placement.gpus}` | `placement_for(hardware_class).gpus` |
| `${placement.tp}` | tensor parallel size |
| `${placement.micro_batch_size}` | must divide `prompts_per_step * group_size` (512 in the shared protocol) |
| `${placement.vllm_gpu_memory_utilization}` | vLLM fraction |
| `${cell.hardware_class}` | the cell's hardware class |
| `${cell.seeds}` | the cell's seed list |
| `${menu.arms}` | the selected menu |

Set the same `chat_template_kwargs` on `audit.generator`, `audit.generator.kwargs`, `audit.features`, `study.train`, and `study.eval_generator.kwargs`. `run_study` refuses to start when those values differ.

`study.train` must stay a `GRPOConfig`: use `model_revision` (not `revision`) and `vllm_tensor_parallel_size` (not a free-form `tp` key).

## Menus

```yaml
menus:
  full: [oracle, ...]
  core: [oracle, ...]
analysis:
  hypotheses: true
  early: {enabled: true, seed: 99, steps: 25}
  by_menu:
    full:
      h1:
        contrasts: [{control: noise_fp_trigger, arm: fp_trigger}]
        beside: []
      h2:
        contrasts: [{control: noise_fp, arm: fp_hash}]
      h3:
        min_pairs: 4
        pairs: [{arm: fp_hash, control: noise_fp}]
      h4:
        shared: [fp_trigger]
        prompt_specific: [fp_hash]
        shared_min: 0.65
        prompt_specific_max: 0.60
      natural: {natural: natural, control: noise_natural, oracle: oracle}
    core:
      # same blocks, and every named arm must be in menus.core
```

Rules the renderer checks:

- `menus` has `full` and `core`. `core` is a subset of `full`.
- `analysis.by_menu` has exactly those keys.
- Every menu has `h1.beside`. Use `beside: []` when there is no beside contrast. If the key is missing, the confirmatory report falls back to the chain-sum decoy pair.
- h1 contrasts, h1 beside, h2 contrasts, h3 pairs, h4 shared and prompt-specific lists, and natural (`natural`, `control`, `oracle`) may name only arms in that menu.
- Render copies the selected block onto `analysis` and drops `menus` and `by_menu`, so the cell file is a normal study config.

Kit arms, on every domain (`configs/arms/kit.yaml`; chain_sum keeps the same arms in `configs/arms/chain_sum.yaml`):

- in both menus: `oracle`, `fp_hash`, `fp_trigger`, `noise_fp`, `noise_fp_trigger`, `fn_behavior`, `fn_prompt_subset`, `noise_fn`;
- in the full menu only: `fp_trigger_enriched`, `noise_fp_trigger_enriched`, `fn_behavior_depleted`, `noise_fn_depleted` (the H10 pairs).

Core menu (E2, E3): the eight kit arms of both menus plus `natural` and `noise_natural` (10 arms); chain_sum has the eight kit arms only.

Full menu (E1):

| Domain | Arms | Count |
| --- | --- | --- |
| chain_sum | the twelve kit arms, `fp_tolerance`, `fp_tolerance_decoy`, `fn_form`, `noise_fn_form`, `combo_fnform_fphash` | 17 |
| math | the twelve kit arms, `natural`, `noise_natural`, `natural_strict`, `noise_natural_strict`, `fp_tolerance`, `noise_fp_tolerance`, `patch_tau`, `patch_rate`, `random` | 21 |
| code, ifeval | the twelve kit arms, `natural`, `noise_natural`, `patch_tau`, `patch_rate` | 16 |
| qa | the code set plus `natural_robust` and `noise_natural_robust` | 18 |

H1 kit contrasts: `noise_fp_trigger - fp_trigger` and `noise_fn - fn_behavior`, plus `noise_fp - fp_tolerance` in chain_sum and `noise_fp_tolerance - fp_tolerance` in math. H2: `noise_fp - fp_hash` and `noise_fn - fn_prompt_subset`. H10 (full menu): `fp_trigger_enriched` against `fp_trigger` and `fn_behavior_depleted` against `fn_behavior`. The audit calibrates every arm in the arms file; withdrawals cover only the cell's menu, and the rest are listed under `outside_menu` in `withdrawals.json`.

Max completion tokens, in both `audit.sampling.max_tokens` and `study.train.max_tokens` / `eval_max_tokens`: chain_sum 1024, math 1536, code 1024, ifeval 768, qa 1024.

## Arms files

The template points `audit.arms.file` at `configs/arms/<domain>.yaml`. The kit (`configs/arms/kit.yaml`) sets targets FPR 0.15 and FNR 0.30. `noise_fp` matches `fp_hash` with support `answered` (on chain_sum it matches `fp_tolerance`), `noise_fp_trigger` and `noise_fp_trigger_enriched` match their rules with support `all`, and `noise_fn` and `noise_fn_depleted` match theirs.

A domain file includes the kit, resolved relative to itself, and then lists its own arms:

```yaml
include: kit.yaml
domain: ifeval
arms:
  - {name: natural, type: ifeval_checker}
  - {name: noise_natural, type: resampled, match: natural, support: all, seed: 0}
  - {name: patch_tau, type: patch_select, base: natural, k: 2, by: tau}
  - {name: patch_rate, type: patch_select, base: natural, k: 2, by: rate}
```

`include` may be a string or a list of paths. Included `arms` come first. The file's own `targets` override included targets. Code's natural verifier is the statement's public examples, math's are the lenient cascade and strip-string match, and qa's is TIGER-Lab's general-verifier.

## Cost model

Each template has a `cost_model`. Rates are single-GPU H100 token rates for a model of `reference_params_b` (1.7 in the shipped templates). The planner multiplies a job's token-hours by `params_b / reference_params_b`. It does not multiply or divide by `placement.gpus`: the rates are per GPU, and a multi-GPU job is assumed to scale throughput with the GPU count.

An audit job prices audit tokens as generation only. A train job prices rollout tokens, eval tokens, and a train-token pass. An early job uses `early_steps`. A transfer job prices `transfer_prompts` (else `eval_prompts`) times `eval_samples` times `mean_completion_len` as generation only.

Domains without a template yet use `default_cost_model` in `configs/program.yaml`, with `domain_mean_completion_len` overriding `mean_completion_len`. Those lengths are placeholders until the domain template exists. Re-run `vfates program plan` after adding the template.

## Variants

A cell may declare `study.variants`. Render copies the list from `configs/program.yaml` into the cell config. `placement: full` is resolved at render time into `train_overrides` (`micro_batch_size`, `vllm_gpu_memory_utilization`, `vllm_tensor_parallel_size`) and a `gpus` count from `ModelSpec.placement_full_for`. Those overrides are not written into the cell's confirmatory `study.train` block.

```yaml
variants:
  - name: kl
    arms: [oracle, fp_trigger]
    seeds: [0]
    train_overrides:
      beta_kl: 0.01
  - name: full
    arms: [oracle, fp_trigger, noise_fp_trigger]
    seeds: [0]
    placement: full
    accelerate_config: configs/accelerate/fsdp_2gpu_full.yaml
    train_overrides:
      full_finetune: true
      lora: null
      lora_init: null
      gradient_checkpointing: true
```

The pooled analysis reads variant runs from exactly this layout (there is no extra `runs/` directory under the variant name):

```
runs/<cell>/variants/<name>/<arm>__s<seed>/
```

Job id: `<cell>/variant/<name>/<arm>/s<seed>`. The job depends on the cell audit, uses that audit and lock, and is exploratory. `vfates study --only train --arm A --seed S --variant NAME --out runs/<cell>` writes the directory above. A variant is not required for the confirmatory cell report. A variant `steps` override is applied after the check that `analysis.final_step` equals the cell's confirmatory `study.train.steps`.
