# P3 estimator: PASS

4 passed, 0 failed, 6 documented. Runtime 6671s.

## Checks

| id | name | metric | value | threshold | status |
| --- | --- | --- | --- | --- | --- |
| P3.1 | held-out base accuracy and coverage (pass@n) | base acc (coverage 0.974) | 0.3997 | n/a | documented |
| P3.2 | velocity direction is reliable across prompt halves (oracle arm) | split-half cosine | 0.9192 | 0.8000 | pass |
| P3.3 | least reliable arm (noise_fn) | split-half cosine | 0.8682 | 0.8000 | pass |
| P3.10 | the noise model reproduces the measured split-half cosine | max \|predicted - measured\| across arms | 0.0062 | 0.0500 | documented |
| P3.4 | CountSketch preserves LoRA-B gradient inner products on train rows | corr(exact, sketched) off-diagonal Gram | 0.9589 | 0.9500 | pass |
| P3.5 | prompt-specific FP (hash) is not decodable on unseen prompts | grouped-CV AUC; as expected | 0.5610 | 0.6000 | documented |
| P3.6 | a shared FP (fp_trigger) is decodable on unseen prompts | grouped-CV AUC; as expected | 0.9757 | 0.6500 | documented |
| P3.7 | hidden-state features rank arms like LoRA-B gradients | Spearman of tau_pool across arms | 0.6625 | n/a | documented |
| P3.9 | LoRA seed 1 does not define the screen by itself | Spearman across arms (max \|d tau_pool\|=0.0312) | 0.8580 | n/a | documented |
| P3.8 | forecasts are locked before training | lock verifies | n/a | n/a | pass |

## Locked forecasts

| arm | FPR0 | FNR0 | J0 | Prec0 mean | Prec0 modal | tau_pool | 95% CI | tau_hard | cos(gV,gO) | reliability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combo_fnform_fphash | 0.1500 | 0.0410 | 0.8090 | 0.7670 | 0.9323 | 0.5781 | [0.4764, 0.6590] | 0.5990 | 0.9573 | 0.9115 |
| fn_behavior | 0.0000 | 0.3020 | 0.6980 | 0.9645 | 0.9645 | 0.5885 | [0.5156, 0.6562] | 0.5130 | 0.3919 | 0.9506 |
| fn_form | 0.0000 | 0.0410 | 0.9590 | 0.9769 | 0.9769 | 0.6719 | [0.5833, 0.7371] | 0.6745 | 0.9972 | 0.9150 |
| fn_prompt_subset | 0.0000 | 0.3003 | 0.6997 | 0.8001 | 0.8001 | 0.6771 | [0.5885, 0.7344] | 0.6719 | 0.9895 | 0.8890 |
| fp_hash | 0.1500 | 0.0000 | 0.8500 | 0.7753 | 0.9412 | 0.5781 | [0.4896, 0.6590] | 0.6094 | 0.9648 | 0.9158 |
| fp_tolerance | 0.1505 | 0.0000 | 0.8495 | 0.7914 | 0.9226 | 0.6250 | [0.5417, 0.6927] | 0.6432 | 0.9793 | 0.9263 |
| fp_tolerance_decoy | 0.1500 | 0.0000 | 0.8500 | 0.8008 | 0.9505 | 0.5938 | [0.5208, 0.6771] | 0.6172 | 0.9677 | 0.9137 |
| fp_trigger | 0.1474 | 0.0000 | 0.8526 | 0.7857 | 0.9390 | 0.5938 | [0.5052, 0.6667] | 0.6224 | 0.9194 | 0.9034 |
| natural_string_match | 0.0000 | 0.0645 | 0.9355 | 0.9741 | 0.9741 | 0.6667 | [0.5781, 0.7292] | 0.6771 | 0.9965 | 0.9139 |
| noise_fn | 0.0000 | 0.2950 | 0.7050 | 0.9580 | 0.9580 | 0.6771 | [0.5858, 0.7344] | 0.6615 | 0.9874 | 0.8682 |
| noise_fn_form | 0.0000 | 0.0413 | 0.9587 | 0.9777 | 0.9777 | 0.6667 | [0.5910, 0.7344] | 0.6719 | 0.9986 | 0.9144 |
| noise_fp | 0.1474 | 0.0000 | 0.8526 | 0.7747 | 0.9440 | 0.6094 | [0.5260, 0.6927] | 0.6354 | 0.9754 | 0.9163 |
| noise_fp_trigger | 0.1483 | 0.0000 | 0.8517 | 0.7748 | 0.9400 | 0.6771 | [0.5781, 0.7292] | 0.6667 | 0.9909 | 0.8812 |
| oracle | 0.0000 | 0.0000 | 1.0000 | 0.9795 | 0.9795 | 0.6667 | [0.5910, 0.7344] | 0.6667 | 1.0000 | 0.9192 |

## Reliability budget

Split-half cosine of `g_V` against the noise model `signal / (signal + noise / m)` with `m` prompts per half. The within-prompt share of the noise (a jackknife estimate, not capped at 1) shrinks with samples per prompt; the rest only with prompts. The last column is the train pool that reaches 0.8 at 16 samples per prompt.

| arm | measured | predicted | within-prompt share | train prompts for gate |
| --- | --- | --- | --- | --- |
| combo_fnform_fphash | 0.9115 | 0.9093 | 1.0254 | 409 |
| fn_behavior | 0.9506 | 0.9485 | 1.0300 | 223 |
| fn_form | 0.9150 | 0.9113 | 1.0127 | 399 |
| fn_prompt_subset | 0.8890 | 0.8828 | 0.9983 | 544 |
| fp_hash | 0.9158 | 0.9137 | 1.0279 | 387 |
| fp_tolerance | 0.9263 | 0.9233 | 1.0387 | 341 |
| fp_tolerance_decoy | 0.9137 | 0.9130 | 1.0202 | 391 |
| fp_trigger | 0.9034 | 0.9000 | 1.0209 | 456 |
| noise_fn | 0.8682 | 0.8650 | 1.0447 | 640 |
| noise_fn_form | 0.9144 | 0.9112 | 1.0196 | 400 |
| noise_fp | 0.9163 | 0.9132 | 1.0307 | 390 |
| noise_fp_trigger | 0.8812 | 0.8783 | 1.0245 | 568 |
| oracle | 0.9192 | 0.9158 | 1.0160 | 377 |

## Decodeability probe

| arm | type | AUC | rows | errors |
| --- | --- | --- | --- | --- |
| fn_behavior | behavior_fn | 0.9991 | 5000 | 1496 |
| fn_form | form_fn | 0.8509 | 5000 | 204 |
| fn_prompt_subset | prompt_subset_fn | 0.4776 | 5000 | 1540 |
| fp_hash | hash_fp | 0.5610 | 5000 | 747 |
| fp_tolerance | tolerance_fp | 0.6741 | 5000 | 750 |
| fp_trigger | trigger_fp | 0.9757 | 5000 | 734 |

## Sketch fidelity

| dim_exact | dim_sketch | gram_rel_error | n | offdiag_corr |
| --- | --- | --- | --- | --- |
| 4866048 | 8192 | 0.0619 | 48 | 0.9589 |

## LoRA seed

Features were extracted again with lora_seed 1 into `lora_seed_1/`. Spearman of tau_pool across arms is 0.8580; the largest absolute tau_pool change is 0.0312. This check is documented, not a gate: it is evidence that the screen does not hinge on one draw of LoRA A.
