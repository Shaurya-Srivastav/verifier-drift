# P3 estimator: FAIL

4 passed, 2 failed, 4 documented. Runtime 3s.

## Checks

| id | name | metric | value | threshold | status |
| --- | --- | --- | --- | --- | --- |
| P3.1 | held-out base accuracy and coverage (pass@n) | base acc (coverage 0.977) | 0.3906 | n/a | documented |
| P3.2 | velocity direction is reliable across prompt halves (oracle arm) | split-half cosine | 0.7192 | 0.8000 | fail |
| P3.3 | least reliable arm (noise_fp_trigger) | split-half cosine | 0.5873 | 0.8000 | fail |
| P3.10 | the noise model reproduces the measured split-half cosine | max \|predicted - measured\| across arms | 0.0127 | n/a | documented |
| P3.4 | CountSketch preserves LoRA-B gradient inner products on train rows | corr(exact, sketched) off-diagonal Gram | 0.9649 | 0.9500 | pass |
| P3.5 | prompt-specific FP (hash) is not decodable on unseen prompts | grouped-CV AUC | 0.5774 | 0.6000 | pass |
| P3.6 | a shared FP (fp_trigger) is decodable on unseen prompts | grouped-CV AUC | 0.7242 | 0.6500 | pass |
| P3.7 | hidden-state features rank arms like LoRA-B gradients | Spearman of tau_pool across arms | 0.5692 | n/a | documented |
| P3.9 | LoRA seed 1 does not define the screen by itself | Spearman across arms (max \|d tau_pool\|=0.0781) | 0.3340 | n/a | documented |
| P3.8 | forecasts are locked before training | lock verifies | n/a | n/a | pass |

## Locked forecasts

| arm | FPR0 | FNR0 | J0 | Prec0 mean | Prec0 modal | tau_pool | 95% CI | tau_hard | cos(gV,gO) | reliability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combo_fnform_fphash | 0.1500 | 0.0593 | 0.7907 | 0.7809 | 0.9367 | 0.5469 | [0.3594, 0.6562] | 0.5234 | 0.9408 | 0.7130 |
| fn_behavior | 0.0000 | 0.2934 | 0.7066 | 0.9707 | 0.9707 | 0.4688 | [0.3438, 0.6250] | 0.4766 | 0.4275 | 0.7956 |
| fn_form | 0.0000 | 0.0593 | 0.9407 | 0.9883 | 0.9883 | 0.5312 | [0.3906, 0.6875] | 0.5703 | 0.9921 | 0.7093 |
| fn_prompt_subset | 0.0000 | 0.2995 | 0.7005 | 0.8071 | 0.8071 | 0.5312 | [0.4062, 0.6875] | 0.5781 | 0.9580 | 0.6719 |
| fp_hash | 0.1500 | 0.0000 | 0.8500 | 0.7872 | 0.9458 | 0.5156 | [0.3594, 0.6645] | 0.5234 | 0.9533 | 0.7225 |
| fp_tolerance | 0.1500 | 0.0000 | 0.8500 | 0.8012 | 0.9313 | 0.5469 | [0.3906, 0.6875] | 0.5781 | 0.9647 | 0.7325 |
| fp_tolerance_decoy | 0.1496 | 0.0000 | 0.8504 | 0.8051 | 0.9521 | 0.4844 | [0.3594, 0.6719] | 0.5547 | 0.9551 | 0.7158 |
| fp_trigger | 0.1496 | 0.0000 | 0.8504 | 0.7797 | 0.9357 | 0.5938 | [0.3750, 0.6957] | 0.5547 | 0.9441 | 0.6363 |
| natural_string_match | 0.0000 | 0.0905 | 0.9095 | 0.9883 | 0.9883 | 0.5469 | [0.4062, 0.6801] | 0.5859 | 0.9896 | 0.7106 |
| noise_fn | 0.0000 | 0.2818 | 0.7182 | 0.9631 | 0.9631 | 0.5156 | [0.3906, 0.6875] | 0.5781 | 0.9589 | 0.5954 |
| noise_fn_form | 0.0000 | 0.0477 | 0.9523 | 0.9846 | 0.9846 | 0.5312 | [0.4062, 0.6719] | 0.5938 | 0.9953 | 0.7134 |
| noise_fp | 0.1455 | 0.0000 | 0.8545 | 0.7830 | 0.9515 | 0.5469 | [0.3668, 0.6719] | 0.5547 | 0.9576 | 0.6934 |
| noise_fp_trigger | 0.1537 | 0.0000 | 0.8463 | 0.7755 | 0.9378 | 0.5469 | [0.4062, 0.6957] | 0.5703 | 0.9681 | 0.5873 |
| oracle | 0.0000 | 0.0000 | 1.0000 | 0.9883 | 0.9883 | 0.5469 | [0.4062, 0.6957] | 0.5938 | 1.0000 | 0.7192 |

## Reliability budget

Split-half cosine of `g_V` against the noise model `signal / (signal + noise / m)` with `m` prompts per half. The within-prompt share of the noise shrinks with samples per prompt; the rest only with prompts. The last column is the train pool that reaches 0.8 at 16 samples per prompt.

| arm | measured | predicted | within-prompt share | train prompts for gate |
| --- | --- | --- | --- | --- |
| combo_fnform_fphash | 0.7130 | 0.7061 | 0.9787 | 427 |
| fn_behavior | 0.7956 | 0.7848 | 0.9295 | 281 |
| fn_form | 0.7093 | 0.7047 | 0.9909 | 430 |
| fn_prompt_subset | 0.6719 | 0.6592 | 0.9142 | 530 |
| fp_hash | 0.7225 | 0.7152 | 0.9826 | 408 |
| fp_tolerance | 0.7325 | 0.7333 | 0.9714 | 373 |
| fp_tolerance_decoy | 0.7158 | 0.7171 | 1.0000 | 405 |
| fp_trigger | 0.6363 | 0.6327 | 0.9776 | 595 |
| noise_fn | 0.5954 | 0.5876 | 0.9759 | 719 |
| noise_fn_form | 0.7134 | 0.7073 | 0.9630 | 424 |
| noise_fp | 0.6934 | 0.6979 | 0.9878 | 444 |
| noise_fp_trigger | 0.5873 | 0.5892 | 0.9856 | 714 |
| oracle | 0.7192 | 0.7146 | 0.9942 | 409 |

## Decodeability probe

| arm | type | AUC | rows | errors |
| --- | --- | --- | --- | --- |
| fn_behavior | behavior_fn | 0.9952 | 1636 | 480 |
| fn_form | form_fn | 0.8010 | 1636 | 97 |
| fn_prompt_subset | prompt_subset_fn | 0.4779 | 1636 | 490 |
| fp_hash | hash_fp | 0.5774 | 2460 | 369 |
| fp_tolerance | tolerance_fp | 0.6235 | 2460 | 369 |
| fp_trigger | trigger_fp | 0.7242 | 2460 | 368 |

## Sketch fidelity

| dim_exact | dim_sketch | gram_rel_error | n | offdiag_corr |
| --- | --- | --- | --- | --- |
| 4866048 | 8192 | 0.0606 | 48 | 0.9649 |

## LoRA seed

Features were extracted again with lora_seed 1 into `lora_seed_1/`. Spearman of tau_pool across arms is 0.3340; the largest absolute tau_pool change is 0.0781. This check is documented, not a gate: it is evidence that the screen does not hinge on one draw of LoRA A.
