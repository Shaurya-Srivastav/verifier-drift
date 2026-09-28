# P2 base audit: PASS

4 passed, 0 failed, 3 documented. Runtime 201s.

Recommended cell for qwen2.5-0.5b: **qwen2.5-0.5b__easy** (base accuracy 0.397).
No feasible cell for qwen2.5-1.5b.
Recommended cell for qwen2.5-1.5b-1024: **qwen2.5-1.5b-1024__egashira** (base accuracy 0.215).

## Checks

| id | name | metric | value | threshold | status |
| --- | --- | --- | --- | --- | --- |
| P2.1 | a cell has base accuracy in band, boxed format, and finishes in budget | feasible cells | 2.0000 | acc in [0.15, 0.5], boxed >= 0.75, trunc <= 0.05 | pass |
| P2.2 | the Mac pilot model (qwen2.5-0.5b) has a feasible cell | feasible cells | 1.0000 | >= 1 | pass |
| P2.3 | every targeted arm hits its rate on qwen2.5-0.5b__easy | max \|achieved - target\| | 0.0052 | 0.0300 | pass |
| P2.4 | every targeted arm flags enough samples to train on | min flagged samples | 93.0000 | 30 | pass |
| P2.5 | generation throughput of the chosen cell | tokens/s | 1560.9637 | n/a | documented |
| P2.6.fn_form | fn_form is capped by how often correct answers are non-canonical | achieved FNR (34 flagged); control matched to it | 0.0835 | n/a | documented |
| P2.7 | the study model (qwen2.5-1.5b-1024) has a feasible cell: qwen2.5-1.5b-1024__egashira | base accuracy of the recommended cell for the H100 gate | 0.2148 | n/a | documented |

## Sweep

| cell | max tok | acc | pass@n | boxed | answered | trunc | tok mean | tok p95 | tok/s | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b__egashira | 512 | 0.0596 | 0.2422 | 0.6621 | 0.6523 | 0.2314 | 371.3857 | 512.0000 | 2877.8053 | no |
| qwen2.5-0.5b__medium | 512 | 0.2510 | 0.8125 | 0.7666 | 0.7607 | 0.0879 | 305.1641 | 512.0000 | 1558.7672 | no |
| qwen2.5-0.5b__easy | 512 | 0.3975 | 0.8984 | 0.8018 | 0.7949 | 0.0430 | 258.8828 | 502.8500 | 1560.9637 | yes |
| qwen2.5-1.5b__egashira | 512 | 0.1963 | 0.6172 | 0.8271 | 0.8232 | 0.1533 | 338.2109 | 512.0000 | 839.6849 | no |
| qwen2.5-1.5b__medium | 512 | 0.4688 | 0.9375 | 0.8418 | 0.8379 | 0.1260 | 295.9785 | 512.0000 | 456.7063 | no |
| qwen2.5-1.5b__easy | 512 | 0.6260 | 0.9766 | 0.9258 | 0.9258 | 0.0586 | 269.5059 | 512.0000 | 493.3666 | no |
| qwen2.5-1.5b-1024__egashira | 1024 | 0.2148 | 0.6562 | 0.9668 | 0.9658 | 0.0059 | 357.0645 | 671.1000 | 670.7952 | yes |
| qwen2.5-1.5b-1024__medium | 1024 | 0.5273 | 0.9844 | 0.9590 | 0.9551 | 0.0049 | 313.7168 | 634.6500 | 599.2567 | no |

## Arms on qwen2.5-0.5b__egashira

| arm | type | FPR | FNR | J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1495 | 0.0000 | 0.8505 | 0.2976 | 144 |
| fp_tolerance | tolerance_fp | 0.1495 | 0.0000 | 0.8505 | 0.2976 | 144 |
| fp_trigger | trigger_fp | 0.1485 | 0.0000 | 0.8515 | 0.2990 | 143 |
| noise_fp | resampled | 0.1495 | 0.0000 | 0.8536 | 0.3020 | n/a |
| fn_form | form_fn | 0.0000 | 0.0656 | 0.9344 | 1.0000 | 4 |
| fn_behavior | behavior_fn | 0.0000 | 0.2951 | 0.7049 | 1.0000 | 18 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.3279 | 0.6721 | 1.0000 | 20 |
| noise_fn | resampled | 0.0000 | 0.2951 | 0.7541 | 1.0000 | n/a |
| noise_fn_form | resampled | 0.0000 | 0.0656 | 0.9180 | 1.0000 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0656 | 0.9344 | 1.0000 | n/a |

## Arms on qwen2.5-0.5b__medium

| arm | type | FPR | FNR | J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1499 | 0.0000 | 0.8501 | 0.6909 | 115 |
| fp_tolerance | tolerance_fp | 0.1499 | 0.0000 | 0.8501 | 0.6909 | 115 |
| fp_trigger | trigger_fp | 0.1499 | 0.0000 | 0.8501 | 0.6909 | 115 |
| noise_fp | resampled | 0.1499 | 0.0000 | 0.8475 | 0.6872 | n/a |
| fn_form | form_fn | 0.0000 | 0.0389 | 0.9611 | 1.0000 | 10 |
| fn_behavior | behavior_fn | 0.0000 | 0.2996 | 0.7004 | 1.0000 | 77 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2957 | 0.7043 | 1.0000 | 76 |
| noise_fn | resampled | 0.0000 | 0.2996 | 0.7626 | 1.0000 | n/a |
| noise_fn_form | resampled | 0.0000 | 0.0389 | 0.9805 | 1.0000 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0389 | 0.9611 | 1.0000 | n/a |

## Arms on qwen2.5-0.5b__easy

| arm | type | FPR | FNR | J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1507 | 0.0000 | 0.8493 | 0.8140 | 93 |
| fp_tolerance | tolerance_fp | 0.1524 | 0.0000 | 0.8476 | 0.8124 | 94 |
| fp_trigger | trigger_fp | 0.1507 | 0.0000 | 0.8493 | 0.8140 | 93 |
| noise_fp | resampled | 0.1524 | 0.0000 | 0.8379 | 0.8028 | n/a |
| fn_form | form_fn | 0.0000 | 0.0835 | 0.9165 | 1.0000 | 34 |
| fn_behavior | behavior_fn | 0.0000 | 0.2948 | 0.7052 | 1.0000 | 120 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2998 | 0.7002 | 1.0000 | 122 |
| noise_fn | resampled | 0.0000 | 0.2948 | 0.7420 | 1.0000 | n/a |
| noise_fn_form | resampled | 0.0000 | 0.0835 | 0.9238 | 1.0000 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0835 | 0.9165 | 1.0000 | n/a |

## Arms on qwen2.5-1.5b__egashira

| arm | type | FPR | FNR | J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1482 | 0.0000 | 0.8518 | 0.6223 | 122 |
| fp_tolerance | tolerance_fp | 0.1543 | 0.0000 | 0.8457 | 0.6128 | 127 |
| fp_trigger | trigger_fp | 0.1507 | 0.0000 | 0.8493 | 0.6185 | 124 |
| noise_fp | resampled | 0.1543 | 0.0000 | 0.8518 | 0.6223 | n/a |
| fn_form | form_fn | 0.0000 | 0.1393 | 0.8607 | 1.0000 | 28 |
| fn_behavior | behavior_fn | 0.0000 | 0.3035 | 0.6965 | 1.0000 | 61 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.3134 | 0.6866 | 1.0000 | 63 |
| noise_fn | resampled | 0.0000 | 0.3035 | 0.7164 | 1.0000 | n/a |
| noise_fn_form | resampled | 0.0000 | 0.1393 | 0.8806 | 1.0000 | n/a |
| natural_string_match | string_match | 0.0000 | 0.1393 | 0.8607 | 1.0000 | n/a |

## Arms on qwen2.5-1.5b__medium

| arm | type | FPR | FNR | J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1507 | 0.0000 | 0.8493 | 0.8541 | 82 |
| fp_tolerance | tolerance_fp | 0.1507 | 0.0000 | 0.8493 | 0.8541 | 82 |
| fp_trigger | trigger_fp | 0.1507 | 0.0000 | 0.8493 | 0.8541 | 82 |
| noise_fp | resampled | 0.1507 | 0.0000 | 0.8566 | 0.8602 | n/a |
| fn_form | form_fn | 0.0000 | 0.0375 | 0.9625 | 1.0000 | 18 |
| fn_behavior | behavior_fn | 0.0000 | 0.3021 | 0.6979 | 1.0000 | 145 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2979 | 0.7021 | 1.0000 | 143 |
| noise_fn | resampled | 0.0000 | 0.3021 | 0.6958 | 1.0000 | n/a |
| noise_fn_form | resampled | 0.0000 | 0.0375 | 0.9750 | 1.0000 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0375 | 0.9625 | 1.0000 | n/a |

## Arms on qwen2.5-1.5b__easy

| arm | type | FPR | FNR | J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1488 | 0.0000 | 0.8512 | 0.9183 | 57 |
| fp_tolerance | tolerance_fp | 0.1488 | 0.0000 | 0.8512 | 0.9183 | 57 |
| fp_trigger | trigger_fp | 0.1488 | 0.0000 | 0.8512 | 0.9183 | 57 |
| noise_fp | resampled | 0.1488 | 0.0000 | 0.8695 | 0.9276 | n/a |
| fn_form | form_fn | 0.0000 | 0.0702 | 0.9298 | 1.0000 | 45 |
| fn_behavior | behavior_fn | 0.0000 | 0.3089 | 0.6911 | 1.0000 | 198 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2995 | 0.7005 | 1.0000 | 192 |
| noise_fn | resampled | 0.0000 | 0.3089 | 0.6927 | 1.0000 | n/a |
| noise_fn_form | resampled | 0.0000 | 0.0702 | 0.9376 | 1.0000 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0702 | 0.9298 | 1.0000 | n/a |

## Arms on qwen2.5-1.5b-1024__egashira

| arm | type | FPR | FNR | J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1505 | 0.0000 | 0.8495 | 0.6452 | 121 |
| fp_tolerance | tolerance_fp | 0.1455 | 0.0000 | 0.8545 | 0.6528 | 117 |
| fp_trigger | trigger_fp | 0.1505 | 0.0000 | 0.8495 | 0.6452 | 121 |
| noise_fp | resampled | 0.1455 | 0.0000 | 0.8619 | 0.6647 | n/a |
| fn_form | form_fn | 0.0000 | 0.1227 | 0.8773 | 1.0000 | 27 |
| fn_behavior | behavior_fn | 0.0000 | 0.2864 | 0.7136 | 1.0000 | 63 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2955 | 0.7045 | 1.0000 | 65 |
| noise_fn | resampled | 0.0000 | 0.2864 | 0.7409 | 1.0000 | n/a |
| noise_fn_form | resampled | 0.0000 | 0.1227 | 0.9136 | 1.0000 | n/a |
| natural_string_match | string_match | 0.0000 | 0.1227 | 0.8773 | 1.0000 | n/a |

## Arms on qwen2.5-1.5b-1024__medium

| arm | type | FPR | FNR | J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1508 | 0.0000 | 0.8492 | 0.8809 | 73 |
| fp_tolerance | tolerance_fp | 0.1488 | 0.0000 | 0.8512 | 0.8824 | 72 |
| fp_trigger | trigger_fp | 0.1508 | 0.0000 | 0.8492 | 0.8809 | 73 |
| noise_fp | resampled | 0.1488 | 0.0000 | 0.8698 | 0.8955 | n/a |
| fn_form | form_fn | 0.0000 | 0.0463 | 0.9537 | 1.0000 | 25 |
| fn_behavior | behavior_fn | 0.0000 | 0.3093 | 0.6907 | 1.0000 | 167 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2963 | 0.7037 | 1.0000 | 160 |
| noise_fn | resampled | 0.0000 | 0.3093 | 0.6926 | 1.0000 | n/a |
| noise_fn_form | resampled | 0.0000 | 0.0463 | 0.9593 | 1.0000 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0463 | 0.9537 | 1.0000 | n/a |
