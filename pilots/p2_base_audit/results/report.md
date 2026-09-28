# P2 base audit: PASS

4 passed, 0 failed, 4 documented. Runtime 1534s.

Recommended cell for qwen2.5-0.5b: **qwen2.5-0.5b__easy** (base accuracy 0.399).
No feasible cell for qwen2.5-1.5b.
Recommended cell for qwen2.5-1.5b-1024: **qwen2.5-1.5b-1024__egashira** (base accuracy 0.215).

## Checks

| id | name | metric | value | threshold | status |
| --- | --- | --- | --- | --- | --- |
| P2.1 | a cell has base accuracy in band, boxed format, and finishes in budget | feasible cells | 2.0000 | acc in [0.15, 0.5], boxed >= 0.75, trunc <= 0.05 | pass |
| P2.2 | the Mac pilot model (qwen2.5-0.5b) has a feasible cell | feasible cells | 1.0000 | >= 1 | pass |
| P2.3 | every targeted arm hits its rate on qwen2.5-0.5b__easy | max \|achieved - target\| | 0.0066 | 0.0200 | pass |
| P2.4 | every targeted arm flags enough samples to train on | min flagged samples | 92.0000 | 30 | pass |
| P2.5 | generation throughput of the chosen cell | tokens/s | 1560.9637 | n/a | documented |
| P2.6.fn_form | fn_form is capped by how often correct answers are non-canonical | channel FNR (19 flagged); control matched to it | 0.0465 | n/a | documented |
| P2.8 | agreement of natural_string_match with fn_form on qwen2.5-0.5b__easy | share of audit rows with equal verdicts | 0.9834 | n/a | documented |
| P2.7 | the study model (qwen2.5-1.5b-1024) has a feasible cell: qwen2.5-1.5b-1024__egashira | base accuracy of the recommended cell for the H100 gate | 0.2148 | n/a | documented |

## Sweep

| cell | max tok | acc | pass@n | boxed | answered | trunc | tok mean | tok p95 | tok/s | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b__egashira | 512 | 0.0596 | 0.2422 | 0.6621 | 0.6523 | 0.2314 | 371.3857 | 512.0000 | 2877.8053 | no |
| qwen2.5-0.5b__medium | 512 | 0.2510 | 0.8125 | 0.7666 | 0.7617 | 0.0879 | 305.1641 | 512.0000 | 1558.7672 | no |
| qwen2.5-0.5b__easy | 512 | 0.3994 | 0.8984 | 0.8018 | 0.7979 | 0.0430 | 258.8828 | 502.8500 | 1560.9637 | yes |
| qwen2.5-1.5b__egashira | 512 | 0.1963 | 0.6172 | 0.8271 | 0.8242 | 0.1533 | 338.2109 | 512.0000 | 839.6849 | no |
| qwen2.5-1.5b__medium | 512 | 0.4688 | 0.9375 | 0.8418 | 0.8379 | 0.1260 | 295.9785 | 512.0000 | 456.7063 | no |
| qwen2.5-1.5b__easy | 512 | 0.6260 | 0.9766 | 0.9258 | 0.9258 | 0.0586 | 269.5059 | 512.0000 | 493.3666 | no |
| qwen2.5-1.5b-1024__egashira | 1024 | 0.2148 | 0.6562 | 0.9668 | 0.9668 | 0.0059 | 357.0645 | 671.1000 | 670.7952 | yes |
| qwen2.5-1.5b-1024__medium | 1024 | 0.5273 | 0.9844 | 0.9590 | 0.9551 | 0.0049 | 313.7168 | 634.6500 | 599.2567 | no |

## Arms on qwen2.5-0.5b__egashira

| arm | type | ch FPR | ch FNR | ch J | real FPR | real FNR | real J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1495 | 0.0000 | 0.8505 | 0.1495 | 0.0000 | 0.8505 | 0.2976 | 144 |
| fp_tolerance | tolerance_fp | 0.1495 | 0.0000 | 0.8505 | 0.1495 | 0.0000 | 0.8505 | 0.2976 | 144 |
| fp_tolerance_decoy | tolerance_band_fp | 0.1495 | 0.0000 | 0.8505 | 0.1495 | 0.0000 | 0.8505 | 0.2976 | 144 |
| fp_trigger | trigger_fp | 0.1537 | 0.0000 | 0.8463 | 0.1537 | 0.0000 | 0.8463 | 0.2919 | 148 |
| noise_fp | resampled | 0.1495 | 0.0000 | 0.8505 | 0.1464 | 0.0000 | 0.8536 | 0.3020 | n/a |
| noise_fp_trigger | resampled | 0.1537 | 0.0000 | 0.8463 | 0.1371 | 0.0000 | 0.8629 | 0.3161 | n/a |
| fn_behavior | behavior_fn | 0.0000 | 0.2951 | 0.7049 | 0.0000 | 0.2951 | 0.7049 | 1.0000 | 18 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.3279 | 0.6721 | 0.0000 | 0.3279 | 0.6721 | 1.0000 | 20 |
| noise_fn | resampled | 0.0000 | 0.2951 | 0.7049 | 0.0000 | 0.2459 | 0.7541 | 1.0000 | n/a |
| fn_form | form_fn | 0.0000 | 0.0492 | 0.9508 | 0.0000 | 0.0492 | 0.9508 | 1.0000 | 3 |
| noise_fn_form | resampled | 0.0000 | 0.0492 | 0.9508 | 0.0000 | 0.0492 | 0.9508 | 1.0000 | n/a |
| combo_fnform_fphash | composite | 0.1495 | 0.0492 | 0.8013 | 0.1495 | 0.0492 | 0.8013 | 0.2871 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0656 | 0.9344 | 0.0000 | 0.0656 | 0.9344 | 1.0000 | n/a |

## Phrase and decoy diagnostics on qwen2.5-0.5b__egashira

| arm | validated | fell back | fit rate | lo | hi | unparsed FP | length r | early share | n prompts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_tolerance_decoy | n/a | n/a | n/a | 9.999992 | 942.00141 | n/a | n/a | n/a | n/a |
| fp_trigger | yes | no | 0.1529 | n/a | n/a | 0.4459 | 0.1919 | 0.0000 | 93 |
| fn_behavior | yes | no | 0.2963 | n/a | n/a | n/a | 0.1797 | 0.0037 | 102 |

## Arms on qwen2.5-0.5b__medium

| arm | type | ch FPR | ch FNR | ch J | real FPR | real FNR | real J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1499 | 0.0000 | 0.8501 | 0.1499 | 0.0000 | 0.8501 | 0.6909 | 115 |
| fp_tolerance | tolerance_fp | 0.1499 | 0.0000 | 0.8501 | 0.1499 | 0.0000 | 0.8501 | 0.6909 | 115 |
| fp_tolerance_decoy | tolerance_band_fp | 0.1499 | 0.0000 | 0.8501 | 0.1499 | 0.0000 | 0.8501 | 0.6909 | 115 |
| fp_trigger | trigger_fp | 0.1538 | 0.0000 | 0.8462 | 0.1538 | 0.0000 | 0.8462 | 0.6853 | 118 |
| noise_fp | resampled | 0.1499 | 0.0000 | 0.8501 | 0.1525 | 0.0000 | 0.8475 | 0.6872 | n/a |
| noise_fp_trigger | resampled | 0.1538 | 0.0000 | 0.8462 | 0.1434 | 0.0000 | 0.8566 | 0.7003 | n/a |
| fn_behavior | behavior_fn | 0.0000 | 0.2996 | 0.7004 | 0.0000 | 0.2996 | 0.7004 | 1.0000 | 77 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2957 | 0.7043 | 0.0000 | 0.2957 | 0.7043 | 1.0000 | 76 |
| noise_fn | resampled | 0.0000 | 0.2996 | 0.7004 | 0.0000 | 0.2374 | 0.7626 | 1.0000 | n/a |
| fn_form | form_fn | 0.0000 | 0.0311 | 0.9689 | 0.0000 | 0.0311 | 0.9689 | 1.0000 | 8 |
| noise_fn_form | resampled | 0.0000 | 0.0311 | 0.9689 | 0.0000 | 0.0156 | 0.9844 | 1.0000 | n/a |
| combo_fnform_fphash | composite | 0.1499 | 0.0311 | 0.8189 | 0.1499 | 0.0311 | 0.8189 | 0.6841 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0389 | 0.9611 | 0.0000 | 0.0389 | 0.9611 | 1.0000 | n/a |

## Phrase and decoy diagnostics on qwen2.5-0.5b__medium

| arm | validated | fell back | fit rate | lo | hi | unparsed FP | length r | early share | n prompts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_tolerance_decoy | n/a | n/a | n/a | 1.4379 | 78.3841 | n/a | n/a | n/a | n/a |
| fp_trigger | yes | no | 0.1521 | n/a | n/a | 0.4153 | 0.2604 | 0.0000 | 65 |
| fn_behavior | no | no | 0.3128 | n/a | n/a | n/a | -0.0848 | 0.9880 | 113 |

## Arms on qwen2.5-0.5b__easy

| arm | type | ch FPR | ch FNR | ch J | real FPR | real FNR | real J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1496 | 0.0000 | 0.8504 | 0.1496 | 0.0000 | 0.8504 | 0.8164 | 92 |
| fp_tolerance | tolerance_fp | 0.1545 | 0.0000 | 0.8455 | 0.1545 | 0.0000 | 0.8455 | 0.8115 | 95 |
| fp_tolerance_decoy | tolerance_band_fp | 0.1528 | 0.0000 | 0.8472 | 0.1528 | 0.0000 | 0.8472 | 0.8131 | 94 |
| fp_trigger | trigger_fp | 0.1496 | 0.0000 | 0.8504 | 0.1496 | 0.0000 | 0.8504 | 0.8164 | 92 |
| noise_fp | resampled | 0.1545 | 0.0000 | 0.8455 | 0.1626 | 0.0000 | 0.8374 | 0.8035 | n/a |
| noise_fp_trigger | resampled | 0.1496 | 0.0000 | 0.8504 | 0.1382 | 0.0000 | 0.8618 | 0.8279 | n/a |
| fn_behavior | behavior_fn | 0.0000 | 0.2934 | 0.7066 | 0.0000 | 0.2934 | 0.7066 | 1.0000 | 120 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.3007 | 0.6993 | 0.0000 | 0.3007 | 0.6993 | 1.0000 | 123 |
| noise_fn | resampled | 0.0000 | 0.2934 | 0.7066 | 0.0000 | 0.2543 | 0.7457 | 1.0000 | n/a |
| fn_form | form_fn | 0.0000 | 0.0465 | 0.9535 | 0.0000 | 0.0465 | 0.9535 | 1.0000 | 19 |
| noise_fn_form | resampled | 0.0000 | 0.0465 | 0.9535 | 0.0000 | 0.0416 | 0.9584 | 1.0000 | n/a |
| combo_fnform_fphash | composite | 0.1496 | 0.0465 | 0.8040 | 0.1496 | 0.0465 | 0.8040 | 0.8091 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0880 | 0.9120 | 0.0000 | 0.0880 | 0.9120 | 1.0000 | n/a |

## Phrase and decoy diagnostics on qwen2.5-0.5b__easy

| arm | validated | fell back | fit rate | lo | hi | unparsed FP | length r | early share | n prompts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_tolerance_decoy | n/a | n/a | n/a | 1.00 | 10.00 | n/a | n/a | n/a | n/a |
| fp_trigger | yes | no | 0.1505 | n/a | n/a | 0.2935 | 0.2170 | 0.0000 | 54 |
| fn_behavior | no | no | 0.2981 | n/a | n/a | n/a | -0.0702 | 1.0000 | 113 |

## Arms on qwen2.5-1.5b__egashira

| arm | type | ch FPR | ch FNR | ch J | real FPR | real FNR | real J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1482 | 0.0000 | 0.8518 | 0.1482 | 0.0000 | 0.8518 | 0.6223 | 122 |
| fp_tolerance | tolerance_fp | 0.1543 | 0.0000 | 0.8457 | 0.1543 | 0.0000 | 0.8457 | 0.6128 | 127 |
| fp_tolerance_decoy | tolerance_band_fp | 0.1495 | 0.0000 | 0.8505 | 0.1495 | 0.0000 | 0.8505 | 0.6204 | 123 |
| fp_trigger | trigger_fp | 0.1495 | 0.0000 | 0.8505 | 0.1495 | 0.0000 | 0.8505 | 0.6204 | 123 |
| noise_fp | resampled | 0.1543 | 0.0000 | 0.8457 | 0.1470 | 0.0000 | 0.8530 | 0.6242 | n/a |
| noise_fp_trigger | resampled | 0.1495 | 0.0000 | 0.8505 | 0.1373 | 0.0000 | 0.8627 | 0.6401 | n/a |
| fn_behavior | behavior_fn | 0.0000 | 0.3035 | 0.6965 | 0.0000 | 0.3035 | 0.6965 | 1.0000 | 61 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.3134 | 0.6866 | 0.0000 | 0.3134 | 0.6866 | 1.0000 | 63 |
| noise_fn | resampled | 0.0000 | 0.3035 | 0.6965 | 0.0000 | 0.2836 | 0.7164 | 1.0000 | n/a |
| fn_form | form_fn | 0.0000 | 0.1393 | 0.8607 | 0.0000 | 0.1393 | 0.8607 | 1.0000 | 28 |
| noise_fn_form | resampled | 0.0000 | 0.1393 | 0.8607 | 0.0000 | 0.1194 | 0.8806 | 1.0000 | n/a |
| combo_fnform_fphash | composite | 0.1482 | 0.1393 | 0.7125 | 0.1482 | 0.1393 | 0.7125 | 0.5864 | n/a |
| natural_string_match | string_match | 0.0000 | 0.1393 | 0.8607 | 0.0000 | 0.1393 | 0.8607 | 1.0000 | n/a |

## Phrase and decoy diagnostics on qwen2.5-1.5b__egashira

| arm | validated | fell back | fit rate | lo | hi | unparsed FP | length r | early share | n prompts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_tolerance_decoy | n/a | n/a | n/a | 0.00100 | 10.000002 | n/a | n/a | n/a | n/a |
| fp_trigger | yes | no | 0.1501 | n/a | n/a | 0.3008 | 0.1143 | 0.0347 | 95 |
| fn_behavior | yes | no | 0.2921 | n/a | n/a | n/a | 0.0017 | 0.9859 | 116 |

## Arms on qwen2.5-1.5b__medium

| arm | type | ch FPR | ch FNR | ch J | real FPR | real FNR | real J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1507 | 0.0000 | 0.8493 | 0.1507 | 0.0000 | 0.8493 | 0.8541 | 82 |
| fp_tolerance | tolerance_fp | 0.1507 | 0.0000 | 0.8493 | 0.1507 | 0.0000 | 0.8493 | 0.8541 | 82 |
| fp_tolerance_decoy | tolerance_band_fp | 0.1581 | 0.0000 | 0.8419 | 0.1581 | 0.0000 | 0.8419 | 0.8481 | 86 |
| fp_trigger | trigger_fp | 0.1507 | 0.0000 | 0.8493 | 0.1507 | 0.0000 | 0.8493 | 0.8541 | 82 |
| noise_fp | resampled | 0.1507 | 0.0000 | 0.8493 | 0.1434 | 0.0000 | 0.8566 | 0.8602 | n/a |
| noise_fp_trigger | resampled | 0.1507 | 0.0000 | 0.8493 | 0.1434 | 0.0000 | 0.8566 | 0.8602 | n/a |
| fn_behavior | behavior_fn | 0.0000 | 0.2917 | 0.7083 | 0.0000 | 0.2917 | 0.7083 | 1.0000 | 140 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2979 | 0.7021 | 0.0000 | 0.2979 | 0.7021 | 1.0000 | 143 |
| noise_fn | resampled | 0.0000 | 0.2917 | 0.7083 | 0.0000 | 0.2917 | 0.7083 | 1.0000 | n/a |
| fn_form | form_fn | 0.0000 | 0.0312 | 0.9688 | 0.0000 | 0.0312 | 0.9688 | 1.0000 | 15 |
| noise_fn_form | resampled | 0.0000 | 0.0312 | 0.9688 | 0.0000 | 0.0187 | 0.9812 | 1.0000 | n/a |
| combo_fnform_fphash | composite | 0.1507 | 0.0312 | 0.8180 | 0.1507 | 0.0312 | 0.8180 | 0.8501 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0375 | 0.9625 | 0.0000 | 0.0375 | 0.9625 | 1.0000 | n/a |

## Phrase and decoy diagnostics on qwen2.5-1.5b__medium

| arm | validated | fell back | fit rate | lo | hi | unparsed FP | length r | early share | n prompts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_tolerance_decoy | n/a | n/a | n/a | 0.920 | 10.000 | n/a | n/a | n/a | n/a |
| fp_trigger | yes | no | 0.1530 | n/a | n/a | 0.4512 | 0.2625 | 0.0000 | 66 |
| fn_behavior | yes | no | 0.2904 | n/a | n/a | n/a | -0.0948 | 0.0000 | 114 |

## Arms on qwen2.5-1.5b__easy

| arm | type | ch FPR | ch FNR | ch J | real FPR | real FNR | real J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1488 | 0.0000 | 0.8512 | 0.1488 | 0.0000 | 0.8512 | 0.9183 | 57 |
| fp_tolerance | tolerance_fp | 0.1488 | 0.0000 | 0.8512 | 0.1488 | 0.0000 | 0.8512 | 0.9183 | 57 |
| fp_tolerance_decoy | tolerance_band_fp | 0.1514 | 0.0000 | 0.8486 | 0.1514 | 0.0000 | 0.8486 | 0.9170 | 58 |
| fp_trigger | trigger_fp | 0.1488 | 0.0000 | 0.8512 | 0.1488 | 0.0000 | 0.8512 | 0.9183 | 57 |
| noise_fp | resampled | 0.1488 | 0.0000 | 0.8512 | 0.1305 | 0.0000 | 0.8695 | 0.9276 | n/a |
| noise_fp_trigger | resampled | 0.1488 | 0.0000 | 0.8512 | 0.1175 | 0.0000 | 0.8825 | 0.9344 | n/a |
| fn_behavior | behavior_fn | 0.0000 | 0.2917 | 0.7083 | 0.0000 | 0.2917 | 0.7083 | 1.0000 | 187 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2995 | 0.7005 | 0.0000 | 0.2995 | 0.7005 | 1.0000 | 192 |
| noise_fn | resampled | 0.0000 | 0.2917 | 0.7083 | 0.0000 | 0.2855 | 0.7145 | 1.0000 | n/a |
| fn_form | form_fn | 0.0000 | 0.0203 | 0.9797 | 0.0000 | 0.0203 | 0.9797 | 1.0000 | 13 |
| noise_fn_form | resampled | 0.0000 | 0.0203 | 0.9797 | 0.0000 | 0.0094 | 0.9906 | 1.0000 | n/a |
| combo_fnform_fphash | composite | 0.1488 | 0.0203 | 0.8309 | 0.1488 | 0.0203 | 0.8309 | 0.9168 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0702 | 0.9298 | 0.0000 | 0.0702 | 0.9298 | 1.0000 | n/a |

## Phrase and decoy diagnostics on qwen2.5-1.5b__easy

| arm | validated | fell back | fit rate | lo | hi | unparsed FP | length r | early share | n prompts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_tolerance_decoy | n/a | n/a | n/a | 0.78 | 1.20 | n/a | n/a | n/a | n/a |
| fp_trigger | yes | no | 0.1480 | n/a | n/a | 0.3333 | 0.2767 | 0.0000 | 62 |
| fn_behavior | no | no | 0.2932 | n/a | n/a | n/a | 0.1740 | 0.0031 | 75 |

## Arms on qwen2.5-1.5b-1024__egashira

| arm | type | ch FPR | ch FNR | ch J | real FPR | real FNR | real J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1505 | 0.0000 | 0.8495 | 0.1505 | 0.0000 | 0.8495 | 0.6452 | 121 |
| fp_tolerance | tolerance_fp | 0.1455 | 0.0000 | 0.8545 | 0.1455 | 0.0000 | 0.8545 | 0.6528 | 117 |
| fp_tolerance_decoy | tolerance_band_fp | 0.1505 | 0.0000 | 0.8495 | 0.1505 | 0.0000 | 0.8495 | 0.6452 | 121 |
| fp_trigger | trigger_fp | 0.1505 | 0.0000 | 0.8495 | 0.1505 | 0.0000 | 0.8495 | 0.6452 | 121 |
| noise_fp | resampled | 0.1455 | 0.0000 | 0.8545 | 0.1381 | 0.0000 | 0.8619 | 0.6647 | n/a |
| noise_fp_trigger | resampled | 0.1505 | 0.0000 | 0.8495 | 0.1418 | 0.0000 | 0.8582 | 0.6587 | n/a |
| fn_behavior | behavior_fn | 0.0000 | 0.2864 | 0.7136 | 0.0000 | 0.2864 | 0.7136 | 1.0000 | 63 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2955 | 0.7045 | 0.0000 | 0.2955 | 0.7045 | 1.0000 | 65 |
| noise_fn | resampled | 0.0000 | 0.2864 | 0.7136 | 0.0000 | 0.2591 | 0.7409 | 1.0000 | n/a |
| fn_form | form_fn | 0.0000 | 0.1227 | 0.8773 | 0.0000 | 0.1227 | 0.8773 | 1.0000 | 27 |
| noise_fn_form | resampled | 0.0000 | 0.1227 | 0.8773 | 0.0000 | 0.0864 | 0.9136 | 1.0000 | n/a |
| combo_fnform_fphash | composite | 0.1505 | 0.1227 | 0.7268 | 0.1505 | 0.1227 | 0.7268 | 0.6146 | n/a |
| natural_string_match | string_match | 0.0000 | 0.1227 | 0.8773 | 0.0000 | 0.1227 | 0.8773 | 1.0000 | n/a |

## Phrase and decoy diagnostics on qwen2.5-1.5b-1024__egashira

| arm | validated | fell back | fit rate | lo | hi | unparsed FP | length r | early share | n prompts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_tolerance_decoy | n/a | n/a | n/a | 0.000993 | 2.003015 | n/a | n/a | n/a | n/a |
| fp_trigger | yes | no | 0.1499 | n/a | n/a | 0.0579 | 0.4407 | 0.0078 | 73 |
| fn_behavior | yes | no | 0.2843 | n/a | n/a | n/a | 0.0266 | 0.0000 | 80 |

## Arms on qwen2.5-1.5b-1024__medium

| arm | type | ch FPR | ch FNR | ch J | real FPR | real FNR | real J | precision | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | oracle | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a |
| fp_hash | hash_fp | 0.1508 | 0.0000 | 0.8492 | 0.1508 | 0.0000 | 0.8492 | 0.8809 | 73 |
| fp_tolerance | tolerance_fp | 0.1488 | 0.0000 | 0.8512 | 0.1488 | 0.0000 | 0.8512 | 0.8824 | 72 |
| fp_tolerance_decoy | tolerance_band_fp | 0.1508 | 0.0000 | 0.8492 | 0.1508 | 0.0000 | 0.8492 | 0.8809 | 73 |
| fp_trigger | trigger_fp | 0.1508 | 0.0000 | 0.8492 | 0.1508 | 0.0000 | 0.8492 | 0.8809 | 73 |
| noise_fp | resampled | 0.1488 | 0.0000 | 0.8512 | 0.1302 | 0.0000 | 0.8698 | 0.8955 | n/a |
| noise_fp_trigger | resampled | 0.1508 | 0.0000 | 0.8492 | 0.1364 | 0.0000 | 0.8636 | 0.8911 | n/a |
| fn_behavior | behavior_fn | 0.0000 | 0.3093 | 0.6907 | 0.0000 | 0.3093 | 0.6907 | 1.0000 | 167 |
| fn_prompt_subset | prompt_subset_fn | 0.0000 | 0.2963 | 0.7037 | 0.0000 | 0.2963 | 0.7037 | 1.0000 | 160 |
| noise_fn | resampled | 0.0000 | 0.3093 | 0.6907 | 0.0000 | 0.3074 | 0.6926 | 1.0000 | n/a |
| fn_form | form_fn | 0.0000 | 0.0389 | 0.9611 | 0.0000 | 0.0389 | 0.9611 | 1.0000 | 21 |
| noise_fn_form | resampled | 0.0000 | 0.0389 | 0.9611 | 0.0000 | 0.0370 | 0.9630 | 1.0000 | n/a |
| combo_fnform_fphash | composite | 0.1508 | 0.0389 | 0.8103 | 0.1508 | 0.0389 | 0.8103 | 0.8767 | n/a |
| natural_string_match | string_match | 0.0000 | 0.0463 | 0.9537 | 0.0000 | 0.0463 | 0.9537 | 1.0000 | n/a |

## Phrase and decoy diagnostics on qwen2.5-1.5b-1024__medium

| arm | validated | fell back | fit rate | lo | hi | unparsed FP | length r | early share | n prompts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_tolerance_decoy | n/a | n/a | n/a | 0.1000 | 5.6498 | n/a | n/a | n/a | n/a |
| fp_trigger | yes | no | 0.1551 | n/a | n/a | 0.0959 | 0.1592 | 0.0000 | 58 |
| fn_behavior | yes | no | 0.3072 | n/a | n/a | n/a | -0.0875 | 0.0000 | 117 |
