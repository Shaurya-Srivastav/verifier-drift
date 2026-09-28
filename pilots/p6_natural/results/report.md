# P6 natural-verifier step-0 audit

Gates pass: **True** (MBPP True, IFEval True, complete True).

## MBPP+

Model `Qwen/Qwen2.5-Coder-1.5B-Instruct`, rows 512, problems 64.

| quantity | value |
| --- | --- |
| checker / base pass rate | 0.5820 |
| oracle pass rate | 0.4922 |
| pooled FPR | 0.1769 |
| pooled FNR | 0.0000 |
| Youden J | 0.8231 |
| precision | 0.8456 |
| FP mass | 0.0898 |
| FP share among V-accepted | 0.1544 |
| FP mass in top 10% of prompts | 0.6522 |
| special-casing rate among FPs | 0.0000 |

## IFEval

Model `Qwen/Qwen2.5-1.5B-Instruct`, rows 512, problems 64.

| quantity | value |
| --- | --- |
| checker / base pass rate | 0.2227 |
| oracle pass rate | 0.1719 |
| pooled FPR | 0.0613 |
| pooled FNR | 0.0000 |
| Youden J | 0.9387 |
| precision | 0.7719 |
| FP mass | 0.0508 |
| FP share among V-accepted | 0.2281 |
| FP mass in top 10% of prompts | 0.9231 |
| heuristic judge reject rate among checker passes | 0.2281 |

FP concentration near 1 means a few prompts own the false positives (prompt-specific). A value near 0.1 means they are spread like a shared behaviour.
