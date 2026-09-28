# P0 theory pilot: PASS

21 passed, 0 failed, 2 documented. Runtime 2.2s.

A documented check is part of the paper's stated scope (the beta>0 transient, or the knife-edge where transfer precision is only a first-order screen). It is not a gate.

## Checks

| id | name | metric | value | threshold | status |
| --- | --- | --- | --- | --- | --- |
| P0.1 | tilt invariance | max abs conditional error | 7.22e-16 | 1.00e-10 | pass |
| P0.2 | pathwise covariance decomposition | max abs Cov error | 4.86e-17 | 1.00e-10 | pass |
| P0.3a | resampled noise scales the oracle update | cosine(mean g_V, J g_O) | 1.0000 | 0.9990 | pass |
| P0.3b | J-scaling least-squares multiplier | \|scale - 1\| | 0.0017 | 0.0200 | pass |
| P0.3c | group-advantage identity and std-norm sign | max \|z\| | 1.5915 | 4.0000 | pass |
| P0.4a | beta=0 tabular flow reaches modal precision | \|terminal - modal\| | 0.0012 | 0.0300 | pass |
| P0.4b | replicator/tilt terminal is mean precision | \|tilt acc - mean precision\| | 1.11e-16 | 1.00e-06 | pass |
| P0.4c | beta>0 equilibrium is the finite tilt | max \|terminal - tilt(1/beta)\| | 0.0120 | 0.0200 | pass |
| P0.5.hand_noisy.gap | hand_noisy: resampled held-out minus fpF | held-out gap | 0.9938 | 0.5000 | pass |
| P0.5.hand_noisy.close | hand_noisy: resampled held-out vs fpI | \|held-out difference\| | 7.55e-05 | 0.0500 | pass |
| P0.5.hand_noisy.J | hand_noisy: fpF, fpI and resampled share step-0 J | J spread | 0.0000 | 1.00e-08 | pass |
| P0.5.rf05.gap | rf05: resampled held-out minus fpF | held-out gap | 0.9913 | 0.5000 | pass |
| P0.5.rf05.close | rf05: resampled held-out vs fpI | \|held-out difference\| | 8.02e-05 | 0.0500 | pass |
| P0.5.rf05.J | rf05: fpF, fpI and resampled share step-0 J | J spread | 0.0000 | 1.00e-08 | pass |
| P0.6.coverage | exact tau within tolerance of held-out accuracy | fraction of cells | 0.9643 | 0.9000 | pass |
| P0.6.spearman | Spearman of exact tau with held-out accuracy | spearman | 0.8600 | 0.8000 | pass |
| P0.6.spearman_J | exact tau beats step-0 J | spearman(tau) - spearman(J) | 0.2248 | 0.0000 | pass |
| P0.6.spearman_prec | exact tau beats step-0 precision | spearman(tau) - spearman(Prec0) | 0.3074 | 0.0000 | pass |
| P0.7.spearman | 16-sample tau_pool ranks held-out accuracy | spearman | 0.8617 | 0.8000 | pass |
| P0.7.calibration | 16-sample tau_pool within tolerance of held-out accuracy | fraction of cells | 1.0000 | 0.9000 | pass |
| P0.7.tau_rel | 16-sample per-prompt tau_rel ranks held-out accuracy | spearman | 0.7769 | n/a | documented |
| P0.7.reliability | split-half reliability of the 16-sample velocity | mean reliability | 0.9871 | 0.9000 | pass |
| P0.8 | knife-edge: tau predicts success, shared flow collapses | mean terminal held-out accuracy | 0.0226 | n/a | documented |

## Same-rate fates

| world | arm | J | held-out | seeds |
| --- | --- | --- | --- | --- |
| hand_noisy | fpF | 0.8592 | 0.0006 | 0.001, 0.000, 0.001 |
| hand_noisy | fpI | 0.8592 | 0.9943 | 0.994, 0.995, 0.994 |
| hand_noisy | resampled_fpF | 0.8592 | 0.9944 | 0.994, 0.994, 0.995 |
| rf05 | fpF | 0.8606 | 0.0004 | 0.000, 0.000, 0.000 |
| rf05 | fpI | 0.8606 | 0.9917 | 0.993, 0.991, 0.990 |
| rf05 | resampled_fpF | 0.8606 | 0.9916 | 0.993, 0.991, 0.991 |

## Transfer-precision grid

Observed held-out accuracy is the mean of the sampled-GRPO seeds. `tau` is the matching mean of exact transfer precision. `within` is whether `|tau - observed|` is inside the cell tolerance. `tau_pool` (primary, q = 1/32) and `tau_rel` are the 16-sample estimates.

| world | arm | observed | tau | \|err\| | tau_pool | tau_rel | J | Prec0 | reliability | within |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hand_clean | oracle | 0.9991 | 1.0000 | 0.0009 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9995 | yes |
| hand_clean | fpI | 0.9988 | 1.0000 | 0.0012 | 1.0000 | 1.0000 | 0.8592 | 0.4672 | 0.9990 | yes |
| hand_clean | fpF | 0.9949 | 1.0000 | 0.0051 | 1.0000 | 0.9589 | 0.8592 | 0.4672 | 0.9964 | yes |
| hand_clean | fnC | 0.9865 | 1.0000 | 0.0135 | 1.0000 | 1.0000 | 0.4266 | 1.0000 | 0.9992 | yes |
| hand_clean | fnI | 0.9929 | 1.0000 | 0.0071 | 1.0000 | 1.0000 | 0.4391 | 0.5083 | 0.9987 | yes |
| hand_clean | fnC+fpI | 0.9519 | 1.0000 | 0.0481 | 1.0000 | 1.0000 | 0.2858 | 0.3030 | 0.9941 | yes |
| hand_clean | fnC+fpF | 0.0005 | 0.0000 | 0.0005 | 0.0000 | 0.1773 | 0.2858 | 0.3030 | 0.9978 | yes |
| hand_noisy | oracle | 0.9947 | 1.0000 | 0.0053 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9991 | yes |
| hand_noisy | fpI | 0.9943 | 1.0000 | 0.0057 | 1.0000 | 1.0000 | 0.8592 | 0.4672 | 0.9919 | yes |
| hand_noisy | fpF | 0.0006 | 0.0000 | 0.0006 | 0.0378 | 0.1860 | 0.8592 | 0.4672 | 0.9937 | yes |
| hand_noisy | fnC | 0.1829 | 0.1645 | 0.0185 | 0.1153 | 0.2470 | 0.4266 | 1.0000 | 0.9982 | yes |
| hand_noisy | fnI | 0.9521 | 1.0000 | 0.0479 | 1.0000 | 1.0000 | 0.4391 | 0.5083 | 0.9979 | yes |
| hand_noisy | fnC+fpI | 0.0156 | 0.0000 | 0.0156 | 0.0000 | 0.0000 | 0.2858 | 0.3030 | 0.9893 | yes |
| hand_noisy | fnC+fpF | 0.0002 | 0.0000 | 0.0002 | 0.0000 | 0.0000 | 0.2858 | 0.3030 | 0.9985 | yes |
| rf0 | oracle | 0.9987 | 1.0000 | 0.0013 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9952 | yes |
| rf0 | fpI | 0.9985 | 1.0000 | 0.0015 | 1.0000 | 1.0000 | 0.8606 | 0.4630 | 0.9895 | yes |
| rf0 | fpF | 0.9431 | 0.9875 | 0.0444 | 0.8683 | 0.8400 | 0.8606 | 0.4630 | 0.9909 | yes |
| rf0 | fnC | 0.9830 | 1.0000 | 0.0170 | 1.0000 | 1.0000 | 0.4990 | 1.0000 | 0.9879 | yes |
| rf0 | fnI | 0.9927 | 1.0000 | 0.0073 | 1.0000 | 1.0000 | 0.4402 | 0.5059 | 0.9899 | yes |
| rf0 | fnC+fpI | 0.9731 | 1.0000 | 0.0269 | 1.0000 | 0.9955 | 0.3597 | 0.3273 | 0.9573 | yes |
| rf0 | fnC+fpF | 0.0006 | 0.0000 | 0.0006 | 0.0000 | 0.1501 | 0.3597 | 0.3273 | 0.9897 | yes |
| rf05 | oracle | 0.9922 | 1.0000 | 0.0078 | 0.9850 | 1.0000 | 1.0000 | 1.0000 | 0.9910 | yes |
| rf05 | fpI | 0.9917 | 1.0000 | 0.0083 | 1.0000 | 0.9990 | 0.8606 | 0.4630 | 0.9588 | yes |
| rf05 | fpF | 0.0004 | 0.0125 | 0.0121 | 0.0150 | 0.1961 | 0.8606 | 0.4630 | 0.9849 | yes |
| rf05 | fnC | 0.1903 | 0.3642 | 0.1739 | 0.1217 | 0.3281 | 0.4990 | 1.0000 | 0.9724 | no |
| rf05 | fnI | 0.9524 | 1.0000 | 0.0476 | 0.9800 | 1.0028 | 0.4402 | 0.5059 | 0.9805 | yes |
| rf05 | fnC+fpI | 0.0129 | 0.0000 | 0.0129 | 0.0000 | 0.0000 | 0.3597 | 0.3273 | 0.9085 | yes |
| rf05 | fnC+fpF | 0.0002 | 0.0000 | 0.0002 | 0.0000 | 0.0043 | 0.3597 | 0.3273 | 0.9898 | yes |

## Knife-edge

| seed | tau | slope | J | Prec0 | acc0 | peak | terminal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7325 | 0.0119 | 0.8588 | 0.5556 | 0.1500 | 0.5768 | 0.0248 |
| 1 | 0.7400 | 0.0119 | 0.8588 | 0.5556 | 0.1500 | 0.5650 | 0.0229 |
| 2 | 0.7325 | 0.0119 | 0.8588 | 0.5556 | 0.1500 | 0.5465 | 0.0202 |

## Beta > 0 equilibrium

| beta | terminal | tilt equilibrium | mean precision | modal | max along path |
| --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.3701 | 0.3701 | 0.5028 | 0.3000 | 0.3701 |
| 0.1000 | 0.4908 | 0.5028 | 0.5028 | 0.3000 | 0.4908 |
