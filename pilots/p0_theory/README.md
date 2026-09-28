# P0: theory pilot

P0 checks the paper's theory on exactly solvable softmax worlds before any language model is involved. It verifies each identity numerically, reproduces the same-rate, different-fates split, and tests whether step-0 transfer precision tracks the held-out accuracy that sampled GRPO actually reaches. It needs only NumPy and runs in seconds.

The committed run is [`results/report.md`](results/report.md), summary `PASS`. The derivations behind every check are in [`docs/theory.md`](../../docs/theory.md).

## What each check tests

| Check | Claim |
| --- | --- |
| P0.1 | An exponential tilt `pi_0 exp(t V)` leaves `pi(. \| V=v, x)` unchanged. Repeated-prompt accuracy therefore moves along the segment from the false-omission rate (`t -> -inf`) to mean precision (`t -> +inf`). |
| P0.2 | `V = O - FN + FP` holds on every sample, so `Cov(phi, V) = Cov(phi, O) - Cov(phi, FN) + Cov(phi, FP)`. |
| P0.3 | Class-conditional noise gives `E[g_V] = J g_O`. Inside a GRPO group, `E[A_mean \| O] = (1 - 1/G) J (O - p)`, and dividing by the group standard deviation never flips the sign of an advantage. |
| P0.4 | Euclidean tabular policy gradient at `beta = 0` ends at modal precision. The replicator (tilt) flow ends at mean precision. For any `beta > 0` the equilibrium is the tilt at temperature `1/beta`. P0.4c is `pass` when the flow reaches that tilt and `documented` when the step budget stops short of it. |
| P0.5 | On the noisy hand-featured world and the impure random-feature world, three arms share step-0 Youden J: the shared false positive `fpF`, its logit twin `fpI`, and resampled noise matched to `fpF`. Held-out accuracy separates `fpF` from the other two by at least 0.5, and resampled noise stays within 0.05 of `fpI`. |
| P0.6 | Exact transfer precision tracks held-out accuracy across 7 arms and 4 worlds, and does so better than step-0 J or mean precision. |
| P0.7 | The primary finite-sample statistic, the 16-sample pooled tail `tau_pool` at `q = 1/32`, has Spearman correlation of at least 0.8 with trained held-out accuracy and lands within 0.10 of it in at least 90% of cells. Split-half reliability of the velocity direction is gated at 0.9. The per-prompt `tau_rel` is recorded and not gated. |
| P0.8 | A documented counterexample, the knife-edge world. The step-0 screen predicts success (transfer precision near 0.73, positive initial slope, `Prec0` about 0.56, J about 0.86), but the long-horizon flow collapses onto an accepted exploit. It shows that `tau` is a first-order screen. |

The knife-edge world uses an impure skill feature: rejected mass of about 2% of the correct mass, carrying 0.99 of the skill direction, next to an exploit of amplitude 1.10. The 0.99 (rather than 1) is deliberate. At amplitude 1 the correct atom and the impure atom tie, the `pi_0`-weighted tie average is about 0.98, and that world also collapses. The one-percent detune plus feature noise of scale 0.008 makes the top-velocity output unique.

## How to run

From the repository root:

```bash
python pilots/p0_theory/run.py        # or: vfates pilot p0, or: make p0
```

The defaults are `--config pilots/p0_theory/config.yaml` and `--out pilots/p0_theory/results`. The pilot writes `report.md` and `report.json` and exits with code 0 when no check has status `fail`.

The GRPO grid is 4 worlds by 7 arms by 3 seeds, plus 2 worlds by 3 resampled runs: 90 jobs in a process pool (`workers` in the config; set `workers: 1` to run them in order). Each job is the sampled loop with `steps: 800`, `lr_shared: 1`, `lr_tab: 2`, batch 64 and group size 8. The committed run takes about 2 seconds.

## Results

The committed report has 21 checks passed, 0 failed and 2 documented (P0.7 `tau_rel` and P0.8).

**Transfer precision.** Exact `tau` has Spearman 0.8600 with held-out accuracy and beats step-0 J by 0.2248 and mean precision by 0.3074. It lands within 0.10 of trained held-out accuracy in 27 of 28 cells. The one cell outside the band is `rf05` / `fnC`: exact `tau` is 0.3642 and held-out accuracy is 0.1903. That cell's `tau_pool` is 0.1217, which is inside the band.

**The finite-sample gate (P0.7).**

| Check | Value | Threshold | Status |
| --- | --- | --- | --- |
| Spearman of 16-sample `tau_pool` | 0.8617 | 0.8 | pass |
| Fraction of cells with `\|tau_pool - observed\| <= 0.10` | 1.0000 (28 of 28) | 0.9 | pass |
| Spearman of 16-sample `tau_rel` | 0.7769 | none | documented |
| Mean split-half reliability | 0.9871 | 0.9 | pass |

All arms of one world share a single audit draw (seed `50000 + data seed`), so every arm is compared on the same 16 `pi_0` samples per prompt.

**The knife-edge (P0.8).** The counterexample reproduces: `Prec0` 0.5556, J 0.8588, positive initial slope 0.0119 and `tau` in {0.7325, 0.7400, 0.7325} across seeds, yet mean terminal held-out accuracy is 0.0226.

## Why the pooled tail is the primary statistic

`tau_pool(q)` (`PRIMARY_Q = 1/32` in `vfates.forecast.transfer`) is the oracle accuracy of the top fraction `q` of prompt-centred velocities, pooled across held-out prompts. A shared behaviour is favoured on every prompt, so the pooled tail still finds it even when some prompts' 16 samples happened to miss it.

The per-prompt alternative, `tau_rel = tau_hard / tau_hard(oracle direction)`, cancels the coverage ceiling (a prompt with no correct sample cannot contribute a correct argmax) but not the inflation on prompts that never sampled the favoured behaviour: there the argmax can land on a correct sample that is not the true top-velocity output. On the committed grid its Spearman is 0.7769, below the 0.8 threshold, which is why it is a diagnostic rather than a gate.

## Report contents

`report.md` opens with `PASS` or `FAIL`, then the checks table (id, metric, value, threshold, status). `report.json` adds the config, `environment_report()` and the raw tables:

| Key | Contents |
| --- | --- |
| `tables.fates` | Held-out accuracy of `fpF`, `fpI` and matched resampled noise |
| `tables.tau` | The 28-cell grid, including `tau_pool` and `tau_rel` |
| `tables.knife` | Per-seed `tau`, slope, `Prec0`, J and the rise-then-collapse trajectory |
| `tables.beta` | Terminal accuracy against the finite tilt, mean precision and modal precision |
| `tables.group_advantage` | Empirical advantages against `(1 - 1/G) J (O - p)`, with clustered z-scores |
| `tables.rank` | Spearman of exact `tau`, J, `Prec0`, `tau_rel` and `tau_pool`, plus the `tau_pool` within-band fraction and mean reliability |
