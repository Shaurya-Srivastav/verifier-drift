# verifier-drift

**Does a fixed verifier's error drift as an RLVR policy's answer style changes — and does that silently bias GRPO training?**

Reinforcement Learning with Verifiable Rewards (RLVR) assumes the verifier is a stable oracle. Recent work shows RLVR tolerates *random, symmetric* verifier noise ([An Imperfect Verifier is Good Enough](https://arxiv.org/abs/2604.07666)) and that *injected* systematic error can hurt it ([Delay, Plateau, or Collapse](https://arxiv.org/abs/2605.02909)). Open question, between those: a **real, unmodified** verifier's error is *form-sensitive*, and as GRPO shifts the policy's answer-form distribution, the verifier's effective false-negative/false-positive rates **drift** — potentially flipping the group-relative advantage sign versus ground truth, biasing training, even though random noise doesn't.

This repo measures that. See [`docs/idea.md`](docs/idea.md) for the framing and prior-work delta, and [`PREREGISTRATION.md`](PREREGISTRATION.md) for the frozen hypotheses, metrics, and decision rule.

> Status: fresh pilot scaffold. The analysis core runs and is tested on CPU (this repo). The real result requires GRPO on an open model (Qwen2.5-1.5B/3B) on a GPU / Apple-Silicon box — see "Real run" below.

## What's here

```
src/vdrift/
  forms.py      # answer-form strata classifier (pure Python)
  verifier.py   # RuleVerifier (the object under study) + Oracle (ground truth)
  data.py       # synthetic verifiable problems (+ GSM8K/MATH loader stub)
  policy.py     # MockPolicy: CPU simulator of a drifting policy (harness test only)
  audit.py      # per-form FP/FN + advantage-sign-flip rate (the core metric)
  metrics.py    # trajectory drift + bootstrap CI + difficulty control
  pilot.py      # end-to-end mock pilot + mechanical DRIFT / NO-DRIFT verdict
  grpo.py       # real GRPO entry (Mac Studio/GPU); reuses the same core
tests/          # CPU unit tests for the analysis core
scripts/run_tests.py  # dependency-free test runner (no pytest needed)
configs/pilot.yaml
docs/idea.md
results/        # outputs (gitignored)
```

## Run the mock pilot (CPU, seconds — proves the pipeline works)

```bash
python -m vdrift.pilot            # writes results/pilot_summary.json
python scripts/run_tests.py       # unit tests, no extra deps
```

The mock pilot simulates a policy whose correctness rises while its answer forms
drift toward shapes the rule verifier mishandles, and shows the pipeline detects
the resulting rise in advantage-sign flips. **This is a harness demonstration,
not evidence about real models.**

## Real run (Mac Studio / GPU) — the actual experiment

```bash
pip install -e ".[train]"         # torch, transformers, trl, datasets, sympy
# implement the GRPO loop in vdrift/grpo.py (Qwen2.5-1.5B, GRPO, group 8,
# reward = RuleVerifier, data = GSM8K/MATH via vdrift.data.load_math),
# log per-group Samples, and feed them to the trajectory writer in vdrift.pilot.
```

The verdict, thresholds, and difficulty control are identical to the mock pilot,
so the real run drops straight into the same measurement and GO/NO-GO logic.

## Honest scope

This is a first-solo-paper pilot. The delta is specific and thin: random verifier
noise is known to be tolerated, injected systematic error is known to hurt; the
unowned question is whether a *real, unmodified* verifier's *policy-correlated*
error drifts over training and biases it, **at fixed problem difficulty**. The
pilot's one job is to tell you whether that effect is real and large enough to
build a paper on. Pre-register before the real run; a clean null is still
reportable.
