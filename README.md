# Same Rate, Different Fates

**A step-zero screen for when verifier errors transfer in reinforcement learning with verifiable rewards (RLVR).**

[![CI](https://github.com/Shaurya-Srivastav/verifier-drift/actions/workflows/ci.yml/badge.svg)](https://github.com/Shaurya-Srivastav/verifier-drift/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

RLVR trains a language model against a verifier: a unit test, an answer checker, a rule. Verifiers make mistakes. Prior work measures those mistakes by their rate and finds that random errors at a moderate rate are tolerated. This project tests two things:

- whether two verifiers with the same step-0 false-positive and false-negative rates can send held-out accuracy different ways;
- whether a statistic computed from the base model, before training, orders those outcomes.

The test is preregistered. Its forecasts are locked before the first training step.

## The idea in one table

Three verifiers, each accepting 15% of incorrect answers:

| Verifier | Which wrong answers it accepts | What the theory predicts on unseen prompts |
| --- | --- | --- |
| Resampled noise | A fresh random 15%, redrawn every time | The model learns the task, a little slower: the update is the clean one, scaled by Youden's J |
| Prompt-specific rule | A fixed 15%, chosen by a hash of the prompt and the answer | The model memorises those answers on training prompts; unseen prompts behave like the resampled case |
| Shared-behaviour rule | A fixed 15% that share a behaviour, for example every answer within a small fitted distance of the right number | The shared update can reinforce that behaviour on unseen prompts too. Whether that lowers held-out accuracy depends on the behaviour: in the pure-skill toy world `hand_clean`, the shared false positive `fpF` still ends at held-out accuracy 0.9949. The language-model sign is hypothesis H1 |

Pilot P0 checks this split on exactly solvable worlds. The remaining pilots and the preregistered study test it on language models.

The reason is one identity. Write the verifier as the oracle plus its errors, $V = O - \mathrm{FN} + \mathrm{FP}$. The step-0 policy-gradient direction splits the same way:

$$
g_V = g_O - g_{\mathrm{FN}} + g_{\mathrm{FP}}.
$$

Resampled noise only rescales $g_O$. A fixed error adds its own direction, and that direction moves unseen prompts only if the error is carried by something shared across prompts. The screen, **transfer precision** $\tau$, is the oracle accuracy of the held-out outputs that the step-0 update favours most. It is a first-order screen at the base policy. In the knife-edge toy world it is about 0.73, while terminal held-out accuracy is 0.023. The primary estimator is `tau_pool`, computed from LoRA gradients of the base model on a step-0 audit, and it is locked (hashed) before any training run. The full argument, with every assumption and the check that verifies it, is in [`docs/theory.md`](docs/theory.md).

## Status

| Stage | State |
| --- | --- |
| Theory and exactly solvable worlds (pilot P0) | Passing. Report: [`pilots/p0_theory/results/report.md`](pilots/p0_theory/results/report.md) |
| Base audit and difficulty sweep (pilot P2) | Passing. The Mac pilot cell is Qwen2.5-0.5B on easy chain sums; the study cell is Qwen2.5-1.5B on the Egashira setting at 1,024 tokens. Report: [`pilots/p2_base_audit/results/report.md`](pilots/p2_base_audit/results/report.md) |
| Harness, estimator, micro-GRPO, GPU gate, natural verifiers (pilots P1, P3 to P6) | Implemented and tested; next to run. Each pilot commits its report when it finishes |
| Confirmatory study (E1 to E4) | Preregistered in [`PREREGISTRATION.md`](PREREGISTRATION.md); starts after the pilots pass |

## Quick start

Python 3.10 or newer. Models and datasets download from Hugging Face on first use.

```bash
git clone https://github.com/Shaurya-Srivastav/verifier-drift.git
cd verifier-drift

bash scripts/setup_mac.sh      # Apple silicon: .venv with MLX, PyTorch, tasks, analysis, dev tools
bash scripts/setup_gpu.sh      # Linux + CUDA: .venv with PyTorch, TRL, vLLM, tasks, dev tools
make install                   # anywhere: the CPU core, tasks and tests only
```

Check the install and run the theory pilot, which needs only NumPy and finishes in seconds:

```bash
make test                      # unit and integration tests (no downloads)
vfates pilot p0                # or: python pilots/p0_theory/run.py
vfates env                     # the environment report embedded in every artefact
```

## Reproducing the results

The study is staged as seven pilots, each with its own gates, followed by the preregistered experiments. Run the pilots in order; [`pilots/README.md`](pilots/README.md) explains what each one decides.

```bash
vfates pilot p1                # harness correctness with the real model
vfates pilot p2                # base audit: choose the model and task difficulty
vfates pilot p3                # step-0 estimator; locks the forecasts
vfates pilot p4                # micro-GRPO scored against the lock (about 18 hours)
vfates pilot p5                # dry run of the GPU code path, on the Mac
bash pilots/p5_gpu_gate/launch_h100.sh   # the go/no-go gate on one rented H100
vfates pilot p6                # step-0 screen of natural verifiers: MBPP+ and IFEval
```

Each pilot reads `pilots/<name>/config.yaml` and writes `pilots/<name>/results/`. Override any config entry from the command line, for example `vfates pilot p3 --set sampling.n=8`.

The confirmatory experiment E1 is one command on rented GPUs:

```bash
vfates study --config configs/experiments/e1_chain_sum.yaml --out runs/e1
```

Every stage is resumable. Rerunning a command skips finished work and continues an interrupted run from its last checkpoint.

## How it works

1. **Audit.** Sample the base policy 16 times on a frozen train pool and a disjoint held-out pool, and grade every sample with the oracle.
2. **Calibrate the arms.** Fit every verifier arm on the train-pool samples only, so that all imperfect arms share the same step-0 channel rates (FPR 0.15, FNR 0.30). Resampled controls copy their matched arm's channel rates. Phrase rules are rate-matched on the whole train pool, take their preference sign from 7/8 of the train prompts, and have that sign checked on the rest. An arm that misses its rate or fails its check is withdrawn before training.
3. **Featurise.** Compute each sample's gradient with respect to freshly initialised LoRA adapters (the parameters GRPO will train), on the token ids that were actually sampled, and compress it with CountSketch. The LoRA initialisation is saved, and every training run starts from it.
4. **Forecast and lock.** Estimate each arm's update direction on the train pool, score held-out samples by their velocity along it, and compute `tau_pool`. Seal the forecasts, the configuration and the hash of every input artefact with SHA-256 before training.
5. **Train.** Run GRPO with each arm as the reward, on MLX (Mac) or TRL with vLLM (CUDA), evaluating held-out oracle accuracy every 25 steps.
6. **Score.** Compare each arm with its rate-matched control, pairing seeds and prompts, with a bootstrap that resamples both. Test whether the locked forecast differences order the accuracy differences within those pairs.

The verifier arms for the decimal chain-sum task are in [`configs/arms/chain_sum.yaml`](configs/arms/chain_sum.yaml). Each fixed rule is compared with the resampled control at its own rate:

| Resampled control | Prompt-specific arms | Shared-behaviour arms |
| --- | --- | --- |
| `noise_fp` (false positives on answered wrong outputs) | `fp_hash` | `fp_tolerance` (near misses), `fp_tolerance_decoy` (errors just beyond that tolerance) |
| `noise_fp_trigger` (false positives on every wrong output) | | `fp_trigger` (a phrase not enriched for correct answers) |
| `noise_fn` (false negatives) | `fn_prompt_subset` | `fn_behavior` (a phrase that predicts correct answers) |
| `noise_fn_form` (false negatives) | | `fn_form` (fewer decimal places than the gold string) |

Plus the clean `oracle`, a `combo_fnform_fphash` composite, and a natural `natural_string_match` verifier that is audited but not trained.

## Repository layout

```
configs/           arms, tasks, training and experiment configurations
docs/              theory, architecture, related work
pilots/            P0 to P6: config, entry point, README and committed report for each
scripts/           environment setup for Apple silicon and Linux + CUDA
src/vfates/
  audit.py         step-0 audit: sample, grade, calibrate, featurise, forecast, lock
  experiment.py    train one arm on an audit's frozen pools
  study.py         the confirmatory study: withdrawals, every planned run, the analysis
  cli.py           the `vfates` command
  tasks/           decimal chain sum, MBPP+, IFEval, answer parsing, code sandbox
  verifiers/       arms, calibration, behaviour monitors, natural verifiers
  gen/             MLX, Hugging Face and vLLM generation behind one interface
  features/        LoRA-gradient and hidden-state features, CountSketch, the saved LoRA init
  forecast/        confusion rates, transfer precision, decodeability probe, lockbox
  train/           GRPO reference, MLX and TRL trainers, evaluation
  analysis/        paired contrasts, the preregistered confirmatory tests, scoring against the lock
  theory/          exactly solvable softmax worlds and identity checks
  pilots/          the library code behind each pilot
tests/             unit and integration tests
```

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/theory.md`](docs/theory.md) | The lemma, proposition and theorem, their assumptions, and the check that verifies each |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, public API, artefacts and conventions |
| [`docs/related_work.md`](docs/related_work.md) | How this work relates to prior studies of verifier noise and reward hacking |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Hypotheses, primary endpoint, estimator, design and analysis plan |
| [`pilots/README.md`](pilots/README.md) | The pilot suite, the go/no-go rules and the decision log |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development setup, tests, style and the rules for changing a reported number |

## Hardware

The pilots run on a single Apple-silicon machine; the confirmatory study runs on rented H100s.

| Setting | Measured or projected |
| --- | --- |
| Mac Studio, M4 Max, 36 GB | MLX sampling at batch 128: 1,600 to 2,900 tokens/s for Qwen2.5-0.5B-Instruct and 460 to 840 tokens/s for Qwen2.5-1.5B-Instruct, depending on output length (pilot P2) |
| One H100 80 GB | The P5 gate (Qwen2.5-1.5B-Instruct, on a development pool disjoint from E1). The dry-run report projects its cost; the gate itself measures hours per arm-seed, which replaces the projection in the E1 budget |

## Citation

If you use this code, please cite:

```bibtex
@misc{srivastav2026samerate,
  title  = {Same Rate, Different Fates: A Step-Zero Screen for When Verifier Errors Transfer in RL},
  author = {Srivastav, Shaurya},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://github.com/Shaurya-Srivastav/verifier-drift}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE). Grading MBPP executes model-written code; read [`SECURITY.md`](SECURITY.md) before running it outside a container.
