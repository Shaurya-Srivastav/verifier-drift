# Same Rate, Different Fates

**A step-zero screen for when verifier errors transfer in reinforcement learning with verifiable rewards (RLVR).**

[![CI](https://github.com/Shaurya-Srivastav/verifier-drift/actions/workflows/ci.yml/badge.svg)](https://github.com/Shaurya-Srivastav/verifier-drift/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

RLVR trains a language model against a verifier: a unit test, an answer checker, a model judge. Verifiers make mistakes. Prior work measures those mistakes by their rate and finds that random errors at a moderate rate are tolerated. This project tests, across five task domains, model sizes from 1.7B to 32B and three model families:

- whether two verifiers with the same step-0 false-positive and false-negative rates can send held-out accuracy different ways;
- whether a statistic computed from the base model before training, with about 200 oracle labels per verifier, orders those outcomes;
- whether it reads the direction of a shared error: two rules with the same rate and the same structure, whose shared phrase predicts correct or wrong answers;
- whether that statistic chooses better verifier fixes than the error count does.

The study is preregistered in two stages: a discovery stage at 1.7B and 4B on lab GPUs, and a prospective replication at 8B to 32B and in two other model families. Every forecast is locked before the first training step of its cell.

## The idea in one table

Three verifiers, each accepting 15% of incorrect answers:

| Verifier | Which wrong answers it accepts | What the theory predicts on unseen prompts |
| --- | --- | --- |
| Resampled noise | A fresh random 15%, redrawn every time | The model learns the task, a little slower: the update is the clean one, scaled by Youden's J |
| Prompt-specific rule | A fixed 15%, chosen by a hash of the prompt and the answer | The model memorises those answers on training prompts; unseen prompts behave like the resampled case |
| Shared-behaviour rule | A fixed 15% that share a behaviour, for example every answer containing a phrase, or every near miss | The shared update can reinforce that behaviour on unseen prompts too, so the same rate can end in a different place |

The reason is one identity. Write the verifier as the oracle plus its errors, $V = O - \mathrm{FN} + \mathrm{FP}$. The step-0 policy-gradient direction splits the same way:

$$
g_V = g_O - g_{\mathrm{FN}} + g_{\mathrm{FP}}.
$$

Resampled noise only rescales $g_O$. A fixed error adds its own direction, and that direction moves unseen prompts only if the error is carried by something shared across prompts. The screen, **transfer precision** $\tau$, is the oracle accuracy of the held-out outputs that the step-0 update favours most. The update direction is built from the verifier's own verdicts, so the oracle is needed only on those favoured outputs: 192 labels per verifier. The primary estimator is `tau_pool`, computed from LoRA gradients of the base model on a step-0 audit and locked (hashed) before any training run. It is a first-order screen at the base policy; the knife-edge toy world in pilot P0 is the documented case where the long run leaves its reach. The full argument, with every assumption and the check that verifies it, is in [`docs/theory.md`](docs/theory.md).

## The study

The unit is a **cell**: one experiment, one domain, one model, with its own audit, lock and hardware class. The 20 cells are listed in [`configs/program.yaml`](configs/program.yaml).

| Experiment | Cells | What it answers |
| --- | --- | --- |
| E1 fates | 5 domains × Qwen3-1.7B, full arm menu, 3 seeds | Same rate, different fates, in every domain |
| E2 scale | Qwen3-4B (5 domains), 8B (math, code, qa), 14B (math, code), 32B (math); core menu, 2 seeds | Does the screen hold at scale, and how does a shared error's effect change with size? |
| E3 families | OLMo-3-7B-SFT and Llama-3.1-8B-Instruct (math, code); core menu, 2 seeds | Does it hold outside the Qwen family? |
| E4 patching | Two patch arms inside the E1 cells of math, code, ifeval and qa | At an equal budget, does the screen pick better fixes than the error count? |

| Stage | Cells | Hardware | Confirmatory tests |
| --- | --- | --- | --- |
| A, discovery | E1 (with E4) and E2 at 4B | lab GPUs | fates (H1, H2), the screen (H3), direction within shared rules (H10), natural verifiers (H7), patching (H8) |
| B, replication | E2 at 8B, 14B, 32B and E3 | rented H100s | the screen on pairs stage A never saw (H3-replication) |

| Domain | Data | Oracle | Natural verifier practitioners use |
| --- | --- | --- | --- |
| chain_sum | Reasoning Gym `decimal_chain_sum` (the Egashira et al. setting) | exact decimal equality | exact string match (audited) |
| math | Big-Math-RL-Verified, decontaminated | symbolic equivalence (math-verify), no tolerance | a lenient cascade with a 1e-4 relative tolerance; a strict string match |
| code | TACO, Python | every hidden test passes | the public examples in the statement |
| ifeval | IF multi-constraint (IFEval and IFBench constraints) | the checkers and a frozen content judge | the checkers alone |
| qa | WebInstruct-verified, canonical answer types | a strict rule | a 1.5B model judge (TIGER-Lab general-verifier) |

Every domain gets the same kit of synthetic verifiers, fitted on the train pool to FPR 0.15 and FNR 0.30. That is what makes the comparison the same in every domain:

| Resampled control | Prompt-specific arm | Shared-behaviour arm |
| --- | --- | --- |
| `noise_fp` (false positives on answered wrong outputs) | `fp_hash` | `fp_tolerance` (near misses; chain_sum and math) |
| `noise_fp_trigger` (false positives on every wrong output) | | `fp_trigger` (a phrase not enriched for correct answers) |
| `noise_fn` (false negatives) | `fn_prompt_subset` | `fn_behavior` (a phrase that predicts correct answers) |
| `noise_natural` (the natural verifier's own rates) | | `natural` |
| `noise_fp_trigger_enriched` | | `fp_trigger_enriched` (a phrase enriched for correct answers; E1) |
| `noise_fn_depleted` | | `fn_behavior_depleted` (a phrase depleted among correct answers; E1) |

The last two rows share their structure and rate with `fp_trigger` and `fn_behavior` and differ only in the direction of the phrase, so the structure alone cannot predict which way they go; the screen has to (H10).
The preregistered hypotheses, endpoint, estimator, exclusions and analysis are in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Status

| Stage | State |
| --- | --- |
| Theory and exactly solvable worlds (pilot P0) | Passing. Report: [`pilots/p0_theory/results/report.md`](pilots/p0_theory/results/report.md) |
| Harness, base audit, estimator, micro-GRPO, GPU gate, natural verifiers (pilots P1 to P6) | Implemented and tested. Each pilot commits its report when it runs |
| Confirmatory study (E1 to E4) | Preregistered in two stages; each cell starts after the pilots pass and its lock is committed |
| Paper | [`paper/`](paper/): ICML draft whose numbers, tables, figures and abstract results sentence are generated from the reports (`make paper`) |

## Quick start

Python 3.10 or newer. Models and datasets download from Hugging Face on first use. Big-Math and Llama 3.1 are gated: accept their terms on the Hub and set `HF_TOKEN`.

```bash
git clone https://github.com/Shaurya-Srivastav/verifier-drift.git
cd verifier-drift

bash scripts/setup_mac.sh      # Apple silicon: .venv with MLX, PyTorch, tasks, analysis, dev tools
bash scripts/setup_gpu.sh      # Linux + CUDA (H100, A100, RTX A6000, RTX 5000 Ada): TRL, vLLM, FlashAttention
make install                   # anywhere: the CPU core, tasks and tests only
```

Check the install and run the theory pilot, which needs only NumPy and finishes in seconds:

```bash
make test                      # unit and integration tests (no downloads)
vfates pilot p0                # or: python pilots/p0_theory/run.py
vfates env                     # the environment report, including the hardware class
```

## Running the study

The pilots come first; [`pilots/README.md`](pilots/README.md) explains what each one decides and where it runs.

```bash
vfates pilot p1                # every stage of every domain on a small model (Mac)
vfates pilot p2                # difficulty and token budget per domain (GPU)
vfates pilot p3                # step-0 estimator reliability and the lock (GPU)
vfates pilot p4                # oracle signal per domain (GPU)
vfates pilot p5                # every model on its hardware class: cost, resume, multi-GPU (GPU)
vfates pilot p6                # natural verifiers and patch categories (GPU)
```

Then the program. Each cell config is rendered from its domain template and model spec, committed, audited and locked before it trains:

```bash
vfates program render                        # configs/cells/<cell>.yaml from templates and model specs
vfates program plan --by-stage               # every job, its GPUs and projected hours by stage and hardware class
vfates program queue --queue runs/queue.jsonl                      # enqueue audits, runs and early baselines
vfates program worker --queue runs/queue.jsonl --hardware-class rtx-a6000-48gb --gpus 1
vfates program status --queue runs/queue.jsonl
vfates program analyze                       # runs/program_report.{json,md}: each complete stage's tests
```

A worker claims only jobs of its hardware class and GPU count, so the same queue feeds lab machines over SSH (`scripts/queue/ssh_workers.sh`), a Slurm cluster (`scripts/slurm/job.sbatch`, preemptible and requeued) and rented H100s. One cell can also run on its own:

```bash
vfates study --config configs/cells/e1-math-qwen3-1.7b.yaml --out runs/e1-math-qwen3-1.7b
```

Every stage is resumable. Rerunning a command skips finished work, and a preempted run continues from its last checkpoint.

## How it works

1. **Audit.** Sample the base policy 16 times on a frozen train pool and a disjoint held-out pool, and grade every sample with the oracle.
2. **Calibrate the arms.** Fit every synthetic arm on the train-pool samples only, so that all imperfect arms share the same step-0 channel rates (FPR 0.15, FNR 0.30). Resampled controls copy their matched arm's channel rates; natural verifiers keep their own. Phrase rules are rate-matched on the whole train pool, take their preference sign from 7/8 of the train prompts, and have that sign checked on the rest. An arm that misses its rate or fails its check is withdrawn before training.
3. **Featurise.** Compute each sample's gradient with respect to freshly initialised LoRA adapters (the parameters GRPO will train), on the token ids that were actually sampled, and compress it with CountSketch. The LoRA initialisation is saved, and every training run starts from it.
4. **Select patches.** For each natural verifier, choose the two error categories to fix by the screen and by the error count.
5. **Forecast and lock.** Estimate each arm's update direction on the train pool, score held-out samples by their velocity along it, and compute `tau_pool`. Seal the forecasts, the configuration, the hardware class and the hash of every input artefact with SHA-256 before training.
6. **Train.** Run GRPO with each arm as the reward, with TRL and vLLM on CUDA (tensor parallel for 14B and 32B) or MLX on a Mac, evaluating held-out oracle accuracy every 25 steps.
7. **Score.** Compare each arm with its rate-matched control, pairing seeds and prompts, with a bootstrap that resamples both. Pool the within-cell comparisons within each stage, and test whether the locked forecast differences order the accuracy differences: across stage A's domains and sizes, then again on stage B's larger models and other families.

## Repository layout

```
configs/
  program.yaml     every cell: experiment, domain, model, hardware class, menu, seeds
  templates/       one template per domain; cells/ holds the rendered, committed cell configs
  models/          model specs: checkpoint, pinned revision, chat template, LoRA keys, placement
  arms/            the kit and each domain's natural, structural and patch arms
  tasks/, train/, accelerate/
docs/              theory, architecture, related work
pilots/            P0 to P6: config, entry point, README and committed report for each
paper/             the ICML draft; generated/ and figures/ are written by vfates.paper
scripts/           environment setup, Slurm and SSH workers
src/vfates/
  audit.py         step-0 audit: sample, grade, calibrate, featurise, select patches, forecast, lock
  experiment.py    train one arm on an audit's frozen pools
  study.py         one cell: withdrawals, every planned run, the per-cell analysis
  program.py       the cells, rendering, job planning and the pooled analysis
  schedule.py      the file-locked job queue and its workers
  models.py        model specs and placement per hardware class
  cli.py           the `vfates` command
  tasks/           chain sum, math, code, instruction following, judged QA, graders and sandbox
  verifiers/       arms, calibration, patches, behaviour monitors, natural verifiers
  gen/             MLX, Hugging Face and vLLM generation behind one interface
  features/        LoRA-gradient and hidden-state features, CountSketch, the saved LoRA init
  forecast/        confusion rates, transfer precision, decodability probe, lockbox
  train/           GRPO reference, MLX and TRL trainers, evaluation, rewards
  analysis/        per-cell and pooled confirmatory tests, cross-domain transfer, power and cost
  theory/          exactly solvable softmax worlds and identity checks
  paper/           the paper's numbers, tables, figures and abstract sentence, from the reports
  pilots/          the library code behind each pilot
tests/             unit and integration tests
```

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/theory.md`](docs/theory.md) | The lemma, proposition and theorem, their assumptions, and the check that verifies each |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, public API, artefacts and conventions |
| [`docs/related_work.md`](docs/related_work.md) | How this work relates to prior studies of verifier noise and reward hacking |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Hypotheses, primary endpoint, estimator, design, stages and analysis plan |
| [`paper/`](paper/) | The ICML draft and how its generated numbers and figures are built |
| [`pilots/README.md`](pilots/README.md) | The pilot suite, its order and commands, the go/no-go rules and the design evidence |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development setup, tests, style and the rules for changing a reported number |

## Hardware

| Hardware class | Used for |
| --- | --- |
| `apple-silicon` | P0 and P1 (Qwen3-0.6B) |
| `rtx-a6000-48gb`, `rtx-5000-ada-32gb` | E1 at Qwen3-1.7B, the 1.7B pilots, and the judge servers |
| `a100-80gb` | the Qwen3-4B cells |
| `h100-80gb` | 8B (one GPU), 14B (two), 32B (four), and E3 |

P5 measures hours per arm-seed on each class, and those measurements replace the projections `vfates program plan` prints.

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

MIT. See [`LICENSE`](LICENSE). Grading code executes model-written programs; read [`SECURITY.md`](SECURITY.md) before running it outside a container.
