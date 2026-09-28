# Contributing

Bug reports, fixes and small extensions are welcome. Please open an issue before starting a large change.

This repository is the code for a paper, so one rule matters more than the rest: **a change that moves a reported number must update the check that produces it**, and the new report must be committed with the change.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Python 3.10 or newer. From the repository root:

```bash
bash scripts/setup_mac.sh          # Apple silicon: .venv with the all extra
bash scripts/setup_gpu.sh          # Linux + CUDA: .venv with the gpu, torch, tasks and dev extras
```

`setup_mac.sh` uses `python3.12` when it is on `PATH` and `python3` otherwise. `setup_gpu.sh` stops unless it is running on Linux with a working `nvidia-smi`.

To install into an existing environment instead:

| Command | Installs | Use for |
| --- | --- | --- |
| `make install` | `.[dev,tasks]` | CPU tests and CI |
| `make install-mac` | `.[all]` (tasks, torch, mlx, analysis, dev) | Apple silicon |
| `make install-gpu` | `.[gpu,torch,tasks,dev]` | Linux + CUDA |

The Makefile uses `.venv/bin/python` when it exists and `python` otherwise. The `mlx` extra installs only on macOS arm64, and the `gpu` extra installs vLLM only on Linux; `all` does not include `gpu`.

Check the environment with `vfates env`. The same report is embedded in every artefact the code writes.

## Tests, linting and types

```bash
make test           # pytest -m 'not slow and not network'
make test-all       # every test, including slow, network and mlx
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy src
```

Ruff is configured in `pyproject.toml` (line length 100, target Python 3.10). The names `O` and `V` follow the paper's notation and are exempt from E741.

Pre-commit runs `ruff-check`, `ruff-format`, the end-of-file and trailing-whitespace fixers, `check-yaml` and `check-added-large-files` (1,000 KB). Enable it once with `pre-commit install`.

Tests that need a downloaded model, the network or a specific accelerator carry the `network`, `slow`, `mlx` or `torch` marker, so `make test` stays fast and offline.

## What a change needs

- **A new identity, forecast, arm or estimator** needs a unit test.
- **A claim that is part of the paper's argument** needs a pointer from `docs/theory.md` to the test or pilot check that verifies it.
- **A gate is never weakened to make a run pass.** A known limitation is recorded as a `documented` check, which is reported but does not fail the pilot (`vfates.pilots.common.Checks`).
- **Held-out prompts are never used** to fit verifiers, calibrations or forecasts, or to choose any setting.
- **Anything a preregistered comparison uses** is locked with `vfates.forecast.lockbox` before the corresponding training starts.

## Changing a reported number or the design

1. Rerun the pilot that produces the number (`vfates pilot <name>`), and commit its new `results/report.md` and `results/report.json` in the same pull request.
2. Update every document that quotes the number. `rg` for the old value.
3. If the change alters the design (an arm, a gate, an endpoint), add an entry to the design evidence in [`pilots/README.md`](pilots/README.md) that names the evidence. After the preregistration is frozen, log it under Deviations in [`PREREGISTRATION.md`](PREREGISTRATION.md) instead, with a date and reason.

## Pilots

Run a pilot with `vfates pilot <name>`, `make <name>` or `python pilots/<folder>/run.py`. All three accept `--config`, `--out` and repeated `--set key=value`. `make pilots-mac` runs P0, P1 with `live=true`, and smoke runs of P2 to P6 into `runs/smoke/`; the scored runs of P2 to P5 need GPUs. The order, hardware and go/no-go rules are in [`pilots/README.md`](pilots/README.md).

| Target | Folder | What it checks |
| --- | --- | --- |
| `p0` | `pilots/p0_theory` | Identities, same-rate fates and transfer precision on exactly solvable worlds |
| `p1` | `pilots/p1_harness` | Every stage of every domain, the program runner and the analysis, on Qwen3-0.6B |
| `p2` | `pilots/p2_base_audit` | Each domain's difficulty bucket and token budget at `qwen3-1.7b` and `qwen3-4b` |
| `p3` | `pilots/p3_estimator` | Reliability of `g_V`, sketch fidelity, bfloat16 agreement and the lock, per domain |
| `p4` | `pilots/p4_micro_grpo` | The oracle arm's 150-step held-out gain per domain |
| `p5` | `pilots/p5_gpu_gate` | Every placement, preemption and resume, per-model gain, the seed rule, throughput |
| `p6` | `pilots/p6_natural` | Step-0 rates of every natural verifier and the human adjudication of its false positives |

## What is tracked

| Tracked | Not tracked |
| --- | --- |
| Every pilot's `results/report.md`, `results/report.json` and `results/decisions.json` | Everything else under `pilots/*/results/` (completions, features, checkpoints, cell dumps) |
| A pilot audit's `calibration.json`, `forecasts.json` and `lock.json`, and P6's labelled `results/adjudication/*.csv` | `runs/`, `checkpoints/`, `wandb/`, `*.pt`, `*.safetensors`, `*.log` |
| `configs/throughput.yaml` once P5 has measured it | Private chat exports (`cursor_*.md`) and any credentials |
| `paper/` sources, `paper/generated/`, `paper/figures/` and `paper/main.pdf` (rebuild with `make paper` after the report changes) | LaTeX build files (`paper/*.aux`, `*.bbl`, `*.blg`, `*.fls`, `*.fdb_latexmk`, `*.out`, `*.log`) |

## Pull requests

Before opening a pull request:

- [ ] `make lint`, `make typecheck` and `make test` pass.
- [ ] New behaviour has a test.
- [ ] Any report whose numbers changed is regenerated and committed.
- [ ] Documentation that describes the change is updated.
- [ ] `CHANGELOG.md` has an entry under "Unreleased".

The pull request template repeats this list.

## Reporting issues

Use the bug report or feature request template. For anything that depends on a machine, a model or a pilot, include the output of `vfates env`. Report security issues privately to the address in [SECURITY.md](SECURITY.md), not in a public issue.
