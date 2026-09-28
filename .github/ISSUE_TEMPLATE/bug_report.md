---
name: Bug report
about: A defect in vfates, a pilot, or a documented check
title: ""
labels: bug
---

## What happened

A short description of the wrong behaviour, and what you expected instead.

## Reproduction

Steps, the command, and the config. A minimal script is better than a full pilot log.

```bash
python -m vfates.cli env
```

Paste that report. Note the commit (`git rev-parse HEAD`).

## Environment

- OS:
- Python:
- Extra installed (`dev`, `tasks`, `torch`, `mlx`, `gpu`, `all`):
- Accelerator (cpu / mps / cuda), if the failure uses a model:

## Logs

The relevant traceback or the failing rows of `report.md`. Do not paste tokens, weights, or private prompts.
