#!/usr/bin/env bash
# Apple-silicon setup: virtualenv plus the all extra (tasks, torch, mlx, analysis, dev).
# The tasks extra includes math-verify. The mlx extra's environment marker skips it off Darwin arm64.
# No sudo. Reuses .venv when it already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v python3.12 >/dev/null 2>&1; then
  base_py=python3.12
else
  base_py=python3
fi

if [ ! -x .venv/bin/python ]; then
  "$base_py" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[all,dev]'
.venv/bin/python -m vfates.cli env
