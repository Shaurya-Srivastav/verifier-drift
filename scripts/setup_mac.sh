#!/usr/bin/env bash
# Apple-silicon setup: virtualenv plus the all extra (tasks, torch, mlx, analysis, dev).
# The mlx extra's environment marker skips it off Darwin arm64.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v python3.12 >/dev/null 2>&1; then
  base_py=python3.12
else
  base_py=python3
fi

"$base_py" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[all,dev]'
.venv/bin/python -m vfates.cli env
