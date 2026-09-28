#!/usr/bin/env bash
# Linux + CUDA setup. The gpu extra does not include torch; both are installed.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Linux" ]; then
  echo "setup_gpu.sh expects Linux with CUDA." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. A CUDA driver is required." >&2
  exit 1
fi

nvidia-smi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[gpu,torch,tasks,dev]'
.venv/bin/python -m vfates.cli env
