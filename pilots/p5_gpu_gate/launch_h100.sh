#!/usr/bin/env bash
# Set up a fresh Linux + CUDA box (one H100 80GB) and run the full P5 gate.
#
#   git clone https://github.com/Shaurya-Srivastav/verifier-drift && cd verifier-drift
#   bash pilots/p5_gpu_gate/launch_h100.sh            # installs, then runs in the background
#   tail -f pilots/p5_gpu_gate/results/run.log
#
# The run is resumable: rerunning this script skips finished audits and runs.
set -euo pipefail
cd "$(dirname "$0")/../.."

command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver on this machine" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[torch,gpu,tasks,analysis]'
python -m vfates.cli env

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p pilots/p5_gpu_gate/results
nohup python pilots/p5_gpu_gate/run.py --set mode=full "$@" \
  > pilots/p5_gpu_gate/results/run.log 2>&1 &
echo "P5 running as PID $! (log: pilots/p5_gpu_gate/results/run.log)"
