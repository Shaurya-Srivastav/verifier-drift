#!/usr/bin/env bash
# Linux + CUDA setup for the E1 H100 run. Pins are the preregistration stack.
#
# trl==0.25.1 requires vllm==0.10.2 on its vllm extra, transformers>=4.56.1,
# and accelerate>=1.4.0. The peft extra requires peft>=0.8.0.
#   https://pypi.org/pypi/trl/0.25.1/json
# vllm==0.10.2 requires torch==2.8.0, torchvision==0.23.0, torchaudio==2.8.0,
# and transformers>=4.55.2.
#   https://pypi.org/pypi/vllm/0.10.2/json
# transformers==4.57.1 satisfies both floors and matches the Mac audit env.
# peft==0.18.0 and accelerate==1.11.0 satisfy TRL.
# reasoning-gym==0.1.24 pins the task generator (pyproject allows >=0.1.19).
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

# CUDA 12.8 wheels. vLLM 0.10.2 is built against torch 2.8.0.
.venv/bin/python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

constraint="$(mktemp)"
cat > "$constraint" << 'EOF'
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
trl==0.25.1
vllm==0.10.2
transformers==4.57.1
peft==0.18.0
accelerate==1.11.0
reasoning-gym==0.1.24
EOF

.venv/bin/python -m pip install \
  -c "$constraint" \
  trl==0.25.1 \
  vllm==0.10.2 \
  transformers==4.57.1 \
  peft==0.18.0 \
  accelerate==1.11.0 \
  reasoning-gym==0.1.24

# The gpu extra allows vllm>=0.8. The constraint file keeps the pins above.
.venv/bin/python -m pip install -c "$constraint" -e '.[gpu,torch,tasks,dev]'
rm -f "$constraint"
.venv/bin/python -m pip check
.venv/bin/python -m vfates.cli env
