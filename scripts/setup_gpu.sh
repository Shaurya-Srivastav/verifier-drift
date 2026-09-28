#!/usr/bin/env bash
# Linux + CUDA setup for GPU runs. Pins are the preregistration stack.
# No sudo. Packages go into .venv in this repo (created if missing, reused if
# present). That works on a lab machine and inside a Slurm job. Ampere (A100,
# RTX A6000) and Ada (RTX 5000 Ada) need a driver new enough for the CUDA 12.8
# wheels (driver 570 or newer).
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

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
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
# math-verify (import math_verify) is in the tasks extra. LLMJudge talks to
# an OpenAI-compatible endpoint with urllib, so the openai package is not
# installed.
.venv/bin/python -m pip install -c "$constraint" -e '.[gpu,torch,tasks,dev]'
rm -f "$constraint"

# Prebuilt flash-attn wheel for torch 2.8 + CUDA 12. A source build needs nvcc
# and is not attempted. Training falls back to sdpa when this step fails.
install_flash_attn() {
  local py="$1"
  local cp cxx ver wheel url
  read -r cp cxx < <("$py" - << 'PY'
import torch, sys
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
print(f"cp{sys.version_info.major}{sys.version_info.minor} {abi}")
PY
)
  for ver in 2.8.3 2.8.2 2.8.1 2.8.0 2.7.4.post1; do
    wheel="flash_attn-${ver}+cu12torch2.8cxx11abi${cxx}-${cp}-${cp}-linux_x86_64.whl"
    url="https://github.com/Dao-AILab/flash-attention/releases/download/v${ver}/${wheel}"
    if "$py" -m pip install "$url"; then
      return 0
    fi
  done
  return 1
}
if ! install_flash_attn .venv/bin/python; then
  echo "flash-attn prebuilt wheel unavailable for this Python/torch/CUDA. Training will use sdpa." >&2
fi

# Gated Llama (meta-llama/Llama-3.1-8B-Instruct) needs a Hub token.
if [ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]; then
  HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" .venv/bin/python - << 'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("huggingface_hub: logged in with HF_TOKEN")
PY
else
  echo "HF_TOKEN is unset. meta-llama/Llama-3.1-8B-Instruct is gated." >&2
  echo "Export HF_TOKEN and rerun, or run: .venv/bin/python -m huggingface_hub.cli.hf auth login" >&2
fi

.venv/bin/python -m pip check
.venv/bin/python -m vfates.cli env
