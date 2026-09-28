#!/bin/bash
# One vfates queue job per allocation.
#
#   #SBATCH --requeue
#   #SBATCH --signal=B:USR1@180
#   #SBATCH --time=7-00:00:00
#
# Partition, account, and GPU count are not hardcoded. `vfates program queue
# --backend slurm` (or scripts/slurm/submit.py) passes them from
# VFATES_SLURM_PARTITION, VFATES_SLURM_ACCOUNT, and the job's GPU count.
# The script execs the worker so USR1 reaches that process. The worker
# forwards USR1 and SIGTERM to the study child.
#SBATCH --job-name=vfates
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1

set -euo pipefail
cd "${VFATES_REPO:-${SLURM_SUBMIT_DIR:-.}}"

queue="${VFATES_QUEUE:?set VFATES_QUEUE}"
hardware="${VFATES_HARDWARE_CLASS:?set VFATES_HARDWARE_CLASS}"
gpus="${VFATES_GPUS:?set VFATES_GPUS}"

if [[ -x .venv/bin/python ]]; then
  py=.venv/bin/python
else
  py=python
fi

exec "$py" -m vfates.cli program worker \
  --queue "$queue" \
  --hardware-class "$hardware" \
  --gpus "$gpus" \
  --once
