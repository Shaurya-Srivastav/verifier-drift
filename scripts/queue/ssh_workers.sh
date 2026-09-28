#!/usr/bin/env bash
# Start one vfates queue worker per line of a hosts file, over SSH.
#
#   scripts/queue/ssh_workers.sh scripts/queue/hosts.example /shared/queue.jsonl
#   scripts/queue/ssh_workers.sh hosts.txt /shared/queue.jsonl /home/me/verifier-drift
#
# Hosts file (whitespace-separated, # comments and blank lines ignored):
#
#   host          hardware_class     gpus
#   gpu01.lab     rtx-a6000-48gb     1
#   user@gpu02    h100-80gb          4
#
# `host` may be a bare hostname or user@host. `hardware_class` is a program
# hardware class (h100-80gb, a100-80gb, rtx-a6000-48gb, ...). `gpus` is the
# integer the worker will claim: it only takes jobs whose placement asks for
# that many GPUs.
#
# The queue path must be the same path on every host (a lock-coherent shared
# filesystem). See the NFS caveat in vfates.schedule. The remote repo defaults
# to the local repo path; pass a third argument when the remote path differs.
# Workers are started with nohup in the background and log under
# runs/workers/ on the remote.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 HOSTS QUEUE [REMOTE_REPO]" >&2
  exit 2
fi

hosts=$1
queue=$2
repo=${3:-$(cd "$(dirname "$0")/../.." && pwd)}

if [[ ! -f "$hosts" ]]; then
  echo "hosts file not found: $hosts" >&2
  exit 1
fi

while read -r host hardware gpus rest; do
  [[ -z "${host:-}" || "$host" == "#"* ]] && continue
  if [[ -n "${rest:-}" ]]; then
    echo "extra field on host line: $host $hardware $gpus $rest" >&2
    exit 1
  fi
  if [[ -z "${hardware:-}" || -z "${gpus:-}" ]]; then
    echo "host line needs hardware_class and gpus: $host" >&2
    exit 1
  fi
  echo "starting $host ($hardware, $gpus GPUs)"
  ssh -n "$host" "mkdir -p '$repo/runs/workers' && cd '$repo' && \
    nohup .venv/bin/python -m vfates.cli program worker \
      --queue '$queue' \
      --hardware-class '$hardware' \
      --gpus '$gpus' \
      > '$repo/runs/workers/${hardware}-${gpus}.log' 2>&1 &"
done < "$hosts"
