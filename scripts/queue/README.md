# Queue workers

`vfates program worker` claims jobs from a JSONL queue. The lock file sits beside it (`queue.jsonl.lock`) and uses `fcntl.flock`.

On NFS, that lock is coherent across hosts only when the lock manager is (NFSv4 with lockd). NFSv3 often locks only on the client, so two machines can claim the same job. Use a lock-coherent mount, or one queue per machine. Details are in the `vfates.schedule` module docstring.

## SSH

`scripts/queue/ssh_workers.sh` starts one worker per line of a hosts file.

```
host  hardware_class  gpus
```

`host` may be `user@host`. Blank lines and `#` comments are ignored. See `hosts.example`.

```bash
scripts/queue/ssh_workers.sh scripts/queue/hosts.example /shared/vfates/queue.jsonl
```

The queue path has to be the same path on every host.

## Slurm

`scripts/slurm/job.sbatch` is one allocation: `--requeue`, `--signal=B:USR1@180`, `--time=7-00:00:00`. Partition and account come from `VFATES_SLURM_PARTITION` and `VFATES_SLURM_ACCOUNT`. The GPU count comes from the job.

```bash
export VFATES_SLURM_PARTITION=gpu
export VFATES_SLURM_ACCOUNT=project
vfates program queue --backend slurm --queue /shared/vfates/queue.jsonl
```

`scripts/slurm/submit.py` is the same submission. A group of jobs that share a hardware class and a GPU count is one sbatch array. `--individual` submits one sbatch per job. Each allocation runs a single worker with `--once`.
