"""File-locked job queue for cell audits, arm runs, and transfer evals.

The queue is a JSONL file. Every update takes an exclusive ``fcntl.flock`` on
a sibling lock file (``<queue>.lock``, so ``queue.jsonl`` locks
``queue.jsonl.lock``). The lock is advisory and POSIX. It is coherent across
processes on one machine, and across machines only when the filesystem's lock
manager is coherent.

NFS caveat: NFSv3 often serves POSIX locks locally, so two clients can both
believe they hold the lock and claim the same job. Put the queue and its lock
file on a filesystem that honors locks across every worker (a local disk, or
NFSv4 with a working lockd). Lab machines that do not share such a filesystem
must not share one queue file: use one queue per machine, or a lock-coherent
mount. The queue file and the lock file must live in the same directory so the
atomic replace stays on one filesystem.

States are ``pending``, ``claimed``, ``done``, ``failed``, and ``preempted``.
``preempted`` is not terminal: the next matching worker claims it again.
A claim with no heartbeat for ``stale_s`` seconds (default 10 minutes) returns
to ``pending``. Each claim increments ``attempt``.

A worker launches one claimed job as a subprocess. Exit code 0 with summary
status ``preempted`` records state ``preempted`` (requeue). Any other exit 0
is ``done``. A non-zero exit is ``failed``. SIGTERM and SIGUSR1 received by
the worker are forwarded to the child process group. Slurm's
``--signal=B:USR1@180`` therefore reaches the trainer, which is expected to
save a checkpoint and exit 0 (the handler itself belongs to the backends
owner).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Single-process accelerate file, named so the backends owner can add it.
# The worker does not launch it: a 1-GPU study job is plain ``python -m``.
ACCELERATE_CONFIGS: dict[int, str] = {
    1: "configs/accelerate/single_gpu.yaml",
    2: "configs/accelerate/fsdp_2gpu.yaml",
    4: "configs/accelerate/fsdp_4gpu.yaml",
}

STATES: tuple[str, ...] = ("pending", "claimed", "done", "failed", "preempted")
CLAIMABLE: frozenset[str] = frozenset({"pending", "preempted"})
# Training jobs share the study CLI shape. Audits stay one process: vLLM uses
# tensor_parallel_size, and accelerate launch would start one audit per rank.
_ACCELERATE_KINDS = frozenset({"train", "early", "variant", "kl", "streaming"})
DEFAULT_STALE_S = 10 * 60
DEFAULT_HEARTBEAT_S = 30.0


def lock_path(queue: str | Path) -> Path:
    """Sibling lock file. ``queue.jsonl`` -> ``queue.jsonl.lock``."""
    path = Path(queue)
    return path.with_name(path.name + ".lock")


def _read_jobs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    jobs: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            jobs.append(json.loads(line))
    return jobs


def _write_jobs(path: Path, jobs: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(job), sort_keys=True, separators=(",", ":")) for job in jobs]
    payload = ("\n".join(lines) + "\n") if lines else ""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _update(queue: str | Path, mutate: Callable[[list[dict[str, Any]]], Any]) -> Any:
    """Read, mutate, and write the queue while holding the lock."""
    import fcntl

    path = Path(queue)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(lock_path(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        jobs = _read_jobs(path)
        result = mutate(jobs)
        _write_jobs(path, jobs)
        return result
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


def read_jobs(queue: str | Path) -> list[dict[str, Any]]:
    """Snapshot of the queue. Takes the lock so a writer cannot tear a line."""

    def mutate(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(job) for job in jobs]

    return _update(queue, mutate)


def enqueue(queue: str | Path, jobs: Sequence[Mapping[str, Any]]) -> int:
    """Add jobs whose ids are new. Existing ids are left untouched.

    A finished job is not reset by a second plan. Returns the number added.
    """

    def mutate(current: list[dict[str, Any]]) -> int:
        known = {str(job["id"]) for job in current}
        added = 0
        for job in jobs:
            if str(job["id"]) in known:
                continue
            current.append(dict(job))
            known.add(str(job["id"]))
            added += 1
        return added

    return int(_update(queue, mutate))


def _claim_rank(job: Mapping[str, Any]) -> int:
    """0 stage A (and unstaged confirmatory), 1 stage B, 2 exploratory."""
    if job.get("exploratory"):
        return 2
    if str(job.get("stage") or "") == "B":
        return 1
    return 0


def _deps_done(job: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> bool:
    states = {str(item["id"]): str(item["state"]) for item in jobs}
    return all(states.get(str(dep)) == "done" for dep in job.get("depends_on") or [])


def _reap_inplace(jobs: list[dict[str, Any]], *, stale_s: float, now: float | None = None) -> int:
    moment = time.time() if now is None else now
    reaped = 0
    for job in jobs:
        if job.get("state") != "claimed":
            continue
        heartbeat = job.get("heartbeat")
        if heartbeat is None:
            heartbeat = job.get("claimed_at")
        if heartbeat is None or moment - float(heartbeat) > stale_s:
            job["state"] = "pending"
            job["claimed_by"] = None
            job["heartbeat"] = None
            job["claimed_at"] = None
            reaped += 1
    return reaped


def reap_stale(
    queue: str | Path, *, stale_s: float = DEFAULT_STALE_S, now: float | None = None
) -> int:
    """Return claimed jobs with no heartbeat for ``stale_s`` seconds to pending.

    ``attempt`` is left as it was at the claim. The default is 10 minutes.
    """

    def mutate(jobs: list[dict[str, Any]]) -> int:
        return _reap_inplace(jobs, stale_s=stale_s, now=now)

    return int(_update(queue, mutate))


def claim_next(
    queue: str | Path,
    *,
    hardware_class: str,
    gpus: int,
    worker_id: str,
    stale_s: float = DEFAULT_STALE_S,
) -> dict[str, Any] | None:
    """Claim the next matching ``pending`` or ``preempted`` job whose deps are done.

    Among jobs for this hardware class and GPU count, stage A confirmatory
    jobs are claimed before stage B, and every confirmatory job before an
    exploratory job. A confirmatory job with no ``stage`` ranks with stage A,
    so a queue written before stages keeps file order. Each rank stays in
    file order, which is plan order when the queue was built by ``plan_jobs``.
    Stale claims return to ``pending`` in the same locked update, before the
    new claim. ``attempt`` increments by one.
    """

    def mutate(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
        _reap_inplace(jobs, stale_s=stale_s)
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for index, job in enumerate(jobs):
            if job.get("state") not in CLAIMABLE:
                continue
            if str(job.get("hardware_class")) != hardware_class:
                continue
            if int(job.get("gpus", 0)) != int(gpus):
                continue
            if not _deps_done(job, jobs):
                continue
            ranked.append((_claim_rank(job), index, job))
        if not ranked:
            return None
        ranked.sort()
        job = ranked[0][2]
        now = time.time()
        job["state"] = "claimed"
        job["attempt"] = int(job.get("attempt") or 0) + 1
        job["claimed_by"] = worker_id
        job["claimed_at"] = now
        job["heartbeat"] = now
        return dict(job)

    return _update(queue, mutate)


def touch_heartbeat(queue: str | Path, job_id: str, worker_id: str) -> bool:
    """Refresh ``heartbeat`` if ``worker_id`` still holds ``job_id``."""

    def mutate(jobs: list[dict[str, Any]]) -> bool:
        for job in jobs:
            if (
                job.get("id") == job_id
                and job.get("state") == "claimed"
                and job.get("claimed_by") == worker_id
            ):
                job["heartbeat"] = time.time()
                return True
        return False

    return bool(_update(queue, mutate))


def _set_state(
    queue: str | Path,
    job_id: str,
    state: str,
    **fields: Any,
) -> None:
    def mutate(jobs: list[dict[str, Any]]) -> None:
        for job in jobs:
            if job.get("id") != job_id:
                continue
            job["state"] = state
            job["claimed_by"] = None
            job["heartbeat"] = None
            job["claimed_at"] = None
            for key, value in fields.items():
                job[key] = value
            return
        raise KeyError(f"unknown job {job_id}")

    _update(queue, mutate)


def mark_done(queue: str | Path, job_id: str, *, exit_code: int = 0) -> None:
    _set_state(queue, job_id, "done", exit_code=exit_code)


def mark_failed(queue: str | Path, job_id: str, *, exit_code: int) -> None:
    _set_state(queue, job_id, "failed", exit_code=exit_code)


def record_preemption(queue: str | Path, job_id: str) -> None:
    """Requeue a preempted job: state ``preempted``, claim cleared, still runnable."""

    def mutate(jobs: list[dict[str, Any]]) -> None:
        for job in jobs:
            if job.get("id") != job_id:
                continue
            job["state"] = "preempted"
            job["preemptions"] = int(job.get("preemptions") or 0) + 1
            job["claimed_by"] = None
            job["heartbeat"] = None
            job["claimed_at"] = None
            job["exit_code"] = 0
            return
        raise KeyError(f"unknown job {job_id}")

    _update(queue, mutate)


def apply_result(
    queue: str | Path, job_id: str, returncode: int, summary_status: str | None
) -> str:
    """Map a subprocess result to a queue state. Return the new state.

    Exit code 0 and summary status ``preempted`` requeues (state ``preempted``).
    Any other exit 0 is ``done``. A non-zero exit is ``failed``.
    """
    if returncode == 0 and summary_status == "preempted":
        record_preemption(queue, job_id)
        return "preempted"
    if returncode == 0:
        mark_done(queue, job_id, exit_code=0)
        return "done"
    mark_failed(queue, job_id, exit_code=int(returncode))
    return "failed"


def accelerate_config(gpus: int) -> str:
    """Accelerate config path for a GPU count.

    2 and 4 GPUs use ``fsdp_2gpu.yaml`` and ``fsdp_4gpu.yaml``. 1 GPU names
    ``single_gpu.yaml`` for the backends owner; the worker does not launch it.
    Any other count uses ``fsdp_<n>gpu.yaml``.
    """
    if gpus in ACCELERATE_CONFIGS:
        return ACCELERATE_CONFIGS[gpus]
    return f"configs/accelerate/fsdp_{gpus}gpu.yaml"


def build_worker_command(job: Mapping[str, Any], *, python: str | None = None) -> list[str]:
    """Subprocess argv for one queue job.

    Single-GPU study jobs are ``python -m vfates.cli study ...``. Training
    jobs with 2 or more GPUs go through ``accelerate launch`` and
    ``configs/accelerate/fsdp_<n>gpu.yaml``. Audits and transfer evals stay
    one process even when the placement has several GPUs: the audit's vLLM
    engine uses ``tensor_parallel_size``, and a transfer eval is one generator.
    """
    py = python or "python"
    kind = str(job["kind"])
    if kind == "transfer":
        return [
            py,
            "-m",
            "vfates.cli",
            "program",
            "transfer-eval",
            "--program",
            str(job.get("program", "configs/program.yaml")),
            "--runs",
            str(job.get("runs_root", "runs")),
            "--cell",
            str(job["cell"]),
            "--arm",
            str(job["arm"]),
            "--seed",
            str(job["seed"]),
            "--target",
            str(job["target_domain"]),
        ]
    args = [
        "study",
        "--config",
        str(job["config"]),
        "--out",
        str(job["out_dir"]),
    ]
    if kind == "audit":
        args += ["--only", "audit"]
    else:
        args += ["--only", "train", "--arm", str(job["arm"]), "--seed", str(job["seed"])]
        if kind == "early":
            args.append("--early")
        if kind == "variant":
            args += ["--variant", str(job["variant"])]
        sets: list[str] = []
        if kind in ("kl", "streaming") and job.get("audit_dir"):
            sets.append(f"audit_dir={job['audit_dir']}")
        if kind == "kl":
            sets.append(f"study.train.beta_kl={job.get('beta_kl', 0.01)}")
        if kind == "streaming":
            sets.append("study.train.data_regime=streaming")
        for item in sets:
            args += ["--set", item]
    gpus = int(job.get("gpus", 1))
    accelerate_file = job.get("accelerate_config")
    if kind in _ACCELERATE_KINDS and (gpus > 1 or accelerate_file):
        config_file = str(accelerate_file) if accelerate_file else accelerate_config(gpus)
        return [
            "accelerate",
            "launch",
            "--config_file",
            config_file,
            "--num_processes",
            str(gpus),
            "-m",
            "vfates.cli",
            *args,
        ]
    return [py, "-m", "vfates.cli", *args]


def summary_status_for(job: Mapping[str, Any]) -> str | None:
    """``summary.json`` status for a study train job, or None if there is no file."""
    if job.get("kind") not in _ACCELERATE_KINDS:
        return None
    out = Path(str(job["out_dir"]))
    arm = str(job.get("arm"))
    seed = int(job["seed"])
    kind = str(job.get("kind"))
    if kind == "early":
        path = out / "early" / f"{arm}__s{seed}" / "summary.json"
    elif kind == "variant":
        path = out / "variants" / str(job.get("variant")) / f"{arm}__s{seed}" / "summary.json"
    else:
        path = out / "runs" / f"{arm}__s{seed}" / "summary.json"
    if not path.is_file():
        return None
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    return None if status is None else str(status)


def _forward_signals(proc: subprocess.Popen[Any]) -> dict[int, Any]:
    import signal

    def forward(signum: int, _frame: object) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            return

    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        previous[signum] = signal.signal(signum, forward)
    return previous


def _restore_signals(previous: Mapping[int, Any]) -> None:
    import signal

    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _heartbeat_loop(
    queue: str | Path,
    job_id: str,
    worker_id: str,
    interval: float,
    stop: threading.Event,
) -> None:
    while not stop.wait(interval):
        try:
            touch_heartbeat(queue, job_id, worker_id)
        except (OSError, KeyError, ValueError):
            continue


def run_worker(
    queue: str | Path,
    *,
    hardware_class: str,
    gpus: int,
    once: bool = False,
    repo: str | Path | None = None,
    poll_s: float = 15.0,
    heartbeat_s: float = DEFAULT_HEARTBEAT_S,
    stale_s: float = DEFAULT_STALE_S,
    python: str | None = None,
) -> int:
    """Claim matching jobs and run them until the queue has none, or once.

    ``python`` defaults to the interpreter running the worker, so a venv
    worker launches study jobs with that same interpreter (``python -m
    vfates.cli``).
    """
    from vfates.program import repo_root

    root = Path(repo) if repo is not None else repo_root()
    interpreter = python or __import__("sys").executable
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    while True:
        job = claim_next(
            queue,
            hardware_class=hardware_class,
            gpus=gpus,
            worker_id=worker_id,
            stale_s=stale_s,
        )
        if job is None:
            if once:
                return 0
            time.sleep(poll_s)
            continue
        command = build_worker_command(job, python=interpreter)
        stop = threading.Event()
        thread = threading.Thread(
            target=_heartbeat_loop,
            args=(queue, str(job["id"]), worker_id, heartbeat_s, stop),
            name=f"vfates-heartbeat-{job['id']}",
            daemon=True,
        )
        thread.start()
        previous: dict[int, Any] = {}
        try:
            proc = subprocess.Popen(command, cwd=root, start_new_session=True)
            previous = _forward_signals(proc)
            code = int(proc.wait())
        finally:
            stop.set()
            if previous:
                _restore_signals(previous)
            thread.join(timeout=max(heartbeat_s, 1.0))
        status = summary_status_for(job)
        apply_result(queue, str(job["id"]), code, status)
        if once:
            return 0


def format_status(queue: str | Path) -> str:
    """Counts by state, then by hardware class."""
    jobs = read_jobs(queue)
    by_state = {state: 0 for state in STATES}
    for job in jobs:
        state = str(job.get("state", "pending"))
        by_state[state] = by_state.get(state, 0) + 1
    lines = [f"{'state':<12} {'count':>6}"]
    for state in STATES:
        lines.append(f"{state:<12} {by_state.get(state, 0):>6}")
    lines.append("")
    header = f"{'hardware_class':<22}" + "".join(f" {state:>10}" for state in STATES)
    lines.append(header)
    classes = sorted({str(job.get("hardware_class")) for job in jobs})
    for name in classes:
        counts = {state: 0 for state in STATES}
        for job in jobs:
            if str(job.get("hardware_class")) == name:
                counts[str(job.get("state"))] = counts.get(str(job.get("state")), 0) + 1
        row = f"{name:<22}" + "".join(f" {counts.get(state, 0):>10}" for state in STATES)
        lines.append(row)
    lines.append(f"jobs: {len(jobs)}")
    return "\n".join(lines)


def _group_submittable(jobs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], int]:
    groups: dict[tuple[str, int], int] = {}
    for job in jobs:
        if job.get("state") not in CLAIMABLE:
            continue
        if not _deps_done(job, jobs):
            continue
        key = (str(job["hardware_class"]), int(job["gpus"]))
        groups[key] = groups.get(key, 0) + 1
    return groups


def job_name_fields(job_name: str) -> tuple[str, str]:
    """Split ``vfates-<hardware>-<gpus>g`` into the class id and GPU count text.

    Hardware class ids contain hyphens (``rtx-a6000-48gb``), so the GPU suffix
    is the segment after the last hyphen.
    """
    body = job_name.removeprefix("vfates-")
    hardware, _, gpu_text = body.rpartition("-")
    if not hardware or not gpu_text.endswith("g") or not gpu_text[:-1].isdigit():
        raise ValueError(f"job name {job_name!r} is not vfates-<hardware>-<gpus>g")
    return hardware, gpu_text[:-1]


def submit_slurm(
    queue: str | Path,
    *,
    repo: str | Path | None = None,
    individual: bool = False,
    dry_run: bool = False,
    partition: str | None = None,
    account: str | None = None,
) -> list[list[str]]:
    """Submit claimable jobs as one allocation each.

    Jobs are grouped by ``(hardware_class, gpus)``. A group of one sbatch is
    a single job. A larger group is a Slurm array of that length unless
    ``individual`` is set, in which case each job is its own sbatch. Each
    allocation runs ``scripts/slurm/job.sbatch``, which starts one worker
    with ``--once``. Partition and account come from ``VFATES_SLURM_PARTITION``
    and ``VFATES_SLURM_ACCOUNT`` when the arguments are omitted.
    """
    from vfates.program import repo_root

    root = Path(repo) if repo is not None else repo_root()
    part = partition if partition is not None else os.environ.get("VFATES_SLURM_PARTITION")
    acct = account if account is not None else os.environ.get("VFATES_SLURM_ACCOUNT")
    if not part or not acct:
        raise ValueError(
            "Slurm partition and account are required "
            "(VFATES_SLURM_PARTITION and VFATES_SLURM_ACCOUNT)"
        )
    jobs = read_jobs(queue)
    groups = _group_submittable(jobs)
    script = root / "scripts" / "slurm" / "job.sbatch"
    commands: list[list[str]] = []
    for (hardware, gpus), count in sorted(groups.items()):
        sizes = [1] * count if individual else [count]
        for size in sizes:
            cmd = [
                "sbatch",
                "--requeue",
                "--signal=B:USR1@180",
                "--time=7-00:00:00",
                f"--partition={part}",
                f"--account={acct}",
                f"--gpus={gpus}",
                f"--job-name=vfates-{hardware}-{gpus}g",
                f"--chdir={root}",
                "--export=ALL",
            ]
            if size > 1:
                cmd.append(f"--array=0-{size - 1}")
            cmd.append(str(script))
            commands.append(cmd)
    if dry_run:
        return commands
    env = os.environ.copy()
    env["VFATES_QUEUE"] = str(Path(queue).resolve())
    env["VFATES_REPO"] = str(root)
    for cmd in commands:
        class_id = ""
        gpu_count = ""
        for part_arg in cmd:
            if part_arg.startswith("--job-name="):
                class_id, gpu_count = job_name_fields(part_arg.removeprefix("--job-name="))
        child_env = dict(env)
        child_env["VFATES_HARDWARE_CLASS"] = class_id
        child_env["VFATES_GPUS"] = gpu_count
        subprocess.run(cmd, cwd=root, env=child_env, check=True)
    return commands


def claim_until_empty(
    queue: str | Path,
    *,
    hardware_class: str,
    gpus: int,
    worker_id: str,
) -> list[str]:
    """Claim and mark done until no matching job is free. For concurrency tests."""
    got: list[str] = []
    while True:
        job = claim_next(queue, hardware_class=hardware_class, gpus=gpus, worker_id=worker_id)
        if job is None:
            return got
        got.append(str(job["id"]))
        mark_done(queue, str(job["id"]))


def claim_until_empty_to_file(
    queue: str,
    dest: str,
    hardware_class: str,
    gpus: int,
    worker_id: str,
) -> None:
    """Claim until empty and write the ids, one per line. Spawn-safe test helper."""
    ids = claim_until_empty(queue, hardware_class=hardware_class, gpus=gpus, worker_id=worker_id)
    Path(dest).write_text("\n".join(ids), encoding="utf-8")
