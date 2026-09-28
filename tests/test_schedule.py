"""Queue lock, claim transitions, worker argv, and Slurm submission."""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from vfates.schedule import (
    ACCELERATE_CONFIGS,
    _heartbeat_loop,
    apply_result,
    build_worker_command,
    claim_next,
    claim_until_empty_to_file,
    enqueue,
    job_name_fields,
    mark_done,
    read_jobs,
    reap_stale,
    submit_slurm,
)


def _pending(job_id: str, **extra: object) -> dict:
    job: dict = {
        "id": job_id,
        "state": "pending",
        "attempt": 0,
        "hardware_class": "h100-80gb",
        "gpus": 1,
        "depends_on": [],
    }
    job.update(extra)
    return job


def _study_job(**extra: object) -> dict:
    job: dict = {
        "kind": "train",
        "config": "configs/cells/e1-math-qwen3-1.7b.yaml",
        "out_dir": "runs/e1-math-qwen3-1.7b",
        "arm": "oracle",
        "seed": 0,
        "gpus": 1,
        "cell": "e1-math-qwen3-1.7b",
        "program": "configs/program.yaml",
        "runs_root": "runs",
    }
    job.update(extra)
    return job


def test_claim_respects_dependencies_and_attempt(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(
        queue,
        [
            _pending("cell/audit"),
            _pending("cell/train/oracle/s0", depends_on=["cell/audit"]),
        ],
    )
    first = claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
    assert first is not None
    assert first["id"] == "cell/audit"
    assert first["attempt"] == 1
    assert claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w2") is None
    mark_done(queue, "cell/audit")
    second = claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
    assert second is not None
    assert second["id"] == "cell/train/oracle/s0"
    mark_done(queue, second["id"])
    assert claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w") is None


def test_claim_takes_confirmatory_jobs_before_exploratory(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(
        queue,
        [
            _pending("explore-first", exploratory=True),
            _pending("conf-a"),
            _pending("explore-second", exploratory="X1"),
            _pending("conf-b"),
            _pending("conf-blocked", depends_on=["not-done"]),
        ],
    )
    claimed = []
    for _ in range(4):
        job = claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
        assert job is not None
        claimed.append(job["id"])
        mark_done(queue, job["id"])
    assert claimed == ["conf-a", "conf-b", "explore-first", "explore-second"]
    assert claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w") is None


def test_claim_takes_stage_a_before_stage_b(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(
        queue,
        [
            _pending("stage-b", stage="B"),
            _pending("explore-on-a", exploratory=True, stage="A"),
            _pending("stage-a", stage="A"),
            _pending("unstaged"),
        ],
    )
    claimed = []
    for _ in range(4):
        job = claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
        assert job is not None
        claimed.append(job["id"])
        mark_done(queue, job["id"])
    assert claimed == ["stage-a", "unstaged", "stage-b", "explore-on-a"]


def test_heartbeat_thread_refreshes_the_claim(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(queue, [_pending("job")])
    claimed = claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
    assert claimed is not None
    first = float(read_jobs(queue)[0]["heartbeat"])
    stop = threading.Event()
    thread = threading.Thread(
        target=_heartbeat_loop,
        args=(queue, "job", "w", 0.05, stop),
    )
    thread.start()
    moved = False
    deadline = time.time() + 2
    while time.time() < deadline:
        current = float(read_jobs(queue)[0]["heartbeat"])
        if current > first:
            moved = True
            break
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=2)
    assert moved


def test_stale_claim_returns_to_pending_without_bumping_attempt(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(queue, [_pending("job")])
    claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
    jobs = read_jobs(queue)
    jobs[0]["heartbeat"] = time.time() - 1000
    queue.write_text("\n".join(json.dumps(job) for job in jobs) + "\n", encoding="utf-8")
    assert reap_stale(queue, stale_s=60) == 1
    pending = read_jobs(queue)[0]
    assert pending["state"] == "pending"
    assert pending["attempt"] == 1
    again = claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w2")
    assert again is not None
    assert again["attempt"] == 2


def test_preempted_exit_requeues_and_other_exits_finish(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(queue, [_pending("job")])
    claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
    assert apply_result(queue, "job", 0, "preempted") == "preempted"
    row = read_jobs(queue)[0]
    assert row["state"] == "preempted"
    assert row["preemptions"] == 1
    again = claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
    assert again is not None
    assert again["state"] == "claimed"
    assert apply_result(queue, "job", 0, None) == "done"
    assert read_jobs(queue)[0]["state"] == "done"

    enqueue(queue, [_pending("bad")])
    claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
    assert apply_result(queue, "bad", 1, None) == "failed"
    failed = next(job for job in read_jobs(queue) if job["id"] == "bad")
    assert failed["state"] == "failed"
    assert failed["exit_code"] == 1


def test_two_processes_claim_disjoint_jobs(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(queue, [_pending(f"j{i}") for i in range(8)])
    ctx = multiprocessing.get_context("spawn")
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    first = ctx.Process(
        target=claim_until_empty_to_file,
        args=(str(queue), str(left), "h100-80gb", 1, "w1"),
    )
    second = ctx.Process(
        target=claim_until_empty_to_file,
        args=(str(queue), str(right), "h100-80gb", 1, "w2"),
    )
    first.start()
    second.start()
    first.join(30)
    second.join(30)
    assert first.exitcode == 0
    assert second.exitcode == 0
    ids_left = {line for line in left.read_text(encoding="utf-8").splitlines() if line}
    ids_right = {line for line in right.read_text(encoding="utf-8").splitlines() if line}
    assert ids_left.isdisjoint(ids_right)
    assert ids_left | ids_right == {f"j{i}" for i in range(8)}


def test_worker_command_for_one_two_and_four_gpus() -> None:
    one = build_worker_command(_study_job(gpus=1), python="python")
    assert one[0] == "python"
    assert "accelerate" not in one
    assert one[one.index("-m") + 1] == "vfates.cli"
    assert "--only" in one and one[one.index("--only") + 1] == "train"
    assert one[one.index("--arm") + 1] == "oracle"
    assert one[one.index("--seed") + 1] == "0"
    assert "single_gpu.yaml" not in " ".join(one)
    assert ACCELERATE_CONFIGS[1] == "configs/accelerate/single_gpu.yaml"

    two = build_worker_command(_study_job(gpus=2), python="python")
    assert two[:3] == ["accelerate", "launch", "--config_file"]
    assert "configs/accelerate/fsdp_2gpu.yaml" in two
    assert two[two.index("--num_processes") + 1] == "2"

    four = build_worker_command(_study_job(gpus=4), python="python")
    assert "configs/accelerate/fsdp_4gpu.yaml" in four
    assert four[four.index("--num_processes") + 1] == "4"

    audit = build_worker_command(_study_job(kind="audit", gpus=4), python="python")
    assert audit[0] == "python"
    assert "accelerate" not in audit
    assert audit[audit.index("--only") + 1] == "audit"

    early = build_worker_command(_study_job(kind="early", seed=99), python="python")
    assert "--early" in early

    kl = build_worker_command(
        _study_job(kind="kl", gpus=1, audit_dir="runs/cell/audit", beta_kl=0.01),
        python="python",
    )
    assert "audit_dir=runs/cell/audit" in kl
    assert "study.train.beta_kl=0.01" in kl

    streaming = build_worker_command(_study_job(kind="streaming", gpus=1), python="python")
    assert "study.train.data_regime=streaming" in streaming

    variant = build_worker_command(
        _study_job(kind="variant", gpus=1, variant="kl"), python="python"
    )
    assert variant[variant.index("--variant") + 1] == "kl"
    assert "accelerate" not in variant

    full = build_worker_command(
        _study_job(
            kind="variant",
            gpus=2,
            variant="full",
            accelerate_config="configs/accelerate/fsdp_2gpu_full.yaml",
        ),
        python="python",
    )
    assert "configs/accelerate/fsdp_2gpu_full.yaml" in full
    assert full[full.index("--num_processes") + 1] == "2"
    assert full[full.index("--variant") + 1] == "full"


def test_submit_slurm_groups_claimable_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(
        queue,
        [
            _pending("a"),
            _pending("b"),
            _pending("train", depends_on=["audit"]),
            _pending("audit", state="failed"),
        ],
    )
    monkeypatch.delenv("VFATES_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("VFATES_SLURM_ACCOUNT", raising=False)
    with pytest.raises(ValueError, match="partition"):
        submit_slurm(queue)
    commands = submit_slurm(queue, partition="gpu", account="lab", dry_run=True)
    assert len(commands) == 1
    text = " ".join(commands[0])
    assert "--array=0-1" in text
    assert "--gpus=1" in text
    assert "--requeue" in text
    assert "--signal=B:USR1@180" in text
    assert "--time=7-00:00:00" in text
    name = next(part for part in commands[0] if part.startswith("--job-name="))
    assert job_name_fields(name.removeprefix("--job-name=")) == ("h100-80gb", "1")
    assert job_name_fields("vfates-rtx-a6000-48gb-4g") == ("rtx-a6000-48gb", "4")
    individual = submit_slurm(queue, partition="gpu", account="lab", dry_run=True, individual=True)
    assert len(individual) == 2
    assert all("--array=" not in " ".join(cmd) for cmd in individual)


def test_second_enqueue_keeps_finished_jobs(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    enqueue(queue, [_pending("keep"), _pending("new-later")])
    claim_next(queue, hardware_class="h100-80gb", gpus=1, worker_id="w")
    mark_done(queue, "keep")
    added = enqueue(queue, [_pending("keep", state="pending"), _pending("fresh")])
    assert added == 1
    states = {job["id"]: job["state"] for job in read_jobs(queue)}
    assert states["keep"] == "done"
    assert states["fresh"] == "pending"


def test_queue_docstring_records_the_nfs_caveat() -> None:
    import vfates.schedule as schedule

    assert schedule.__doc__ is not None
    assert "NFS" in schedule.__doc__
