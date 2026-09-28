"""Command line: ``vfates {env,audit,train,score,study,pilot,program}``.

``vfates audit --config configs/cells/e1-chain_sum-qwen3-1.7b.yaml --out runs/e1-chain_sum-qwen3-1.7b/audit``
``vfates train --audit runs/e1/audit --arm fp_tolerance --config configs/train/trl_h100.yaml --backend trl --out runs/e1/fp_tolerance``
``vfates score --lock runs/e1/audit/lock.json --run oracle=runs/e1/oracle --run fp_tolerance=runs/e1/fp_tolerance``
``vfates study --config configs/cells/e1-chain_sum-qwen3-1.7b.yaml --out runs/e1-chain_sum-qwen3-1.7b``
``vfates study --config configs/cells/e1-chain_sum-qwen3-1.7b.yaml --out runs/e1-chain_sum-qwen3-1.7b --only train --arm oracle --seed 0``
``vfates program plan``
``vfates pilot p2 --set sampling.n=4``
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PILOTS: dict[str, str] = {
    "p0": "p0_theory",
    "p1": "p1_harness",
    "p2": "p2_base_audit",
    "p3": "p3_estimator",
    "p4": "p4_micro_grpo",
    "p5": "p5_gpu_gate",
    "p6": "p6_natural",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cmd_env(args: argparse.Namespace) -> int:
    from vfates.utils.env import environment_report

    print(json.dumps(environment_report(), indent=2, sort_keys=True, default=str))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from vfates.audit import run_audit
    from vfates.pilots.common import apply_override
    from vfates.utils.io import read_yaml

    config = read_yaml(args.config)
    for item in args.set:
        apply_override(config, item)
    run = run_audit(config, args.out)
    tr = run.table.split == "train"
    print(
        f"audit: {len(run.table)} rows, train acc {run.table.O[tr].mean():.3f}, "
        f"held-out acc {run.table.O[~tr].mean():.3f} -> {args.out}"
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from vfates.experiment import train_arm
    from vfates.pilots.common import apply_override
    from vfates.utils.io import read_yaml

    config = read_yaml(args.config)
    for item in args.set:
        apply_override(config, item)
    summary = train_arm(
        args.audit,
        args.arm,
        config,
        args.out,
        backend=args.backend,
        resume=args.resume,
        max_train=args.max_train,
        max_heldout=args.max_heldout,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from vfates.analysis.outcomes import score_lock

    runs = {}
    for item in args.run:
        arm, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--run expects arm=dir, got {item!r}")
        runs[arm] = path
    print(json.dumps(score_lock(args.lock, runs), indent=2, default=str))
    return 0


def _cmd_study(args: argparse.Namespace) -> int:
    """Run a confirmatory study. Exit 0 exactly when ``report['ok']`` is true.

    A preempted single job (``summary == 'PREEMPTED'``) also exits 0 so the
    queue can requeue it. The cell report is not that summary: the run's
    ``summary.json`` carries ``status: preempted``.
    """
    from vfates.pilots.common import apply_override
    from vfates.study import run_study
    from vfates.utils.io import read_yaml

    config = read_yaml(args.config)
    for item in args.set:
        apply_override(config, item)
    kwargs: dict[str, Any] = {}
    if args.only is not None:
        kwargs["only"] = args.only
    if args.arm is not None:
        kwargs["arm"] = args.arm
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.early:
        kwargs["early"] = True
    if args.variant is not None:
        kwargs["variant"] = args.variant
    report = run_study(config, args.out, **kwargs)
    print(f"{report['summary']}: {Path(args.out) / 'report.md'}")
    if report.get("summary") == "PREEMPTED":
        return 0
    return 0 if report["ok"] else 1


def _cmd_program_render(args: argparse.Namespace) -> int:
    from vfates.program import load_program, render_program

    paths = render_program(args.program, root=args.root, only=args.cell or None)
    program = load_program(args.program, root=args.root)
    written = {path.stem for path in paths}
    for cell in program["cells"]:
        if args.cell and cell["id"] not in args.cell:
            continue
        if cell["id"] not in written:
            print(f"skip {cell['id']}: template configs/templates/{cell['domain']}.yaml is missing")
    for path in paths:
        print(path)
    return 0


def _cmd_program_plan(args: argparse.Namespace) -> int:
    from vfates.program import format_budget, load_program, plan_jobs
    from vfates.schedule import enqueue

    program = load_program(args.program, root=args.root)
    jobs = plan_jobs(program, root=args.root, runs_root=str(args.runs))
    print(
        format_budget(
            jobs,
            program=program,
            by_cell=bool(args.by_cell),
            by_stage=bool(args.by_stage),
            budget_usd=args.budget_usd,
            gpus=args.gpus,
        )
    )
    print(f"jobs: {len(jobs)}")
    if args.queue is not None:
        added = enqueue(args.queue, jobs)
        print(f"queue: {args.queue} ({added} new)")
    return 0


def _cmd_program_queue(args: argparse.Namespace) -> int:
    from vfates.schedule import submit_slurm

    if args.backend != "slurm":
        raise SystemExit(f"unknown queue backend {args.backend}")
    commands = submit_slurm(
        args.queue,
        repo=args.root,
        individual=args.individual,
        dry_run=args.dry_run,
    )
    for cmd in commands:
        print(" ".join(cmd))
    return 0


def _cmd_program_worker(args: argparse.Namespace) -> int:
    from vfates.schedule import run_worker

    return run_worker(
        args.queue,
        hardware_class=args.hardware_class,
        gpus=args.gpus,
        once=args.once,
        repo=args.root,
    )


def _cmd_program_status(args: argparse.Namespace) -> int:
    from vfates.schedule import format_status

    print(format_status(args.queue))
    return 0


def _cmd_program_analyze(args: argparse.Namespace) -> int:
    from vfates.program import analyze_program, load_program

    program = load_program(args.program, root=args.root)
    analyze_program(program, args.runs, budget_usd=args.budget_usd, root=args.root)
    print(Path(args.runs) / "program_report.md")
    return 0


def _cmd_program_transfer(args: argparse.Namespace) -> int:
    from vfates.program import load_program, transfer_eval

    program = load_program(args.program, root=args.root)
    paths = transfer_eval(
        program,
        args.runs,
        cell=args.cell,
        arm=args.arm,
        seed=args.seed,
        targets=args.target or None,
        root=args.root,
    )
    for path in paths:
        print(path)
    return 0


def _cmd_pilot(args: argparse.Namespace) -> int:
    from vfates.pilots.common import cli

    name = PILOTS[args.name]
    module = importlib.import_module(f"vfates.pilots.{name}")
    folder = _repo_root() / "pilots" / name
    forwarded = []
    if args.config:
        forwarded += ["--config", str(args.config)]
    if args.out:
        forwarded += ["--out", str(args.out)]
    for item in args.set:
        forwarded += ["--set", item]
    return cli(module.run, folder, forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vfates", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="print the environment report").set_defaults(func=_cmd_env)

    audit = sub.add_parser("audit", help="run or resume a step-0 audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    audit.set_defaults(func=_cmd_audit)

    train = sub.add_parser("train", help="train one arm on an audit's pools")
    train.add_argument("--audit", type=Path, required=True)
    train.add_argument("--arm", required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--backend", choices=("mlx", "trl"), default="mlx")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-train", type=int, default=None)
    train.add_argument("--max-heldout", type=int, default=None)
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.set_defaults(func=_cmd_train)

    score = sub.add_parser("score", help="score a lock file against trained runs")
    score.add_argument("--lock", type=Path, required=True)
    score.add_argument("--run", action="append", default=[], metavar="ARM=DIR", required=True)
    score.set_defaults(func=_cmd_score)

    study = sub.add_parser("study", help="audit, lock, train every arm and seed, and score")
    study.add_argument("--config", type=Path, required=True)
    study.add_argument("--out", type=Path, required=True)
    study.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    study.add_argument("--only", choices=("audit", "train"), default=None)
    study.add_argument("--arm", default=None)
    study.add_argument("--seed", type=int, default=None)
    study.add_argument("--early", action="store_true")
    study.add_argument("--variant", default=None, help="train one named study.variants entry")
    study.set_defaults(func=_cmd_study)

    program = sub.add_parser("program", help="render, plan, queue, and analyse the study program")
    program_sub = program.add_subparsers(dest="program_command", required=True)

    render = program_sub.add_parser("render", help="write configs/cells from templates")
    render.add_argument("--program", type=Path, default=Path("configs/program.yaml"))
    render.add_argument("--root", type=Path, default=None)
    render.add_argument("--cell", action="append", default=[], help="render only this cell id")
    render.set_defaults(func=_cmd_program_render)

    plan = program_sub.add_parser("plan", help="print the GPU-hour budget and optional queue")
    plan.add_argument("--program", type=Path, default=Path("configs/program.yaml"))
    plan.add_argument("--root", type=Path, default=None)
    plan.add_argument("--runs", type=Path, default=Path("runs"))
    plan.add_argument(
        "--queue", type=Path, default=None, help="append new jobs to this queue.jsonl"
    )
    plan.add_argument(
        "--by-cell", action="store_true", help="print hours, class, stage, and USD per cell"
    )
    plan.add_argument(
        "--by-stage",
        action="store_true",
        help="print jobs, hours, and USD per stage and hardware class",
    )
    plan.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="drop cells in compute.cut_order until rented USD fits",
    )
    plan.add_argument(
        "--gpus",
        default=None,
        help="class=count,... wall-clock days, for example rtx-a6000-48gb=4",
    )
    plan.set_defaults(func=_cmd_program_plan)

    queue = program_sub.add_parser("queue", help="submit claimable jobs")
    queue.add_argument("--backend", choices=("slurm",), required=True)
    queue.add_argument("--queue", type=Path, required=True)
    queue.add_argument("--root", type=Path, default=None)
    queue.add_argument("--individual", action="store_true")
    queue.add_argument("--dry-run", action="store_true")
    queue.set_defaults(func=_cmd_program_queue)

    worker = program_sub.add_parser(
        "worker", help="claim jobs for one hardware class and GPU count"
    )
    worker.add_argument("--queue", type=Path, required=True)
    worker.add_argument("--hardware-class", required=True)
    worker.add_argument("--gpus", type=int, required=True)
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--root", type=Path, default=None)
    worker.set_defaults(func=_cmd_program_worker)

    status = program_sub.add_parser("status", help="count queue jobs by state")
    status.add_argument("--queue", type=Path, required=True)
    status.set_defaults(func=_cmd_program_status)

    analyze = program_sub.add_parser("analyze", help="write runs/program_report.json and .md")
    analyze.add_argument("--program", type=Path, default=Path("configs/program.yaml"))
    analyze.add_argument("--root", type=Path, default=None)
    analyze.add_argument("--runs", type=Path, default=Path("runs"))
    analyze.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="drop cells in compute.cut_order until rented USD fits, then require the rest",
    )
    analyze.set_defaults(func=_cmd_program_analyze)

    transfer = program_sub.add_parser(
        "transfer-eval", help="score one finished adapter on other domains"
    )
    transfer.add_argument("--program", type=Path, default=Path("configs/program.yaml"))
    transfer.add_argument("--root", type=Path, default=None)
    transfer.add_argument("--runs", type=Path, default=Path("runs"))
    transfer.add_argument("--cell", required=True)
    transfer.add_argument("--arm", required=True)
    transfer.add_argument("--seed", type=int, required=True)
    transfer.add_argument(
        "--target", action="append", default=[], help="target domain (repeatable)"
    )
    transfer.set_defaults(func=_cmd_program_transfer)

    pilot = sub.add_parser("pilot", help="run a pilot (p0-p6)")
    pilot.add_argument("name", choices=sorted(PILOTS))
    pilot.add_argument("--config", type=Path, default=None)
    pilot.add_argument("--out", type=Path, default=None)
    pilot.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    pilot.set_defaults(func=_cmd_pilot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
