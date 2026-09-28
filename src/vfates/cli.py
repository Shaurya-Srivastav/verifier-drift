"""Command line: ``vfates {env,audit,train,score,study,pilot}``.

``vfates audit --config configs/experiments/e1_chain_sum.yaml --out runs/e1/audit``
``vfates train --audit runs/e1/audit --arm fp_tolerance --config configs/train/trl_h100.yaml --backend trl --out runs/e1/fp_tolerance``
``vfates score --lock runs/e1/audit/lock.json --run oracle=runs/e1/oracle --run fp_tolerance=runs/e1/fp_tolerance``
``vfates study --config configs/experiments/e1_chain_sum.yaml --out runs/e1``
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
    """Run a confirmatory study. Exit 0 exactly when ``report['ok']`` is true."""
    from vfates.pilots.common import apply_override
    from vfates.study import run_study
    from vfates.utils.io import read_yaml

    config = read_yaml(args.config)
    for item in args.set:
        apply_override(config, item)
    report = run_study(config, args.out)
    print(f"{report['summary']}: {Path(args.out) / 'report.md'}")
    return 0 if report["ok"] else 1


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
    study.set_defaults(func=_cmd_study)

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
