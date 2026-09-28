#!/usr/bin/env python3
"""Submit claimable queue jobs to Slurm.

Equivalent to ``vfates program queue --backend slurm``. Partition and account
come from VFATES_SLURM_PARTITION and VFATES_SLURM_ACCOUNT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--individual",
        action="store_true",
        help="one sbatch per job instead of one array per hardware class and GPU count",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from vfates.schedule import submit_slurm

    commands = submit_slurm(
        args.queue,
        repo=args.root,
        individual=args.individual,
        dry_run=args.dry_run,
    )
    for cmd in commands:
        print(" ".join(cmd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
