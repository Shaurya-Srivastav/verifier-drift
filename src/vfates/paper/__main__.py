"""``python -m vfates.paper assets`` writes the manuscript's generated files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vfates.paper.assets import build_assets
from vfates.program import repo_root


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vfates.paper")
    sub = parser.add_subparsers(dest="cmd", required=True)
    assets = sub.add_parser("assets", help="write macros, tables, and figures")
    assets.add_argument(
        "--runs", default="runs", help="directory that may hold program_report.json"
    )
    assets.add_argument("--out", default="paper/generated", help="directory for tex assets")
    assets.add_argument("--figures", default="paper/figures", help="directory for pdf figures")
    args = parser.parse_args(argv)
    if args.cmd != "assets":
        parser.error(f"unknown command {args.cmd}")
        return 2
    root = repo_root()
    runs = _resolve(root, args.runs)
    mismatches = build_assets(
        repo=root,
        runs=runs,
        out=_resolve(root, args.out),
        figures=_resolve(root, args.figures),
    )
    for item in mismatches:
        print(f"mismatch: {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
