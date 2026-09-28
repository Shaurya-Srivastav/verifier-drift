"""Shared plumbing for pilots: gate records, Markdown reports, and the CLI entry point.

Every pilot exposes ``run(config, out_dir) -> dict`` and writes
``report.json`` (machine-readable, with the environment report) and
``report.md`` (the checks table first, then pilot-specific sections).
A check's ``status`` is ``pass``, ``fail`` or ``documented``; documented
checks record a stated limitation and never fail a pilot.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vfates.utils.env import environment_report
from vfates.utils.io import read_yaml, write_json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Checks:
    """Ordered gate results for one pilot."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        id: str,
        name: str,
        value: float | None,
        passed: bool,
        *,
        threshold: Any = None,
        metric: str = "",
        documented: bool = False,
    ) -> bool:
        status = "documented" if documented else ("pass" if passed else "fail")
        self.rows.append(
            {
                "id": id,
                "name": name,
                "metric": metric,
                "threshold": threshold,
                "value": None if value is None else float(value),
                "passed": bool(passed),
                "status": status,
            }
        )
        return bool(passed)

    @property
    def ok(self) -> bool:
        return all(row["status"] != "fail" for row in self.rows)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.2e}"
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def table_cell(value: Any) -> str:
    """``fmt`` with pipes escaped, so ``|x - y|`` stays inside its column."""
    return fmt(value).replace("|", "\\|")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(table_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(table_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(
    out_dir: str | Path,
    title: str,
    checks: Checks,
    *,
    config: Mapping[str, Any],
    started: float,
    tables: Mapping[str, Any] | None = None,
    sections: Sequence[tuple[str, str]] = (),
    intro: str = "",
) -> dict[str, Any]:
    """Write ``report.json`` and ``report.md``; return the JSON payload."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - started
    counts = {s: sum(r["status"] == s for r in checks.rows) for s in ("pass", "fail", "documented")}
    summary = "PASS" if checks.ok else "FAIL"
    report = {
        "summary": summary,
        "ok": checks.ok,
        "elapsed_sec": elapsed,
        "counts": counts,
        "checks": checks.rows,
        "tables": dict(tables or {}),
        "config": dict(config),
        "environment": environment_report(),
    }
    write_json(out / "report.json", report)
    md = [
        f"# {title}: {summary}",
        "",
        f"{counts['pass']} passed, {counts['fail']} failed, {counts['documented']} documented. "
        f"Runtime {elapsed:.0f}s.",
        "",
    ]
    if intro:
        md += [intro, ""]
    md += [
        "## Checks",
        "",
        markdown_table(
            ["id", "name", "metric", "value", "threshold", "status"],
            [
                [r["id"], r["name"], r["metric"], r["value"], r["threshold"], r["status"]]
                for r in checks.rows
            ],
        ),
        "",
    ]
    for heading, body in sections:
        md += [f"## {heading}", "", body, ""]
    (out / "report.md").write_text("\n".join(md))
    print(f"{title}: {summary} ({counts}) in {elapsed:.0f}s -> {out / 'report.md'}", flush=True)
    return report


def cli(
    run: Callable[[dict[str, Any], Path], dict[str, Any]],
    default_dir: str | Path,
    argv: Sequence[str] | None = None,
) -> int:
    """``--config`` (default ``<dir>/config.yaml``) and ``--out`` (default ``<dir>/results``)."""
    here = Path(default_dir)
    parser = argparse.ArgumentParser(description=(run.__doc__ or "").strip().split("\n")[0])
    parser.add_argument("--config", type=Path, default=here / "config.yaml")
    parser.add_argument("--out", type=Path, default=here / "results")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config entry, e.g. --set sampling.n=4 (YAML-parsed value)",
    )
    args = parser.parse_args(argv)
    config = read_yaml(args.config)
    for item in args.set:
        apply_override(config, item)
    report = run(config, args.out)
    return 0 if report.get("ok") else 1


def apply_override(config: dict[str, Any], item: str) -> None:
    """Apply ``a.b.c=value`` (value parsed as YAML) to a nested mapping in place."""
    import yaml

    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"override {item!r} must look like key=value")
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(raw)


def main_guard(run: Callable[[dict[str, Any], Path], dict[str, Any]], file: str) -> None:
    sys.exit(cli(run, Path(file).resolve().parent))
