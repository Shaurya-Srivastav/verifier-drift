"""Shared plumbing for pilots: gate records, Markdown reports, and the CLI entry point.

Every pilot exposes ``run(config, out_dir) -> dict`` and writes
``report.json`` (machine-readable, with the environment report) and
``report.md`` (the checks table first, then pilot-specific sections).
A check's ``status`` is ``pass``, ``fail`` or ``documented``; documented
checks record a stated limitation and never fail a pilot.
"""

from __future__ import annotations

import argparse
import copy
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


def repo_root() -> Path:
    """Repository root (the directory that contains ``configs`` and ``src``)."""
    return Path(__file__).resolve().parents[3]


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    if not key or not isinstance(key, str):
        raise ValueError(f"override key {key!r} must be a non-empty dotted string")
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any] | Sequence[str]) -> None:
    """Dotted mapping, or ``key=value`` strings parsed as YAML (the ``--set`` form)."""
    if isinstance(overrides, str) or not isinstance(overrides, (Mapping, Sequence)):
        raise TypeError("overrides must be a mapping of dotted keys or a sequence of key=value")
    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            _set_dotted(config, str(key), value)
        return
    for item in overrides:
        if not isinstance(item, str):
            raise TypeError(f"override {item!r} must be a key=value string")
        apply_override(config, item)


def cell_config(
    cell_id: str,
    *,
    dev: bool = True,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Load ``configs/cells/<cell_id>.yaml`` and return a detached copy.

    When ``dev`` is true, ``audit.pools.from`` is set to ``dev`` and the
    top-level ``pilot`` flag is set to true. The study refuses a
    development-pool audit unless that flag is set. ``dev`` false leaves
    ``pilot`` unset. ``overrides`` is applied after those fields: a mapping
    of dotted keys to values, or a sequence of ``key=value`` strings
    (YAML-parsed, same as ``--set``). ``root`` defaults to the repository root.
    """
    if not isinstance(cell_id, str) or not cell_id or "/" in cell_id or "\\" in cell_id:
        raise ValueError(f"cell id {cell_id!r} must be a single file name")
    base = Path(root) if root is not None else repo_root()
    path = base / "configs" / "cells" / f"{cell_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"cell config not found: {path}")
    loaded = read_yaml(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping")
    config: dict[str, Any] = copy.deepcopy(loaded)
    if dev:
        audit = config.setdefault("audit", {})
        if not isinstance(audit, dict):
            raise ValueError("audit must be a mapping")
        pools = audit.setdefault("pools", {})
        if not isinstance(pools, dict):
            raise ValueError("audit.pools must be a mapping")
        pools["from"] = "dev"
        config["pilot"] = True
    if overrides:
        _apply_overrides(config, overrides)
    return config


def dev_pools_honored() -> bool:
    """True when ``load_pools`` reads ``pools['from']`` (the development-pool flag).

    Pilots set ``audit.pools.from: dev`` and refuse to sample until the audit
    draws those pseudo-pools from the task's ``dev`` split. Sampling before
    that lands would score confirmatory prompts.
    """
    import inspect

    from vfates.audit import load_pools

    source = inspect.getsource(load_pools)
    markers = ('["from"]', '.get("from"', ".get('from'", "_pools_from")
    return any(token in source for token in markers)


def write_decisions(out_dir: str | Path, document: Mapping[str, Any]) -> Path:
    """Write ``decisions.json``. The document must carry a ``schema`` id."""
    if "schema" not in document or not isinstance(document["schema"], str):
        raise ValueError("decisions need a schema id")
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "decisions.json"
    write_json(path, dict(document))
    return path


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
