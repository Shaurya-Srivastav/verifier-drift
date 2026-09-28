"""Exploratory cross-domain transfer (X1).

Final adapters are scored on the other domains' held-out pools.
``vfates.program.transfer_eval`` writes one file per target::

    runs/<cell>/transfer/<arm>__s<seed>/<target_domain>.json

    schema vfates.transfer_eval/1
    {source_cell, arm, seed, target_domain, target_cell, adapter, step,
     n_prompts, n_samples, accuracy,
     per_prompt: [{pid, accuracy, n}]}

Each row is the seed-averaged per-prompt accuracy of an arm minus the same
cell's oracle arm, on that target domain, with a prompt bootstrap interval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vfates.utils.io import read_json

SCHEMA = "vfates.transfer/1"
_ORACLE = "oracle"


def transfer_report(
    runs_root: str | Path,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Accuracy of each transferred arm relative to that cell's oracle arm."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    root = Path(runs_root)
    grouped = _load_groups(root)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        cell, arm, target = key
        if arm == _ORACLE:
            continue
        rows.append(
            _row(
                cell,
                arm,
                target,
                grouped[key],
                grouped.get((cell, _ORACLE, target), []),
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
            )
        )
    return {
        "schema": SCHEMA,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "level": 1.0 - alpha,
        "oracle_arm": _ORACLE,
        "n_rows": len(rows),
        "rows": rows,
        "input_schema": "vfates.transfer_eval/1",
        "rule": (
            "X1, exploratory. Files are vfates.transfer_eval/1 at "
            "runs/<cell>/transfer/<arm>__s<seed>/<target_domain>.json. "
            "For each source cell, arm and target domain, accuracy is the "
            "seed average of per_prompt accuracy. The contrast is arm minus "
            "the same cell's oracle arm on that target. The interval is a "
            "prompt bootstrap of that paired difference."
        ),
    }


def render_transfer_markdown(report: Mapping[str, Any]) -> str:
    """A table of arm minus oracle on each target domain."""
    lines = [
        "# Cross-domain transfer (X1)",
        "",
        f"Decision: exploratory. {report.get('rule', '')}",
        "",
        "| cell | arm | target | target cell | acc | oracle | arm - oracle | interval | seeds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {cell} | {arm} | {target} | {target_cell} | {acc} | {oracle} | {diff} | {interval} | {seeds} |".format(
                cell=row.get("source_cell") or row.get("cell"),
                arm=row.get("arm"),
                target=row.get("target_domain"),
                target_cell=row.get("target_cell") or "",
                acc=_fmt(row.get("acc")),
                oracle=_fmt(row.get("oracle_acc")),
                diff=_fmt(row.get("diff")),
                interval=_interval(row),
                seeds=row.get("n_seeds"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _load_groups(root: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    if not root.is_dir():
        raise ValueError(f"runs root is not a directory: {root}")
    for cell_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        transfer = cell_dir / "transfer"
        if not transfer.is_dir():
            continue
        for run_dir in sorted(path for path in transfer.iterdir() if path.is_dir()):
            arm, seed = _arm_seed(run_dir.name)
            if arm is None or seed is None:
                continue
            for path in sorted(run_dir.glob("*.json")):
                doc = read_json(path)
                if not isinstance(doc, Mapping):
                    raise ValueError(f"{path} is not an object")
                target = str(doc.get("target_domain") or path.stem)
                arm_name = str(doc.get("arm") or arm)
                source_cell = str(doc.get("source_cell") or cell_dir.name)
                record = {
                    "cell": cell_dir.name,
                    "source_cell": source_cell,
                    "arm": arm_name,
                    "seed": int(doc.get("seed", seed)),
                    "target_domain": target,
                    "target_cell": doc.get("target_cell"),
                    "adapter": doc.get("adapter"),
                    "step": doc.get("step"),
                    "n_prompts_file": doc.get("n_prompts"),
                    "n_samples": doc.get("n_samples"),
                    "accuracy": doc.get("accuracy"),
                    "prompts": _per_prompt(doc.get("per_prompt"), path),
                }
                grouped.setdefault((cell_dir.name, arm_name, target), []).append(record)
    return grouped


def _arm_seed(name: str) -> tuple[str | None, int | None]:
    arm, sep, seed_s = name.rpartition("__s")
    if not sep or not seed_s.lstrip("-").isdigit():
        return None, None
    return arm, int(seed_s)


def _per_prompt(raw: Any, path: Path) -> dict[str, float]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{path} is missing per_prompt")
    out: dict[str, float] = {}
    for row in raw:
        if not isinstance(row, Mapping) or "pid" not in row or "accuracy" not in row:
            raise ValueError(f"{path} per_prompt rows need pid and accuracy")
        value = float(row["accuracy"])
        if not math.isfinite(value):
            raise ValueError(f"{path} has a non-finite accuracy")
        out[str(row["pid"])] = value
    if not out:
        raise ValueError(f"{path} has no per_prompt rows")
    return out


def _row(
    cell: str,
    arm: str,
    target: str,
    arm_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    first = arm_rows[0] if arm_rows else {}
    base: dict[str, Any] = {
        "cell": cell,
        "source_cell": first.get("source_cell", cell),
        "arm": arm,
        "target_domain": target,
        "target_cell": first.get("target_cell"),
        "adapter": first.get("adapter"),
        "step": first.get("step"),
        "n_samples": first.get("n_samples"),
        "status": "tested",
    }
    if not oracle_rows:
        base["status"] = "missing_oracle"
        base["n_seeds"] = 0
        base["reason"] = "no oracle file for this cell and target"
        return base
    by_seed_arm = {int(row["seed"]): row["prompts"] for row in arm_rows}
    by_seed_oracle = {int(row["seed"]): row["prompts"] for row in oracle_rows}
    common_seeds = sorted(set(by_seed_arm) & set(by_seed_oracle))
    if not common_seeds:
        base["status"] = "missing_oracle"
        base["n_seeds"] = 0
        base["reason"] = "arm and oracle share no seed"
        return base
    pids = set(by_seed_arm[common_seeds[0]]) & set(by_seed_oracle[common_seeds[0]])
    for seed_i in common_seeds[1:]:
        pids &= set(by_seed_arm[seed_i]) & set(by_seed_oracle[seed_i])
    ordered = sorted(pids)
    if not ordered:
        raise ValueError(f"{cell} {arm} on {target} shares no prompts with the oracle")
    arm_mean = np.array(
        [float(np.mean([by_seed_arm[s][pid] for s in common_seeds])) for pid in ordered],
        dtype=float,
    )
    oracle_mean = np.array(
        [float(np.mean([by_seed_oracle[s][pid] for s in common_seeds])) for pid in ordered],
        dtype=float,
    )
    diff = arm_mean - oracle_mean
    point, lo, hi = _prompt_bootstrap(diff, n_boot=n_boot, seed=seed, alpha=alpha)
    base.update(
        {
            "n_seeds": len(common_seeds),
            "seeds": common_seeds,
            "n_prompts": len(ordered),
            "acc": float(arm_mean.mean()),
            "oracle_acc": float(oracle_mean.mean()),
            "diff": point,
            "lo": lo,
            "hi": hi,
            "level": 1.0 - alpha,
        }
    )
    return base


def _prompt_bootstrap(
    values: np.ndarray, *, n_boot: int, seed: int, alpha: float
) -> tuple[float, float, float]:
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def _interval(row: Mapping[str, Any]) -> str:
    if row.get("lo") is None:
        return ""
    return f"[{_fmt(row.get('lo'))}, {_fmt(row.get('hi'))}]"
