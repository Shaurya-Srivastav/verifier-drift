"""Pooled confirmatory report across the cells of a program.

Each finished cell is ``runs/<cell_id>/`` with ``audit/``, ``runs/``,
``early/``, and ``withdrawals.json`` directly under that root. Per-cell
contrasts use the crossed seed-by-prompt bootstrap. Pooled means resample
inside each cell, then average cells with equal weight.

``pooled_report`` is the flat report (every cell in the mapping). A program
with ``stages`` is written by ``staged_report``: stage A is discovery
(H1, H1-pooled, H2, H3, H7, H8, H10, and the stage secondaries) and stage B
is replication (H3 on stage B cells only). The union of the two stages is
a secondary block, computed only when both stages are complete.

Figure lists live at ``figures`` and are plain lists of dicts:

* ``h3_pairs``: one row per tested H3 pair. Fields: ``cell``, ``stage``,
  ``experiment``, ``domain``, ``model``, ``params_b``, ``arm``, ``control``,
  ``class`` (the arm's structure class, -1 shared behaviour, 0 otherwise),
  ``is_natural``, ``dtau``, ``dacc``, and ``baseline_<name>`` for each
  baseline difference (arm minus control), including ``baseline_class``.
* ``h9_budgets``: one row per oracle-label budget. Fields: ``stage``,
  ``budget``, ``status``, ``median_rho``, ``q05``, ``frac_positive``,
  ``n_draws``, ``n_pairs``.
* ``natural_rows``: one row per natural pair. Fields: ``cell``, ``stage``,
  ``experiment``, ``domain``, ``natural``, ``control``, ``point``, ``lo``,
  ``hi``, ``differs``, ``locality``, ``screen``, ``step0_fpr``, ``step0_fnr``,
  ``step0_error_rate`` (their sum), the locked ``dtau`` (natural minus control)
  and ``dacc`` (natural minus control, the negated contrast).
* ``h8_rows``: one row per included patch cell. Fields: ``cell``, ``stage``,
  ``domain``, ``point``, ``per_seed``.
* ``h10_rows``: one row per H10 pair. Fields: ``stage``, ``name``, ``a``,
  ``b``, ``status``, ``point``, ``p_le_0``, ``holm_adjusted_p``, ``n_cells``,
  ``rejects``.
* ``cost_rows``: one row per cell and one rollup per stage. Fields:
  ``scope`` (``cell`` or ``stage``), ``cell``, ``stage``,
  ``audit_gpu_hours``, ``early_gpu_hours_per_arm``, ``n_trained_arms``,
  ``break_even_arms``. GPU-hours are ``wall_seconds * gpus / 3600`` from
  recorded ``timing``. A missing timing field is null.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vfates.analysis.confirmatory import _forecast_arm_items, _jsonify, natural_pair_blocks
from vfates.analysis.outcomes import (
    crossed_pair_dacc,
    difference_matrix,
    forecast_row,
    holm,
    holm_adjusted,
    ols_slope,
    prompt_accuracy,
)
from vfates.analysis.transfer import transfer_report
from vfates.forecast.lockbox import load_lock
from vfates.utils.io import canonical_json, read_json, read_yaml
from vfates.utils.stats import group_indices, spearman

SCHEMA = "vfates.pooled/1"
H3_MIN_PAIRS = 20
H9_BUDGETS = (32, 64, 128, 192)
_EXPERIMENTS = frozenset({"E1", "E2", "E3"})
_PATCH_DOMAINS = frozenset({"math", "code", "ifeval", "qa"})
_KIT_H1 = (
    ("noise_fp_trigger", "fp_trigger"),
    ("noise_fn", "fn_behavior"),
)
_H1_RULE = (
    "Holm-Bonferroni across every tested E1 (cell, contrast) at alpha. "
    "p_sup is the crossed-bootstrap proportion of resamples with mean <= margin. "
    "A contrast rejects when its Holm-adjusted p_sup is <= alpha and every "
    "paired seed's contrast is > 0. H1 holds if at least one tested contrast "
    "rejects. Inconclusive cells stay in this family."
)
_H1_POOLED_RULE = (
    "Equal-weight mean, over conclusive E1 domains, of each kit contrast "
    "(noise_fp_trigger - fp_trigger and noise_fn - fn_behavior). A domain with "
    "several cells contributes their equal-weight mean. The crossed bootstrap "
    "resamples seeds and prompts inside each cell, then averages cells and "
    "domains with equal weight. The one-sided p-value is the proportion of "
    "resamples with mean <= 0. Holm-Bonferroni across the two kit contrasts. "
    "Inconclusive cells are excluded here and included in the sensitivity."
)
_H2_RULE = (
    "TOST at +/- margin. p_tost is the larger of the two one-sided bootstrap "
    "tails. Holm-Bonferroni across every tested E1 H2 contrast at alpha. "
    "H2 holds if at least one contrast remains and every tested contrast rejects."
)
_H3_RULE = (
    "Spearman rho of dtau with dacc over non-withdrawn matched pairs of "
    "conclusive cells in E1, E2 and E3. dtau is locked tau_pool(arm) minus "
    "tau_pool(control). dacc is seed-averaged final accuracy(arm) minus "
    "accuracy(control). The p-value permutes dacc within cells "
    "(Monte Carlo, (1 + count) / (n_perm + 1)). The 5th percentile of rho "
    "is a crossed bootstrap inside each cell with forecasts fixed. H3 holds "
    "when at least min_pairs pairs remain, p < alpha, and the 5th percentile "
    "is > 0. Inconclusive cells are excluded and reported in the sensitivity."
)
_H7_RULE = (
    "Per domain, the per-cell natural block: acc(control) - acc(natural), "
    "the two-sided interval, TOST locality, and the screen direction against "
    "locked tau_pool. A list config is one block per pair (math has natural "
    "and natural_strict). The primary rows are the E1 cell of that domain."
)
_H8_RULE = (
    "Equal-weight mean of final accuracy(patch_tau) - accuracy(patch_rate) "
    "over E1 domains whose patch selections differ and whose two arms are not "
    "withdrawn. The crossed bootstrap is stratified by cell. The one-sided "
    "p-value is the proportion of resamples with mean <= 0. H8 holds when "
    "that p-value is <= alpha."
)
_H9_RULE = (
    "H3's Spearman recomputed from stored tau_pool_budget draws at m in "
    "{32, 64, 128, 192}. Draw d uses draw d of the arm minus draw d of the "
    "control on the same tested pairs as H3. The report is the median rho, "
    "the 5th percentile, and the fraction of draws with rho > 0. A budget "
    "that covers the top-q set (sd 0, or budget >= oracle_labels_needed) is "
    "exact: a constant draw list or the locked mean. Any non-exact arm "
    "without a finite draw list makes that budget missing. Draws are never "
    "imputed, resampled, or recomputed from audit.npz."
)
_EXPLORATORY_RULE = (
    "Exploratory, reported and not tested. These rows stay outside every "
    "H1 to H9 family, Holm set, and pooled statistic. Base runs are "
    "runs/<cell>/runs/<arm>__s<seed>. Early baselines are "
    "runs/<cell>/early/<arm>__s<seed>. Withdrawals are "
    "runs/<cell>/withdrawals.json. Variant runs are "
    "runs/<cell>/variants/<name>/<arm>__s<seed>. Experiment X cells are "
    "compared with the E1 cell of the same domain. Reported arms are read "
    "from analysis.reported, or detected as random and natural_robust. "
    "X1 transfer is included here."
)
_SCALE_RULE = (
    "Per kit contrast and domain, the OLS slope of the cell contrast on "
    "log10(params_b) across qwen3 ladder cells. The interval is a cell-level "
    "bootstrap. params_b comes from the program cell, else from "
    "vfates.models.load_model_spec. Secondary."
)
_H3_STAGE_A_RULE = (
    "The H3 test on the conclusive cells of stage A only (the E1 cells and "
    "the E2 cells at qwen3-4b). Same Spearman, within-cell permutation "
    "(100,000 draws, seed 0), crossed-bootstrap 5th percentile, and minimum "
    "pair count as the program H3."
)
_H3_REPLICATION_RULE = (
    "H3-replication: the H3 test on the conclusive cells of stage B only. "
    "Spearman of dtau with dacc, within-cell permutation p from 100,000 draws "
    "seed 0, crossed-bootstrap 5th percentile above 0, and at least 20 "
    "non-withdrawn pairs."
)
_H3_UNION_RULE = (
    "Secondary. The H3 test on the union of conclusive cells from stages A "
    "and B. Same three gates as H3. Computed only when both stages are complete."
)
_H10_RULE = (
    "Stage A. For each pair (a, b) in analysis.h10.pairs, over conclusive E1 "
    "cells where neither arm is withdrawn: D_cell = sign(dtau) * mean over "
    "seeds and prompts of accuracy(a) - accuracy(b), with dtau = tau_pool(a) - "
    "tau_pool(b) from the lock. dtau exactly 0 gives D_cell 0. The estimate is "
    "the equal-weight mean of D_cell. The crossed bootstrap resamples seeds "
    "and prompts inside each cell independently (10,000 draws). The one-sided "
    "p-value is the share of draws at or below 0. Holm-Bonferroni across the "
    "pairs. A pair with fewer than 3 cells is insufficient."
)
_H10_PAIRS = (
    ("fp_trigger_enriched", "fp_trigger"),
    ("fn_behavior_depleted", "fn_behavior"),
)
_H10_MIN_CELLS = 3
_COST_RULE = (
    "Measured GPU-hours of the shared audit and of the step-25 early baseline. "
    "GPU-hours = wall_seconds * gpus / 3600 from timing on audit.meta.json and "
    "on each early summary.json. The audit is one per cell. early_gpu_hours_per_arm "
    "is the mean over trained arms. break_even_arms = audit_gpu_hours / "
    "early_gpu_hours_per_arm. A missing timing field is null, never filled in."
)
_WITHIN_SHARED_RULE = (
    "H3's Spearman, within-cell permutation p, and crossed-bootstrap 5th "
    "percentile, restricted to pairs whose arm is a shared behaviour rule "
    "(class -1) or a natural verifier."
)
# Structure class is -1 for a shared-behaviour rule and 0 otherwise.
# Composite arms are -1 when any part is -1. Natural verifiers and patch
# arms are 0. Class is read from the arm type in the arms file.
SHARED_ARM_TYPES = frozenset(
    {"trigger_fp", "behavior_fn", "tolerance_fp", "tolerance_band_fp", "form_fn"}
)
NATURAL_ARM_TYPES = frozenset(
    {
        "string_match",
        "math_lenient",
        "math_strip_string",
        "code_public_tests",
        "ifeval_checker",
        "model_judge",
        "master_rm",
    }
)
ZERO_ARM_TYPES = frozenset({"oracle", "hash_fp", "prompt_subset_fn", "resampled"})
PATCH_ARM_TYPES = frozenset({"patch_select", "patched"})
KNOWN_ARM_TYPES = (
    SHARED_ARM_TYPES | NATURAL_ARM_TYPES | ZERO_ARM_TYPES | PATCH_ARM_TYPES | {"composite"}
)


def leaf_arm_class(type_name: str) -> tuple[int, bool]:
    """Structure class and natural flag for one non-composite verifier type.

    Shared behaviour rules are class -1. Every other type is class 0.
    ``is_natural`` is true only for a natural verifier type.
    """
    if type_name in SHARED_ARM_TYPES:
        return -1, False
    if type_name in NATURAL_ARM_TYPES:
        return 0, True
    if type_name in ZERO_ARM_TYPES or type_name in PATCH_ARM_TYPES:
        return 0, False
    raise ValueError(f"unclassified verifier type {type_name!r}")


def classify_arms(specs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, bool]]:
    """Map each arm name to ``(class, is_natural)``.

    A composite is class -1 when any ``fn`` or ``fp`` part is class -1.
    Parts are arm names in the same arms list, or nested specs with a ``type``.
    """
    by_name: dict[str, Mapping[str, Any]] = {}
    for spec in specs:
        if isinstance(spec, Mapping) and spec.get("name") and spec.get("type"):
            by_name[str(spec["name"])] = spec
    classes: dict[str, int] = {}
    natural: dict[str, bool] = {}

    def walk(name: str, stack: frozenset[str]) -> tuple[int, bool]:
        if name in classes:
            return classes[name], natural[name]
        spec = by_name.get(name)
        if spec is None:
            raise ValueError(f"composite part {name!r} is not an arm in this file")
        if name in stack:
            raise ValueError(f"composite cycle at {name}")
        typ = str(spec["type"])
        if typ == "composite":
            shared = False
            for slot in ("fn", "fp"):
                parts = spec.get(slot) or []
                if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
                    continue
                for part in parts:
                    if _part_class(part, walk, stack | {name}) == -1:
                        shared = True
            found = (-1 if shared else 0, False)
        else:
            found = leaf_arm_class(typ)
        classes[name] = found[0]
        natural[name] = found[1]
        return found

    for name in by_name:
        walk(name, frozenset())
    return classes, natural


def _part_class(
    part: Any,
    walk: Any,
    stack: frozenset[str],
) -> int:
    if isinstance(part, str):
        return int(walk(part, stack)[0])
    if isinstance(part, Mapping):
        if part.get("type"):
            typ = str(part["type"])
            if typ == "composite":
                name = str(part.get("name") or "")
                if name:
                    return int(walk(name, stack)[0])
            return leaf_arm_class(typ)[0]
        if part.get("name"):
            return int(walk(str(part["name"]), stack)[0])
    raise ValueError(f"composite part {part!r} is not an arm name or a verifier spec")


def _arm_structure(
    config: Mapping[str, Any], domain: str
) -> tuple[dict[str, int], dict[str, bool]]:
    """Classes from the cell's arms file. An absent arms block leaves both empty."""
    audit = config.get("audit")
    if not isinstance(audit, Mapping):
        return {}, {}
    arms = audit.get("arms")
    if not isinstance(arms, Mapping):
        return {}, {}
    from vfates.audit import resolve_arms

    resolved = resolve_arms(arms, domain=domain or None)
    specs = resolved.get("arms") or []
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
        return {}, {}
    return classify_arms([spec for spec in specs if isinstance(spec, Mapping)])


_CONCLUSIVE_RULE = (
    "A cell is conclusive when the oracle arm's mean seed gain (final held-out "
    "accuracy minus step 0) is >= analysis.oracle_gain_min. Inconclusive cells "
    "are excluded from H1-pooled and from pooled H3, and included in those "
    "sensitivities."
)
_BASELINES = (
    "J",
    "mean_precision",
    "modal_precision",
    "cos_gV_gO",
    "probe_heldout_auc",
    "early_acc",
    "tau_pool_std",
    "behaviour_alignment",
    "class",
)


@dataclass
class _Cell:
    id: str
    experiment: str
    domain: str
    model: str
    params_b: float | None
    conclusive: bool
    oracle_gain: dict[str, Any]
    withdrawn: dict[str, list[str]]
    runs: dict[str, dict[int, Path]]
    planned: dict[str, list[int]]
    forecasts: dict[str, Any]
    patch: dict[str, Any]
    final_step: int
    h1_pairs: list[tuple[str, str]]
    h2_pairs: list[tuple[str, str]]
    h3_pairs: list[tuple[str, str]]
    natural: Any
    analysis: dict[str, Any]
    root: Path
    variant_runs: dict[str, dict[str, dict[int, Path]]] = field(default_factory=dict)
    variant_specs: list[dict[str, Any]] = field(default_factory=list)
    withdrawn_rows: list[dict[str, Any]] = field(default_factory=list)
    stage: str | None = None
    arm_class: dict[str, int] = field(default_factory=dict)
    arm_natural: dict[str, bool] = field(default_factory=dict)


def within_cell_spearman_p(
    dtau: Sequence[float] | np.ndarray,
    dacc: Sequence[float] | np.ndarray,
    cells: Sequence[object] | np.ndarray,
    *,
    n_perm: int,
    seed: int = 0,
) -> dict[str, Any]:
    """One-sided Spearman p-value, permuting ``dacc`` inside each cell.

    ``p = (1 + #{rho* >= rho}) / (n_perm + 1)``. Cells with one pair are
    unchanged by the permutation. This is the null that, within each cell,
    ``dacc`` is exchangeable with respect to ``dtau``.
    """
    xx = np.asarray(list(dtau), dtype=float)
    yy = np.asarray(list(dacc), dtype=float)
    labels = [str(label) for label in cells]
    if xx.shape != yy.shape or xx.size != len(labels):
        raise ValueError("dtau, dacc and cells must have the same length")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    if xx.size < 2:
        return {
            "rho": float("nan"),
            "p": float("nan"),
            "n": int(xx.size),
            "method": "undefined",
            "n_perm": 0,
        }
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("non-finite values in a within-cell permutation")
    observed = spearman(xx, yy)
    if not math.isfinite(observed):
        return {
            "rho": float(observed),
            "p": float("nan"),
            "n": int(xx.size),
            "method": "undefined",
            "n_perm": 0,
        }
    groups = group_indices(labels)
    rng = np.random.default_rng(seed)
    count = 0
    tol = 1e-9
    scrambled = yy.copy()
    for _ in range(n_perm):
        for idx in groups:
            scrambled[idx] = rng.permutation(yy[idx])
        if spearman(xx, scrambled) >= observed - tol:
            count += 1
    return {
        "rho": float(observed),
        "p": (count + 1) / (n_perm + 1),
        "n": int(xx.size),
        "method": "within_cell_monte_carlo",
        "n_perm": int(n_perm),
    }


def pooled_report(
    program: Mapping[str, Any],
    runs_root: str | Path,
    *,
    n_boot: int = 10_000,
    n_perm: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Confirmatory report for a program of cells.

    ``program`` is the program mapping (``cells``, or ``experiments`` each
    with ``cells``). ``runs_root/<cell id>`` holds ``audit/``, ``runs/``,
    ``early/``, ``variants/``, ``transfer/``, and ``withdrawals.json``.
    """
    if n_boot < 1 or n_perm < 1:
        raise ValueError("n_boot and n_perm must be >= 1")
    root = Path(runs_root)
    analysis = program.get("analysis") if isinstance(program.get("analysis"), Mapping) else {}
    alpha = float(analysis.get("alpha", 0.05)) if isinstance(analysis, Mapping) else 0.05
    margin = float(analysis.get("margin", 0.05)) if isinstance(analysis, Mapping) else 0.05
    level = float(analysis.get("level", 0.95)) if isinstance(analysis, Mapping) else 0.95
    min_pairs = (
        int(analysis.get("h3_min_pairs", H3_MIN_PAIRS))
        if isinstance(analysis, Mapping)
        else H3_MIN_PAIRS
    )
    cells = [_load_cell(row, root) for row in _program_cells(program)]
    h3, h3_tested = _h3_block(
        cells, n_boot=n_boot, n_perm=n_perm, seed=seed, alpha=alpha, min_pairs=min_pairs
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "n_boot": n_boot,
        "n_perm": n_perm,
        "seed": seed,
        "alpha": alpha,
        "margin": margin,
        "level": level,
        "cells": [_cell_public(cell) for cell in cells],
        "conclusiveness": _conclusiveness(cells),
        "h1": _h1(cells, n_boot=n_boot, seed=seed, alpha=alpha, margin=margin, level=level),
        "h1_pooled": _h1_pooled(cells, n_boot=n_boot, seed=seed, alpha=alpha, level=level),
        "h2": _h2(cells, n_boot=n_boot, seed=seed, alpha=alpha, margin=margin, level=level),
        "h3": h3,
        "h7": _h7(cells, n_boot=n_boot, seed=seed, alpha=alpha, margin=margin, level=level),
        "h8": _h8(cells, n_boot=n_boot, seed=seed, alpha=alpha, level=level),
        "h9": _h9(h3_tested, h3.get("rho")),
        "scale": _scale(cells, n_boot=n_boot, seed=seed, level=level),
        "h10": _h10(cells, n_boot=n_boot, seed=seed, alpha=alpha, level=level),
        "cost": _cost(cells),
        "exploratory": _exploratory(
            cells,
            root,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            margin=margin,
            level=level,
        ),
    }
    report["figures"] = _figures(
        h3_pairs=report["h3"].get("pairs") or [],
        h9=report["h9"],
        h7=report["h7"],
        h8=report["h8"],
        h10=report["h10"],
        cost=report["cost"],
        stage=None,
    )
    return _jsonify(report)


def _h3_block(
    cells: Sequence[_Cell],
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
    alpha: float,
    min_pairs: int,
    rule: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """H3 primary, sensitivity, slices, and the tested pair records used by H9."""
    tested, everyone, withdrawn, excluded = _scan_h3(cells)
    primary = _h3_statistics(
        tested, n_boot=n_boot, n_perm=n_perm, seed=seed, alpha=alpha, min_pairs=min_pairs
    )
    sensitivity: dict[str, Any]
    if not excluded:
        sensitivity = {"same_as_primary": True, "note": "no inconclusive cells"}
    else:
        sensitivity = _h3_statistics(
            everyone, n_boot=n_boot, n_perm=n_perm, seed=seed, alpha=alpha, min_pairs=min_pairs
        )
    block = {
        **primary,
        "withdrawn": withdrawn,
        "excluded_inconclusive": excluded,
        "sensitivity_including_inconclusive": sensitivity,
    }
    if rule is not None:
        block["rule"] = rule
    _fill_slices(block, n_perm, seed)
    _strip_private(block)
    _strip_private(sensitivity)
    return block, tested


def _h10_pair_specs(cells: Sequence[_Cell]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    declared = False
    for cell in cells:
        if cell.experiment != "E1":
            continue
        block = cell.analysis.get("h10")
        if not isinstance(block, Mapping) or "pairs" not in block:
            continue
        declared = True
        pairs = block.get("pairs")
        if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
            continue
        for row in pairs:
            if not isinstance(row, Mapping) or "a" not in row or "b" not in row:
                continue
            pair = (str(row["a"]), str(row["b"]))
            if pair not in seen:
                seen.add(pair)
                found.append(pair)
    if declared:
        return found
    return list(_H10_PAIRS)


def _h10(
    cells: Sequence[_Cell], *, n_boot: int, seed: int, alpha: float, level: float
) -> dict[str, Any]:
    rows = [
        _h10_one(cells, left, right, n_boot=n_boot, seed=seed, level=level)
        for left, right in _h10_pair_specs(cells)
    ]
    _apply_mean_holm(rows, alpha)
    tested = [row for row in rows if row["status"] == "tested"]
    insufficient = [row for row in rows if row["status"] == "insufficient"]
    holds = any(row.get("rejects") for row in tested)
    decision = "insufficient" if not tested and insufficient else _decision(holds, tested)
    return {
        "rule": _H10_RULE,
        "decision": decision,
        "holds": bool(holds) if tested else False,
        "alpha": alpha,
        "min_cells": _H10_MIN_CELLS,
        "pairs": rows,
        "withdrawn": [item for row in rows for item in row.get("withdrawn") or []],
    }


def _h10_one(
    cells: Sequence[_Cell],
    left: str,
    right: str,
    *,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    name = f"{left} - {right}"
    withdrawn: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    members: list[_Cell] = []
    for cell in cells:
        if cell.experiment != "E1":
            continue
        status, reasons = _pair_status(cell, left, right)
        if status == "withdrawn":
            withdrawn.append({"cell": cell.id, "domain": cell.domain, "reasons": reasons})
            continue
        if status != "tested":
            continue
        if not cell.conclusive:
            excluded.append(
                {
                    "cell": cell.id,
                    "domain": cell.domain,
                    "reason": cell.oracle_gain.get("reason"),
                }
            )
            continue
        if left not in cell.forecasts or right not in cell.forecasts:
            continue
        members.append(cell)
    row: dict[str, Any] = {
        "name": name,
        "a": left,
        "b": right,
        "withdrawn": withdrawn,
        "excluded_inconclusive": excluded,
        "n_cells": len(members),
        "cells": [],
    }
    if len(members) < _H10_MIN_CELLS:
        row["status"] = "insufficient"
        row["point"] = None
        row["p_le_0"] = None
        return row
    points: list[float] = []
    draws: list[np.ndarray] = []
    cell_rows: list[dict[str, Any]] = []
    for cell in members:
        dtau = _tau(cell.forecasts[left]) - _tau(cell.forecasts[right])
        sign = 0.0 if dtau == 0.0 else (1.0 if dtau > 0.0 else -1.0)
        stats = _contrast_stats(
            cell,
            left,
            right,
            n_boot=n_boot,
            seed=seed,
            margin=0.0,
            level=level,
            label="h10",
        )
        if sign == 0.0:
            cell_draws = np.zeros(n_boot, dtype=float)
            point = 0.0
        else:
            cell_draws = sign * stats["draws"]
            point = sign * float(stats["point"])
        draws.append(cell_draws)
        points.append(point)
        cell_rows.append(
            {
                "cell": cell.id,
                "domain": cell.domain,
                "dtau": dtau,
                "point": point,
            }
        )
    mean_draws = np.mean(np.stack(draws, axis=0), axis=0)
    point = float(np.mean(points))
    lo, hi = _interval(mean_draws, level)
    row.update(
        {
            "status": "tested",
            "point": point,
            "lo": lo,
            "hi": hi,
            "p_le_0": float(np.mean(mean_draws <= 0.0)),
            "cells": cell_rows,
        }
    )
    return row


def _timing_gpu_hours(doc: Mapping[str, Any] | None) -> float | None:
    """``wall_seconds * gpus / 3600``. None when either field is missing."""
    if not isinstance(doc, Mapping):
        return None
    timing = doc.get("timing")
    if not isinstance(timing, Mapping):
        return None
    wall = timing.get("wall_seconds")
    gpus = timing.get("gpus")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)):
        return None
    if isinstance(gpus, bool) or not isinstance(gpus, (int, float)):
        return None
    if not math.isfinite(float(wall)) or not math.isfinite(float(gpus)):
        return None
    if float(wall) < 0.0 or float(gpus) < 0.0:
        return None
    return float(wall) * float(gpus) / 3600.0


def _read_timing_doc(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        doc = read_json(path)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _audit_gpu_hours(cell: _Cell) -> float | None:
    return _timing_gpu_hours(_read_timing_doc(cell.root / "audit" / "audit.meta.json"))


def _early_gpu_hours(cell: _Cell, arm: str) -> float | None:
    early = cell.analysis.get("early") or {}
    if isinstance(early, Mapping) and early.get("enabled") is False:
        return None
    seed = int(early.get("seed", 99)) if isinstance(early, Mapping) else 99
    path = cell.root / "early" / f"{arm}__s{seed}" / "summary.json"
    return _timing_gpu_hours(_read_timing_doc(path))


def _cost_row(cell: _Cell) -> dict[str, Any]:
    audit_hours = _audit_gpu_hours(cell)
    arms = sorted(cell.runs)
    early_values: list[float] = []
    missing = False
    for arm in arms:
        hours = _early_gpu_hours(cell, arm)
        if hours is None:
            missing = True
        else:
            early_values.append(hours)
    mean_early = None if (not arms or missing) else float(np.mean(early_values))
    break_even = None
    if audit_hours is not None and mean_early is not None and mean_early > 0.0:
        break_even = audit_hours / mean_early
    return {
        "scope": "cell",
        "cell": cell.id,
        "stage": cell.stage,
        "audit_gpu_hours": audit_hours,
        "early_gpu_hours_per_arm": mean_early,
        "n_trained_arms": len(arms),
        "break_even_arms": break_even,
    }


def _rollup_cost(stage: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audits = [row.get("audit_gpu_hours") for row in rows]
    if not rows or any(value is None for value in audits):
        audit_hours: float | None = None
    else:
        audit_hours = float(sum(float(value) for value in audits if value is not None))
    weighted: list[tuple[float, int]] = []
    missing = False
    n_arms = 0
    for row in rows:
        count = int(row.get("n_trained_arms") or 0)
        n_arms += count
        if count == 0:
            continue
        value = row.get("early_gpu_hours_per_arm")
        if value is None:
            missing = True
        else:
            weighted.append((float(value), count))
    if missing or not weighted:
        mean_early = None
    else:
        total = sum(count for _value, count in weighted)
        mean_early = sum(value * count for value, count in weighted) / total
    break_even = None
    if audit_hours is not None and mean_early is not None and mean_early > 0.0:
        break_even = audit_hours / mean_early
    return {
        "scope": "stage",
        "cell": None,
        "stage": stage,
        "audit_gpu_hours": audit_hours,
        "early_gpu_hours_per_arm": mean_early,
        "n_trained_arms": n_arms,
        "break_even_arms": break_even,
    }


def _cost(cells: Sequence[_Cell]) -> dict[str, Any]:
    rows = [_cost_row(cell) for cell in cells]
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stage = row.get("stage")
        if stage is not None:
            by_stage.setdefault(str(stage), []).append(row)
    return {
        "rule": _COST_RULE,
        "decision": "reported",
        "cells": rows,
        "stages": [_rollup_cost(stage, group) for stage, group in by_stage.items()],
        "withdrawn": [],
    }


def _h3_figure_row(pair: Mapping[str, Any]) -> dict[str, Any]:
    baselines = pair.get("baselines") if isinstance(pair.get("baselines"), Mapping) else {}
    row: dict[str, Any] = {
        "cell": pair.get("cell"),
        "stage": pair.get("stage"),
        "experiment": pair.get("experiment"),
        "domain": pair.get("domain"),
        "model": pair.get("model"),
        "params_b": pair.get("params_b"),
        "arm": pair.get("arm"),
        "control": pair.get("control"),
        "class": pair.get("arm_class"),
        "is_natural": bool(pair.get("is_natural")),
        "dtau": pair.get("dtau"),
        "dacc": pair.get("dacc"),
    }
    for name in _BASELINES:
        row[f"baseline_{name}"] = baselines.get(name) if isinstance(baselines, Mapping) else None
    return row


def _h9_figure_rows(h9: Mapping[str, Any] | None, stage: str | None) -> list[dict[str, Any]]:
    if not isinstance(h9, Mapping):
        return []
    budgets = h9.get("budgets")
    if not isinstance(budgets, Mapping):
        return []
    rows = []
    for name, block in budgets.items():
        if not isinstance(block, Mapping):
            continue
        budget: int | str = int(name) if str(name).isdigit() else str(name)
        rows.append(
            {
                "stage": stage,
                "budget": budget,
                "status": block.get("status"),
                "median_rho": block.get("median_rho"),
                "q05": block.get("q05"),
                "frac_positive": block.get("frac_positive"),
                "n_draws": block.get("n_draws"),
                "n_pairs": block.get("n_pairs"),
            }
        )
    return rows


def _natural_step0(cell: _Cell, natural: object, control: object) -> dict[str, Any]:
    """Step-0 channel rates of a natural verifier and the locked tau gap to its control.

    ``dtau`` is ``tau_pool(natural) - tau_pool(control)``. A missing forecast
    or a non-finite value stays null.
    """

    def finite(value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    out: dict[str, Any] = {"fpr": None, "fnr": None, "error_rate": None, "dtau": None}
    entry = cell.forecasts.get(str(natural)) if isinstance(natural, str) else None
    base = cell.forecasts.get(str(control)) if isinstance(control, str) else None
    if not isinstance(entry, Mapping):
        return out
    try:
        row = forecast_row(entry)
    except (KeyError, TypeError):
        return out
    out["fpr"] = finite(row.get("fpr0"))
    out["fnr"] = finite(row.get("fnr0"))
    if out["fpr"] is not None and out["fnr"] is not None:
        out["error_rate"] = out["fpr"] + out["fnr"]
    tau = finite(row.get("tau_pool"))
    if isinstance(base, Mapping) and tau is not None:
        try:
            control_tau = finite(forecast_row(base).get("tau_pool"))
        except (KeyError, TypeError):
            control_tau = None
        if control_tau is not None:
            out["dtau"] = tau - control_tau
    return out


def _natural_figure_rows(h7: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(h7, Mapping) or not isinstance(h7.get("domains"), Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for domain, block in h7["domains"].items():
        if not isinstance(block, Mapping):
            continue
        for pair in block.get("pairs") or []:
            if not isinstance(pair, Mapping):
                continue
            contrast = pair.get("contrast") if isinstance(pair.get("contrast"), Mapping) else {}
            differs = pair.get("differs") if isinstance(pair.get("differs"), Mapping) else {}
            locality = pair.get("locality") if isinstance(pair.get("locality"), Mapping) else {}
            screen = (
                pair.get("screen_direction")
                if isinstance(pair.get("screen_direction"), Mapping)
                else {}
            )
            raw_step0 = pair.get("step0")
            step0: Mapping[str, Any] = raw_step0 if isinstance(raw_step0, Mapping) else {}
            point = contrast.get("point") if isinstance(contrast, Mapping) else None
            rows.append(
                {
                    "cell": pair.get("cell"),
                    "stage": pair.get("stage"),
                    "experiment": pair.get("experiment"),
                    "domain": pair.get("domain") or domain,
                    "natural": pair.get("natural"),
                    "control": pair.get("control"),
                    "point": contrast.get("point") if isinstance(contrast, Mapping) else None,
                    "lo": contrast.get("lo") if isinstance(contrast, Mapping) else None,
                    "hi": contrast.get("hi") if isinstance(contrast, Mapping) else None,
                    "differs": differs.get("decision") if isinstance(differs, Mapping) else None,
                    "locality": locality.get("decision") if isinstance(locality, Mapping) else None,
                    "screen": screen.get("decision") if isinstance(screen, Mapping) else None,
                    "step0_fpr": step0.get("fpr"),
                    "step0_fnr": step0.get("fnr"),
                    "step0_error_rate": step0.get("error_rate"),
                    "dtau": step0.get("dtau"),
                    "dacc": None if point is None else -float(point),
                }
            )
    return rows


def _h8_figure_rows(h8: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(h8, Mapping):
        return []
    rows = []
    for row in h8.get("cells") or []:
        if isinstance(row, Mapping):
            rows.append(
                {
                    "cell": row.get("cell"),
                    "stage": row.get("stage"),
                    "domain": row.get("domain"),
                    "point": row.get("point"),
                    "per_seed": list(row.get("per_seed") or []),
                }
            )
    return rows


def _h10_figure_rows(h10: Mapping[str, Any] | None, stage: str | None) -> list[dict[str, Any]]:
    if not isinstance(h10, Mapping):
        return []
    rows = []
    for row in h10.get("pairs") or []:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "stage": stage if row.get("stage") is None else row.get("stage"),
                "name": row.get("name"),
                "a": row.get("a"),
                "b": row.get("b"),
                "status": row.get("status"),
                "point": row.get("point"),
                "p_le_0": row.get("p_le_0"),
                "holm_adjusted_p": row.get("holm_adjusted_p"),
                "n_cells": row.get("n_cells"),
                "rejects": row.get("rejects"),
            }
        )
    if stage is not None:
        for row in rows:
            row["stage"] = stage
    return rows


def _cost_figure_rows(cost: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(cost, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for row in list(cost.get("cells") or []) + list(cost.get("stages") or []):
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _figures(
    *,
    h3_pairs: Sequence[Any],
    h9: Mapping[str, Any] | None,
    h7: Mapping[str, Any] | None,
    h8: Mapping[str, Any] | None,
    h10: Mapping[str, Any] | None,
    cost: Mapping[str, Any] | None,
    stage: str | None,
) -> dict[str, list[dict[str, Any]]]:
    """Plain lists the paper figures read. See the module docstring."""
    return {
        "h3_pairs": [_h3_figure_row(pair) for pair in h3_pairs if isinstance(pair, Mapping)],
        "h9_budgets": _h9_figure_rows(h9, stage),
        "natural_rows": _natural_figure_rows(h7),
        "h8_rows": _h8_figure_rows(h8),
        "h10_rows": _h10_figure_rows(h10, stage),
        "cost_rows": _cost_figure_rows(cost),
    }


def _analysis_settings(
    program: Mapping[str, Any],
) -> tuple[float, float, float, int]:
    analysis = program.get("analysis") if isinstance(program.get("analysis"), Mapping) else {}
    if not isinstance(analysis, Mapping):
        return 0.05, 0.05, 0.95, H3_MIN_PAIRS
    alpha = float(analysis.get("alpha", 0.05))
    margin = float(analysis.get("margin", 0.05))
    level = float(analysis.get("level", 0.95))
    min_pairs = int(analysis.get("h3_min_pairs", H3_MIN_PAIRS))
    return alpha, margin, level, min_pairs


def _load_stage_cells(
    program: Mapping[str, Any], runs_root: Path, stage: str, cell_ids: Sequence[str]
) -> list[_Cell]:
    by_id = {str(row["id"]): row for row in _program_cells(program)}
    cells: list[_Cell] = []
    for cell_id in cell_ids:
        if cell_id not in by_id:
            raise KeyError(f"stage {stage} cell {cell_id} is not in the program")
        row = dict(by_id[cell_id])
        row["stage"] = stage
        cells.append(_load_cell(row, runs_root))
    return cells


def _stage_a_block(
    cells: Sequence[_Cell],
    runs_root: Path,
    *,
    name: Any,
    dropped: Sequence[str],
    n_boot: int,
    n_perm: int,
    seed: int,
    alpha: float,
    margin: float,
    level: float,
    min_pairs: int,
) -> dict[str, Any]:
    h3, tested = _h3_block(
        cells,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
        alpha=alpha,
        min_pairs=min_pairs,
        rule=_H3_STAGE_A_RULE,
    )
    return {
        "name": name,
        "status": "complete",
        "dropped": list(dropped),
        "conclusiveness": _conclusiveness(cells),
        "h1": _h1(cells, n_boot=n_boot, seed=seed, alpha=alpha, margin=margin, level=level),
        "h1_pooled": _h1_pooled(cells, n_boot=n_boot, seed=seed, alpha=alpha, level=level),
        "h2": _h2(cells, n_boot=n_boot, seed=seed, alpha=alpha, margin=margin, level=level),
        "h3": h3,
        "h7": _h7(cells, n_boot=n_boot, seed=seed, alpha=alpha, margin=margin, level=level),
        "h8": _h8(cells, n_boot=n_boot, seed=seed, alpha=alpha, level=level),
        "h9": _h9(tested, h3.get("rho")),
        "h10": _h10(cells, n_boot=n_boot, seed=seed, alpha=alpha, level=level),
        "scale": _scale(cells, n_boot=n_boot, seed=seed, level=level),
        "cost": _cost(cells),
        "exploratory": _exploratory(
            cells,
            runs_root,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            margin=margin,
            level=level,
        ),
    }


def _stage_b_block(
    cells: Sequence[_Cell],
    *,
    name: Any,
    dropped: Sequence[str],
    n_boot: int,
    n_perm: int,
    seed: int,
    alpha: float,
    level: float,
    min_pairs: int,
) -> dict[str, Any]:
    h3, tested = _h3_block(
        cells,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
        alpha=alpha,
        min_pairs=min_pairs,
        rule=_H3_REPLICATION_RULE,
    )
    return {
        "name": name,
        "status": "complete",
        "dropped": list(dropped),
        "conclusiveness": _conclusiveness(cells),
        "h3_replication": h3,
        "h9": _h9(tested, h3.get("rho")),
        "cost": _cost(cells),
    }


def _cross_stage_secondary(
    cells_a: Sequence[_Cell] | None,
    cells_b: Sequence[_Cell] | None,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
    alpha: float,
    level: float,
    min_pairs: int,
) -> dict[str, Any]:
    """Union H3, its strata, and the scale slopes on the full ladder."""
    missing = [key for key, cells in (("A", cells_a), ("B", cells_b)) if cells is None]
    if missing:
        return {
            "status": "pending",
            "missing_stages": missing,
            "rule": _H3_UNION_RULE,
        }
    both = [*(cells_a or ()), *(cells_b or ())]
    h3, _tested = _h3_block(
        both,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
        alpha=alpha,
        min_pairs=min_pairs,
        rule=_H3_UNION_RULE,
    )
    return {
        "status": "complete",
        "h3": h3,
        "h3_by_stratum": {
            "by_experiment": h3.get("by_experiment") or {},
            "by_domain": h3.get("by_domain") or {},
            "by_model": h3.get("by_model") or {},
        },
        "scale": _scale(both, n_boot=n_boot, seed=seed, level=level),
    }


def _stage_name(program: Mapping[str, Any], key: str) -> Any:
    stages = program.get("stages")
    if not isinstance(stages, Mapping):
        return None
    block = stages.get(key)
    if isinstance(block, Mapping):
        return block.get("name")
    return None


def _merge_figures(blocks: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    figures: dict[str, list[dict[str, Any]]] = {
        "h3_pairs": [],
        "h9_budgets": [],
        "natural_rows": [],
        "h8_rows": [],
        "h10_rows": [],
        "cost_rows": [],
    }
    for key in ("A", "B"):
        block = blocks.get(key)
        if not isinstance(block, Mapping) or block.get("status") != "complete":
            continue
        for source_key in ("h3", "h3_replication"):
            source = block.get(source_key)
            if isinstance(source, Mapping):
                pairs = source.get("pairs") or []
                figures["h3_pairs"].extend(
                    _h3_figure_row(pair) for pair in pairs if isinstance(pair, Mapping)
                )
        if isinstance(block.get("h9"), Mapping):
            figures["h9_budgets"].extend(_h9_figure_rows(block["h9"], key))
        if isinstance(block.get("h7"), Mapping):
            figures["natural_rows"].extend(_natural_figure_rows(block["h7"]))
        if isinstance(block.get("h8"), Mapping):
            figures["h8_rows"].extend(_h8_figure_rows(block["h8"]))
        if isinstance(block.get("h10"), Mapping):
            figures["h10_rows"].extend(_h10_figure_rows(block["h10"], key))
        if isinstance(block.get("cost"), Mapping):
            figures["cost_rows"].extend(_cost_figure_rows(block["cost"]))
    return figures


def staged_report(
    program: Mapping[str, Any],
    runs_root: str | Path,
    *,
    ready: Mapping[str, Sequence[str]],
    pending: Mapping[str, Mapping[str, Any]],
    dropped: Mapping[str, Sequence[str]] | None = None,
    n_boot: int = 10_000,
    n_perm: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Stage A and stage B confirmatory blocks, plus the union secondary.

    ``ready`` maps a stage id to the cell ids whose confirmatory train runs
    are complete (budget-cut cells already removed). ``pending`` maps a stage
    id to a block with ``status`` ``pending`` and ``missing`` job ids. A
    pending stage contributes no hypothesis numbers. The secondary union is
    pending until both stages are ready.
    """
    if n_boot < 1 or n_perm < 1:
        raise ValueError("n_boot and n_perm must be >= 1")
    root = Path(runs_root)
    alpha, margin, level, min_pairs = _analysis_settings(program)
    loaded: dict[str, list[_Cell]] = {}
    blocks: dict[str, Any] = {}
    for key in ("A", "B"):
        if key in pending:
            blocks[key] = dict(pending[key])
            continue
        if key not in ready:
            continue
        cells = _load_stage_cells(program, root, key, ready[key])
        loaded[key] = cells
        name = _stage_name(program, key)
        stage_dropped = list((dropped or {}).get(key) or [])
        if key == "A":
            blocks[key] = _stage_a_block(
                cells,
                root,
                name=name,
                dropped=stage_dropped,
                n_boot=n_boot,
                n_perm=n_perm,
                seed=seed,
                alpha=alpha,
                margin=margin,
                level=level,
                min_pairs=min_pairs,
            )
        else:
            blocks[key] = _stage_b_block(
                cells,
                name=name,
                dropped=stage_dropped,
                n_boot=n_boot,
                n_perm=n_perm,
                seed=seed,
                alpha=alpha,
                level=level,
                min_pairs=min_pairs,
            )
    secondary = _cross_stage_secondary(
        loaded.get("A"),
        loaded.get("B"),
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
        alpha=alpha,
        level=level,
        min_pairs=min_pairs,
    )
    report = {
        "schema": SCHEMA,
        "n_boot": n_boot,
        "n_perm": n_perm,
        "seed": seed,
        "alpha": alpha,
        "margin": margin,
        "level": level,
        "stages": {key: blocks[key] for key in ("A", "B") if key in blocks},
        "secondary": secondary,
        "figures": _merge_figures(blocks),
    }
    return _jsonify(report)


def render_pooled_markdown(report: Mapping[str, Any]) -> str:
    """One section per hypothesis, decision line first."""
    if isinstance(report.get("stages"), Mapping):
        return _render_staged_markdown(report)
    lines = ["# Pooled confirmatory report", ""]
    sections = (
        ("h1", "H1"),
        ("h1_pooled", "H1 pooled"),
        ("h2", "H2"),
        ("h3", "H3"),
        ("h7", "H7"),
        ("h8", "H8"),
        ("h9", "H9"),
        ("h10", "H10"),
        ("scale", "Scale"),
        ("cost", "Cost"),
        ("conclusiveness", "Conclusiveness"),
        ("exploratory", "Exploratory"),
    )
    for key, title in sections:
        block = report.get(key)
        if not isinstance(block, Mapping):
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"Decision: {block.get('decision')}.")
        lines.append("")
        if block.get("rule"):
            lines.append(str(block["rule"]))
            lines.append("")
        lines.append("Withdrawn: " + _fmt_withdrawn(block.get("withdrawn") or []))
        lines.append("")
        _append_block_body(lines, key, block)
        lines.append("")
    return "\n".join(lines)


def _render_staged_markdown(report: Mapping[str, Any]) -> str:
    """Markdown for a two-stage program report. Pending stages list missing runs."""
    lines = ["# Pooled confirmatory report", ""]
    stages = report.get("stages")
    if not isinstance(stages, Mapping):
        return "\n".join(lines)
    for key in ("A", "B"):
        block = stages.get(key)
        if not isinstance(block, Mapping):
            continue
        lines.append(f"## Stage {key}")
        lines.append("")
        name = block.get("name")
        if name:
            lines.append(f"Name: {name}.")
            lines.append("")
        lines.append(f"Status: {block.get('status')}.")
        lines.append("")
        if block.get("status") == "pending":
            missing = [str(item) for item in (block.get("missing") or [])]
            lines.append(f"Missing runs ({len(missing)}):")
            lines.extend(f"- {item}" for item in missing)
            dropped = block.get("dropped") or []
            if dropped:
                lines.append("Dropped: " + ", ".join(str(item) for item in dropped) + ".")
            lines.append("")
            continue
        if block.get("dropped"):
            lines.append("Dropped: " + ", ".join(str(item) for item in block["dropped"]) + ".")
            lines.append("")
        body = {k: v for k, v in block.items() if k not in {"name", "status", "dropped"}}
        if "h3_replication" in body and "h3" not in body:
            body = dict(body)
            body["h3"] = body["h3_replication"]
        nested = render_pooled_markdown(body)
        nested_lines = nested.splitlines()
        if nested_lines and nested_lines[0].startswith("# "):
            nested_lines = nested_lines[1:]
            if nested_lines and nested_lines[0] == "":
                nested_lines = nested_lines[1:]
        lines.extend(nested_lines)
        lines.append("")
    secondary = report.get("secondary")
    lines.append("## Secondary")
    lines.append("")
    if isinstance(secondary, Mapping):
        lines.append(f"Status: {secondary.get('status')}.")
        if secondary.get("status") == "pending":
            missing_stages = ", ".join(str(item) for item in secondary.get("missing_stages") or [])
            lines.append(f"Missing stages: {missing_stages}.")
        elif isinstance(secondary.get("h3"), Mapping):
            h3 = secondary["h3"]
            lines.append(
                f"Union H3: n_pairs={h3.get('n_pairs')}, rho={h3.get('rho')}, "
                f"permutation_p={h3.get('permutation_p')}, q05={h3.get('q05')}, "
                f"decision={h3.get('decision')}."
            )
    lines.append("")
    return "\n".join(lines)


def _program_cells(program: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(program, Mapping):
        raise ValueError("program must be a mapping")
    if "cells" in program:
        cells = program["cells"]
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            raise ValueError("program cells must be a list")
        return [row for row in cells if isinstance(row, Mapping)]
    found: list[Mapping[str, Any]] = []
    experiments = program.get("experiments") or []
    if isinstance(experiments, Mapping):
        experiments = [
            {"id": key, **(value if isinstance(value, Mapping) else {})}
            for key, value in experiments.items()
        ]
    if not isinstance(experiments, Sequence) or isinstance(experiments, (str, bytes)):
        raise ValueError("program has no cells")
    for exp in experiments:
        if not isinstance(exp, Mapping):
            continue
        for cell in exp.get("cells") or []:
            if not isinstance(cell, Mapping):
                continue
            row = dict(cell)
            row.setdefault("experiment", exp.get("id") or exp.get("name") or exp.get("experiment"))
            found.append(row)
    if not found:
        raise ValueError("program has no cells")
    return found


def _load_cell(row: Mapping[str, Any], runs_root: Path) -> _Cell:
    cell_id = str(row["id"])
    root = runs_root / cell_id
    if not root.is_dir():
        raise ValueError(f"missing cell directory: {root}")
    config = _cell_config(row)
    analysis: dict[str, Any] = {}
    if isinstance(row.get("analysis"), Mapping):
        analysis.update(row["analysis"])
    if isinstance(config.get("analysis"), Mapping):
        analysis.update(config["analysis"])
    study_raw = config.get("study")
    if not isinstance(study_raw, Mapping) and isinstance(row.get("study"), Mapping):
        study_raw = row.get("study")
    study: Mapping[str, Any] = study_raw if isinstance(study_raw, Mapping) else {}
    runs = _discover_runs(root / "runs")
    variant_runs = _discover_variants(root / "variants")
    planned = _planned(study, runs)
    withdrawn = _read_withdrawals(root / "withdrawals.json")
    for arm, seeds in planned.items():
        have = runs.get(arm, {})
        if seeds and any(seed not in have for seed in seeds):
            bucket = withdrawn.setdefault(arm, [])
            if "incomplete_seeds" not in bucket:
                bucket.append("incomplete_seeds")
    forecasts, patch = _split_forecasts(_forecast_document(root / "audit"))
    if patch.get("same_selection") is True:
        bucket = withdrawn.setdefault("patch_rate", [])
        if "same_selection" not in bucket:
            bucket.append("same_selection")
    final_step = int(analysis.get("final_step", row.get("final_step", 300)))
    gain_min = float(analysis.get("oracle_gain_min", 0.10))
    oracle = _oracle_gain(runs, planned, final_step, gain_min)
    domain = str(row.get("domain") or "")
    arm_class, arm_natural = _arm_structure(config, domain)
    stage = row.get("stage")
    return _Cell(
        id=cell_id,
        experiment=_experiment(row.get("experiment")),
        domain=domain,
        model=str(row.get("model") or ""),
        params_b=_params_b(row, config),
        conclusive=not bool(oracle["inconclusive"]),
        oracle_gain=oracle,
        withdrawn=withdrawn,
        runs=runs,
        planned=planned,
        forecasts=forecasts,
        patch=patch,
        final_step=final_step,
        h1_pairs=_pairs(analysis.get("h1"), "contrasts", _default_h1(domain)),
        h2_pairs=_pairs(analysis.get("h2"), "contrasts", _default_h2()),
        h3_pairs=_h3_pair_list(analysis.get("h3"), domain),
        natural=analysis.get("natural"),
        analysis=analysis,
        root=root,
        variant_runs=variant_runs,
        variant_specs=_variant_specs(study),
        withdrawn_rows=_withdrawn_rows(cell_id, withdrawn),
        stage=None if stage is None else str(stage),
        arm_class=arm_class,
        arm_natural=arm_natural,
    )


def _cell_config(row: Mapping[str, Any]) -> dict[str, Any]:
    cfg = row.get("config")
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if isinstance(cfg, str) and cfg:
        path = Path(cfg)
        if not path.is_file():
            raise ValueError(f"cell config not found: {cfg}")
        loaded = read_yaml(path)
        if not isinstance(loaded, dict):
            raise ValueError(f"cell config is not a mapping: {cfg}")
        return loaded
    return {}


def _experiment(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split()[0] if text else ""


def _params_b(row: Mapping[str, Any], config: Mapping[str, Any]) -> float | None:
    for source in (row, config):
        if source.get("params_b") is not None:
            return float(source["params_b"])
    model = row.get("model")
    if not isinstance(model, str) or not model:
        return None
    try:
        from vfates.models import load_model_spec
    except ImportError:
        return None
    try:
        spec = load_model_spec(model)
    except (FileNotFoundError, OSError, ValueError, KeyError, AttributeError):
        return None
    value = getattr(spec, "params_b", None)
    if value is None and isinstance(spec, Mapping):
        value = spec.get("params_b")
    return None if value is None else float(value)


def _discover_runs(root: Path) -> dict[str, dict[int, Path]]:
    """``<arm>__s<seed>`` directories, the layout of ``vfates.study._run_dir``."""
    found: dict[str, dict[int, Path]] = {}
    if not root.is_dir():
        return found
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        parsed = _arm_seed(path.name)
        if parsed is None:
            continue
        arm, seed = parsed
        found.setdefault(arm, {})[seed] = path
    return found


def _discover_variants(root: Path) -> dict[str, dict[str, dict[int, Path]]]:
    """``variants/<name>/<arm>__s<seed>`` under a cell root."""
    found: dict[str, dict[str, dict[int, Path]]] = {}
    if not root.is_dir():
        return found
    for variant_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        found[variant_dir.name] = _discover_runs(variant_dir)
    return found


def _arm_seed(name: str) -> tuple[str, int] | None:
    arm, sep, seed_s = name.rpartition("__s")
    if not sep or not arm or not seed_s.lstrip("-").isdigit():
        return None
    return arm, int(seed_s)


def _variant_specs(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = study.get("variants")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    specs: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping) and item.get("name"):
            specs.append(dict(item))
    return specs


def _planned(
    study: Mapping[str, Any], runs: Mapping[str, Mapping[int, Path]]
) -> dict[str, list[int]]:
    if "arms" in study and "seeds" in study:
        from vfates.study import planned_seeds

        return {arm: list(seeds) for arm, seeds in planned_seeds(study).items()}
    return {arm: sorted(seeds) for arm, seeds in runs.items()}


def _read_withdrawals(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    doc = read_json(path)
    if not isinstance(doc, Mapping):
        return {}
    reasons: dict[str, list[str]] = {}
    for bucket in ("audit", "training"):
        block = doc.get(bucket)
        if not isinstance(block, Mapping):
            continue
        for arm, info in block.items():
            found: list[str] = []
            if isinstance(info, Mapping):
                raw = info.get("reasons") or []
                if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                    found = [str(item) for item in raw]
            elif isinstance(info, Sequence) and not isinstance(info, (str, bytes)):
                found = [str(item) for item in info]
            reasons.setdefault(str(arm), []).extend(found)
    listed = doc.get("withdrawn") or []
    if isinstance(listed, Sequence) and not isinstance(listed, (str, bytes)):
        for arm in listed:
            reasons.setdefault(str(arm), [])
    for arm, items in reasons.items():
        if not items:
            reasons[arm] = ["withdrawn"]
    return reasons


def _forecast_document(audit_dir: Path) -> dict[str, Any]:
    path = audit_dir / "forecasts.json"
    if not path.is_file():
        raise ValueError(f"{audit_dir} is missing forecasts.json")
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not an object")
    lock_path = audit_dir / "lock.json"
    if not lock_path.is_file():
        return doc
    try:
        locked = load_lock(lock_path)
    except ValueError:
        return doc
    if not isinstance(locked, dict):
        return doc
    if canonical_json(doc) == canonical_json(locked):
        return locked
    merged = dict(locked)
    extra = doc.get("patch_selection")
    if "patch_selection" not in merged and isinstance(extra, dict):
        merged["patch_selection"] = extra
    for arm, entry in _forecast_arm_items(doc):
        current = merged.get(arm)
        if not isinstance(current, dict):
            merged[arm] = entry
            continue
        budget = entry.get("tau_pool_budget")
        if "tau_pool_budget" not in current and isinstance(budget, dict):
            merged[arm] = {**current, "tau_pool_budget": budget}
    return merged


def _split_forecasts(doc: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = {name: dict(entry) for name, entry in _forecast_arm_items(doc)}
    patch = doc.get("patch_selection")
    return arms, dict(patch) if isinstance(patch, Mapping) else {}


def _oracle_gain(
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    final_step: int,
    gain_min: float,
) -> dict[str, Any]:
    seeds = list(planned.get("oracle") or [])
    oracle_runs = runs.get("oracle") or {}
    if not seeds:
        seeds = sorted(oracle_runs)
    if not seeds or any(seed not in oracle_runs for seed in seeds):
        return {
            "value": None,
            "threshold": gain_min,
            "inconclusive": True,
            "reason": "oracle arm is missing",
            "n_seeds": 0,
        }
    gains = []
    for seed in seeds:
        path = oracle_runs[int(seed)]
        final, final_map = _mean_at(path, final_step)
        step0, step0_map = _mean_at(path, 0)
        if set(final_map) != set(step0_map):
            raise ValueError(f"{path}: step 0 and the final step do not share prompts")
        gains.append(final - step0)
    value = float(np.mean(gains))
    low = value < gain_min
    return {
        "value": value,
        "threshold": gain_min,
        "inconclusive": low,
        "reason": "gain < threshold" if low else "gain >= threshold",
        "n_seeds": len(seeds),
    }


def _mean_at(path: Path, step: int) -> tuple[float, dict[str, float]]:
    raw = prompt_accuracy(path, "heldout", step)
    acc = {str(key): float(value) for key, value in raw.items()}
    if not acc:
        raise ValueError(f"{path}: empty prompt map at step {step}")
    return float(np.mean(list(acc.values()))), acc


def _pairs(block: Any, key: str, default: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    if not isinstance(block, Mapping) or key not in block:
        return list(default)
    return [(str(row["control"]), str(row["arm"])) for row in block[key]]


def _h3_pair_list(block: Any, domain: str) -> list[tuple[str, str]]:
    if isinstance(block, Mapping) and "pairs" in block:
        return [(str(row["arm"]), str(row["control"])) for row in block["pairs"]]
    return _default_h3(domain)


def _default_h1(domain: str) -> list[tuple[str, str]]:
    pairs = [("noise_fp_trigger", "fp_trigger"), ("noise_fn", "fn_behavior")]
    if domain in {"chain_sum", "math"}:
        return [("noise_fp", "fp_tolerance"), *pairs]
    return pairs


def _default_h2() -> list[tuple[str, str]]:
    return [("noise_fp", "fp_hash"), ("noise_fn", "fn_prompt_subset")]


def _default_h3(domain: str) -> list[tuple[str, str]]:
    pairs = [
        ("fp_hash", "noise_fp"),
        ("fp_trigger", "noise_fp_trigger"),
        ("fn_behavior", "noise_fn"),
        ("fn_prompt_subset", "noise_fn"),
    ]
    if domain == "chain_sum":
        return [
            ("fp_tolerance", "noise_fp"),
            ("fp_tolerance_decoy", "noise_fp"),
            ("fn_form", "noise_fn_form"),
            *pairs,
        ]
    if domain == "math":
        return [("fp_tolerance", "noise_fp"), *pairs]
    return pairs


def _withdrawn_rows(cell_id: str, withdrawn: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    return [
        {"cell": cell_id, "arm": arm, "reasons": list(reasons)}
        for arm, reasons in sorted(withdrawn.items())
    ]


def _cell_public(cell: _Cell) -> dict[str, Any]:
    return {
        "id": cell.id,
        "experiment": cell.experiment,
        "domain": cell.domain,
        "model": cell.model,
        "params_b": cell.params_b,
        "stage": cell.stage,
        "conclusive": cell.conclusive,
        "oracle_gain": cell.oracle_gain,
        "withdrawn": cell.withdrawn_rows,
    }


def _seeds_for(cell: _Cell, arm: str) -> list[int]:
    if arm in cell.planned:
        return list(cell.planned[arm])
    return sorted(cell.runs.get(arm, {}))


def _pair_status(cell: _Cell, left: str, right: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for arm in (left, right):
        if arm in cell.withdrawn:
            arm_reasons = list(cell.withdrawn[arm]) or ["withdrawn"]
            reasons.extend(f"{arm}: {reason}" for reason in arm_reasons)
    if reasons:
        return "withdrawn", reasons
    for arm in (left, right):
        seeds = _seeds_for(cell, arm)
        if arm not in cell.runs or not seeds or any(seed not in cell.runs[arm] for seed in seeds):
            return "not_in_study", []
    if _seeds_for(cell, left) != _seeds_for(cell, right):
        return "seed_mismatch", []
    return "tested", []


def _stream_seed(master: int, *parts: object) -> int:
    text = repr((int(master), *parts))
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _crossed_mean_draws(matrix: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n_seeds, n_prompts = matrix.shape
    prompt_idx = rng.integers(0, n_prompts, size=(n_boot, n_prompts))
    seed_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    prompt_means = np.stack(
        [matrix[s, prompt_idx].mean(axis=1) for s in range(n_seeds)],
        axis=1,
    )
    return prompt_means[np.arange(n_boot)[:, None], seed_idx].mean(axis=1)


def _interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return float(lo), float(hi)


def _diff_matrix(cell: _Cell, left: str, right: str) -> np.ndarray:
    seeds = _seeds_for(cell, left)
    matrix, _pids = difference_matrix(
        [(cell.runs[left][seed], cell.runs[right][seed]) for seed in seeds],
        step=cell.final_step,
    )
    return matrix


def _contrast_stats(
    cell: _Cell,
    left: str,
    right: str,
    *,
    n_boot: int,
    seed: int,
    margin: float,
    level: float,
    label: str,
) -> dict[str, Any]:
    matrix = _diff_matrix(cell, left, right)
    rng = np.random.default_rng(_stream_seed(seed, label, cell.id, left, right))
    draws = _crossed_mean_draws(matrix, n_boot, rng)
    lo, hi = _interval(draws, level)
    per_seed = [float(value) for value in matrix.mean(axis=1)]
    return {
        "point": float(matrix.mean()),
        "per_seed": per_seed,
        "lo": lo,
        "hi": hi,
        "p_sup": float(np.mean(draws <= margin)),
        "p_tost": float(max(np.mean(draws <= -margin), np.mean(draws >= margin))),
        "p_le_0": float(np.mean(draws <= 0.0)),
        "draws": draws,
        "n_seeds": int(matrix.shape[0]),
        "n_prompts": int(matrix.shape[1]),
        "all_seeds_positive": all(value > 0.0 for value in per_seed),
    }


def _drop_draws(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "draws"}


def _h1(
    cells: Sequence[_Cell],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    margin: float,
    level: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    withdrawn: list[dict[str, Any]] = []
    for cell in cells:
        if cell.experiment != "E1":
            continue
        for control, arm in cell.h1_pairs:
            status, reasons = _pair_status(cell, control, arm)
            row: dict[str, Any] = {
                "cell": cell.id,
                "experiment": cell.experiment,
                "domain": cell.domain,
                "name": f"{control} - {arm}",
                "control": control,
                "arm": arm,
                "status": status,
                "reasons": reasons,
                "inconclusive_cell": not cell.conclusive,
            }
            if status == "withdrawn":
                withdrawn.append(row)
                records.append(row)
                continue
            if status != "tested":
                records.append(row)
                continue
            stats = _contrast_stats(
                cell,
                control,
                arm,
                n_boot=n_boot,
                seed=seed,
                margin=margin,
                level=level,
                label="h1",
            )
            row.update(_drop_draws(stats))
            row["holm_id"] = f"{cell.id} | {row['name']}"
            records.append(row)
    tested = [row for row in records if row["status"] == "tested"]
    if tested:
        pvalues = {row["holm_id"]: row["p_sup"] for row in tested}
        adjusted = holm_adjusted(pvalues)
        rejected = holm(pvalues, alpha)
        for row in tested:
            row["holm_adjusted_p"] = adjusted[row["holm_id"]]
            row["holm_reject"] = rejected[row["holm_id"]]
            row["rejects"] = bool(rejected[row["holm_id"]] and row["all_seeds_positive"])
    holds = any(row.get("rejects") for row in tested)
    return {
        "rule": _H1_RULE,
        "decision": _decision(holds, tested),
        "holds": holds if tested else False,
        "alpha": alpha,
        "margin": margin,
        "n_family": len(tested),
        "n_rejected": sum(bool(row.get("rejects")) for row in tested),
        "contrasts": records,
        "withdrawn": withdrawn,
        "inconclusive_cells": [
            cell.id for cell in cells if cell.experiment == "E1" and not cell.conclusive
        ],
    }


def _h2(
    cells: Sequence[_Cell],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    margin: float,
    level: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    withdrawn: list[dict[str, Any]] = []
    for cell in cells:
        if cell.experiment != "E1":
            continue
        for control, arm in cell.h2_pairs:
            status, reasons = _pair_status(cell, control, arm)
            row: dict[str, Any] = {
                "cell": cell.id,
                "domain": cell.domain,
                "name": f"{control} - {arm}",
                "control": control,
                "arm": arm,
                "status": status,
                "reasons": reasons,
            }
            if status == "withdrawn":
                withdrawn.append(row)
                records.append(row)
                continue
            if status != "tested":
                records.append(row)
                continue
            stats = _contrast_stats(
                cell,
                control,
                arm,
                n_boot=n_boot,
                seed=seed,
                margin=margin,
                level=level,
                label="h2",
            )
            row.update(_drop_draws(stats))
            row["holm_id"] = f"{cell.id} | {row['name']}"
            records.append(row)
    tested = [row for row in records if row["status"] == "tested"]
    if tested:
        pvalues = {row["holm_id"]: row["p_tost"] for row in tested}
        adjusted = holm_adjusted(pvalues)
        rejected = holm(pvalues, alpha)
        for row in tested:
            row["holm_adjusted_p"] = adjusted[row["holm_id"]]
            row["rejects"] = bool(rejected[row["holm_id"]])
    holds = bool(tested) and all(row.get("rejects") for row in tested)
    return {
        "rule": _H2_RULE,
        "decision": _decision(holds, tested),
        "holds": holds if tested else False,
        "alpha": alpha,
        "margin": margin,
        "n_family": len(tested),
        "contrasts": records,
        "withdrawn": withdrawn,
    }


def _h1_pooled(
    cells: Sequence[_Cell], *, n_boot: int, seed: int, alpha: float, level: float
) -> dict[str, Any]:
    primary = [
        _pooled_kit(
            cells, control, arm, n_boot=n_boot, seed=seed, level=level, conclusive_only=True
        )
        for control, arm in _KIT_H1
    ]
    sensitivity = [
        _pooled_kit(
            cells, control, arm, n_boot=n_boot, seed=seed, level=level, conclusive_only=False
        )
        for control, arm in _KIT_H1
    ]
    _apply_mean_holm(primary, alpha)
    _apply_mean_holm(sensitivity, alpha)
    tested = [row for row in primary if row["status"] == "tested"]
    holds = any(row.get("rejects") for row in tested)
    withdrawn = [item for row in primary for item in row["withdrawn"]]
    return {
        "rule": _H1_POOLED_RULE,
        "decision": _decision(holds, tested),
        "holds": holds if tested else False,
        "alpha": alpha,
        "contrasts": primary,
        "withdrawn": withdrawn,
        "sensitivity_including_inconclusive": {
            "contrasts": sensitivity,
            "holds": any(row.get("rejects") for row in sensitivity if row["status"] == "tested"),
            "decision": _decision(
                any(row.get("rejects") for row in sensitivity if row["status"] == "tested"),
                [row for row in sensitivity if row["status"] == "tested"],
            ),
        },
    }


def _pooled_kit(
    cells: Sequence[_Cell],
    control: str,
    arm: str,
    *,
    n_boot: int,
    seed: int,
    level: float,
    conclusive_only: bool,
) -> dict[str, Any]:
    name = f"{control} - {arm}"
    withdrawn: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    by_domain: dict[str, list[_Cell]] = {}
    for cell in cells:
        if cell.experiment != "E1":
            continue
        status, reasons = _pair_status(cell, control, arm)
        if status == "withdrawn":
            withdrawn.append(
                {"cell": cell.id, "domain": cell.domain, "contrast": name, "reasons": reasons}
            )
            continue
        if status != "tested":
            continue
        if conclusive_only and not cell.conclusive:
            excluded.append(
                {
                    "cell": cell.id,
                    "domain": cell.domain,
                    "oracle_gain": cell.oracle_gain.get("value"),
                    "reason": cell.oracle_gain.get("reason"),
                }
            )
            continue
        by_domain.setdefault(cell.domain, []).append(cell)
    if not by_domain:
        return {
            "name": name,
            "control": control,
            "arm": arm,
            "status": "not_tested",
            "withdrawn": withdrawn,
            "excluded_inconclusive": excluded,
            "domains": [],
        }
    domain_rows = []
    domain_draws = []
    domain_points = []
    for domain, members in by_domain.items():
        cell_draws = []
        cell_points = []
        cell_rows = []
        for cell in members:
            stats = _contrast_stats(
                cell,
                control,
                arm,
                n_boot=n_boot,
                seed=seed,
                margin=0.0,
                level=level,
                label="h1-pooled" if conclusive_only else "h1-pooled-sensitivity",
            )
            cell_draws.append(stats["draws"])
            cell_points.append(stats["point"])
            cell_rows.append(
                {
                    "cell": cell.id,
                    "point": stats["point"],
                    "per_seed": stats["per_seed"],
                    "conclusive": cell.conclusive,
                }
            )
        domain_draws.append(np.mean(np.stack(cell_draws, axis=0), axis=0))
        point = float(np.mean(cell_points))
        domain_points.append(point)
        domain_rows.append({"domain": domain, "point": point, "cells": cell_rows})
    draws = np.mean(np.stack(domain_draws, axis=0), axis=0)
    point = float(np.mean(domain_points))
    lo, hi = _interval(draws, level)
    return {
        "name": name,
        "control": control,
        "arm": arm,
        "status": "tested",
        "point": point,
        "lo": lo,
        "hi": hi,
        "p_le_0": float(np.mean(draws <= 0.0)),
        "n_domains": len(domain_rows),
        "domains": domain_rows,
        "withdrawn": withdrawn,
        "excluded_inconclusive": excluded,
    }


def _apply_mean_holm(rows: list[dict[str, Any]], alpha: float) -> None:
    tested = [row for row in rows if row["status"] == "tested"]
    if not tested:
        return
    pvalues = {row["name"]: row["p_le_0"] for row in tested}
    adjusted = holm_adjusted(pvalues)
    rejected = holm(pvalues, alpha)
    for row in tested:
        row["holm_adjusted_p"] = adjusted[row["name"]]
        row["rejects"] = bool(rejected[row["name"]])


def _scan_h3(
    cells: Sequence[_Cell],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    conclusive: list[dict[str, Any]] = []
    everyone: list[dict[str, Any]] = []
    withdrawn: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for cell in cells:
        if cell.experiment not in _EXPERIMENTS:
            continue
        if not cell.conclusive:
            excluded.append(
                {
                    "cell": cell.id,
                    "experiment": cell.experiment,
                    "domain": cell.domain,
                    "oracle_gain": cell.oracle_gain.get("value"),
                    "reason": cell.oracle_gain.get("reason"),
                }
            )
        for arm, control in cell.h3_pairs:
            status, reasons = _pair_status(cell, arm, control)
            info = {
                "cell": cell.id,
                "experiment": cell.experiment,
                "domain": cell.domain,
                "model": cell.model,
                "params_b": cell.params_b,
                "stage": cell.stage,
                "arm": arm,
                "control": control,
                "arm_class": cell.arm_class.get(arm),
                "is_natural": bool(cell.arm_natural.get(arm, False)),
                "status": status,
                "reasons": reasons,
                "conclusive": cell.conclusive,
            }
            if status == "withdrawn":
                withdrawn.append(info)
                continue
            if status != "tested":
                continue
            built = _h3_outcome(cell, info)
            everyone.append(built)
            if cell.conclusive:
                conclusive.append(built)
    return conclusive, everyone, withdrawn, excluded


def _h3_outcome(cell: _Cell, info: dict[str, Any]) -> dict[str, Any]:
    arm, control = info["arm"], info["control"]
    if arm not in cell.forecasts or control not in cell.forecasts:
        raise ValueError(f"{cell.id}: H3 pair {arm}/{control} is missing a forecast")
    tau_arm = _tau(cell.forecasts[arm])
    tau_control = _tau(cell.forecasts[control])
    seeds = _seeds_for(cell, arm)
    arm_maps = [_prompt_map(cell.runs[arm][seed], cell.final_step) for seed in seeds]
    control_maps = [_prompt_map(cell.runs[control][seed], cell.final_step) for seed in seeds]
    pids = sorted(arm_maps[0])
    if any(set(item) != set(pids) for item in arm_maps + control_maps):
        raise ValueError(f"{cell.id}: H3 pair {arm}/{control} does not share prompts")
    arm_cube = np.array([[item[pid] for pid in pids] for item in arm_maps], dtype=float)
    control_cube = np.array([[item[pid] for pid in pids] for item in control_maps], dtype=float)
    info = dict(info)
    info.update(
        {
            "dtau": tau_arm - tau_control,
            "tau_arm": tau_arm,
            "tau_control": tau_control,
            "seeds": seeds,
            "pids": pids,
            "arm_cube": arm_cube,
            "control_cube": control_cube,
            "scalars_arm": _baseline_scalars(cell, arm),
            "scalars_control": _baseline_scalars(cell, control),
            "forecast_arm": cell.forecasts[arm],
            "forecast_control": cell.forecasts[control],
            "_root": cell.root,
        }
    )
    return info


def _prompt_map(path: Path, step: int) -> dict[str, float]:
    _mean, acc = _mean_at(path, step)
    return acc


def _tau(forecast: Mapping[str, Any]) -> float:
    value = forecast_row(forecast)["tau_pool"]
    if value is None or not math.isfinite(float(value)):
        raise ValueError("H3 pair is missing a finite tau_pool")
    return float(value)


def _h3_statistics(
    pairs: Sequence[dict[str, Any]],
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
    alpha: float,
    min_pairs: int,
) -> dict[str, Any]:
    if not pairs:
        return {
            "rule": _H3_RULE,
            "decision": "not_tested",
            "holds": False,
            "min_pairs": min_pairs,
            "n_pairs": 0,
            "rho": None,
            "permutation_p": None,
            "permutation_method": "undefined",
            "q05": None,
            "pairs": [],
            "baselines": {},
            "within_shared": {
                "rule": _WITHIN_SHARED_RULE,
                "n_pairs": 0,
                "rho": None,
                "permutation_p": None,
                "permutation_method": "undefined",
                "q05": None,
            },
        }
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_cell.setdefault(str(pair["cell"]), []).append(pair)
    points: list[np.ndarray] = []
    boots: list[np.ndarray] = []
    ordered: list[dict[str, Any]] = []
    for cell_id, group in by_cell.items():
        point, boot = crossed_pair_dacc(
            [pair["arm_cube"] for pair in group],
            [pair["control_cube"] for pair in group],
            n_boot=n_boot,
            seed=_stream_seed(seed, "h3", cell_id),
            seed_lists=[pair["seeds"] for pair in group],
        )
        points.append(point)
        boots.append(boot)
        for pair, dacc_value in zip(group, point, strict=True):
            pair["dacc"] = float(dacc_value)
            ordered.append(pair)
    dtau = np.array([pair["dtau"] for pair in ordered], dtype=float)
    dacc = np.array([pair["dacc"] for pair in ordered], dtype=float)
    boot_mat = np.concatenate(boots, axis=1) if boots else np.zeros((n_boot, 0))
    labels = [str(pair["cell"]) for pair in ordered]
    perm = within_cell_spearman_p(dtau, dacc, labels, n_perm=n_perm, seed=seed)
    q05 = _rho_q05(dtau, boot_mat)
    p_value = perm.get("p")
    p_ok = isinstance(p_value, float) and math.isfinite(p_value) and p_value < alpha
    q_ok = q05 is not None and q05 > 0.0
    holds = len(ordered) >= min_pairs and p_ok and q_ok
    public = [_public_pair(pair) for pair in ordered]
    return {
        "rule": _H3_RULE,
        "decision": _decision(holds, ordered),
        "holds": holds,
        "min_pairs": min_pairs,
        "n_pairs": len(ordered),
        "rho": perm.get("rho"),
        "permutation_p": p_value,
        "permutation_method": perm.get("method"),
        "n_perm": perm.get("n_perm"),
        "q05": q05,
        "pairs": public,
        "baselines": _h3_baselines(ordered, dtau, dacc, boot_mat),
        "within_shared": _within_shared(ordered, dtau, dacc, boot_mat, n_perm=n_perm, seed=seed),
        "_dtau": dtau,
        "_dacc": dacc,
        "_boots": boot_mat,
        "_labels": labels,
    }


def _public_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell": pair["cell"],
        "stage": pair.get("stage"),
        "experiment": pair["experiment"],
        "domain": pair["domain"],
        "model": pair["model"],
        "params_b": pair.get("params_b"),
        "arm": pair["arm"],
        "control": pair["control"],
        "arm_class": pair.get("arm_class"),
        "is_natural": bool(pair.get("is_natural")),
        "status": "tested",
        "dtau": pair["dtau"],
        "dacc": pair["dacc"],
        "seeds": list(pair["seeds"]),
        "conclusive": pair["conclusive"],
        "baselines": _pair_baseline_values(pair),
    }


def _pair_baseline_values(pair: Mapping[str, Any]) -> dict[str, float | None]:
    arm_scalars = pair.get("scalars_arm") if isinstance(pair.get("scalars_arm"), Mapping) else {}
    control_scalars = (
        pair.get("scalars_control") if isinstance(pair.get("scalars_control"), Mapping) else {}
    )
    out: dict[str, float | None] = {}
    for name in _BASELINES:
        left = arm_scalars.get(name) if isinstance(arm_scalars, Mapping) else None
        right = control_scalars.get(name) if isinstance(control_scalars, Mapping) else None
        if left is None or right is None:
            out[name] = None
        else:
            out[name] = float(left) - float(right)
    return out


def _within_shared(
    pairs: Sequence[Mapping[str, Any]],
    dtau: np.ndarray,
    dacc: np.ndarray,
    boots: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """H3 statistics on shared-behaviour arms and natural verifiers."""
    indexes = [
        i
        for i, pair in enumerate(pairs)
        if pair.get("arm_class") == -1 or pair.get("is_natural") is True
    ]
    empty: dict[str, Any] = {
        "rule": _WITHIN_SHARED_RULE,
        "n_pairs": len(indexes),
        "rho": None,
        "permutation_p": None,
        "permutation_method": "undefined",
        "q05": None,
    }
    if len(indexes) < 2:
        return empty
    idx = np.asarray(indexes, dtype=int)
    sub_boots = boots[:, idx] if isinstance(boots, np.ndarray) and boots.size else boots
    perm = within_cell_spearman_p(
        dtau[idx],
        dacc[idx],
        [str(pairs[i]["cell"]) for i in indexes],
        n_perm=n_perm,
        seed=seed,
    )
    return {
        "rule": _WITHIN_SHARED_RULE,
        "n_pairs": int(idx.size),
        "rho": perm.get("rho"),
        "permutation_p": perm.get("p"),
        "permutation_method": perm.get("method"),
        "q05": _rho_q05(dtau[idx], sub_boots if isinstance(sub_boots, np.ndarray) else boots),
    }


def _rho_q05(dtau: np.ndarray, boots: np.ndarray) -> float | None:
    if dtau.size < 2 or boots.size == 0:
        return None
    draws = [spearman(dtau, row) for row in boots]
    finite = np.asarray([value for value in draws if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.quantile(finite, 0.05))


def _fill_slices(h3: dict[str, Any], n_perm: int, seed: int) -> None:
    pairs = h3.get("pairs") or []
    dtau = h3.get("_dtau")
    dacc = h3.get("_dacc")
    boots = h3.get("_boots")
    if not isinstance(dtau, np.ndarray) or not isinstance(dacc, np.ndarray) or not pairs:
        for key in ("by_experiment", "by_domain", "by_model"):
            h3[key] = {}
        return
    boot_mat = boots if isinstance(boots, np.ndarray) else np.zeros((0, 0))
    for field_name, report_key in (
        ("experiment", "by_experiment"),
        ("domain", "by_domain"),
        ("model", "by_model"),
    ):
        groups: dict[str, list[int]] = {}
        for index, pair in enumerate(pairs):
            groups.setdefault(str(pair[field_name]), []).append(index)
        block = {}
        for name, indexes in groups.items():
            idx = np.asarray(indexes, dtype=int)
            sub_boots = boot_mat[:, idx] if boot_mat.size else boot_mat
            perm = within_cell_spearman_p(
                dtau[idx],
                dacc[idx],
                [pairs[i]["cell"] for i in indexes],
                n_perm=n_perm,
                seed=seed,
            )
            block[name] = {
                "n_pairs": int(idx.size),
                "rho": perm.get("rho"),
                "permutation_p": perm.get("p"),
                "permutation_method": perm.get("method"),
                "q05": _rho_q05(dtau[idx], sub_boots),
            }
        h3[report_key] = block


def _h3_baselines(
    pairs: Sequence[Mapping[str, Any]],
    dtau: np.ndarray,
    dacc: np.ndarray,
    boots: np.ndarray,
) -> dict[str, Any]:
    rho_tau = spearman(dtau, dacc) if dtau.size >= 2 else float("nan")
    tau_draws = [spearman(dtau, row) for row in boots] if boots.size else []
    out: dict[str, Any] = {}
    for name in _BASELINES:
        xs: list[float] = []
        ok = True
        for pair in pairs:
            arm_v = pair["scalars_arm"].get(name)
            control_v = pair["scalars_control"].get(name)
            if arm_v is None or control_v is None:
                ok = False
                break
            xs.append(float(arm_v) - float(control_v))
        if not ok or dtau.size < 2:
            out[name] = {
                "rho": None,
                "reason": "baseline missing on a tested pair or fewer than 2 pairs",
            }
            continue
        x = np.asarray(xs, dtype=float)
        rho_b = spearman(x, dacc)
        diffs = []
        if tau_draws and boots.size:
            for row, rho_tau_b in zip(boots, tau_draws, strict=False):
                rho_draw = spearman(x, row)
                if math.isfinite(rho_draw) and math.isfinite(rho_tau_b):
                    diffs.append(rho_tau_b - rho_draw)
        diff_lo = diff_hi = None
        if diffs:
            diff_lo, diff_hi = (float(v) for v in np.quantile(np.asarray(diffs), [0.05, 0.95]))
        out[name] = {
            "rho": None if not math.isfinite(rho_b) else float(rho_b),
            "rho_tau_minus_baseline": None
            if not (math.isfinite(rho_tau) and math.isfinite(rho_b))
            else float(rho_tau - rho_b),
            "diff_q05": diff_lo,
            "diff_q95": diff_hi,
        }
    return out


def _baseline_scalars(cell: _Cell, arm: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {name: None for name in _BASELINES}
    forecast = cell.forecasts.get(arm)
    if forecast is not None:
        parsed = forecast_row(forecast)
        for key in (
            "J",
            "mean_precision",
            "modal_precision",
            "cos_gV_gO",
            "tau_pool_std",
            "behaviour_alignment",
        ):
            out[key] = _finite(parsed.get(key))
        out["probe_heldout_auc"] = _probe_auc(parsed.get("probe_heldout_auc"))
    out["early_acc"] = _early_acc(cell, arm)
    if arm in cell.arm_class:
        out["class"] = float(cell.arm_class[arm])
    return out


def _probe_auc(value: Any) -> float | None:
    if isinstance(value, Mapping):
        numbers = [_finite(item) for item in value.values()]
        if not numbers or any(item is None for item in numbers):
            return None
        return float(np.mean([item for item in numbers if item is not None]))
    return _finite(value)


def _early_acc(cell: _Cell, arm: str) -> float | None:
    early = cell.analysis.get("early") or {}
    if isinstance(early, Mapping) and early.get("enabled") is False:
        return None
    step = int(early.get("steps", 25)) if isinstance(early, Mapping) else 25
    seed = int(early.get("seed", 99)) if isinstance(early, Mapping) else 99
    path = cell.root / "early" / f"{arm}__s{seed}"
    if not (path / "eval_prompts.jsonl").exists():
        return None
    mean, _acc = _mean_at(path, step)
    return mean


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _h7(
    cells: Sequence[_Cell],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    margin: float,
    level: float,
) -> dict[str, Any]:
    domains: dict[str, list[dict[str, Any]]] = {}
    withdrawn: list[dict[str, Any]] = []
    for cell in cells:
        if cell.natural is None:
            continue
        blocks = natural_pair_blocks(
            cell.natural,
            forecasts=cell.forecasts,
            runs=cell.runs,
            planned=cell.planned,
            withdrawn=set(cell.withdrawn),
            allow_partial=True,
            final_step=cell.final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=_stream_seed(seed, "h7", cell.id),
            alpha=alpha,
            inconclusive=not cell.conclusive,
        )
        for block in blocks:
            wrapped = {
                "cell": cell.id,
                "stage": cell.stage,
                "experiment": cell.experiment,
                "domain": cell.domain,
                **block,
                "step0": _natural_step0(cell, block.get("natural"), block.get("control")),
            }
            domains.setdefault(cell.domain, []).append(wrapped)
            contrast = block.get("contrast") or {}
            if contrast.get("status") == "withdrawn":
                withdrawn.append(
                    {
                        "cell": cell.id,
                        "arm": block.get("natural"),
                        "control": block.get("control"),
                        "reasons": ["withdrawn"],
                    }
                )
    domain_rows = {}
    for domain, blocks in domains.items():
        primary = [block for block in blocks if block.get("experiment") == "E1"] or blocks
        domain_rows[domain] = {
            "decision": _natural_decision(domain, primary),
            "pairs": blocks,
        }
    decision = (
        "not_tested"
        if not domain_rows
        else "; ".join(str(row["decision"]) for row in domain_rows.values())
    )
    return {
        "rule": _H7_RULE,
        "decision": decision,
        "domains": domain_rows,
        "withdrawn": withdrawn,
    }


def _natural_decision(domain: str, blocks: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for block in blocks:
        differs = (
            (block.get("differs") or {}).get("decision")
            if isinstance(block.get("differs"), Mapping)
            else None
        )
        locality = (
            (block.get("locality") or {}).get("decision")
            if isinstance(block.get("locality"), Mapping)
            else None
        )
        screen = (
            (block.get("screen_direction") or {}).get("decision")
            if isinstance(block.get("screen_direction"), Mapping)
            else None
        )
        parts.append(
            f"{domain} {block.get('natural')}: differs={differs}, locality={locality}, screen={screen}"
        )
    return "; ".join(parts)


def _h8(
    cells: Sequence[_Cell], *, n_boot: int, seed: int, alpha: float, level: float
) -> dict[str, Any]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    withdrawn: list[dict[str, Any]] = []
    draws_list: list[np.ndarray] = []
    points: list[float] = []
    for cell in cells:
        if not _h8_candidate(cell):
            continue
        if cell.patch.get("same_selection") is True or (
            "patch_rate" in cell.withdrawn and "same_selection" in cell.withdrawn["patch_rate"]
        ):
            excluded.append({"cell": cell.id, "domain": cell.domain, "reason": "same_selection"})
            withdrawn.append(
                {
                    "cell": cell.id,
                    "arm": "patch_rate",
                    "reasons": list(cell.withdrawn.get("patch_rate") or ["same_selection"]),
                }
            )
            continue
        status, reasons = _pair_status(cell, "patch_tau", "patch_rate")
        if status == "withdrawn":
            excluded.append(
                {"cell": cell.id, "domain": cell.domain, "reason": "withdrawn", "reasons": reasons}
            )
            withdrawn.append(
                {"cell": cell.id, "contrast": "patch_tau - patch_rate", "reasons": reasons}
            )
            continue
        if status != "tested":
            excluded.append(
                {"cell": cell.id, "domain": cell.domain, "reason": status or "missing_arm"}
            )
            continue
        stats = _contrast_stats(
            cell,
            "patch_tau",
            "patch_rate",
            n_boot=n_boot,
            seed=seed,
            margin=0.0,
            level=level,
            label="h8",
        )
        draws_list.append(stats["draws"])
        points.append(stats["point"])
        included.append(
            {
                "cell": cell.id,
                "stage": cell.stage,
                "domain": cell.domain,
                "point": stats["point"],
                "per_seed": stats["per_seed"],
            }
        )
    if not draws_list:
        return {
            "rule": _H8_RULE,
            "decision": "not_tested",
            "holds": False,
            "alpha": alpha,
            "point": None,
            "p_le_0": None,
            "cells": [],
            "excluded": excluded,
            "withdrawn": withdrawn,
        }
    # One weight per domain: average cells inside a domain, then average domains.
    by_domain: dict[str, list[int]] = {}
    for index, row in enumerate(included):
        by_domain.setdefault(str(row["domain"]), []).append(index)
    domain_draws = []
    domain_points = []
    for indexes in by_domain.values():
        domain_draws.append(np.mean(np.stack([draws_list[i] for i in indexes], axis=0), axis=0))
        domain_points.append(float(np.mean([points[i] for i in indexes])))
    draws = np.mean(np.stack(domain_draws, axis=0), axis=0)
    point = float(np.mean(domain_points))
    p_le_0 = float(np.mean(draws <= 0.0))
    lo, hi = _interval(draws, level)
    holds = p_le_0 <= alpha
    return {
        "rule": _H8_RULE,
        "decision": "holds" if holds else "does_not_hold",
        "holds": holds,
        "alpha": alpha,
        "point": point,
        "lo": lo,
        "hi": hi,
        "p_le_0": p_le_0,
        "n_domains": len(by_domain),
        "cells": included,
        "excluded": excluded,
        "withdrawn": withdrawn,
    }


def _h8_candidate(cell: _Cell) -> bool:
    if cell.experiment not in {"E1", "E4"}:
        return False
    if cell.domain in _PATCH_DOMAINS or cell.patch:
        return True
    return "patch_tau" in cell.runs or "patch_rate" in cell.runs


def _h9(pairs: Sequence[dict[str, Any]], h3_rho: Any) -> dict[str, Any]:
    """Oracle-light Spearman from stored draws only.

    For each budget m, and each draw index d shared by the stored lists:
    ``dtau_d(pair) = draw_d(arm) - draw_d(control)`` and
    ``rho_d = Spearman(dtau_d, dacc)`` on the H3 tested pairs. ``dacc`` is the
    H3 point, not a bootstrap. The budget reports the median and 5th
    percentile of ``rho_d`` and the fraction of indexes with ``rho_d > 0``.

    An entry is exact when ``sd`` is 0 or ``budget >= oracle_labels_needed``.
    Exact entries may omit ``draws``; the locked mean is then the single
    draw. A non-exact entry without a finite ``draws`` list, or draw lists
    whose lengths disagree, makes the whole budget ``missing``. No normal
    fill-in and no recomputation from ``audit.npz``.
    """
    budgets = {str(budget): _h9_budget(pairs, budget, h3_rho) for budget in H9_BUDGETS}
    return {
        "rule": _H9_RULE,
        "decision": "reported",
        "budgets": budgets,
        "withdrawn": [],
        "h3_rho": h3_rho if isinstance(h3_rho, float) and math.isfinite(h3_rho) else None,
    }


def _h9_budget(pairs: Sequence[dict[str, Any]], budget: int, h3_rho: Any) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "status": "not_tested",
        "median_rho": None,
        "q05": None,
        "frac_positive": None,
        "n_draws": 0,
        "n_pairs": len(pairs),
        "missing": [],
        "matches_h3": False,
    }
    if len(pairs) < 2:
        return empty
    stored: dict[tuple[str, str], tuple[bool, float | None, np.ndarray | None]] = {}
    missing: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        for arm in (str(pair["arm"]), str(pair["control"])):
            key = (str(pair["cell"]), arm)
            if key in stored or key in seen:
                continue
            seen.add(key)
            parsed = _stored_budget(pair, arm, budget)
            if parsed is None:
                missing.append({"cell": key[0], "arm": key[1]})
            else:
                stored[key] = parsed
    if missing:
        return {
            "status": "missing",
            "median_rho": None,
            "q05": None,
            "frac_positive": None,
            "n_draws": 0,
            "n_pairs": len(pairs),
            "missing": missing,
            "matches_h3": False,
        }
    lengths = {int(values.size) for _exact, _mean, values in stored.values() if values is not None}
    if len(lengths) > 1:
        bad = [
            {"cell": cell, "arm": arm}
            for (cell, arm), (_exact, _mean, values) in stored.items()
            if values is not None
        ]
        return {
            "status": "missing",
            "median_rho": None,
            "q05": None,
            "frac_positive": None,
            "n_draws": 0,
            "n_pairs": len(pairs),
            "missing": bad,
            "matches_h3": False,
        }
    width = next(iter(lengths)) if lengths else 1
    columns: list[np.ndarray] = []
    dacc: list[float] = []
    all_exact = True
    for pair in pairs:
        arm_row = stored[(str(pair["cell"]), str(pair["arm"]))]
        control_row = stored[(str(pair["cell"]), str(pair["control"]))]
        all_exact = all_exact and arm_row[0] and control_row[0]
        columns.append(_draw_series(arm_row, width) - _draw_series(control_row, width))
        dacc.append(float(pair["dacc"]))
    mat = np.stack(columns, axis=1)
    observed = np.asarray(dacc, dtype=float)
    rhos = np.asarray([spearman(mat[i], observed) for i in range(width)], dtype=float)
    finite = rhos[np.isfinite(rhos)]
    median = None if finite.size == 0 else float(np.median(finite))
    q05 = None if finite.size == 0 else float(np.quantile(finite, 0.05))
    frac = float(np.mean(np.isfinite(rhos) & (rhos > 0.0)))
    h3_value = h3_rho if isinstance(h3_rho, float) and math.isfinite(h3_rho) else None
    matches = (
        all_exact
        and median is not None
        and q05 is not None
        and h3_value is not None
        and abs(median - h3_value) <= 1e-8
        and abs(q05 - h3_value) <= 1e-8
    )
    return {
        "status": "exact" if all_exact else "draws",
        "median_rho": median,
        "q05": q05,
        "frac_positive": frac,
        "n_draws": int(width),
        "n_pairs": len(pairs),
        "missing": [],
        "matches_h3": matches,
    }


def _stored_budget(
    pair: Mapping[str, Any], arm: str, budget: int
) -> tuple[bool, float | None, np.ndarray | None] | None:
    """``(exact, mean, draws)``. ``None`` when this arm cannot be used.

    ``draws`` is ``None`` only for an exact entry whose list was omitted.
    """
    stored = pair.get("forecast_arm") if arm == pair.get("arm") else pair.get("forecast_control")
    if not isinstance(stored, Mapping):
        return None
    budget_map = stored.get("tau_pool_budget")
    block = budget_map.get(str(budget)) if isinstance(budget_map, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    mean = _finite(block.get("mean"))
    sd = _finite(block.get("sd"))
    needed = _finite(block.get("oracle_labels_needed"))
    stated = _finite(block.get("budget"))
    budget_n = budget if stated is None else stated
    exact = (sd == 0.0) or (needed is not None and needed > 0 and budget_n >= needed)
    values = _finite_draws(block.get("draws"))
    if values is not None:
        return exact, mean, values
    if exact and mean is not None:
        return True, mean, None
    return None


def _finite_draws(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) == 0:
        return None
    try:
        values = np.asarray([float(item) for item in raw], dtype=float)
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return values


def _draw_series(parsed: tuple[bool, float | None, np.ndarray | None], width: int) -> np.ndarray:
    _exact, mean, values = parsed
    if values is not None:
        return values
    return np.full(width, float(mean if mean is not None else np.nan))


def _exploratory(
    cells: Sequence[_Cell],
    runs_root: Path,
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    margin: float,
    level: float,
) -> dict[str, Any]:
    """Variants, experiment X, reported arms, and X1 transfer.

    Nothing here is added to an H1 to H9 contrast list or to a Holm family.
    """
    return {
        "rule": _EXPLORATORY_RULE,
        "decision": "reported",
        "run_naming": {
            "base": "runs/<cell>/runs/<arm>__s<seed>",
            "early": "runs/<cell>/early/<arm>__s<seed>",
            "withdrawals": "runs/<cell>/withdrawals.json",
            "variant": "runs/<cell>/variants/<name>/<arm>__s<seed>",
            "transfer": "runs/<cell>/transfer/<arm>__s<seed>/<target_domain>.json",
        },
        "variants": _exploratory_variants(cells, n_boot=n_boot, seed=seed, level=level),
        "experiment_x": _exploratory_x(cells, n_boot=n_boot, seed=seed, level=level),
        "reported_arms": _reported_arms(
            cells, n_boot=n_boot, seed=seed, alpha=alpha, margin=margin, level=level
        ),
        "transfer": transfer_report(runs_root, n_boot=n_boot, seed=seed, alpha=alpha),
        "withdrawn": [],
    }


def _exploratory_variants(
    cells: Sequence[_Cell], *, n_boot: int, seed: int, level: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        names: list[str] = []
        seen: set[str] = set()
        for spec in cell.variant_specs:
            name = str(spec.get("name"))
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        for name in sorted(cell.variant_runs):
            if name not in seen:
                names.append(name)
        for name in names:
            spec = next(
                (item for item in cell.variant_specs if str(item.get("name")) == name),
                {},
            )
            runs = cell.variant_runs.get(name, {})
            overrides = spec.get("train_overrides")
            rows.append(
                {
                    "cell": cell.id,
                    "experiment": cell.experiment,
                    "domain": cell.domain,
                    "name": name,
                    "status": "reported" if runs else "missing_runs",
                    "arms_planned": list(spec.get("arms") or []),
                    "seeds_planned": list(spec.get("seeds") or []),
                    "train_overrides": dict(overrides) if isinstance(overrides, Mapping) else {},
                    "accelerate_config": spec.get("accelerate_config"),
                    "placement": spec.get("placement"),
                    "arms": _variant_arm_rows(cell, runs),
                    "contrasts": _variant_contrasts(
                        cell, runs, n_boot=n_boot, seed=seed, level=level, variant=name
                    ),
                }
            )
    return rows


def _variant_arm_rows(cell: _Cell, runs: Mapping[str, Mapping[int, Path]]) -> list[dict[str, Any]]:
    rows = []
    for arm in sorted(runs):
        seeds = []
        for seed_i in sorted(runs[arm]):
            final, _acc = _mean_at(runs[arm][seed_i], cell.final_step)
            seeds.append({"seed": int(seed_i), "final": final})
        rows.append({"arm": arm, "seeds": seeds})
    return rows


def _variant_contrasts(
    cell: _Cell,
    runs: Mapping[str, Mapping[int, Path]],
    *,
    n_boot: int,
    seed: int,
    level: float,
    variant: str,
) -> list[dict[str, Any]]:
    rows = []
    for control, arm in cell.h1_pairs:
        if control not in runs or arm not in runs:
            continue
        common = sorted(set(runs[control]) & set(runs[arm]))
        name = f"{control} - {arm}"
        base_status, _reasons = _pair_status(cell, control, arm)
        base_point = None
        if base_status == "tested":
            base_point = _exploratory_contrast(
                cell.runs,
                _seeds_for(cell, control),
                control,
                arm,
                step=cell.final_step,
                n_boot=n_boot,
                seed=seed,
                level=level,
                label=f"exploratory-base-{variant}",
                cell_id=cell.id,
            )["point"]
        if not common:
            rows.append(
                {
                    "name": name,
                    "control": control,
                    "arm": arm,
                    "status": "not_in_study",
                    "base_point": base_point,
                    "base_status": base_status,
                    "sign_agrees": None,
                }
            )
            continue
        stats = _exploratory_contrast(
            runs,
            common,
            control,
            arm,
            step=cell.final_step,
            n_boot=n_boot,
            seed=seed,
            level=level,
            label=f"exploratory-variant-{variant}",
            cell_id=cell.id,
        )
        rows.append(
            {
                "name": name,
                "control": control,
                "arm": arm,
                "status": "tested",
                **stats,
                "base_point": base_point,
                "base_status": base_status,
                "sign_agrees": _sign_agrees(stats["point"], base_point),
            }
        )
    return rows


def _exploratory_x(
    cells: Sequence[_Cell], *, n_boot: int, seed: int, level: float
) -> list[dict[str, Any]]:
    e1 = [cell for cell in cells if cell.experiment == "E1"]
    rows = []
    for cell in cells:
        if cell.experiment != "X":
            continue
        contrasts = []
        for control, arm in cell.h1_pairs:
            status, reasons = _pair_status(cell, control, arm)
            name = f"{control} - {arm}"
            row: dict[str, Any] = {
                "name": name,
                "control": control,
                "arm": arm,
                "status": status,
                "reasons": reasons,
            }
            point = None
            if status == "tested":
                stats = _exploratory_contrast(
                    cell.runs,
                    _seeds_for(cell, control),
                    control,
                    arm,
                    step=cell.final_step,
                    n_boot=n_boot,
                    seed=seed,
                    level=level,
                    label="exploratory-x",
                    cell_id=cell.id,
                )
                row.update(stats)
                point = stats["point"]
            references: list[dict[str, Any]] = []
            for other in e1:
                if other.domain != cell.domain:
                    continue
                other_status, _other_reasons = _pair_status(other, control, arm)
                if other_status != "tested":
                    references.append(
                        {
                            "cell": other.id,
                            "status": other_status,
                            "point": None,
                            "sign_agrees": None,
                        }
                    )
                    continue
                other_point = _exploratory_contrast(
                    other.runs,
                    _seeds_for(other, control),
                    control,
                    arm,
                    step=other.final_step,
                    n_boot=n_boot,
                    seed=seed,
                    level=level,
                    label="exploratory-x-e1",
                    cell_id=other.id,
                )["point"]
                references.append(
                    {
                        "cell": other.id,
                        "status": "tested",
                        "point": other_point,
                        "sign_agrees": _sign_agrees(point, other_point),
                    }
                )
            agreed = [item["sign_agrees"] for item in references if item["sign_agrees"] is not None]
            row["e1"] = references
            row["sign_agrees"] = None if not agreed else all(agreed)
            contrasts.append(row)
        rows.append(
            {
                "cell": cell.id,
                "domain": cell.domain,
                "model": cell.model,
                "arms": _variant_arm_rows(cell, cell.runs),
                "contrasts": contrasts,
            }
        )
    return rows


def _reported_arms(
    cells: Sequence[_Cell],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    margin: float,
    level: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        for spec in _reported_specs(cell):
            if spec["kind"] == "random":
                rows.append(_reported_random(cell, spec["arm"]))
            elif spec["kind"] == "natural":
                rows.append(
                    _reported_natural(
                        cell,
                        spec["arm"],
                        spec["control"],
                        n_boot=n_boot,
                        seed=seed,
                        alpha=alpha,
                        margin=margin,
                        level=level,
                    )
                )
    return rows


def _reported_specs(cell: _Cell) -> list[dict[str, str]]:
    if "reported" in cell.analysis:
        raw = cell.analysis.get("reported")
        specs: list[dict[str, str]] = []
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return specs
        for item in raw:
            if not isinstance(item, Mapping) or not item.get("arm"):
                continue
            arm = str(item["arm"])
            kind = str(item.get("kind") or _reported_kind(arm) or "")
            if kind not in {"random", "natural"}:
                continue
            control = ""
            if kind == "natural":
                control = str(item["control"]) if item.get("control") else _natural_control(arm)
            specs.append({"arm": arm, "kind": kind, "control": control})
        return specs
    found: list[dict[str, str]] = []
    if "random" in cell.runs or "random" in cell.planned:
        found.append({"arm": "random", "kind": "random", "control": ""})
    if "natural_robust" in cell.runs or "natural_robust" in cell.planned:
        found.append(
            {"arm": "natural_robust", "kind": "natural", "control": "noise_natural_robust"}
        )
    return found


def _reported_kind(arm: str) -> str | None:
    if arm == "random":
        return "random"
    if arm == "natural_robust":
        return "natural"
    return None


def _natural_control(arm: str) -> str:
    if arm == "natural_robust":
        return "noise_natural_robust"
    return f"noise_{arm}"


def _reported_random(cell: _Cell, arm: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cell": cell.id,
        "experiment": cell.experiment,
        "domain": cell.domain,
        "arm": arm,
        "kind": "random",
        "status": "reported",
        "seeds": [],
    }
    if arm not in cell.runs:
        row["status"] = "not_in_study"
        return row
    seeds = []
    for seed_i in _seeds_for(cell, arm):
        if seed_i not in cell.runs[arm]:
            continue
        final, _final_map = _mean_at(cell.runs[arm][seed_i], cell.final_step)
        step0, _step0_map = _mean_at(cell.runs[arm][seed_i], 0)
        oracle_final = None
        oracle_runs = cell.runs.get("oracle") or {}
        if seed_i in oracle_runs:
            oracle_final, _oracle_map = _mean_at(oracle_runs[seed_i], cell.final_step)
        seeds.append(
            {
                "seed": int(seed_i),
                "final": final,
                "step0": step0,
                "minus_step0": final - step0,
                "oracle_final": oracle_final,
                "minus_oracle": None if oracle_final is None else final - oracle_final,
            }
        )
    row["seeds"] = seeds
    return row


def _reported_natural(
    cell: _Cell,
    arm: str,
    control: str,
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    margin: float,
    level: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cell": cell.id,
        "experiment": cell.experiment,
        "domain": cell.domain,
        "arm": arm,
        "control": control,
        "kind": "natural",
        "status": "reported",
    }
    try:
        blocks = natural_pair_blocks(
            {"natural": arm, "control": control, "oracle": "oracle"},
            forecasts=cell.forecasts,
            runs=cell.runs,
            planned=cell.planned,
            withdrawn=set(cell.withdrawn),
            allow_partial=True,
            final_step=cell.final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=_stream_seed(seed, "reported", cell.id, arm),
            alpha=alpha,
            inconclusive=not cell.conclusive,
        )
    except ValueError as exc:
        row["status"] = "missing_forecast"
        row["reason"] = str(exc)
        return row
    row["block"] = blocks[0] if blocks else {}
    return row


def _exploratory_contrast(
    runs: Mapping[str, Mapping[int, Path]],
    seeds: Sequence[int],
    left: str,
    right: str,
    *,
    step: int,
    n_boot: int,
    seed: int,
    level: float,
    label: str,
    cell_id: str,
) -> dict[str, Any]:
    matrix, _pids = difference_matrix(
        [(runs[left][int(item)], runs[right][int(item)]) for item in seeds],
        step=step,
    )
    rng = np.random.default_rng(_stream_seed(seed, label, cell_id, left, right))
    draws = _crossed_mean_draws(matrix, n_boot, rng)
    lo, hi = _interval(draws, level)
    return {
        "point": float(matrix.mean()),
        "per_seed": [float(value) for value in matrix.mean(axis=1)],
        "lo": lo,
        "hi": hi,
        "n_seeds": int(matrix.shape[0]),
        "n_prompts": int(matrix.shape[1]),
    }


def _sign_agrees(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None or not math.isfinite(left) or not math.isfinite(right):
        return None
    return (left > 0.0) == (right > 0.0) and (left < 0.0) == (right < 0.0)


def _scale(cells: Sequence[_Cell], *, n_boot: int, seed: int, level: float) -> dict[str, Any]:
    contrasts = []
    missing_params: list[dict[str, Any]] = []
    domains = sorted(
        {cell.domain for cell in cells if _is_qwen3(cell.model) and cell.experiment != "X"}
    )
    for domain in domains:
        for control, arm in _KIT_H1:
            name = f"{control} - {arm}"
            points = []
            for cell in cells:
                if cell.experiment == "X" or cell.domain != domain or not _is_qwen3(cell.model):
                    continue
                status, _reasons = _pair_status(cell, control, arm)
                if status != "tested":
                    continue
                if cell.params_b is None:
                    missing_params.append({"cell": cell.id, "contrast": name})
                    continue
                matrix = _diff_matrix(cell, control, arm)
                points.append(
                    {
                        "cell": cell.id,
                        "model": cell.model,
                        "params_b": cell.params_b,
                        "log10_params": math.log10(cell.params_b),
                        "point": float(matrix.mean()),
                        "conclusive": cell.conclusive,
                    }
                )
            row: dict[str, Any] = {
                "domain": domain,
                "name": name,
                "control": control,
                "arm": arm,
                "n_cells": len(points),
                "cells": points,
            }
            if len(points) < 2:
                row["slope"] = None
                row["lo"] = None
                row["hi"] = None
                row["status"] = "not_tested"
            else:
                x = np.array([item["log10_params"] for item in points], dtype=float)
                y = np.array([item["point"] for item in points], dtype=float)
                slope, lo, hi = _slope_interval(
                    x, y, n_boot=n_boot, seed=_stream_seed(seed, "scale", domain, name), level=level
                )
                row.update({"slope": slope, "lo": lo, "hi": hi, "status": "tested"})
            contrasts.append(row)
    return {
        "rule": _SCALE_RULE,
        "decision": "reported",
        "contrasts": contrasts,
        "missing_params": missing_params,
        "withdrawn": [],
    }


def _is_qwen3(model: str) -> bool:
    return model.lower().replace("_", "-").startswith("qwen3")


def _slope_interval(
    x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int, level: float
) -> tuple[float | None, float | None, float | None]:
    point = ols_slope(x.tolist(), y.tolist())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n = x.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = ols_slope(x[idx].tolist(), y[idx].tolist())
    finite = draws[np.isfinite(draws)]
    if finite.size == 0 or not math.isfinite(point):
        return (None if not math.isfinite(point) else float(point)), None, None
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(finite, [tail, 1.0 - tail])
    return float(point), float(lo), float(hi)


def _conclusiveness(cells: Sequence[_Cell]) -> dict[str, Any]:
    rows = [
        {
            "cell": cell.id,
            "experiment": cell.experiment,
            "domain": cell.domain,
            "conclusive": cell.conclusive,
            "oracle_gain": cell.oracle_gain,
        }
        for cell in cells
    ]
    n_yes = sum(cell.conclusive for cell in cells)
    n_no = len(cells) - n_yes
    return {
        "rule": _CONCLUSIVE_RULE,
        "decision": f"{n_yes} conclusive, {n_no} inconclusive",
        "n_conclusive": n_yes,
        "n_inconclusive": n_no,
        "cells": rows,
        "withdrawn": [row for cell in cells for row in cell.withdrawn_rows],
    }


def _decision(holds: bool, tested: Sequence[Any]) -> str:
    if not tested:
        return "not_tested"
    return "holds" if holds else "does_not_hold"


def _fmt_withdrawn(rows: Sequence[Any]) -> str:
    if not rows:
        return "none"
    parts = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        reasons = ", ".join(str(item) for item in row.get("reasons") or []) or str(
            row.get("reason") or "withdrawn"
        )
        cell = row.get("cell")
        who = row.get("contrast") or row.get("name") or row.get("arm") or cell
        if cell and who != cell:
            parts.append(f"{cell} {who} ({reasons})")
        else:
            parts.append(f"{who} ({reasons})")
    return "; ".join(parts) if parts else "none"


def _append_block_body(lines: list[str], key: str, block: Mapping[str, Any]) -> None:
    if key in {"h1", "h2"}:
        for row in block.get("contrasts") or []:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"- {row.get('cell')} {row.get('name')}: status={row.get('status')}, "
                f"point={row.get('point')}, rejects={row.get('rejects')}, "
                f"p_sup={row.get('p_sup')}, p_tost={row.get('p_tost')}"
            )
    elif key == "h1_pooled":
        for row in block.get("contrasts") or []:
            if isinstance(row, Mapping):
                lines.append(
                    f"- {row.get('name')}: status={row.get('status')}, point={row.get('point')}, "
                    f"p_le_0={row.get('p_le_0')}, rejects={row.get('rejects')}"
                )
    elif key == "h3":
        lines.append(
            f"n_pairs={block.get('n_pairs')}, rho={block.get('rho')}, "
            f"permutation_p={block.get('permutation_p')}, q05={block.get('q05')}."
        )
        shared = block.get("within_shared")
        if isinstance(shared, Mapping):
            lines.append(
                f"Within shared: n_pairs={shared.get('n_pairs')}, rho={shared.get('rho')}, "
                f"permutation_p={shared.get('permutation_p')}, q05={shared.get('q05')}."
            )
        baselines = block.get("baselines")
        if isinstance(baselines, Mapping) and isinstance(baselines.get("class"), Mapping):
            klass = baselines["class"]
            lines.append(
                f"Class baseline: rho={klass.get('rho')}, "
                f"rho_tau_minus_baseline={klass.get('rho_tau_minus_baseline')}."
            )
        excluded = block.get("excluded_inconclusive") or []
        if excluded:
            names = ", ".join(str(row.get("cell")) for row in excluded if isinstance(row, Mapping))
            lines.append(f"Excluded inconclusive cells: {names}.")
    elif key == "h8":
        lines.append(
            f"point={block.get('point')}, p_le_0={block.get('p_le_0')}, "
            f"n_domains={block.get('n_domains')}."
        )
        excluded = block.get("excluded") or []
        if excluded:
            lines.append(
                "Excluded: "
                + "; ".join(
                    f"{row.get('cell')} ({row.get('reason')})"
                    for row in excluded
                    if isinstance(row, Mapping)
                )
            )
    elif key == "h10":
        for row in block.get("pairs") or []:
            if isinstance(row, Mapping):
                lines.append(
                    f"- {row.get('name')}: status={row.get('status')}, point={row.get('point')}, "
                    f"p_le_0={row.get('p_le_0')}, n_cells={row.get('n_cells')}, "
                    f"rejects={row.get('rejects')}"
                )
    elif key == "cost":
        for row in block.get("cells") or []:
            if isinstance(row, Mapping):
                lines.append(
                    f"- {row.get('cell')}: audit_gpu_hours={row.get('audit_gpu_hours')}, "
                    f"early_gpu_hours_per_arm={row.get('early_gpu_hours_per_arm')}, "
                    f"n_trained_arms={row.get('n_trained_arms')}, "
                    f"break_even_arms={row.get('break_even_arms')}"
                )
    elif key == "h9":
        budgets = block.get("budgets") or {}
        if isinstance(budgets, Mapping):
            for name, row in budgets.items():
                if isinstance(row, Mapping):
                    lines.append(
                        f"- m={name}: status={row.get('status')}, "
                        f"median_rho={row.get('median_rho')}, q05={row.get('q05')}, "
                        f"frac_positive={row.get('frac_positive')}, "
                        f"matches_h3={row.get('matches_h3')}"
                    )
    elif key == "scale":
        for row in block.get("contrasts") or []:
            if isinstance(row, Mapping):
                lines.append(
                    f"- {row.get('domain')} {row.get('name')}: slope={row.get('slope')}, "
                    f"interval=[{row.get('lo')}, {row.get('hi')}]"
                )
    elif key == "conclusiveness":
        lines.append(
            f"{block.get('n_conclusive')} conclusive, {block.get('n_inconclusive')} inconclusive."
        )
    elif key == "exploratory":
        naming = block.get("run_naming") or {}
        lines.append(f"Variant runs: {naming.get('variant')}.")
        for row in block.get("variants") or []:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"- variant {row.get('cell')} {row.get('name')}: status={row.get('status')}"
            )
            for contrast in row.get("contrasts") or []:
                if isinstance(contrast, Mapping):
                    lines.append(
                        f"  - {contrast.get('name')}: point={contrast.get('point')}, "
                        f"sign_agrees={contrast.get('sign_agrees')}"
                    )
        for row in block.get("experiment_x") or []:
            if not isinstance(row, Mapping):
                continue
            lines.append(f"- experiment X {row.get('cell')} ({row.get('domain')})")
            for contrast in row.get("contrasts") or []:
                if isinstance(contrast, Mapping):
                    lines.append(
                        f"  - {contrast.get('name')}: point={contrast.get('point')}, "
                        f"sign_agrees={contrast.get('sign_agrees')}"
                    )
        for row in block.get("reported_arms") or []:
            if isinstance(row, Mapping):
                lines.append(
                    f"- reported {row.get('cell')} {row.get('arm')}: "
                    f"kind={row.get('kind')}, status={row.get('status')}"
                )
        transfer = block.get("transfer") or {}
        if isinstance(transfer, Mapping):
            lines.append(f"Transfer rows: {transfer.get('n_rows')}.")


def _strip_private(block: dict[str, Any]) -> None:
    for key in ("_dtau", "_dacc", "_boots", "_labels"):
        block.pop(key, None)
