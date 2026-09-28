"""Write the macros, tables, and figures the manuscript inputs.

``read_program_report`` is the only reader of the pooled confirmatory report.
Key names are reconciled there. Outcome macros are ``\\pending`` until the
report marks that stage complete.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from vfates.program import (
    cell_stage,
    class_gpu_hours,
    load_program,
    plan_jobs,
    repo_root,
)
from vfates.utils.io import read_json, read_yaml

PENDING = r"\pending"
_FORBIDDEN = ("--", "\u2013", "\u2014", "\u2212")
_WORD = re.compile(r"\b(final|legacy|v2|old)\b", re.IGNORECASE)

# Confirmatory hypotheses, in abstract order. A failed one contributes no sentence.
_CONFIRMATORY = (
    "h1",
    "h1_pooled",
    "h2",
    "h3",
    "h10",
    "h7",
    "h8",
    "h3_replication",
)

_STAGE_A_KEYS = ("h1", "h1_pooled", "h2", "h3", "h7", "h8", "h10")
_H9_BUDGETS = (32, 64, 128, 192)

_BASELINE_ROWS = (
    ("J", "step-0 Youden J", "shared audit"),
    ("mean_precision", "mean precision", "shared audit"),
    ("modal_precision", "modal precision", "shared audit"),
    ("cos_gV_gO", r"$\cos(g_V, g_O)$", "shared audit"),
    ("probe_heldout_auc", "held-out probe AUC", "shared audit"),
    ("early_acc", "step-25 accuracy", "one 25-step run per verifier"),
    ("behaviour_alignment", r"Egashira et al.\ $f \times C$", "shared audit"),
    ("class", "shared-behaviour label", "rule label, no training"),
)

_BASELINE_ALIASES = {
    "J": ("J", "j", "youden_j"),
    "mean_precision": ("mean_precision", "mean_prec"),
    "modal_precision": ("modal_precision", "modal_prec"),
    "cos_gV_gO": ("cos_gV_gO", "cos_gv_go", "cos"),
    "probe_heldout_auc": ("probe_heldout_auc", "probe_auc", "auc"),
    "early_acc": ("early_acc", "step25", "step_25"),
    "behaviour_alignment": ("behaviour_alignment", "behavior_alignment", "f_times_c", "egashira"),
    "class": ("class", "structure_class", "class_label"),
}


@dataclass
class Hypothesis:
    key: str
    stage: str
    present: bool
    passed: bool
    values: dict[str, float | None] = field(default_factory=dict)


@dataclass
class PairRow:
    cell: str
    stage: str
    experiment: str
    domain: str
    model: str
    params_b: float | None
    arm: str
    control: str
    structure_class: str
    is_natural: bool
    dtau: float | None
    dacc: float | None
    baselines: dict[str, float | None]


@dataclass
class ProgramReport:
    """Pooled report after key names have been reconciled."""

    present: bool
    mismatches: list[str]
    stage_status: dict[str, str]
    hypotheses: dict[str, Hypothesis]
    pairs: list[PairRow]
    h9: dict[int, dict[str, float | None]]
    natural_rows: list[dict[str, Any]]
    h8_rows: list[dict[str, Any]]
    h10_rows: list[dict[str, Any]]
    cost_rows: list[dict[str, Any]]
    deviations: list[str]
    locks: list[tuple[str, str]]
    scale_rows: list[dict[str, Any]]
    baselines: dict[str, dict[str, float | None]]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def fmt_stat(value: float | None) -> str:
    """Four decimal places, or scientific notation below 1e-4."""
    if value is None or not math.isfinite(value):
        return PENDING
    ax = abs(value)
    if ax != 0.0 and ax < 1e-4:
        exp = math.floor(math.log10(ax))
        mant = value / (10.0**exp)
        return f"\\ensuremath{{{mant:.2f} \\times 10^{{{exp}}}}}"
    return f"{value:.4f}"


def fmt_hours(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return PENDING
    return f"{value:.2f}"


def fmt_count(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return PENDING
    if abs(value - round(value)) < 1e-8:
        return str(round(value))
    return fmt_stat(value)


def tex_text(value: object) -> str:
    """Escape a report string for LaTeX and strip dash characters."""
    text = str(value)
    text = text.replace("\u2013", " to ").replace("\u2014", ", ").replace("\u2212", "-")
    text = text.replace("---", ", ").replace("--", " to ")
    text = _WORD.sub(
        lambda match: {
            "final": "endpoint",
            "legacy": "earlier",
            "v2": "revision",
            "old": "earlier",
        }[match.group(0).lower()],
        text,
    )
    mapping = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = "".join(mapping.get(char, char) for char in text)
    return escaped.replace("-", r"-\allowbreak{}")


def _clean(text: str, label: str) -> str:
    for token in _FORBIDDEN:
        if token in text:
            raise ValueError(f"{label} contains a forbidden dash")
    if _WORD.search(text):
        raise ValueError(f"{label} contains a forbidden word")
    return text


def _first_float(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        if name in row:
            return _finite(row.get(name))
    return None


def _stage_block(doc: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    stages = doc.get("stages")
    if not isinstance(stages, Mapping):
        return None
    aliases = {
        "A": ("A", "a", "stage_a", "stageA"),
        "B": ("B", "b", "stage_b", "stageB"),
    }[key]
    for alias in aliases:
        block = _mapping(stages.get(alias))
        if block is not None:
            return block
    return None


def _stage_status(block: Mapping[str, Any] | None) -> str:
    if block is None:
        return "missing"
    if isinstance(block.get("complete"), bool):
        return "complete" if block["complete"] else "pending"
    status = str(block.get("status", "")).strip().lower().replace("-", "_")
    if status in {"complete", "done", "ready", "finished"}:
        return "complete"
    if status in {"pending", "incomplete", "missing", "not_started"}:
        return "pending"
    return "unmarked"


def _passed(block: Mapping[str, Any]) -> bool:
    if isinstance(block.get("holds"), bool):
        return bool(block["holds"])
    decision = str(block.get("decision", "")).strip().lower().replace(" ", "_").replace("-", "_")
    return decision in {"holds", "passed", "pass"}


def _hypothesis_block(stage: Mapping[str, Any] | None, *names: str) -> dict[str, Any] | None:
    if stage is None:
        return None
    for name in names:
        block = _mapping(stage.get(name))
        if block is not None:
            return block
    return None


def _contrast_stats(block: Mapping[str, Any], arm: str) -> tuple[float | None, float | None]:
    rows = block.get("contrasts")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = block.get("pairs")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None, None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("arm", ""))
        label = str(row.get("name", ""))
        if name == arm or arm in label.split():
            point = _first_float(row, ("point", "mean", "estimate", "aligned", "statistic"))
            p_value = _first_float(row, ("p_le_0", "p", "p_sup", "permutation_p"))
            return point, p_value
    return None, None


def _h3_values(block: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "rho": _first_float(block, ("rho", "spearman")),
        "p": _first_float(block, ("permutation_p", "p", "p_value")),
        "q05": _first_float(block, ("q05", "rho_q05")),
        "n": _first_float(block, ("n_pairs", "n")),
    }


def _baseline_block(h3: Mapping[str, Any] | None) -> dict[str, dict[str, float | None]]:
    raw = _mapping(h3.get("baselines")) if h3 else None
    out: dict[str, dict[str, float | None]] = {}
    if raw is None:
        return out
    for key, aliases in _BASELINE_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in raw:
                found = raw[alias]
                break
        block = _mapping(found)
        if block is None and _finite(found) is not None:
            block = {"rho": found}
        if block is None:
            continue
        out[key] = {
            "rho": _first_float(block, ("rho", "spearman")),
            "gap": _first_float(block, ("rho_tau_minus_baseline", "gap", "delta")),
        }
    return out


def _stratum_values(
    h3: Mapping[str, Any] | None, stage: Mapping[str, Any] | None
) -> dict[str, float | None]:
    candidates: list[Mapping[str, Any]] = []
    if h3 is not None:
        for key in ("within_shared", "shared_and_natural", "shared_natural"):
            block = _mapping(h3.get(key))
            if block is not None:
                candidates.append(block)
        strata = _mapping(h3.get("by_stratum")) or _mapping(h3.get("strata"))
        if strata is not None:
            for key in ("shared_and_natural", "shared_natural", "shared"):
                block = _mapping(strata.get(key))
                if block is not None:
                    candidates.append(block)
    if stage is not None:
        for key in ("h3_shared", "h3_shared_and_natural", "shared_and_natural"):
            block = _mapping(stage.get(key))
            if block is not None:
                candidates.append(block)
    if not candidates:
        return {}
    return _h3_values(candidates[0])


def _pair_from(row: Mapping[str, Any], *, default_stage: str) -> tuple[PairRow, set[str]]:
    missing: set[str] = set()

    def need(name: str, *aliases: str) -> str:
        for alias in (name, *aliases):
            if row.get(alias) not in (None, ""):
                return str(row[alias])
        missing.add(name)
        return ""

    arm = need("arm")
    control = need("control")
    natural_flag = row.get("is_natural", row.get("natural"))
    if isinstance(natural_flag, bool):
        is_natural = natural_flag
    else:
        is_natural = arm.startswith("natural")
        if "is_natural" not in row and "natural" not in row:
            missing.add("is_natural")
    structure = None
    for key in ("class", "arm_class", "structure_class", "class_label"):
        if key in row and row.get(key) is not None:
            structure = row.get(key)
            break
    if structure is None:
        missing.add("class")
        structure_text = ""
    elif isinstance(structure, (int, float)) and not isinstance(structure, bool):
        structure_text = fmt_count(float(structure))
    else:
        structure_text = str(structure)
    dtau = _first_float(row, ("dtau", "delta_tau", "delta_tau_pool"))
    dacc = _first_float(row, ("dacc", "delta_acc", "delta_accuracy"))
    if dtau is None:
        missing.add("dtau")
    if dacc is None:
        missing.add("dacc")
    baselines: dict[str, float | None] = {}
    raw_base = _mapping(row.get("baselines")) or {}
    for key, aliases in _BASELINE_ALIASES.items():
        baselines[key] = None
        for alias in aliases:
            if alias in raw_base:
                baselines[key] = _finite(raw_base.get(alias))
                break
            flat = row.get(f"baseline_{alias}")
            if flat is not None:
                baselines[key] = _finite(flat)
                break
    params = _first_float(row, ("params_b", "params"))
    return (
        PairRow(
            cell=need("cell", "cell_id"),
            stage=str(row.get("stage") or default_stage),
            experiment=str(row.get("experiment") or ""),
            domain=str(row.get("domain") or ""),
            model=str(row.get("model") or ""),
            params_b=params,
            arm=arm,
            control=control,
            structure_class=structure_text,
            is_natural=is_natural,
            dtau=dtau,
            dacc=dacc,
            baselines=baselines,
        ),
        missing,
    )


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _h9_from(block: Mapping[str, Any] | None) -> dict[int, dict[str, float | None]]:
    out: dict[int, dict[str, float | None]] = {
        budget: {"median": None, "q05": None, "frac": None} for budget in _H9_BUDGETS
    }
    if block is None:
        return out
    budgets = block.get("budgets")
    if isinstance(budgets, Mapping):
        items = [(str(key), value) for key, value in budgets.items()]
    elif isinstance(budgets, Sequence) and not isinstance(budgets, (str, bytes)):
        items = []
        for row in budgets:
            if isinstance(row, Mapping):
                items.append((str(row.get("budget", row.get("m", ""))), row))
    else:
        items = []
    for key, raw in items:
        number = _finite(key)
        if number is None or int(number) not in out or not isinstance(raw, Mapping):
            continue
        out[int(number)] = {
            "median": _first_float(raw, ("median_rho", "median", "rho")),
            "q05": _first_float(raw, ("q05", "rho_q05")),
            "frac": _first_float(raw, ("frac_positive", "frac", "fraction_positive")),
        }
    return out


def _natural_rate(row: Mapping[str, Any]) -> float | None:
    explicit = _first_float(row, ("error_rate", "step0_error_rate", "err"))
    if explicit is not None:
        return explicit
    fpr = _first_float(row, ("fpr", "FPR"))
    fnr = _first_float(row, ("fnr", "FNR"))
    if fpr is not None and fnr is not None:
        return fpr + fnr
    youden = _first_float(row, ("J", "j"))
    if youden is not None:
        return 1.0 - youden
    return None


def _natural_coords(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """Step-0 error rate, locked dtau, and endpoint dacc for one natural row.

    The published figure row stores the contrast as ``point`` (control minus
    natural) and the screen gap as ``screen_direction.dtau`` (the same sign).
    The paper's dtau and dacc are natural minus control, so both are negated.
    """
    rate = _natural_rate(row)
    dtau = _first_float(row, ("dtau", "delta_tau"))
    dacc = _first_float(row, ("dacc", "delta_acc"))
    screen = row.get("screen_direction")
    if dtau is None and isinstance(screen, Mapping):
        raw = _finite(screen.get("dtau"))
        if raw is not None:
            dtau = -raw
    contrast = row.get("contrast")
    if dacc is None and isinstance(contrast, Mapping):
        raw = _finite(contrast.get("point"))
        if raw is not None:
            dacc = -raw
    if rate is None or dtau is None or dacc is None:
        return None
    return rate, dtau, dacc


def _natural_plottable(row: Mapping[str, Any]) -> bool:
    return _natural_coords(row) is not None


def _h7_diff_count(block: Mapping[str, Any]) -> float | None:
    stated = _first_float(block, ("n_differs", "n_diff"))
    if stated is not None:
        return stated
    count = 0
    seen = False
    domains = block.get("domains")
    if not isinstance(domains, Mapping):
        return None
    for domain_block in domains.values():
        if not isinstance(domain_block, Mapping):
            continue
        for pair in domain_block.get("pairs") or []:
            if not isinstance(pair, Mapping):
                continue
            if pair.get("experiment") not in (None, "E1"):
                continue
            seen = True
            differs = pair.get("differs")
            mapping_hit = isinstance(differs, Mapping) and (
                differs.get("value") is True or differs.get("decision") == "differs"
            )
            if differs is True or mapping_hit or differs == "differs":
                count += 1
    return float(count) if seen else None


def _measured_break_even(report: ProgramReport) -> float | None:
    """Stage A break-even from recorded timing. None when timing was not written."""
    stage_rows = [
        row
        for row in report.cost_rows
        if str(row.get("scope")) == "stage" and str(row.get("stage")) == "A"
    ]
    if stage_rows:
        return _finite(stage_rows[0].get("break_even_arms"))
    values = [
        value
        for row in report.cost_rows
        if str(row.get("scope")) == "cell" and str(row.get("stage")) == "A"
        for value in (_finite(row.get("break_even_arms")),)
        if value is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _deviations(doc: Mapping[str, Any]) -> list[str]:
    raw = doc.get("deviations")
    if raw is None:
        return []
    lines: list[str] = []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    for item in raw:
        if isinstance(item, str):
            lines.append(item)
        elif isinstance(item, Mapping):
            date = str(item.get("date") or "")
            reason = str(item.get("reason") or item.get("text") or item.get("note") or "")
            lines.append(f"{date}: {reason}".strip(": "))
    return lines


def _locks(doc: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw = doc.get("locks")
    if raw is None:
        raw = doc.get("lock_hashes")
    rows: list[tuple[str, str]] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            rows.append((str(key), str(value)))
        return rows
    for row in _rows(raw):
        cell = str(row.get("cell") or row.get("cell_id") or "")
        digest = str(row.get("hash") or row.get("sha256") or row.get("lock") or "")
        if cell or digest:
            rows.append((cell, digest))
    return rows


def _empty_report() -> ProgramReport:
    hypotheses = {
        key: Hypothesis(key=key, stage="", present=False, passed=False) for key in _CONFIRMATORY
    }
    return ProgramReport(
        present=False,
        mismatches=[],
        stage_status={"A": "missing", "B": "missing"},
        hypotheses=hypotheses,
        pairs=[],
        h9={budget: {"median": None, "q05": None, "frac": None} for budget in _H9_BUDGETS},
        natural_rows=[],
        h8_rows=[],
        h10_rows=[],
        cost_rows=[],
        deviations=[],
        locks=[],
        scale_rows=[],
        baselines={},
    )


def read_program_report(path: Path) -> ProgramReport:
    """Map a pooled report into :class:`ProgramReport`.

    Expected shape, reconciled here and nowhere else:

    * ``stages.A`` and ``stages.B``, each ``complete`` or ``status: pending``.
    * Hypothesis blocks inside the stage: ``h1``, ``h1_pooled``, ``h2``, ``h3``,
      ``h7``, ``h8``, ``h10``, and on stage B ``h3_replication`` (``h3`` is the
      alias).
    * Figure lists under ``figures`` (``h3_pairs``, ``h9_budgets``,
      ``natural_rows``, ``h8_rows``, ``h10_rows``, ``cost_rows``), with the
      same lists accepted at the top level. Pair class may be ``class`` or
      ``arm_class``. The shared stratum is ``h3.within_shared``.
    * ``secondary.h3`` and ``secondary.scale`` when ``secondary.status`` is
      ``complete``. Top-level hypothesis blocks outside ``stages`` are ignored.
    """
    if not path.is_file():
        return _empty_report()
    loaded = read_json(path)
    doc = _mapping(loaded)
    report = _empty_report()
    report.present = True
    if doc is None:
        report.mismatches.append("program report is not a mapping")
        return report
    if "stages" not in doc:
        report.mismatches.append(
            "program report has no stages block; confirmatory macros stay pending"
        )
    stage_a = _stage_block(doc, "A")
    stage_b = _stage_block(doc, "B")
    report.stage_status["A"] = _stage_status(stage_a)
    report.stage_status["B"] = _stage_status(stage_b)
    if stage_a is not None and report.stage_status["A"] == "unmarked":
        report.mismatches.append("stages.A has no status or complete flag")
        report.stage_status["A"] = "pending"
    if stage_b is not None and report.stage_status["B"] == "unmarked":
        report.mismatches.append("stages.B has no status or complete flag")
        report.stage_status["B"] = "pending"
    if doc.get("stages") is not None and stage_a is None:
        report.mismatches.append("stages.A is missing")
    if doc.get("stages") is not None and stage_b is None:
        report.mismatches.append("stages.B is missing")

    def take(key: str, stage_name: str, block: Mapping[str, Any] | None) -> None:
        present = block is not None
        report.hypotheses[key] = Hypothesis(
            key=key,
            stage=stage_name,
            present=present,
            passed=block is not None and _passed(block),
            values=_h3_values(block) if block else {},
        )

    if report.stage_status["A"] == "complete" and stage_a is not None:
        for key in _STAGE_A_KEYS:
            aliases = {
                "h1_pooled": ("h1_pooled", "h1pooled"),
                "h10": ("h10", "H10"),
            }.get(key, (key,))
            block = _hypothesis_block(stage_a, *aliases)
            if block is None:
                report.mismatches.append(f"stages.A.{key} is missing")
            take(key, "A", block)
        h3 = _hypothesis_block(stage_a, "h3")
        if h3 is not None:
            report.hypotheses["h3"].values = _h3_values(h3)
            report.baselines = _baseline_block(h3)
            shared = _stratum_values(h3, stage_a)
            if shared:
                report.hypotheses["h3"].values["shared_rho"] = shared.get("rho")
                report.hypotheses["h3"].values["shared_p"] = shared.get("p")
                report.hypotheses["h3"].values["shared_q05"] = shared.get("q05")
                report.hypotheses["h3"].values["shared_n"] = shared.get("n")
            else:
                report.mismatches.append("stages.A h3 has no shared_and_natural stratum")
            if "class" not in report.baselines:
                report.mismatches.append("stages.A h3 baselines have no class row")
        h1 = report.hypotheses["h1"]
        h1_block = _hypothesis_block(stage_a, "h1")
        if h1_block is not None:
            h1.values["n_rejected"] = _first_float(h1_block, ("n_rejected",))
            h1.values["n_family"] = _first_float(h1_block, ("n_family", "n"))
        pooled = _hypothesis_block(stage_a, "h1_pooled", "h1pooled")
        if pooled is not None:
            point, p_value = _contrast_stats(pooled, "fp_trigger")
            report.hypotheses["h1_pooled"].values["trigger"] = point
            report.hypotheses["h1_pooled"].values["trigger_p"] = p_value
            point, p_value = _contrast_stats(pooled, "fn_behavior")
            report.hypotheses["h1_pooled"].values["fn"] = point
            report.hypotheses["h1_pooled"].values["fn_p"] = p_value
        h10 = _hypothesis_block(stage_a, "h10", "H10")
        if h10 is not None:
            point, p_value = _contrast_stats(h10, "fp_trigger")
            # The false-positive pair's arm field may be either member.
            if point is None:
                for row in _rows(h10.get("pairs") or h10.get("contrasts")):
                    arms = {str(row.get("a")), str(row.get("arm")), str(row.get("b"))}
                    if "fp_trigger" in arms or "fp_trigger_enriched" in arms:
                        if "fn_behavior" in arms:
                            continue
                        point = _first_float(row, ("point", "mean", "aligned", "statistic"))
                        p_value = _first_float(row, ("p_le_0", "p"))
                        break
            report.hypotheses["h10"].values["trigger"] = point
            report.hypotheses["h10"].values["trigger_p"] = p_value
            point_fn, p_fn = _contrast_stats(h10, "fn_behavior")
            if point_fn is None:
                for row in _rows(h10.get("pairs") or h10.get("contrasts")):
                    arms = {str(row.get("a")), str(row.get("arm")), str(row.get("b"))}
                    if "fn_behavior" in arms or "fn_behavior_depleted" in arms:
                        point_fn = _first_float(row, ("point", "mean", "aligned", "statistic"))
                        p_fn = _first_float(row, ("p_le_0", "p"))
                        break
            report.hypotheses["h10"].values["fn"] = point_fn
            report.hypotheses["h10"].values["fn_p"] = p_fn
            report.hypotheses["h10"].values["n"] = _first_float(h10, ("n_cells", "n"))
        h7 = _hypothesis_block(stage_a, "h7")
        if h7 is not None:
            report.hypotheses["h7"].values["n_diff"] = _h7_diff_count(h7)
        h8 = _hypothesis_block(stage_a, "h8")
        if h8 is not None:
            report.hypotheses["h8"].values["point"] = _first_float(h8, ("point", "mean"))
            report.hypotheses["h8"].values["p"] = _first_float(h8, ("p_le_0", "p"))
            report.hypotheses["h8"].values["n"] = _first_float(h8, ("n_domains", "n"))
        report.h9 = _h9_from(_hypothesis_block(stage_a, "h9") or _mapping(doc.get("h9")))
        if _hypothesis_block(stage_a, "h9") is None and not isinstance(doc.get("h9"), Mapping):
            report.mismatches.append("H9 budgets are missing")
    elif isinstance(doc.get("h1"), Mapping) or isinstance(doc.get("h3"), Mapping):
        report.mismatches.append(
            "top-level hypothesis blocks are not inside stages; they were not used"
        )

    if report.stage_status["B"] == "complete" and stage_b is not None:
        replication = _hypothesis_block(stage_b, "h3_replication", "h3_rep")
        alias = _hypothesis_block(stage_b, "h3")
        if replication is None and alias is not None:
            replication = alias
            report.mismatches.append("stages.B.h3 used as H3-replication")
        if replication is None:
            report.mismatches.append("stages.B.h3_replication is missing")
        take("h3_replication", "B", replication)
        if replication is not None:
            report.hypotheses["h3_replication"].values = _h3_values(replication)

    both_done = report.stage_status["A"] == "complete" and report.stage_status["B"] == "complete"
    pooled = _mapping(doc.get("h3_all")) or _mapping(doc.get("h3_pooled"))
    secondary = _mapping(doc.get("secondary"))
    if pooled is None and isinstance(secondary, Mapping) and secondary.get("status") == "complete":
        pooled = _mapping(secondary.get("h3"))
    if both_done and pooled is None:
        report.mismatches.append("A+B pooled H3 is missing")
    if both_done and isinstance(secondary, Mapping) and secondary.get("status") == "pending":
        report.mismatches.append("secondary is pending although both stages are complete")
    if isinstance(secondary, Mapping) and secondary.get("status") == "complete":
        scale = _mapping(secondary.get("scale")) or {}
        report.scale_rows = [dict(row) for row in _rows(scale.get("contrasts"))]
        if not report.scale_rows:
            report.mismatches.append("secondary.scale contrasts are missing")
    if pooled is not None and both_done:
        # Stored on h3 as extra values for the secondary macros. Not a pass flag.
        report.hypotheses["h3"].values["all_rho"] = _h3_values(pooled).get("rho")
        report.hypotheses["h3"].values["all_p"] = _h3_values(pooled).get("p")
        report.hypotheses["h3"].values["all_q05"] = _h3_values(pooled).get("q05")
        report.hypotheses["h3"].values["all_n"] = _h3_values(pooled).get("n")

    figures = _mapping(doc.get("figures")) or {}
    pair_source = figures.get("h3_pairs")
    pair_origin = "figures.h3_pairs"
    if pair_source is None:
        pair_source = doc.get("h3_pairs")
        pair_origin = "h3_pairs"
    if pair_source is None:
        pair_source = doc.get("pairs")
        pair_origin = "pairs"
    pair_rows = _rows(pair_source)
    if not pair_rows and (stage_a or stage_b):
        collected: list[Mapping[str, Any]] = []
        for stage_name, block in (("A", stage_a), ("B", stage_b)):
            if block is None:
                continue
            h3 = _hypothesis_block(block, "h3", "h3_replication")
            for row in _rows((h3 or {}).get("pairs")):
                copied = dict(row)
                copied.setdefault("stage", stage_name)
                collected.append(copied)
        if collected:
            pair_rows = collected
            pair_origin = "stages.*.h3.pairs"
            report.mismatches.append("h3_pairs missing; used stage h3 pair lists")
    missing_fields: set[str] = set()
    for row in pair_rows:
        parsed, missing = _pair_from(row, default_stage="")
        report.pairs.append(parsed)
        missing_fields |= missing
    if pair_rows and missing_fields:
        fields = ", ".join(sorted(missing_fields))
        report.mismatches.append(f"{pair_origin} rows missing fields: {fields}")

    report.natural_rows = [dict(row) for row in _rows(figures.get("natural_rows"))]
    if not report.natural_rows:
        report.natural_rows = [dict(row) for row in _rows(doc.get("natural_rows"))]
    if not report.natural_rows and stage_a is not None:
        report.natural_rows = [dict(row) for row in _rows(stage_a.get("natural_rows"))]
    report.h8_rows = [dict(row) for row in _rows(figures.get("h8_rows"))]
    if not report.h8_rows:
        report.h8_rows = [dict(row) for row in _rows(doc.get("h8_rows"))]
    if not report.h8_rows and stage_a is not None:
        h8 = _hypothesis_block(stage_a, "h8")
        report.h8_rows = [dict(row) for row in _rows((h8 or {}).get("cells"))]
    report.h10_rows = [dict(row) for row in _rows(figures.get("h10_rows"))]
    if not report.h10_rows:
        report.h10_rows = [dict(row) for row in _rows(doc.get("h10_rows"))]
    if not report.h10_rows and stage_a is not None:
        h10 = _hypothesis_block(stage_a, "h10", "H10")
        report.h10_rows = [dict(row) for row in _rows((h10 or {}).get("pairs"))]
    report.cost_rows = [dict(row) for row in _rows(figures.get("cost_rows"))]
    if not report.cost_rows:
        report.cost_rows = [dict(row) for row in _rows(doc.get("cost_rows") or doc.get("costs"))]
    if report.stage_status["A"] == "complete" and not report.cost_rows:
        report.mismatches.append("cost_rows are missing")
    if (
        report.stage_status["A"] == "complete"
        and report.natural_rows
        and not any(_natural_plottable(row) for row in report.natural_rows)
    ):
        report.mismatches.append(
            "natural_rows do not include a step-0 error rate together with locked dtau and endpoint dacc"
        )
    figure_budgets = _rows(figures.get("h9_budgets"))
    if figure_budgets and all(
        (report.h9.get(budget) or {}).get("median") is None for budget in _H9_BUDGETS
    ):
        report.h9 = _h9_from({"budgets": figure_budgets})
    report.deviations = _deviations(doc)
    report.locks = _locks(doc)
    return report


def _p0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = read_json(path)
    return loaded if isinstance(loaded, dict) else {}


def _check_value(doc: Mapping[str, Any], check_id: str) -> float | None:
    for row in doc.get("checks") or []:
        if isinstance(row, Mapping) and row.get("id") == check_id:
            return _finite(row.get("value"))
    return None


def _fate(doc: Mapping[str, Any], world: str, arm: str) -> tuple[float | None, float | None]:
    tables = _mapping(doc.get("tables")) or {}
    for row in tables.get("fates") or []:
        if isinstance(row, Mapping) and row.get("world") == world and row.get("arm") == arm:
            return _finite(row.get("J")), _finite(row.get("obs"))
    return None, None


def _tau_row(doc: Mapping[str, Any], world: str, arm: str) -> dict[str, Any] | None:
    tables = _mapping(doc.get("tables")) or {}
    for row in tables.get("tau") or []:
        if isinstance(row, Mapping) and row.get("world") == world and row.get("arm") == arm:
            return dict(row)
    return None


def _knife_taus(doc: Mapping[str, Any]) -> list[float]:
    tables = _mapping(doc.get("tables")) or {}
    values = []
    for row in tables.get("knife") or []:
        if isinstance(row, Mapping):
            number = _finite(row.get("tau"))
            if number is not None:
                values.append(number)
    return values


@dataclass
class PlanNumbers:
    rate_word: str
    buckets: dict[str, dict[str, float]]
    rented_usd: float
    stage_class_rows: list[tuple[str, str, int, float, float, float]]


def _plan_numbers(repo: Path) -> PlanNumbers:
    program = load_program(repo / "configs" / "program.yaml", root=repo)
    jobs = plan_jobs(program, root=repo)
    policy = program.get("hardware_policy") or {}
    speed = {str(key): float(value) for key, value in (policy.get("speed_vs_h100") or {}).items()}
    usd = {str(key): float(value) for key, value in (policy.get("usd_per_gpu_hour") or {}).items()}
    sources = {str(job.get("rate_source") or "placeholder") for job in jobs}
    if sources == {"measured"}:
        rate_word = "measured"
    elif sources <= {"placeholder"}:
        rate_word = "projected"
    else:
        rate_word = "mixed"
    buckets: dict[str, dict[str, float]] = {}
    rented = 0.0
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for job in jobs:
        exploratory = bool(job.get("exploratory"))
        stage = job.get("stage")
        bucket = "exploratory" if exploratory or stage in (None, "") else str(stage)
        hardware = str(job["hardware_class"])
        hours = float(job["gpu_hours"])
        klass = class_gpu_hours(hours, hardware, speed)
        dollars = klass * float(usd.get(hardware, 0.0))
        slot = buckets.setdefault(bucket, {"jobs": 0.0, "h100": 0.0, "usd": 0.0})
        slot["jobs"] += 1
        slot["h100"] += hours
        slot["usd"] += dollars
        rented += dollars
        grouped.setdefault((bucket, hardware), []).append(job)
    rows: list[tuple[str, str, int, float, float, float]] = []
    for bucket, hardware in sorted(grouped):
        group = grouped[(bucket, hardware)]
        hours = sum(float(job["gpu_hours"]) for job in group)
        klass = class_gpu_hours(hours, hardware, speed)
        dollars = klass * float(usd.get(hardware, 0.0))
        rows.append((bucket, hardware, len(group), hours, klass, dollars))
    return PlanNumbers(
        rate_word=rate_word,
        buckets=buckets,
        rented_usd=rented,
        stage_class_rows=rows,
    )


def _macro(name: str, body: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{body}}}"


def _numbers_tex(p0: Mapping[str, Any], report: ProgramReport, plan: PlanNumbers) -> str:
    hand_j, hand_fpf = _fate(p0, "hand_noisy", "fpF")
    _, hand_fpi = _fate(p0, "hand_noisy", "fpI")
    _, hand_noise = _fate(p0, "hand_noisy", "resampled_fpF")
    rf_j, rf_fpf = _fate(p0, "rf05", "fpF")
    _, rf_fpi = _fate(p0, "rf05", "fpI")
    _, rf_noise = _fate(p0, "rf05", "resampled_fpF")
    rank = _mapping((_mapping(p0.get("tables")) or {}).get("rank")) or {}
    fnc = _tau_row(p0, "rf05", "fnC") or {}
    clean = _tau_row(p0, "hand_clean", "fpF") or {}
    hand_tau = _tau_row(p0, "hand_noisy", "fpF") or {}
    rf_tau = _tau_row(p0, "rf05", "fpF") or {}
    taus = _knife_taus(p0)
    knife_j, _ = _fate_knife_summary(p0)
    macros: dict[str, str] = {
        "PZeroHandJ": fmt_stat(hand_j),
        "PZeroHandFpF": fmt_stat(hand_fpf),
        "PZeroHandFpI": fmt_stat(hand_fpi),
        "PZeroHandNoise": fmt_stat(hand_noise),
        "PZeroRfJ": fmt_stat(rf_j),
        "PZeroRfFpF": fmt_stat(rf_fpf),
        "PZeroRfFpI": fmt_stat(rf_fpi),
        "PZeroRfNoise": fmt_stat(rf_noise),
        "PZeroCleanJ": fmt_stat(_finite(clean.get("J"))),
        "PZeroCleanFpF": fmt_stat(_finite(clean.get("obs"))),
        "PZeroCleanTau": fmt_stat(_finite(clean.get("tau_pool"))),
        "PZeroHandTau": fmt_stat(_finite(hand_tau.get("tau_pool"))),
        "PZeroRfTau": fmt_stat(_finite(rf_tau.get("tau_pool"))),
        "PZeroTauPoolRho": fmt_stat(_finite(rank.get("spearman_tau_pool"))),
        "PZeroExactRho": fmt_stat(_finite(rank.get("spearman_tau"))),
        "PZeroJRho": fmt_stat(_finite(rank.get("spearman_J"))),
        "PZeroPrecRho": fmt_stat(_finite(rank.get("spearman_prec0"))),
        "PZeroTauRelRho": fmt_stat(_finite(rank.get("spearman_tau_rel"))),
        "PZeroWithin": fmt_stat(_finite(rank.get("tau_pool_within"))),
        "PZeroReliability": fmt_stat(_finite(rank.get("mean_reliability"))),
        "PZeroTiltErr": fmt_stat(_check_value(p0, "P0.1")),
        "PZeroDecompErr": fmt_stat(_check_value(p0, "P0.2")),
        "PZeroCosine": fmt_stat(_check_value(p0, "P0.3a")),
        "PZeroScaleGap": fmt_stat(_check_value(p0, "P0.3b")),
        "PZeroGroupZ": fmt_stat(_check_value(p0, "P0.3c")),
        "PZeroModalGap": fmt_stat(_check_value(p0, "P0.4a")),
        "PZeroPrecGap": fmt_stat(_check_value(p0, "P0.4b")),
        "PZeroBetaGap": fmt_stat(_check_value(p0, "P0.4c")),
        "PZeroHandGap": fmt_stat(_check_value(p0, "P0.5.hand_noisy.gap")),
        "PZeroHandClose": fmt_stat(_check_value(p0, "P0.5.hand_noisy.close")),
        "PZeroRfGap": fmt_stat(_check_value(p0, "P0.5.rf05.gap")),
        "PZeroRfClose": fmt_stat(_check_value(p0, "P0.5.rf05.close")),
        "PZeroKnifeTerminal": fmt_stat(_check_value(p0, "P0.8")),
        "PZeroKnifeTauMin": fmt_stat(min(taus) if taus else None),
        "PZeroKnifeTauMax": fmt_stat(max(taus) if taus else None),
        "PZeroKnifeJ": fmt_stat(knife_j),
        "PZeroFnCTau": fmt_stat(_finite(fnc.get("tau"))),
        "PZeroFnCObs": fmt_stat(_finite(fnc.get("obs"))),
        "PZeroFnCPool": fmt_stat(_finite(fnc.get("tau_pool"))),
    }
    hyp = report.hypotheses
    macros.update(
        {
            "HOneRejected": fmt_count(hyp["h1"].values.get("n_rejected")),
            "HOneFamily": fmt_count(hyp["h1"].values.get("n_family")),
            "HOnePooledTrigger": fmt_stat(hyp["h1_pooled"].values.get("trigger")),
            "HOnePooledTriggerP": fmt_stat(hyp["h1_pooled"].values.get("trigger_p")),
            "HOnePooledFn": fmt_stat(hyp["h1_pooled"].values.get("fn")),
            "HOnePooledFnP": fmt_stat(hyp["h1_pooled"].values.get("fn_p")),
            "HThreeRho": fmt_stat(hyp["h3"].values.get("rho")),
            "HThreeP": fmt_stat(hyp["h3"].values.get("p")),
            "HThreeQFive": fmt_stat(hyp["h3"].values.get("q05")),
            "HThreeN": fmt_count(hyp["h3"].values.get("n")),
            "HThreeClassRho": fmt_stat((report.baselines.get("class") or {}).get("rho")),
            "HThreeClassGap": fmt_stat((report.baselines.get("class") or {}).get("gap")),
            "HThreeSharedRho": fmt_stat(hyp["h3"].values.get("shared_rho")),
            "HThreeSharedP": fmt_stat(hyp["h3"].values.get("shared_p")),
            "HThreeSharedQFive": fmt_stat(hyp["h3"].values.get("shared_q05")),
            "HThreeSharedN": fmt_count(hyp["h3"].values.get("shared_n")),
            "HSevenNDiff": fmt_count(hyp["h7"].values.get("n_diff")),
            "HEightPoint": fmt_stat(hyp["h8"].values.get("point")),
            "HEightP": fmt_stat(hyp["h8"].values.get("p")),
            "HEightN": fmt_count(hyp["h8"].values.get("n")),
            "HTenTrigger": fmt_stat(hyp["h10"].values.get("trigger")),
            "HTenTriggerP": fmt_stat(hyp["h10"].values.get("trigger_p")),
            "HTenFn": fmt_stat(hyp["h10"].values.get("fn")),
            "HTenFnP": fmt_stat(hyp["h10"].values.get("fn_p")),
            "HThreeRepRho": fmt_stat(hyp["h3_replication"].values.get("rho")),
            "HThreeRepP": fmt_stat(hyp["h3_replication"].values.get("p")),
            "HThreeRepQFive": fmt_stat(hyp["h3_replication"].values.get("q05")),
            "HThreeRepN": fmt_count(hyp["h3_replication"].values.get("n")),
            "HThreeAllRho": fmt_stat(hyp["h3"].values.get("all_rho")),
            "HThreeAllP": fmt_stat(hyp["h3"].values.get("all_p")),
            "HThreeAllQFive": fmt_stat(hyp["h3"].values.get("all_q05")),
            "HThreeAllN": fmt_count(hyp["h3"].values.get("all_n")),
        }
    )
    budget_names = {
        32: "ThirtyTwo",
        64: "SixtyFour",
        128: "OneTwoEight",
        192: "OneNineTwo",
    }
    for budget, suffix in budget_names.items():
        block = report.h9.get(budget) or {}
        macros[f"HNineMed{suffix}"] = fmt_stat(block.get("median"))
        macros[f"HNineQ{suffix}"] = fmt_stat(block.get("q05"))
        macros[f"HNinePos{suffix}"] = fmt_stat(block.get("frac"))
    for bucket, prefix in (("A", "StageA"), ("B", "StageB"), ("exploratory", "Exploratory")):
        slot = plan.buckets.get(bucket) or {}
        macros[f"Plan{prefix}Jobs"] = fmt_count(slot.get("jobs"))
        macros[f"Plan{prefix}Hours"] = fmt_hours(slot.get("h100"))
        macros[f"Plan{prefix}Usd"] = fmt_hours(slot.get("usd"))
    macros["PlanRentedUsd"] = fmt_hours(plan.rented_usd)
    macros["PlanRateWord"] = plan.rate_word
    macros["BreakEvenMeasured"] = fmt_hours(_measured_break_even(report))
    lines = [
        "% Generated by vfates.paper. Do not edit by hand.",
        r"\DeclareRobustCommand{\pending}{\textcolor{red}{[pending]}}",
    ]
    for name in sorted(macros):
        lines.append(_macro(name, macros[name]))
    return "\n".join(lines) + "\n"


def _fate_knife_summary(doc: Mapping[str, Any]) -> tuple[float | None, float | None]:
    tables = _mapping(doc.get("tables")) or {}
    rows = [row for row in (tables.get("knife") or []) if isinstance(row, Mapping)]
    if not rows:
        return None, None
    return _finite(rows[0].get("J")), _finite(rows[0].get("prec0"))


_ABSTRACT = {
    "h1": (
        "H1 passed its preregistered test: at least one stage A shared-rule contrast "
        "rejects the margin of 0.05 after Holm correction "
        r"(\HOneRejected{} of \HOneFamily{} tested contrasts)."
    ),
    "h1_pooled": (
        "H1-pooled passed its preregistered test: the equal-weight kit contrasts are "
        r"\HOnePooledTrigger{} and \HOnePooledFn{}."
    ),
    "h2": (
        "H2 passed its preregistered test: the prompt-specific contrasts are "
        "equivalent to zero within the margin."
    ),
    "h3": (
        "H3 passed its preregistered test: the stage A within-pair correlation is "
        r"\HThreeRho{} (permutation $p = \HThreeP{}$, 5th percentile \HThreeQFive{}, "
        r"$n = \HThreeN{}$)."
    ),
    "h10": (
        "H10 passed its preregistered test: the screen's sign aligns with the accuracy "
        r"gap inside each same-class phrase pair (\HTenTrigger{} and \HTenFn{})."
    ),
    "h7": (
        "H7 passed its preregistered test: natural verifiers differ from rate-matched "
        r"resampled noise (\HSevenNDiff{} pairs whose interval excludes zero)."
    ),
    "h8": (
        "H8 passed its preregistered test: patches chosen by the screen end "
        r"\HEightPoint{} above patches chosen by the error count "
        r"(one-sided $p = \HEightP{}$)."
    ),
    "h3_replication": (
        "H3-replication passed its preregistered test: on stage B pairs the correlation "
        r"is \HThreeRepRho{} (permutation $p = \HThreeRepP{}$, 5th percentile "
        r"\HThreeRepQFive{}, $n = \HThreeRepN{}$)."
    ),
}


def _abstract_tex(report: ProgramReport) -> str:
    stage_ready = (
        report.stage_status.get("A") == "complete" or report.stage_status.get("B") == "complete"
    )
    if not report.present or not stage_ready:
        body = r"{\color{red}[pending: confirmatory report not available]}"
        return _clean(body + "\n", "abstract")
    sentences = [_ABSTRACT[key] for key in _CONFIRMATORY if report.hypotheses[key].passed]
    if not sentences:
        body = "% No confirmatory hypothesis passed its preregistered test.\n"
        return _clean(body, "abstract")
    return _clean(" ".join(sentences) + "\n", "abstract")


def _tabular(headers: Sequence[str], rows: Sequence[Sequence[str]], spec: str) -> str:
    head = " & ".join(headers) + r" \\"
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return (
        "\\begin{tabular}{"
        + spec
        + "}\n\\toprule\n"
        + head
        + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _domains_tex() -> str:
    rows = [
        (
            "chain sum",
            "Reasoning Gym decimal chain sum",
            "exact decimal equality",
            "exact string match, audited",
        ),
        (
            "math",
            "Big-Math-RL-Verified, decontaminated",
            "symbolic equivalence",
            "lenient cascade; strict string match",
        ),
        ("code", "TACO, Python", "every hidden test passes", "public examples in the statement"),
        (
            "instruction following",
            "IFEval and IFBench constraints",
            "checkers and a content judge",
            "checkers alone",
        ),
        ("qa", "WebInstruct-verified", "strict rule", "1.5B model judge"),
    ]
    return _tabular(
        ("Domain", "Source", "Oracle", "Natural verifier"),
        rows,
        r"p{0.16\textwidth}p{0.26\textwidth}p{0.22\textwidth}p{0.22\textwidth}",
    )


def _kit_tex(repo: Path, mismatches: list[str]) -> str:
    loaded = read_yaml(repo / "configs" / "arms" / "kit.yaml")
    arms = loaded.get("arms") if isinstance(loaded, Mapping) else None
    targets = loaded.get("targets") if isinstance(loaded, Mapping) else None
    fpr = _finite((targets or {}).get("fpr")) if isinstance(targets, Mapping) else None
    fnr = _finite((targets or {}).get("fnr")) if isinstance(targets, Mapping) else None
    if fpr is None or fnr is None:
        mismatches.append("kit.yaml is missing targets.fpr or targets.fnr")
        fpr_text = fnr_text = PENDING
    else:
        fpr_text = f"{fpr:.2f}"
        fnr_text = f"{fnr:.2f}"
    rows: list[tuple[str, str, str]] = []
    if not isinstance(arms, list):
        mismatches.append("kit.yaml has no arms list")
        return _tabular(("Arm", "Channel", "Structure"), [("pending", PENDING, PENDING)], "lll")
    for arm in arms:
        if not isinstance(arm, Mapping):
            continue
        name = tex_text(arm.get("name", ""))
        kind = str(arm.get("type", ""))
        prefer = str(arm.get("prefer", ""))
        match = tex_text(arm.get("match", ""))
        support = str(arm.get("support", ""))
        if kind == "oracle":
            channel, blurb = "none", "clean reward"
        elif kind == "hash_fp":
            channel, blurb = f"FPR {fpr_text}", "prompt-specific false positives"
        elif kind == "trigger_fp" and prefer == "not_enriched":
            channel, blurb = f"FPR {fpr_text}", "shared, phrase not enriched"
        elif kind == "trigger_fp" and prefer == "enriched":
            channel, blurb = f"FPR {fpr_text}", "shared, phrase enriched for correct answers"
        elif kind == "behavior_fn" and prefer == "enriched":
            channel, blurb = f"FNR {fnr_text}", "shared, phrase predicts correctness"
        elif kind == "behavior_fn" and prefer == "depleted":
            channel, blurb = f"FNR {fnr_text}", "shared, phrase depleted among correct answers"
        elif kind == "prompt_subset_fn":
            channel, blurb = f"FNR {fnr_text}", "prompt-specific false negatives"
        elif kind == "resampled":
            matched = next(
                (
                    row
                    for row in arms
                    if isinstance(row, Mapping) and row.get("name") == arm.get("match")
                ),
                {},
            )
            if str(matched.get("type", "")).endswith("_fn"):
                where = "correct outputs"
            elif support == "all":
                where = "every wrong output"
            else:
                where = "answered wrong outputs"
            channel = f"matched to {match}"
            blurb = f"resampled, {where}"
        else:
            channel, blurb = PENDING, "see the arms file"
            mismatches.append(f"kit arm {arm.get('name')} has no paper blurb")
        rows.append((f"\\texttt{{{name}}}", channel, blurb))
    return _tabular(
        ("Arm", "Channel", "Structure"),
        rows,
        r"p{0.30\textwidth}lp{0.32\textwidth}",
    )


def _hypotheses_tex() -> str:
    rows = [
        (
            "H1",
            "Stage A, E1",
            "Superiority over 0.05 on 12 contrasts; Holm; every seed positive; at least one rejection.",
        ),
        (
            "H1-pooled",
            "Stage A, E1",
            "Equal-weight mean of each kit contrast above 0; at least 3 cells; Holm on the two; one-sided bootstrap.",
        ),
        ("H2", "Stage A, E1", r"TOST at $\pm 0.05$ on 10 prompt-specific contrasts; Holm."),
        (
            "H3",
            "Stage A pairs",
            r"Spearman of locked $\Delta\tau$ with $\Delta$acc; permutation $p < 0.05$; 5th percentile above 0; at least 20 pairs.",
        ),
        (
            "H10",
            "Stage A, E1",
            r"Mean of $\mathrm{sign}(\Delta\tau)\,(\mathrm{acc}_a - \mathrm{acc}_b)$ above 0; Holm on 2 pairs; at least 3 cells.",
        ),
        (
            "H7",
            "Stage A, E1",
            r"Natural minus its resampled control differs from 0; Holm on 5 pairs; sign against locked $\Delta\tau$.",
        ),
        (
            "H8",
            "Stage A, E4",
            "Mean of screen patch minus error-count patch above 0; at least 2 domains; one-sided bootstrap.",
        ),
        (
            "H3-replication",
            "Stage B pairs",
            "The H3 test on stage B pairs only. Stage A does not enter it.",
        ),
    ]
    return _tabular(
        ("Hypothesis", "Sample", "Preregistered test"),
        rows,
        r"p{0.16\textwidth}p{0.14\textwidth}p{0.48\textwidth}",
    )


def _baseline_tex(report: ProgramReport) -> str:
    cost_note = (
        "The audit, with its features, is paid once per cell and serves every verifier; "
        "a step-25 run is one short training job per verifier. "
        r"From recorded wall time, one audit costs as much as \BreakEvenMeasured{} step-25 runs. "
        "No projected ratio is quoted: the plan does not price the audit's gradient pass."
    )
    rows = []
    for key, label, cost in _BASELINE_ROWS:
        block = report.baselines.get(key) or {}
        rows.append((label, fmt_stat(block.get("rho")), fmt_stat(block.get("gap")), cost))
    table = _tabular(
        ("Baseline", r"$\rho$", r"$\rho(\tau)-\rho(b)$", "Cost"),
        rows,
        r"lrrp{0.28\textwidth}",
    )
    return table + "\n" + cost_note + "\n"


def _cell_rows(repo: Path, report: ProgramReport) -> str:
    """Confirmatory cells from the program. Outcome columns stay pending.

    A complete stage does not yet publish a per-cell oracle-gain field in the
    reconciled report, so those two columns are pending until that key exists.
    """
    del report
    program = load_program(repo / "configs" / "program.yaml", root=repo)
    lines_src = []
    for cell in program["cells"]:
        cell_id = str(cell["id"])
        stage = cell_stage(program, cell_id) or "exploratory"
        if stage == "exploratory":
            continue
        lines_src.append(
            (
                f"\\texttt{{{tex_text(cell_id)}}}",
                stage,
                tex_text(cell["domain"]),
                tex_text(cell["model"]),
                tex_text(cell["hardware_class"]),
                PENDING,
                PENDING,
            )
        )
    return _tabular(
        ("Cell", "Stage", "Domain", "Model", "Hardware", "Gain", "Conclusive"),
        lines_src,
        r"p{0.22\textwidth}cp{0.12\textwidth}p{0.14\textwidth}p{0.16\textwidth}cc",
    )


def _pairs_tex(report: ProgramReport) -> str:
    if not report.pairs:
        return "No pair rows are in the report yet.\n"
    rows = []
    for pair in report.pairs:
        rows.append(
            (
                f"\\texttt{{{tex_text(pair.cell)}}}",
                tex_text(pair.stage),
                f"\\texttt{{{tex_text(pair.arm)}}}",
                f"\\texttt{{{tex_text(pair.control)}}}",
                tex_text(pair.structure_class) if pair.structure_class else PENDING,
                "yes" if pair.is_natural else "no",
                fmt_stat(pair.dtau),
                fmt_stat(pair.dacc),
            )
        )
    return _tabular(
        ("Cell", "Stage", "Arm", "Control", "Class", "Natural", r"$\Delta\tau$", r"$\Delta$acc"),
        rows,
        "llllllll",
    )


def _h9_tex(report: ProgramReport) -> str:
    rows = []
    for budget in _H9_BUDGETS:
        block = report.h9.get(budget) or {}
        rows.append(
            (
                str(budget),
                fmt_stat(block.get("median")),
                fmt_stat(block.get("q05")),
                fmt_stat(block.get("frac")),
            )
        )
    return _tabular(
        ("Labels", r"Median $\rho$", "5th pct.", r"Frac.\ $>0$"),
        rows,
        "lrrr",
    )


def _scale_tex(report: ProgramReport) -> str:
    if not report.scale_rows:
        return "Scale slopes are pending until stage B is complete.\n"
    rows = []
    for row in report.scale_rows:
        rows.append(
            (
                tex_text(row.get("domain", "")),
                tex_text(row.get("name", "")),
                fmt_count(_finite(row.get("n_cells"))),
                fmt_stat(_finite(row.get("slope"))),
                fmt_stat(_finite(row.get("lo"))),
                fmt_stat(_finite(row.get("hi"))),
            )
        )
    return _tabular(
        ("Domain", "Contrast", "Cells", "Slope", "Low", "High"),
        rows,
        "llllll",
    )


def _compute_tex(plan: PlanNumbers) -> str:
    rows = []
    for bucket, hardware, jobs, hours, klass, dollars in plan.stage_class_rows:
        rows.append(
            (
                tex_text(bucket),
                tex_text(hardware),
                str(jobs),
                f"{hours:.2f}",
                f"{klass:.2f}",
                f"{dollars:.2f}",
            )
        )
    note = (
        f"Rates are {plan.rate_word}. "
        "Placeholder token rates stay projected until pilot P5 writes measured rates. "
        "H100-equivalent hours are the planner's \\texttt{gpu\\_hours}. "
        "Class hours divide those by \\texttt{speed\\_vs\\_h100}. "
        "Rented dollars use \\texttt{usd\\_per\\_gpu\\_hour}. "
        r"The rented total is \PlanRentedUsd{}."
    )
    return (
        _tabular(
            ("Stage", "Hardware", "Jobs", "H100 hours", "Class hours", "USD"),
            rows,
            "llrrrr",
        )
        + "\n"
        + note
        + "\n"
    )


def _locks_tex(report: ProgramReport) -> str:
    if not report.locks:
        return "No lock hashes are in the report yet. Each cell's lock is the SHA-256 of the forecasts, the cell configuration, the environment, and every input artefact, written before that cell's first training step.\n"
    rows = []
    for cell, digest in report.locks:
        hex_chars = re.sub(r"[^0-9a-fA-F]", "", digest)
        if len(hex_chars) >= 32:
            parts = [hex_chars[index : index + 16] for index in range(0, len(hex_chars), 16)]
            shown = r"\texttt{" + r"\allowbreak ".join(parts) + "}"
        else:
            shown = f"\\texttt{{{tex_text(digest)}}}"
        rows.append((f"\\texttt{{{tex_text(cell)}}}", shown))
    return _tabular(("Cell", "Lock hash"), rows, "ll")


def _deviations_tex(report: ProgramReport) -> str:
    if not report.present or not report.deviations:
        return "The preregistration's deviation log is empty. The report lists none.\n"
    items = "\n".join(f"\\item {tex_text(line)}" for line in report.deviations)
    return "\\begin{itemize}\n" + items + "\n\\end{itemize}\n"


def _p0_fates_tex(doc: Mapping[str, Any]) -> str:
    tables = _mapping(doc.get("tables")) or {}
    rows = []
    for row in tables.get("fates") or []:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            (
                tex_text(row.get("world", "")),
                tex_text(row.get("arm", "")),
                fmt_stat(_finite(row.get("J"))),
                fmt_stat(_finite(row.get("obs"))),
            )
        )
    if not rows:
        return "Pilot P0 fates are not in the committed report.\n"
    return _tabular(("World", "Arm", "$J$", "Held-out accuracy"), rows, "llrr")


def _p0_checks_tex(doc: Mapping[str, Any]) -> str:
    rows = []
    for row in doc.get("checks") or []:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            (
                tex_text(row.get("id", "")),
                tex_text(row.get("name", "")),
                fmt_stat(_finite(row.get("value"))),
                tex_text(row.get("status", "")),
            )
        )
    if not rows:
        return "Pilot P0 checks are not in the committed report.\n"
    return _tabular(
        ("Check", "Name", "Value", "Status"),
        rows,
        r"p{0.24\textwidth}p{0.42\textwidth}rl",
    )


def _style_plots() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": "vfates", "Title": path.stem})
    plt.close(fig)


def _fig_schematic(path: Path, doc: Mapping[str, Any]) -> None:
    _style_plots()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.75, 2.3),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    ax = axes[0]
    patterns = (
        ("Resampled", {(0, 1), (0, 4), (1, 3), (2, 0), (3, 5), (4, 2)}, True),
        ("Prompt-specific", {(0, 0), (1, 4), (2, 1), (3, 5), (4, 3)}, False),
        ("Shared", {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)}, False),
    )
    colors = ("#0072B2", "#E69F00", "#009E73")
    for panel, (label, filled, redrawn) in enumerate(patterns):
        origin = panel * 7.5
        for row in range(5):
            for col in range(6):
                hit = (row, col) in filled
                ax.add_patch(
                    Rectangle(
                        (origin + col, 4 - row),
                        0.85,
                        0.85,
                        facecolor="white" if not hit or redrawn else colors[panel],
                        edgecolor=colors[panel] if hit else "#444444",
                        hatch="/////" if hit and redrawn else None,
                        linewidth=0.8 if hit else 0.4,
                    )
                )
        ax.text(origin + 2.9, -0.5, label, ha="center", va="top", fontsize=8)
    ax.text(-0.5, 2.4, "prompts", rotation=90, ha="right", va="center", fontsize=7)
    ax.text(2.9, 5.2, "output behaviours", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-1.4, 21.2)
    ax.set_ylim(-1.4, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    worlds = ("hand_clean", "hand_noisy", "rf05")
    arms = (
        ("fpF", "shared (fpF)", colors[2]),
        ("fpI", "prompt-specific (fpI)", colors[1]),
        ("resampled_fpF", "resampled", colors[0]),
    )
    width = 0.26
    for index, (arm, label, color) in enumerate(arms):
        positions, heights = [], []
        for group, world in enumerate(worlds):
            if arm == "resampled_fpF":
                _jay, obs = _fate(doc, world, arm)
            else:
                found = _tau_row(doc, world, arm)
                obs = None if found is None else _finite(found.get("obs"))
            if obs is None:
                continue
            positions.append(group + (index - 1) * width)
            heights.append(obs)
        ax.bar(positions, heights, width=width, color=color, label=label)
        for x, height in zip(positions, heights, strict=True):
            ax.text(x, height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=6)
    labels = []
    for world in worlds:
        shared = _tau_row(doc, world, "fpF") or {}
        youden = _finite(shared.get("J"))
        tau = _finite(shared.get("tau_pool"))
        parts = [world.replace("_", " ")]
        if youden is not None:
            parts.append(f"J = {youden:.2f}")
        if tau is not None:
            parts.append(rf"$\tau_{{\mathrm{{pool}}}}$ = {tau:.2f}")
        labels.append("\n".join(parts))
    ax.set_xticks(list(range(len(worlds))))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Trained held-out accuracy")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def _awaiting(path: Path, sentence: str, *, figsize: tuple[float, float]) -> None:
    _style_plots()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, sentence, ha="center", va="center", fontsize=11)
    _save(fig, path)


def _fig_h3(path: Path, report: ProgramReport) -> None:
    usable = [row for row in report.pairs if row.dtau is not None and row.dacc is not None]
    if not usable:
        _awaiting(path, "awaiting stage A report", figsize=(3.3, 2.2))
        return
    _style_plots()
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    other = [row for row in usable if not row.is_natural]
    natural = [row for row in usable if row.is_natural]
    if other:
        ax.scatter(
            [float(row.dtau) for row in other if row.dtau is not None],
            [float(row.dacc) for row in other if row.dacc is not None],
            s=18,
            c="#0072B2",
            label="matched pair",
        )
    if natural:
        ax.scatter(
            [float(row.dtau) for row in natural if row.dtau is not None],
            [float(row.dacc) for row in natural if row.dacc is not None],
            s=28,
            c="#E69F00",
            marker="D",
            label="natural pair",
        )
    ax.axhline(0.0, color="#888888", linewidth=0.4)
    ax.axvline(0.0, color="#888888", linewidth=0.4)
    ax.set_xlabel(r"locked $\Delta\tau$")
    ax.set_ylabel(r"endpoint $\Delta$acc")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def _fig_natural(path: Path, report: ProgramReport) -> None:
    points = []
    for row in report.natural_rows:
        coords = _natural_coords(row)
        if coords is None:
            continue
        points.append((*coords, str(row.get("name") or row.get("arm") or row.get("natural") or "")))
    if not points:
        _awaiting(path, "awaiting stage A report", figsize=(6.5, 1.7))
        return
    _style_plots()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.7))
    rates = [item[0] for item in points]
    axes[0].scatter(rates, [item[1] for item in points], c="#0072B2", s=22)
    axes[1].scatter(rates, [item[2] for item in points], c="#009E73", s=22)
    axes[0].set_xlabel("Step-0 error rate")
    axes[1].set_xlabel("Step-0 error rate")
    axes[0].set_ylabel(r"locked $\Delta\tau$")
    axes[1].set_ylabel(r"endpoint $\Delta$acc")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def _fig_h9(path: Path, report: ProgramReport) -> None:
    xs = []
    ys = []
    for budget in _H9_BUDGETS:
        median = (report.h9.get(budget) or {}).get("median")
        if median is None:
            continue
        xs.append(budget)
        ys.append(median)
    if len(xs) < 2:
        _awaiting(path, "awaiting stage A report", figsize=(3.3, 1.8))
        return
    _style_plots()
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    ax.plot(xs, ys, marker="o", color="#0072B2")
    ax.set_xticks(list(_H9_BUDGETS))
    ax.set_xlabel("Oracle labels")
    ax.set_ylabel("Median correlation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_clean(text, path.name), encoding="utf-8")


def build_assets(
    *,
    repo: Path | None = None,
    runs: Path | None = None,
    out: Path,
    figures: Path,
    report_path: Path | None = None,
    p0_path: Path | None = None,
) -> list[str]:
    """Write macros, tables, and figures. Returns report-key mismatches."""
    root = repo or repo_root()
    report_file = report_path
    if report_file is None and runs is not None:
        report_file = runs / "program_report.json"
    if report_file is None:
        report_file = root / "runs" / "program_report.json"
    p0_file = p0_path or (root / "pilots" / "p0_theory" / "results" / "report.json")
    report = read_program_report(report_file)
    p0 = _p0(p0_file)
    plan = _plan_numbers(root)
    mismatches = list(report.mismatches)
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    _write(out / "numbers.tex", _numbers_tex(p0, report, plan))
    _write(out / "abstract_results.tex", _abstract_tex(report))
    _write(out / "domains.tex", _domains_tex())
    _write(out / "kit.tex", _kit_tex(root, mismatches))
    _write(out / "hypotheses.tex", _hypotheses_tex())
    _write(out / "baselines.tex", _baseline_tex(report))
    _write(out / "per_cell.tex", _cell_rows(root, report))
    _write(out / "pairs.tex", _pairs_tex(report))
    _write(out / "h9.tex", _h9_tex(report))
    _write(out / "scale.tex", _scale_tex(report))
    _write(out / "compute.tex", _compute_tex(plan))
    _write(out / "locks.tex", _locks_tex(report))
    _write(out / "deviations.tex", _deviations_tex(report))
    _write(out / "p0_fates.tex", _p0_fates_tex(p0))
    _write(out / "p0_checks.tex", _p0_checks_tex(p0))
    _fig_schematic(figures / "fig1.pdf", p0)
    _fig_h3(figures / "fig_h3.pdf", report)
    _fig_natural(figures / "fig_natural.pdf", report)
    _fig_h9(figures / "fig_h9.pdf", report)
    return mismatches
