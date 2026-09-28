"""Confirmatory report for E1. Decisions follow the preregistered rules.

Training stays outside this module: pass finished run directories and the
withdrawal record written before training. ``ok`` on the study is execution
success; hypothesis outcomes live in this report.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vfates.analysis.outcomes import (
    adjusted_contrast,
    crossed_pair_dacc,
    difference_matrix,
    eval_curve,
    forecast_row,
    holm,
    holm_adjusted,
    knife_edge,
    ols_slope,
    paired_contrast_crossed,
    permutation_spearman_p,
    prompt_accuracy,
    rank_bootstrap,
    rate_path,
    require_final_step,
    score_forecasts,
    time_averaged,
)
from vfates.forecast.lockbox import assert_matches_lock, load_lock
from vfates.utils.io import read_json
from vfates.utils.stats import spearman

PHRASE_TYPES = frozenset({"trigger_fp", "behavior_fn"})

# (control, arm). The contrast is acc(control) - acc(arm).
H1_FAMILY: tuple[tuple[str, str], ...] = (
    ("noise_fp", "fp_tolerance"),
    ("noise_fp_trigger", "fp_trigger"),
    ("noise_fn", "fn_behavior"),
)
H1_BESIDE: tuple[tuple[str, str], ...] = (("noise_fp", "fp_tolerance_decoy"),)
H2_FAMILY: tuple[tuple[str, str], ...] = (
    ("noise_fp", "fp_hash"),
    ("noise_fn", "fn_prompt_subset"),
)
# (arm, control). dtau and dacc are arm minus control.
H3_PAIRS: tuple[tuple[str, str], ...] = (
    ("fp_tolerance", "noise_fp"),
    ("fp_tolerance_decoy", "noise_fp"),
    ("fp_hash", "noise_fp"),
    ("fp_trigger", "noise_fp_trigger"),
    ("fn_behavior", "noise_fn"),
    ("fn_prompt_subset", "noise_fn"),
    ("fn_form", "noise_fn_form"),
)
FP_RULES = frozenset(
    {
        "fp_tolerance",
        "fp_tolerance_decoy",
        "fp_hash",
        "fp_trigger",
        "fp_trigger_enriched",
        "noise_fp",
        "noise_fp_trigger",
        "noise_fp_trigger_enriched",
    }
)
FN_RULES = frozenset(
    {
        "fn_behavior",
        "fn_behavior_depleted",
        "fn_prompt_subset",
        "fn_form",
        "noise_fn",
        "noise_fn_depleted",
        "noise_fn_form",
        "combo_fnform_fphash",
    }
)
H4_SHARED = (
    "fp_tolerance",
    "fp_tolerance_decoy",
    "fp_trigger",
    "fp_trigger_enriched",
    "fn_behavior",
    "fn_behavior_depleted",
    "fn_form",
)
H4_PROMPT_SPECIFIC = ("fp_hash", "fn_prompt_subset")

H1_RULE = (
    "Superiority: p_sup = P*(stat <= margin). A family contrast is rejected when "
    "its Holm-adjusted p_sup is <= alpha and every paired seed's contrast is > 0. "
    "H1 holds if at least one in-study, non-withdrawn family contrast is rejected. "
    "The beside contrast uses the same superiority and seed-sign rules and is not "
    "in the Holm family."
)
H2_RULE = (
    "TOST: p_tost = max(P*(stat <= -margin), P*(stat >= margin)). "
    "Holm-Bonferroni across the in-study, non-withdrawn contrasts at alpha. "
    "H2 holds if at least one such contrast remains and every one of them is rejected."
)
H3_RULE = (
    "dtau = tau_pool(arm) - tau_pool(control), dacc = final accuracy(arm) - "
    "accuracy(control) averaged over the pair's own planned seeds, rho = Spearman "
    "with average ranks. H3 holds if at least min_pairs pairs are non-withdrawn, "
    "the one-sided permutation p-value of rho is < alpha, and the 5th percentile "
    "of rho under the crossed bootstrap (forecasts fixed, accuracies resampled; "
    "one prompt draw for every pair, one seed draw per distinct seed list) is > 0."
)
H3_NOT_TESTED = (
    "analysis.h3.tested is false: H3 is not a hypothesis of this study. The "
    "statistics above are descriptive and carry no decision."
)
H10_RULE = (
    "Per cell. For each pair (a, b) the contrast is acc(a) - acc(b), with the "
    "crossed seed-by-prompt bootstrap used by the other contrasts. "
    "dtau = tau_pool(a) - tau_pool(b) from the lock. "
    "signed = sign(dtau) * (acc(a) - acc(b)), and its interval is that "
    "bootstrap multiplied by sign(dtau). A pair is withdrawn when either arm "
    "is withdrawn."
)
ORACLE_RULE = (
    "Oracle held-out gain is the mean over seeds of (final prompt-mean accuracy "
    "minus step-0 prompt-mean accuracy). Gain < oracle_gain_min labels H1, H2, H3 "
    "and the E2 natural-verifier tests inconclusive. The statistical results are "
    "still computed."
)
NATURAL_RULE = (
    "E2. The contrast is acc(control) - acc(natural), where the control is resampled "
    "noise matched to the natural verifier's step-0 channel rates. It differs from 0 "
    "when its two-sided crossed-bootstrap interval at level excludes 0. Locality is "
    "TOST at +/- margin: p_tost <= alpha. The screen direction counts only when the "
    "interval excludes 0; it matches when the interval's sign equals the sign of "
    "tau_pool(control) - tau_pool(natural) from the lock. oracle - natural is "
    "reported for scale and is not a test."
)
ADJUST_RULE = (
    "Adjusted contrast = observed contrast - b * dJbar. dJbar is the "
    "control-minus-arm difference in time-averaged on-policy J, the trapezoid "
    "of train-split eval J from step 0 to the final step. b is the OLS slope of "
    "final held-out accuracy on that time average, across every non-withdrawn run."
)
KNIFE_RULE = (
    "A run is knife-edge when held-out accuracy rises by >= rise above step 0 "
    "and a later eval falls by >= fall below the running peak. The label does "
    "not change a hypothesis. The endpoint stays the final step."
)
RELIABILITY_RULE = (
    "Documented. The split-half cosine of g_V on the locked audit, per arm, "
    "against the P3 gate. An arm below it has a noisier forecast; no hypothesis "
    "changes."
)


def audit_withdrawal_reasons(
    info: Mapping[str, Any],
    *,
    rate_miss_max: float = 0.02,
) -> list[str]:
    """Why an arm is withdrawn from the audit, or an empty list.

    A targeted arm whose ``rate_miss`` exceeds ``rate_miss_max``, or is
    undefined, is withdrawn.
    A phrase arm (``trigger_fp`` or ``behavior_fn``) is withdrawn when
    ``fell_back`` is true or ``validated`` is false. ``capped`` withdraws only
    the decoy (``tolerance_band_fp``) that cannot reach its target; ``form_fn``
    is capped by design and runs at its achievable rate. Equality with the
    threshold stays in.
    """
    reasons: list[str] = []
    rate_miss = info.get("rate_miss")
    if info.get("targeted"):
        if rate_miss is None:
            reasons.append("rate_undefined")
        elif float(rate_miss) > rate_miss_max:
            reasons.append("rate_miss")
    if info.get("type") in PHRASE_TYPES:
        if info.get("fell_back"):
            reasons.append("fell_back")
        if info.get("validated") is False:
            reasons.append("validated")
    if info.get("type") == "tolerance_band_fp" and info.get("capped"):
        reasons.append("capped")
    return reasons


def decide_audit_withdrawals(
    validity: Mapping[str, Mapping[str, Any]],
    *,
    rate_miss_max: float = 0.02,
    forecasts: Mapping[str, Any] | None = None,
    menu: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Withdrawal record decided from :func:`arm_validity` before any training.

    ``forecasts`` is the ``forecasts.json`` document. When it carries
    ``patch_selection.same_selection`` true, ``patch_rate`` is withdrawn with
    reason ``same_selection``. A missing ``patch_selection`` block leaves the
    rate rules unchanged.

    ``menu`` is the cell's trained arms. The audit calibrates every arm in the
    arms file, so a core cell also fits full-menu arms it never trains; those
    are listed under ``outside_menu`` and never withdrawn, so they cannot
    appear as withdrawals of that cell.
    """
    audit: dict[str, Any] = {}
    for arm, info in validity.items():
        reasons = audit_withdrawal_reasons(info, rate_miss_max=rate_miss_max)
        if reasons:
            audit[str(arm)] = {"reasons": reasons, "validity": dict(info)}
    _apply_same_selection(audit, validity, forecasts)
    record: dict[str, Any] = {"schema": "vfates.withdrawals/1"}
    if menu is not None:
        trained = {str(arm) for arm in menu}
        record["outside_menu"] = sorted(str(arm) for arm in validity if str(arm) not in trained)
        audit = {arm: entry for arm, entry in audit.items() if arm in trained}
    record.update({"audit": audit, "training": {}, "withdrawn": sorted(audit)})
    return record


def _apply_same_selection(
    audit: dict[str, Any],
    validity: Mapping[str, Mapping[str, Any]],
    forecasts: Mapping[str, Any] | None,
) -> None:
    if not isinstance(forecasts, Mapping):
        return
    patch = forecasts.get("patch_selection")
    if not isinstance(patch, Mapping) or patch.get("same_selection") is not True:
        return
    entry = audit.get("patch_rate")
    if entry is None:
        info = validity.get("patch_rate", {})
        entry = {"reasons": [], "validity": dict(info) if isinstance(info, Mapping) else {}}
        audit["patch_rate"] = entry
    reasons = list(entry.get("reasons") or [])
    if "same_selection" not in reasons:
        reasons.append("same_selection")
    entry["reasons"] = reasons


def _pairs(
    block: Mapping[str, Any] | None, default: Sequence[tuple[str, str]], *, key: str
) -> list[tuple[str, str]]:
    if not block or key not in block:
        return list(default)
    rows = []
    for row in block[key]:
        rows.append((str(row["control"]), str(row["arm"])))
    return rows


def _h3_pairs(block: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Return ``(arm, control)`` pairs."""
    if not block or "pairs" not in block:
        return list(H3_PAIRS)
    return [(str(row["arm"]), str(row["control"])) for row in block["pairs"]]


def _h10_pairs(block: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Return ``(a, b)`` pairs. The contrast is ``acc(a) - acc(b)``."""
    if not block or "pairs" not in block:
        return []
    return [(str(row["a"]), str(row["b"])) for row in block["pairs"]]


def _contrast_name(control: str, arm: str) -> str:
    return f"{control} - {arm}"


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


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.floating | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer | int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    return value


def _seeds_of(planned: Mapping[str, Sequence[int]], arm: str) -> list[int]:
    return [int(s) for s in planned[arm]]


def _status(
    control: str,
    arm: str,
    *,
    planned: Mapping[str, Sequence[int]],
    withdrawn: set[str],
    allow_partial: bool,
) -> str:
    in_study = control in planned and arm in planned
    if not in_study:
        if allow_partial:
            return "not_in_study"
        raise ValueError(f"contrast {_contrast_name(control, arm)} uses an arm that is not trained")
    if control in withdrawn or arm in withdrawn:
        return "withdrawn"
    if _seeds_of(planned, control) != _seeds_of(planned, arm):
        raise ValueError(f"{control} and {arm} have different planned seeds")
    return "tested"


def _prompt_means(run_dir: Path, step: int) -> dict[str, float]:
    acc = prompt_accuracy(run_dir, "heldout", step)
    if any(not math.isfinite(float(v)) for v in acc.values()):
        raise ValueError(f"non-finite prompt accuracy in {run_dir} at step {step}")
    return {str(k): float(v) for k, v in acc.items()}


def _mean_acc(acc: Mapping[str, float]) -> float:
    if not acc:
        raise ValueError("prompt accuracy map is empty")
    return float(np.mean(list(acc.values())))


def _aligned_prompts(maps: Sequence[Mapping[str, float]], *, what: str) -> None:
    sets = [set(m) for m in maps]
    if any(s != sets[0] for s in sets):
        raise ValueError(f"{what} runs do not share an identical held-out prompt set")


def _contrast_record(
    control: str,
    arm: str,
    *,
    status: str,
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    final_step: int,
    n_boot: int,
    level: float,
    margin: float,
    boot_seed: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": _contrast_name(control, arm),
        "control": control,
        "arm": arm,
        "status": status,
    }
    if status != "tested":
        base["rule"] = f"contrast {status}; not tested"
        return base
    seeds = _seeds_of(planned, control)
    pair_dirs = [(runs[control][s], runs[arm][s]) for s in seeds]
    maps = [_prompt_means(path, final_step) for pair in pair_dirs for path in pair]
    _aligned_prompts(maps, what=_contrast_name(control, arm))
    diff, _pids = difference_matrix(pair_dirs, step=final_step)
    crossed = paired_contrast_crossed(
        diff, n_boot=n_boot, level=level, seed=boot_seed, margin=margin
    )
    base.update(
        {
            "seeds": seeds,
            "point": crossed["point"],
            "per_seed": crossed["per_seed"],
            "lo": crossed["lo"],
            "hi": crossed["hi"],
            "level": level,
            "margin": margin,
            "p_sup": crossed["p_sup"],
            "p_tost": crossed["p_tost"],
            "p_le_0": crossed["p_le_0"],
            "n_prompts": crossed["n_prompts"],
            "n_seeds": crossed["n_seeds"],
            "n_boot": n_boot,
            "prompt_only": crossed["prompt_only"],
            "seed_t": crossed["seed_t"],
        }
    )
    return base


def _apply_superiority(records: list[dict[str, Any]], *, alpha: float, in_family: bool) -> None:
    tested = [row for row in records if row["status"] == "tested"]
    if in_family and tested:
        pvalues = {row["name"]: row["p_sup"] for row in tested}
        adjusted = holm_adjusted(pvalues)
        rejected = holm(pvalues, alpha)
    else:
        adjusted = {}
        rejected = {}
    for row in tested:
        seeds_positive = all(value > 0.0 for value in row["per_seed"])
        row["all_seeds_positive"] = seeds_positive
        if in_family:
            row["holm_adjusted_p"] = adjusted[row["name"]]
            row["holm_reject"] = rejected[row["name"]]
            row["rejects"] = bool(rejected[row["name"]] and seeds_positive)
            row["rule"] = (
                "Rejected when Holm-adjusted p_sup(margin) <= alpha and every "
                "paired seed contrast is > 0."
            )
        else:
            row["holm_adjusted_p"] = None
            row["holm_reject"] = None
            row["rejects"] = bool(row["p_sup"] <= alpha and seeds_positive)
            row["rule"] = (
                "Beside H1, outside the Holm family. Rejected when p_sup(margin) <= alpha "
                "and every paired seed contrast is > 0."
            )


def _apply_tost(records: list[dict[str, Any]], *, alpha: float) -> None:
    tested = [row for row in records if row["status"] == "tested"]
    if tested:
        pvalues = {row["name"]: row["p_tost"] for row in tested}
        adjusted = holm_adjusted(pvalues)
        rejected = holm(pvalues, alpha)
    else:
        adjusted = {}
        rejected = {}
    for row in tested:
        row["holm_adjusted_p"] = adjusted[row["name"]]
        row["holm_reject"] = rejected[row["name"]]
        row["rejects"] = bool(rejected[row["name"]])
        row["rule"] = "Rejected when Holm-adjusted p_tost(margin) <= alpha."


def _family_holds(records: list[dict[str, Any]], *, how: str) -> bool:
    tested = [row for row in records if row["status"] == "tested"]
    if not tested:
        return False
    if how == "any":
        return any(row["rejects"] for row in tested)
    return all(row["rejects"] for row in tested)


def _label(holds: bool, inconclusive: bool) -> str:
    if inconclusive:
        return "inconclusive"
    return "holds" if holds else "does_not_hold"


def _natural_block(
    cfg: Mapping[str, Any],
    *,
    forecasts: Mapping[str, Any],
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    withdrawn: set[str],
    allow_partial: bool,
    final_step: int,
    n_boot: int,
    level: float,
    margin: float,
    boot_seed: int,
    alpha: float,
    inconclusive: bool,
) -> dict[str, Any]:
    """E2's preregistered tests of one natural verifier against its matched control."""
    natural = str(cfg["natural"])
    control = str(cfg.get("control", "resampled"))
    oracle = str(cfg.get("oracle", "oracle"))

    def record(first: str, second: str) -> dict[str, Any]:
        status = _status(
            first, second, planned=planned, withdrawn=withdrawn, allow_partial=allow_partial
        )
        return _contrast_record(
            first,
            second,
            status=status,
            runs=runs,
            planned=planned,
            final_step=final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=boot_seed,
        )

    contrast = record(control, natural)
    block: dict[str, Any] = {
        "rule": NATURAL_RULE,
        "natural": natural,
        "control": control,
        "oracle": oracle,
        "alpha": alpha,
        "contrast": contrast,
        "scale": record(oracle, natural),
    }
    if contrast["status"] != "tested":
        block.update({"differs": None, "locality": None, "screen_direction": None})
        return block
    lo, hi = float(contrast["lo"]), float(contrast["hi"])
    differs = lo > 0.0 or hi < 0.0
    equivalent = float(contrast["p_tost"]) <= alpha
    taus: dict[str, float] = {}
    for arm in (control, natural):
        entry = forecasts.get(arm)
        tau = None if entry is None else _finite(forecast_row(entry).get("tau_pool"))
        if tau is None:
            raise ValueError(f"E2 arm {arm} has no finite tau_pool in the lock")
        taus[arm] = tau
    dtau = taus[control] - taus[natural]
    matches: bool | None = None
    if not differs:
        direction = "not_counted"
    elif dtau == 0.0:
        direction = "undefined"
    else:
        matches = (lo > 0.0) == (dtau > 0.0)
        direction = "inconclusive" if inconclusive else ("matches" if matches else "does_not_match")
    block.update(
        {
            "differs": {
                "value": differs,
                "lo": lo,
                "hi": hi,
                "level": level,
                "decision": "inconclusive"
                if inconclusive
                else ("differs" if differs else "does_not_differ"),
            },
            "locality": {
                "p_tost": float(contrast["p_tost"]),
                "margin": margin,
                "equivalent": equivalent,
                "decision": "inconclusive"
                if inconclusive
                else ("equivalent" if equivalent else "not_equivalent"),
            },
            "screen_direction": {
                "tau_control": taus[control],
                "tau_natural": taus[natural],
                "dtau": dtau,
                "counted": differs,
                "matches": matches,
                "decision": direction,
            },
        }
    )
    return block


def _natural_configs(cfg: Mapping[str, Any] | Sequence[Any]) -> list[Mapping[str, Any]]:
    """One mapping, or a list of ``{natural, control, oracle}`` mappings."""
    if isinstance(cfg, Mapping):
        return [cfg]
    if isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
        items: list[Mapping[str, Any]] = []
        for item in cfg:
            if not isinstance(item, Mapping):
                raise ValueError("analysis.natural entries must be mappings")
            items.append(item)
        if not items:
            raise ValueError("analysis.natural list is empty")
        return items
    raise ValueError("analysis.natural must be a mapping or a list of mappings")


def natural_pair_blocks(
    cfg: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    forecasts: Mapping[str, Any],
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    withdrawn: set[str],
    allow_partial: bool,
    final_step: int,
    n_boot: int,
    level: float,
    margin: float,
    boot_seed: int,
    alpha: float,
    inconclusive: bool,
) -> list[dict[str, Any]]:
    """One natural-verifier block per configured pair.

    A mapping config is the single pair used by the original E2 block. A list
    is one block each (math has ``natural`` and ``natural_strict``).
    """
    return [
        _natural_block(
            item,
            forecasts=forecasts,
            runs=runs,
            planned=planned,
            withdrawn=withdrawn,
            allow_partial=allow_partial,
            final_step=final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=boot_seed,
            alpha=alpha,
            inconclusive=inconclusive,
        )
        for item in _natural_configs(cfg)
    ]


def _j_of(row: Mapping[str, Any]) -> float | None:
    if row.get("J") is not None and math.isfinite(float(row["J"])):
        return float(row["J"])
    fpr, fnr = row.get("fpr"), row.get("fnr")
    if fpr is None or fnr is None:
        return None
    if not (math.isfinite(float(fpr)) and math.isfinite(float(fnr))):
        return None
    return (1.0 - float(fnr)) - float(fpr)


def _time_averaged_j(run_dir: Path, split: str, final_step: int) -> float:
    rows = [r for r in eval_curve(run_dir, split) if int(r["step"]) <= final_step]
    usable = [(int(r["step"]), _j_of(r)) for r in rows]
    usable = [(step, jay) for step, jay in usable if jay is not None]
    if not usable or usable[0][0] != 0 or usable[-1][0] != final_step:
        raise ValueError(f"{run_dir}: {split} J path must include step 0 and step {final_step}")
    steps, values = zip(*usable, strict=True)
    return time_averaged(steps, values)


def _monitor(outcome_row: Mapping[str, Any], channel: str, name: str) -> float | None:
    key = "monitors_wrong" if channel == "fp" else "monitors_correct"
    raw = outcome_row.get(key) or {}
    if name not in raw or raw[name] is None:
        return None
    return float(raw[name])


def _eval_row(run_dir: Path, split: str, step: int) -> dict[str, Any]:
    matches = [r for r in eval_curve(run_dir, split) if int(r["step"]) == step]
    if not matches:
        raise ValueError(f"{run_dir}: no {split} eval at step {step}")
    return matches[-1]


def confirmatory_report(
    *,
    audit_dir: str | Path,
    runs: Mapping[str, Mapping[int, str | Path]],
    early_runs: Mapping[str, str | Path],
    analysis: Mapping[str, Any],
    withdrawals: Mapping[str, Any],
    planned_seeds: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """H1 to H6, H10 when configured, withdrawals, sensitivities, and labels.

    Every planned seed of a non-withdrawn arm must be present and must have an
    eval at the preregistered final step. Hypothesis fields are omitted when
    ``analysis['hypotheses']`` is false (KL check and other descriptive runs).
    """
    audit_dir = Path(audit_dir)
    run_map: dict[str, dict[int, Path]] = {
        arm: {int(seed): Path(path) for seed, path in seeds.items()} for arm, seeds in runs.items()
    }
    early_map = {arm: Path(path) for arm, path in early_runs.items()}
    planned = {arm: [int(s) for s in seeds] for arm, seeds in planned_seeds.items()}
    withdrawn = {str(arm) for arm in withdrawals.get("withdrawn", [])}
    final_step = int(analysis["final_step"])
    hypotheses = bool(analysis.get("hypotheses", True))
    allow_partial = bool(analysis.get("allow_partial_menu", False))
    alpha = float(analysis.get("alpha", 0.05))
    margin = float(analysis.get("margin", 0.05))
    level = float(analysis.get("level", 0.95))
    n_boot = int(analysis.get("n_boot", 10_000))
    boot_seed = int(analysis.get("seed", 0))
    gain_min = float(analysis.get("oracle_gain_min", 0.10))
    knife_cfg = analysis.get("knife_edge") or {}
    rise = float(knife_cfg.get("rise", 0.05))
    fall = float(knife_cfg.get("fall", 0.10))
    early_cfg = analysis.get("early") or {}
    early_enabled = bool(early_cfg.get("enabled", hypotheses))
    early_step = int(early_cfg.get("steps", 25))
    j_split = str(analysis.get("j_split", "train"))

    missing: list[str] = []
    for arm, seeds in planned.items():
        if arm in withdrawn:
            continue
        have = run_map.get(arm, {})
        for seed in seeds:
            if seed not in have:
                missing.append(f"{arm} seed {seed}")
        if early_enabled and arm not in early_map:
            missing.append(f"{arm} early")
    if missing:
        raise ValueError("incomplete planned runs: " + ", ".join(missing))

    active_dirs = [
        run_map[arm][seed]
        for arm, seeds in planned.items()
        if arm not in withdrawn
        for seed in seeds
    ]
    require_final_step(active_dirs, final_step)
    if early_enabled:
        early_dirs = [early_map[arm] for arm in planned if arm not in withdrawn]
        require_final_step(early_dirs, early_step)

    active_arms = [arm for arm in planned if arm not in withdrawn]
    endpoints = _endpoints(run_map, planned, active_arms, final_step)
    knives = _knife_labels(run_map, planned, active_arms, final_step, rise, fall)
    paths = _rate_paths(run_map, planned, active_arms, final_step)
    oracle = _oracle_gain(endpoints, gain_min)
    inconclusive = bool(oracle["inconclusive"])

    report: dict[str, Any] = {
        "schema": "vfates.confirmatory/1",
        "hypotheses": hypotheses,
        "final_step": final_step,
        "alpha": alpha,
        "margin": margin,
        "level": level,
        "n_boot": n_boot,
        "bootstrap_seed": boot_seed,
        "withdrawals": _jsonify(dict(withdrawals)),
        "endpoints": endpoints,
        "knife_edge": {"rule": KNIFE_RULE, "rise": rise, "fall": fall, "runs": knives},
        "on_policy": paths,
        "oracle_gain": oracle,
        "inconclusive": inconclusive,
    }
    natural_cfg = analysis.get("natural")
    forecasts = _locked_forecasts(audit_dir) if hypotheses or natural_cfg else {}
    if forecasts:
        report["reliability"] = _reliability_block(
            forecasts, float(analysis.get("reliability_min", 0.8))
        )
    if natural_cfg:
        blocks = natural_pair_blocks(
            natural_cfg,
            forecasts=forecasts,
            runs=run_map,
            planned=planned,
            withdrawn=withdrawn,
            allow_partial=allow_partial,
            final_step=final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=boot_seed,
            alpha=alpha,
            inconclusive=inconclusive,
        )
        report["natural"] = blocks[0] if isinstance(natural_cfg, Mapping) else blocks
    if not hypotheses:
        report["rule"] = (
            "E2: H1 to H6 are not tested; the natural-verifier tests are under 'natural'."
            if natural_cfg
            else "Descriptive run: H1 to H6 are not tested."
        )
        return _jsonify(report)

    h1_cfg = analysis.get("h1") or {}
    h2_cfg = analysis.get("h2") or {}
    h3_cfg = analysis.get("h3") or {}
    h3_tested = bool(h3_cfg.get("tested", True))
    family = [
        _contrast_record(
            control,
            arm,
            status=_status(
                control, arm, planned=planned, withdrawn=withdrawn, allow_partial=allow_partial
            ),
            runs=run_map,
            planned=planned,
            final_step=final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=boot_seed,
        )
        for control, arm in _pairs(h1_cfg, H1_FAMILY, key="contrasts")
    ]
    beside = [
        _contrast_record(
            control,
            arm,
            status=_status(
                control, arm, planned=planned, withdrawn=withdrawn, allow_partial=allow_partial
            ),
            runs=run_map,
            planned=planned,
            final_step=final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=boot_seed,
        )
        for control, arm in _pairs(h1_cfg, H1_BESIDE, key="beside")
    ]
    _apply_superiority(family, alpha=alpha, in_family=True)
    _apply_superiority(beside, alpha=alpha, in_family=False)
    h2 = [
        _contrast_record(
            control,
            arm,
            status=_status(
                control, arm, planned=planned, withdrawn=withdrawn, allow_partial=allow_partial
            ),
            runs=run_map,
            planned=planned,
            final_step=final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=boot_seed,
        )
        for control, arm in _pairs(h2_cfg, H2_FAMILY, key="contrasts")
    ]
    _apply_tost(h2, alpha=alpha)
    slope, jbars = _fit_j_slope(run_map, planned, active_arms, final_step, j_split, endpoints)
    _attach_adjustment(family + beside, jbars, slope, j_split)
    h1_holds = _family_holds(family, how="any")
    h2_holds = _family_holds(h2, how="all")
    h3 = _h3_block(
        pairs=_h3_pairs(h3_cfg),
        forecasts=forecasts,
        runs=run_map,
        early_runs=early_map,
        planned=planned,
        withdrawn=withdrawn,
        allow_partial=allow_partial,
        final_step=final_step,
        early_step=early_step,
        early_enabled=early_enabled,
        n_boot=n_boot,
        boot_seed=boot_seed,
        alpha=alpha,
        min_pairs=int(h3_cfg.get("min_pairs", 5)),
        active_arms=active_arms,
    )
    h4 = _h4_block(forecasts, analysis.get("h4") or {}, active_arms, withdrawn)
    describes, calibration_loaded = _arm_describes(audit_dir)
    h5 = _h5_block(
        run_map,
        planned,
        _h3_pairs(h3_cfg),
        withdrawn,
        allow_partial,
        final_step,
        describes,
        calibration_loaded,
    )
    h6 = _h6_block(forecasts, endpoints, active_arms)
    report.update(
        {
            "h1": {
                "rule": H1_RULE,
                "holds": h1_holds,
                "decision": _label(h1_holds, inconclusive),
                "contrasts": family,
                "beside": beside,
            },
            "h2": {
                "rule": H2_RULE,
                "holds": h2_holds,
                "decision": _label(h2_holds, inconclusive),
                "contrasts": h2,
            },
            "h3": (
                {**h3, "tested": True, "decision": _label(bool(h3["holds"]), inconclusive)}
                if h3_tested
                else {
                    **h3,
                    "tested": False,
                    "holds": None,
                    "decision": "not_tested",
                    "note": H3_NOT_TESTED,
                }
            ),
            "h4": h4,
            "h5": h5,
            "h6": h6,
            "adjustment": {
                "rule": ADJUST_RULE,
                "split": j_split,
                "slope": slope,
                "n_runs": len(jbars),
            },
        }
    )
    h10_cfg = analysis.get("h10")
    if isinstance(h10_cfg, Mapping):
        report["h10"] = _h10_block(
            pairs=_h10_pairs(h10_cfg),
            forecasts=forecasts,
            runs=run_map,
            planned=planned,
            withdrawn=withdrawn,
            allow_partial=allow_partial,
            final_step=final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=boot_seed,
        )
    return _jsonify(report)


def _arm_describes(audit_dir: Path) -> tuple[dict[str, dict[str, Any]], bool]:
    """Calibration ``describe()`` dicts, merged with report details.

    Returns ``({}, False)`` when ``calibration.json`` is absent, which keeps a
    chain-sum report that never wrote one on the name-list path.
    """
    path = audit_dir / "calibration.json"
    if not path.exists():
        return {}, False
    doc = read_json(path)
    if not isinstance(doc, Mapping):
        raise ValueError(f"calibration.json is not a mapping: {path}")
    described = doc.get("arms")
    report = doc.get("report")
    reported: Mapping[str, Any] = {}
    if isinstance(report, Mapping) and isinstance(report.get("arms"), Mapping):
        reported = report["arms"]
    names: set[str] = set()
    if isinstance(described, Mapping):
        names.update(str(key) for key in described)
    names.update(str(key) for key in reported)
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        merged: dict[str, Any] = {}
        raw_report = reported.get(name)
        if isinstance(raw_report, Mapping):
            merged.update(raw_report)
            details = raw_report.get("details")
            if isinstance(details, Mapping):
                for key in ("matched_arm", "match", "base"):
                    if key in details and key not in merged:
                        merged[key] = details[key]
        raw_desc = described.get(name) if isinstance(described, Mapping) else None
        if isinstance(raw_desc, Mapping) and ("kind" in raw_desc or "type" in raw_desc):
            merged.update(dict(raw_desc))
        if merged:
            merged.setdefault("name", name)
            out[name] = merged
    return out, True


def _locked_forecasts(audit_dir: Path) -> dict[str, Any]:
    lock_path = audit_dir / "lock.json"
    forecasts_path = audit_dir / "forecasts.json"
    if not forecasts_path.exists():
        raise ValueError(f"audit is missing forecasts.json: {audit_dir}")
    forecasts = read_json(forecasts_path)
    assert_matches_lock(lock_path, forecasts)
    locked = load_lock(lock_path)
    if not isinstance(locked, dict):
        raise ValueError(f"locked payload is not an arm map: {lock_path}")
    return locked


def _endpoints(
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    arms: Sequence[str],
    final_step: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in arms:
        per_seed = []
        for seed in planned[arm]:
            final_map = _prompt_means(runs[arm][seed], final_step)
            step0_map = _prompt_means(runs[arm][seed], 0)
            _aligned_prompts([final_map, step0_map], what=f"{arm} seed {seed}")
            final = _mean_acc(final_map)
            step0 = _mean_acc(step0_map)
            per_seed.append({"seed": seed, "step0": step0, "final": final, "gain": final - step0})
        finals = [row["final"] for row in per_seed]
        step0s = [row["step0"] for row in per_seed]
        out[arm] = {
            "seeds": per_seed,
            "final_mean": float(np.mean(finals)),
            "step0_mean": float(np.mean(step0s)),
            "gain_mean": float(np.mean(finals) - np.mean(step0s)),
        }
    return out


def _forecast_arm_items(
    forecasts: Mapping[str, Any],
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Arm records in a forecasts document.

    Uses ``vfates.audit.iter_forecast_arms`` when that function is importable
    (it skips the non-arm key ``patch_selection``). If the import is missing,
    skips any key whose value is not a mapping or does not carry ``transfer``.
    """
    try:
        from vfates.audit import iter_forecast_arms
    except (ImportError, AttributeError):
        for name, entry in forecasts.items():
            if isinstance(entry, Mapping) and "transfer" in entry:
                yield str(name), entry
        return
    yield from iter_forecast_arms(forecasts)


def _reliability_block(forecasts: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    arms = {}
    for arm, forecast in _forecast_arm_items(forecasts):
        transfer = forecast.get("transfer")
        raw = transfer.get("reliability") if isinstance(transfer, Mapping) else None
        value = _finite(raw)
        arms[arm] = {"value": value, "below": value is not None and value < threshold}
    return {
        "rule": RELIABILITY_RULE,
        "threshold": threshold,
        "arms": arms,
        "below": sorted(arm for arm, row in arms.items() if row["below"]),
    }


def _oracle_gain(endpoints: Mapping[str, Any], gain_min: float) -> dict[str, Any]:
    if "oracle" not in endpoints:
        return {
            "value": None,
            "threshold": gain_min,
            "inconclusive": True,
            "rule": ORACLE_RULE,
            "reason": "oracle arm is not in the analysed runs",
        }
    value = float(endpoints["oracle"]["gain_mean"])
    return {
        "value": value,
        "threshold": gain_min,
        "inconclusive": value < gain_min,
        "rule": ORACLE_RULE,
        "reason": "gain < threshold" if value < gain_min else "gain >= threshold",
    }


def _knife_labels(
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    arms: Sequence[str],
    final_step: int,
    rise: float,
    fall: float,
) -> list[dict[str, Any]]:
    labels = []
    for arm in arms:
        for seed in planned[arm]:
            curve = [
                (int(row["step"]), float(row["oracle_acc"]))
                for row in eval_curve(runs[arm][seed], "heldout")
                if int(row["step"]) <= final_step
            ]
            if not curve or curve[0][0] != 0:
                raise ValueError(f"{arm} seed {seed}: held-out curve must start at step 0")
            accs = [acc for _, acc in curve]
            labels.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "knife_edge": knife_edge(accs, rise=rise, fall=fall),
                    "step0": accs[0],
                    "peak": max(accs),
                    "final": accs[-1],
                }
            )
    return labels


def _rate_paths(
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    arms: Sequence[str],
    final_step: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in arms:
        per_seed = {}
        for seed in planned[arm]:
            per_seed[str(seed)] = {
                split: [
                    row for row in rate_path(runs[arm][seed], split) if row["step"] <= final_step
                ]
                for split in ("train", "heldout")
            }
        out[arm] = per_seed
    return out


def _fit_j_slope(
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    arms: Sequence[str],
    final_step: int,
    split: str,
    endpoints: Mapping[str, Any],
) -> tuple[float | None, list[dict[str, Any]]]:
    rows = []
    for arm in arms:
        by_seed = {row["seed"]: row["final"] for row in endpoints[arm]["seeds"]}
        for seed in planned[arm]:
            jbar = _time_averaged_j(runs[arm][seed], split, final_step)
            rows.append({"arm": arm, "seed": seed, "jbar": jbar, "final": by_seed[seed]})
    slope = ols_slope([row["jbar"] for row in rows], [row["final"] for row in rows])
    return (None if not math.isfinite(slope) else float(slope)), rows


def _attach_adjustment(
    records: Sequence[dict[str, Any]],
    jbars: Sequence[Mapping[str, Any]],
    slope: float | None,
    split: str,
) -> None:
    index = {(row["arm"], row["seed"]): row["jbar"] for row in jbars}
    for record in records:
        if record["status"] != "tested":
            continue
        seeds = record["seeds"]
        djbars = [index[(record["control"], s)] - index[(record["arm"], s)] for s in seeds]
        d_jbar = float(np.mean(djbars))
        record["dJbar"] = d_jbar
        record["j_split"] = split
        if slope is None:
            record["adjusted"] = None
            record["adjustment_rule"] = ADJUST_RULE + " Slope is undefined because J did not vary."
        else:
            record["adjusted"] = adjusted_contrast(record["point"], d_jbar, slope)
            record["adjustment_slope"] = slope
            record["adjustment_rule"] = ADJUST_RULE


def _h3_block(
    *,
    pairs: Sequence[tuple[str, str]],
    forecasts: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[int, Path]],
    early_runs: Mapping[str, Path],
    planned: Mapping[str, Sequence[int]],
    withdrawn: set[str],
    allow_partial: bool,
    final_step: int,
    early_step: int,
    early_enabled: bool,
    n_boot: int,
    boot_seed: int,
    alpha: float,
    min_pairs: int,
    active_arms: Sequence[str],
) -> dict[str, Any]:
    listed: list[dict[str, Any]] = []
    for arm, control in pairs:
        # _status expects (control, arm) order for the seed-list check.
        if arm not in planned or control not in planned:
            if allow_partial:
                listed.append({"arm": arm, "control": control, "status": "not_in_study"})
                continue
            raise ValueError(f"H3 pair {arm} / {control} is not in the trained menu")
        if arm in withdrawn or control in withdrawn:
            listed.append({"arm": arm, "control": control, "status": "withdrawn"})
            continue
        if _seeds_of(planned, arm) != _seeds_of(planned, control):
            raise ValueError(f"H3 pair {arm} / {control} has different planned seeds")
        listed.append({"arm": arm, "control": control, "status": "tested"})
    tested = [row for row in listed if row["status"] == "tested"]
    scalars: dict[str, dict[str, float | None]] = {}
    for arm in {name for row in tested for name in (row["arm"], row["control"])}:
        scalars[arm] = _baseline_scalars(
            arm,
            forecasts.get(arm),
            early_runs.get(arm),
            early_step if early_enabled else None,
        )
    pair_rows: list[dict[str, Any]] = []
    arm_cube = []
    control_cube = []
    pair_seeds: list[list[int]] = []
    dtau = []
    expected_pids: list[str] | None = None
    for row in tested:
        arm, control = row["arm"], row["control"]
        tau_arm = scalars[arm].get("tau_pool")
        tau_control = scalars[control].get("tau_pool")
        if tau_arm is None or tau_control is None:
            raise ValueError(f"H3 pair {arm} / {control} is missing tau_pool")
        seeds = _seeds_of(planned, arm)
        if not seeds:
            raise ValueError(f"H3 pair {arm} / {control} has no planned seed")
        arm_maps = [_prompt_means(runs[arm][s], final_step) for s in seeds]
        control_maps = [_prompt_means(runs[control][s], final_step) for s in seeds]
        _aligned_prompts(arm_maps + control_maps, what=f"H3 {arm}/{control}")
        pids = sorted(arm_maps[0])
        if expected_pids is None:
            expected_pids = pids
        elif pids != expected_pids:
            raise ValueError("H3 pairs do not share the held-out prompt ids")
        arm_cube.append(np.array([[m[p] for p in pids] for m in arm_maps], dtype=float))
        control_cube.append(np.array([[m[p] for p in pids] for m in control_maps], dtype=float))
        pair_seeds.append(seeds)
        d_tau = float(tau_arm - tau_control)
        dtau.append(d_tau)
        pair_rows.append(
            {
                **row,
                "dtau": d_tau,
                "seeds": seeds,
                "tau_arm": tau_arm,
                "tau_control": tau_control,
            }
        )
    point = np.zeros(0)
    boots = np.zeros((0, 0))
    if pair_rows:
        point, boots = crossed_pair_dacc(
            arm_cube,
            control_cube,
            n_boot=n_boot,
            seed=boot_seed,
            seed_lists=pair_seeds,
        )
        for i, row in enumerate(pair_rows):
            row["dacc"] = float(point[i])
    dtau_arr = np.asarray(dtau, dtype=float)
    dacc_arr = np.asarray(point, dtype=float)
    perm: dict[str, Any]
    if dtau_arr.size >= 2:
        perm = permutation_spearman_p(dtau_arr, dacc_arr)
    else:
        perm = {
            "rho": float("nan"),
            "p": float("nan"),
            "n": int(dtau_arr.size),
            "method": "undefined",
        }
    q05 = None
    quantile_draws: list[float] = []
    if dtau_arr.size >= 2 and boots.size:
        for draw in boots:
            quantile_draws.append(spearman(dtau_arr, draw))
        finite = np.asarray([v for v in quantile_draws if math.isfinite(v)], dtype=float)
        if finite.size:
            q05 = float(np.quantile(finite, 0.05))
    p_value = perm.get("p")
    p_ok = isinstance(p_value, float) and math.isfinite(p_value) and p_value < alpha
    q_ok = q05 is not None and q05 > 0.0
    holds = len(pair_rows) >= min_pairs and p_ok and q_ok
    baselines = _h3_baselines(tested, scalars, dtau_arr, dacc_arr, boots, quantile_draws)
    concordance = _concordance(pair_rows)
    overall = _overall_spearman(
        forecasts, runs, planned, active_arms, final_step, n_boot, boot_seed
    )
    return {
        "rule": H3_RULE,
        "holds": holds,
        "min_pairs": min_pairs,
        "n_pairs": len(pair_rows),
        "seed_lists": [list(s) for s in dict.fromkeys(tuple(s) for s in pair_seeds)],
        "rho": perm.get("rho"),
        "permutation_p": p_value,
        "permutation_method": perm.get("method"),
        "q05": q05,
        "pairs": listed if not pair_rows else _merge_pair_status(listed, pair_rows),
        "baselines": baselines,
        "concordance": concordance,
        "overall_spearman": overall,
    }


def _merge_pair_status(
    listed: list[dict[str, Any]], tested_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {(row["arm"], row["control"]): row for row in tested_rows}
    merged = []
    for row in listed:
        key = (row["arm"], row["control"])
        merged.append(by_key.get(key, row))
    return merged


def _baseline_scalars(
    arm: str,
    forecast: Mapping[str, Any] | None,
    early_dir: Path | None,
    early_step: int | None,
) -> dict[str, float | None]:
    row: dict[str, float | None] = {
        "tau_pool": None,
        "J": None,
        "mean_precision": None,
        "modal_precision": None,
        "cos_gV_gO": None,
        "probe_heldout_auc": None,
        "tau_pool_std": None,
        "behaviour_alignment": None,
        "early_acc": None,
    }
    if forecast is not None:
        parsed = forecast_row(forecast)
        for key in (
            "tau_pool",
            "J",
            "mean_precision",
            "modal_precision",
            "cos_gV_gO",
            "tau_pool_std",
            "behaviour_alignment",
        ):
            row[key] = _finite(parsed.get(key))
        row["probe_heldout_auc"] = _probe_auc(parsed.get("probe_heldout_auc"))
    if early_dir is not None and early_step is not None:
        row["early_acc"] = _mean_acc(_prompt_means(early_dir, early_step))
    return row


def _h3_baselines(
    tested: Sequence[Mapping[str, Any]],
    scalars: Mapping[str, Mapping[str, float | None]],
    dtau: np.ndarray,
    dacc: np.ndarray,
    boots: np.ndarray,
    tau_draws: Sequence[float],
) -> dict[str, Any]:
    names = (
        "J",
        "mean_precision",
        "modal_precision",
        "cos_gV_gO",
        "probe_heldout_auc",
        "early_acc",
        "tau_pool_std",
        "behaviour_alignment",
    )
    out: dict[str, Any] = {}
    rho_tau = spearman(dtau, dacc) if dtau.size >= 2 else float("nan")
    for name in names:
        xs = []
        ok = True
        for row in tested:
            arm_v = scalars[row["arm"]].get(name)
            control_v = scalars[row["control"]].get(name)
            if arm_v is None or control_v is None:
                ok = False
                break
            xs.append(arm_v - control_v)
        if not ok or dtau.size < 2:
            out[name] = {
                "rho": None,
                "reason": "baseline missing on a tested pair or fewer than 2 pairs",
            }
            continue
        x = np.asarray(xs, dtype=float)
        rho_b = spearman(x, dacc)
        diffs = []
        if boots.size and tau_draws:
            for draw, rho_tau_b in zip(boots, tau_draws, strict=True):
                rho_draw = spearman(x, draw)
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
            "rule": (
                "Secondary. Forecast or baseline scalars stay fixed. Accuracies are "
                "resampled with the same crossed draws as rho(tau_pool). The interval "
                "is the 5th and 95th percentiles of rho(tau_pool) - rho(baseline)."
            ),
        }
    return out


def _concordance(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sign agreement of dtau and dacc on the tested H3 pairs that are also H1 pairs.

    Reported for every tested H3 pair. A zero on either side is not concordant.
    """
    rows = []
    h1_arms = {arm for _, arm in (*H1_FAMILY, *H1_BESIDE)}
    for row in pair_rows:
        dtau = float(row["dtau"])
        dacc = float(row["dacc"])
        same = (dtau > 0 and dacc > 0) or (dtau < 0 and dacc < 0)
        rows.append(
            {
                "arm": row["arm"],
                "control": row["control"],
                "dtau": dtau,
                "dacc": dacc,
                "concordant": same,
                "h1_pair": row["arm"] in h1_arms,
            }
        )
    h1_rows = [row for row in rows if row["h1_pair"]]
    return {
        "rule": (
            "Secondary. Concordant when dtau and dacc are both positive or both negative. "
            "H1 pairs are flagged; concordance is not a gate."
        ),
        "pairs": rows,
        "n_h1_concordant": sum(row["concordant"] for row in h1_rows),
        "n_h1_pairs": len(h1_rows),
    }


def _overall_spearman(
    forecasts: Mapping[str, Any],
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    arms: Sequence[str],
    final_step: int,
    n_boot: int,
    boot_seed: int,
) -> dict[str, Any]:
    present = [arm for arm in arms if arm in forecasts]
    outcomes = {
        arm: {
            "heldout_final": float(
                np.mean([_mean_acc(_prompt_means(runs[arm][s], final_step)) for s in planned[arm]])
            ),
            "heldout_delta": 0.0,
        }
        for arm in present
    }
    scored = (
        score_forecasts(forecasts, outcomes)
        if present
        else {
            "spearman": {},
            "tau_pool_mae": None,
            "n_arms": 0,
        }
    )
    keys = [
        key
        for key in (
            "tau_pool",
            "J",
            "mean_precision",
            "modal_precision",
            "cos_gV_gO",
            "tau_pool_std",
            "behaviour_alignment",
        )
        if present
        and all(_finite(forecast_row(forecasts[arm]).get(key)) is not None for arm in present)
    ]
    seeds = {arm: list(planned[arm]) for arm in present}
    bootstrap = None
    if len(present) >= 3 and keys:
        bootstrap = rank_bootstrap(
            {arm: forecasts[arm] for arm in present},
            {arm: [runs[arm][s] for s in seeds[arm]] for arm in present},
            step=final_step,
            keys=keys,
            quantiles=(0.05, 0.95),
            n_boot=n_boot,
            seed=boot_seed,
            seed_lists=seeds,
        )
    return {
        "rule": (
            "Secondary. Overall Spearman of each locked forecast with seed-averaged "
            "final accuracy, across non-withdrawn trained arms, each over its own "
            "planned seeds. The interval is a crossed bootstrap with explicit 5th and "
            "95th percentiles; arms with the same seed list share one seed draw."
        ),
        "point": scored["spearman"],
        "tau_pool_mae": scored["tau_pool_mae"],
        "bootstrap": bootstrap,
        "seeds": seeds,
        "n_arms": len(present),
    }


def _h4_block(
    forecasts: Mapping[str, Any],
    cfg: Mapping[str, Any],
    active_arms: Sequence[str],
    withdrawn: set[str],
) -> dict[str, Any]:
    shared = [str(a) for a in cfg.get("shared", H4_SHARED)]
    specific = [str(a) for a in cfg.get("prompt_specific", H4_PROMPT_SPECIFIC)]
    shared_min = float(cfg.get("shared_min", 0.65))
    specific_max = float(cfg.get("prompt_specific_max", 0.60))
    rows: list[dict[str, Any]] = []
    for arm, kind, threshold, compare in (
        *((arm, "shared", shared_min, "gt") for arm in shared),
        *((arm, "prompt_specific", specific_max, "le") for arm in specific),
    ):
        if arm in withdrawn:
            rows.append({"arm": arm, "kind": kind, "status": "withdrawn"})
            continue
        if arm not in active_arms:
            rows.append({"arm": arm, "kind": kind, "status": "not_in_study"})
            continue
        if arm not in forecasts:
            rows.append({"arm": arm, "kind": kind, "status": "missing_forecast"})
            continue
        raw = forecast_row(forecasts[arm]).get("probe_heldout_auc")
        channels = (
            {str(key): _finite(value) for key, value in raw.items()}
            if isinstance(raw, Mapping)
            else None
        )
        auc = None if channels is not None else _finite(raw)
        if auc is None:
            call = None
        elif compare == "gt":
            call = auc > threshold
        else:
            call = auc <= threshold
        row: dict[str, Any] = {
            "arm": arm,
            "kind": kind,
            "auc": auc,
            "threshold": threshold,
            "call": call,
            "status": "reported",
        }
        if channels is not None:
            row["channels"] = channels
        rows.append(row)
    return {
        "rule": (
            "Secondary. Held-out probe AUC from the lock (ridge on train-audit features, "
            f"scored on held-out audit rows). Shared errors > {shared_min}, "
            f"prompt-specific <= {specific_max}. An arm probed on both channels "
            "reports both under 'channels' and makes no call. Not a gate."
        ),
        "rows": rows,
    }


def _probe_auc(value: Any) -> float | None:
    """One probe AUC per arm: the mean of the channels for a two-channel arm."""
    if isinstance(value, Mapping):
        channels = [_finite(v) for v in value.values()]
        if not channels or any(v is None for v in channels):
            return None
        return float(np.mean([v for v in channels if v is not None]))
    return _finite(value)


_FP_TYPES = frozenset({"hash_fp", "tolerance_fp", "tolerance_band_fp", "trigger_fp"})
_FN_TYPES = frozenset({"form_fn", "behavior_fn", "prompt_subset_fn"})
_NATURAL_FN_TYPES = frozenset({"string_match"})


def arm_error_channel(
    describe: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """``fp``, ``fn`` or ``oracle`` from a calibration ``describe()`` dict.

    ``arms`` is the calibration arm map, used when a resampled or patched arm
    points at another arm. The chain-sum name lists are not consulted.
    """
    return _arm_error_channel(describe, arms or {}, set())


def _arm_error_channel(
    describe: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    seen: set[str],
) -> str:
    name = str(describe.get("name") or "")
    if name:
        if name in seen:
            return "fp"
        seen.add(name)
    kind = str(describe.get("kind") or "").lower()
    typ = str(describe.get("type") or "").lower()
    if kind in {"fp", "fn"}:
        return kind
    if typ in _FP_TYPES or typ.endswith("_fp"):
        return "fp"
    if typ in _FN_TYPES or typ.endswith("_fn"):
        return "fn"
    if kind in {"resampled", "noise"} or typ == "resampled":
        return _resampled_channel(describe, arms, seen)
    if kind == "composite" or typ == "composite":
        return _composite_channel(describe, arms, seen)
    if kind == "patched" or typ in {"patched", "patch_select"}:
        return _patched_channel(describe, arms, seen)
    if kind == "natural" or typ in {"natural", "string_match", "ifeval_checker"}:
        return _natural_channel(describe)
    if kind == "oracle" or typ == "oracle":
        return "oracle"
    rated = _rate_channel(describe)
    return rated if rated is not None else "fp"


def _resampled_channel(
    describe: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    seen: set[str],
) -> str:
    matched = describe.get("matched_arm")
    if matched is None:
        matched = describe.get("match")
    if isinstance(matched, str) and matched in arms and matched not in seen:
        return _arm_error_channel(arms[matched], arms, seen)
    rho0 = _finite(describe.get("rho0"))
    rho1 = _finite(describe.get("rho1"))
    if rho1 is not None and (rho0 is None or rho1 > rho0):
        return "fn"
    rated = _rate_channel(describe)
    return rated if rated is not None else "fp"


def _composite_channel(
    describe: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    seen: set[str],
) -> str:
    # A composite that carries both rules is reported on the FN monitor.
    # That matches combo_fnform_fphash, which the chain-sum menu treats as FN.
    found: set[str] = set()
    for slot in ("fn", "fp"):
        items = describe.get(slot) or []
        if isinstance(items, str) or not isinstance(items, Sequence):
            continue
        for item in items:
            if isinstance(item, Mapping):
                found.add(_arm_error_channel(item, arms, seen))
            elif isinstance(item, str) and item in arms:
                found.add(_arm_error_channel(arms[item], arms, seen))
    if "fn" in found:
        return "fn"
    if "fp" in found:
        return "fp"
    return "fp"


def _patched_channel(
    describe: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    seen: set[str],
) -> str:
    base = describe.get("base")
    if isinstance(base, Mapping):
        return _arm_error_channel(base, arms, seen)
    if isinstance(base, str) and base in arms and base not in seen:
        return _arm_error_channel(arms[base], arms, seen)
    if isinstance(base, str) and "strict" in base:
        return "fn"
    rated = _rate_channel(describe)
    return rated if rated is not None else "fp"


def _natural_channel(describe: Mapping[str, Any]) -> str:
    name = str(describe.get("name") or "")
    typ = str(describe.get("type") or "").lower()
    if "strict" in name or "strict" in typ or typ in _NATURAL_FN_TYPES:
        return "fn"
    rated = _rate_channel(describe)
    return rated if rated is not None else "fp"


def _rate_channel(describe: Mapping[str, Any]) -> str | None:
    fpr = _finite(describe.get("achieved_fpr"))
    fnr = _finite(describe.get("achieved_fnr"))
    channel = describe.get("channel")
    if isinstance(channel, Mapping):
        if fpr is None:
            fpr = _finite(channel.get("fpr"))
        if fnr is None:
            fnr = _finite(channel.get("fnr"))
    if fpr is None and fnr is None:
        return None
    if fnr is not None and (fpr is None or fnr > fpr):
        return "fn"
    return "fp"


def _name_channel(arm: str) -> str | None:
    if arm in FP_RULES:
        return "fp"
    if arm in FN_RULES:
        return "fn"
    return None


def _h5_channel(
    arm: str,
    describes: Mapping[str, Mapping[str, Any]],
    calibration_loaded: bool,
) -> str | None:
    """Channel for one H5 arm.

    Calibration ``describe()`` dicts win. Without a calibration file the
    chain-sum name lists keep the previous report. With a calibration file an
    arm that is not described still receives a channel, so it is never
    ``no_channel``.
    """
    if calibration_loaded and arm in describes:
        return arm_error_channel(describes[arm], describes)
    named = _name_channel(arm)
    if named is not None:
        return named
    if calibration_loaded:
        if arm.startswith("fn") or arm.endswith("_fn"):
            return "fn"
        return "fp"
    return None


def _h5_block(
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    pairs: Sequence[tuple[str, str]],
    withdrawn: set[str],
    allow_partial: bool,
    final_step: int,
    describes: Mapping[str, Mapping[str, Any]] | None = None,
    calibration_loaded: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    arm_describes = describes or {}
    for arm, control in pairs:
        if arm not in planned or control not in planned:
            if allow_partial:
                continue
            raise ValueError(f"H5 pair {arm} / {control} is not in the trained menu")
        if arm in withdrawn or control in withdrawn:
            rows.append({"arm": arm, "control": control, "status": "withdrawn"})
            continue
        channel = _h5_channel(arm, arm_describes, calibration_loaded)
        if channel is None:
            rows.append({"arm": arm, "control": control, "status": "no_channel"})
            continue
        if channel == "oracle":
            rows.append(
                {
                    "arm": arm,
                    "control": control,
                    "status": "reported",
                    "channel": "oracle",
                    "field": None,
                    "per_seed": [],
                    "mean": {},
                }
            )
            continue
        field = "monitors_wrong" if channel == "fp" else "monitors_correct"
        per_seed: list[dict[str, Any]] = []
        for seed in planned[arm]:
            entry: dict[str, Any] = {"seed": seed}
            for who, path in (("arm", runs[arm][seed]), ("control", runs[control][seed])):
                for when, step in (("step0", 0), ("final", final_step)):
                    row = _eval_row(path, "heldout", step)
                    entry[f"{who}_{when}"] = _monitor(row, channel, arm)
            per_seed.append(entry)
        means: dict[str, float | None] = {}
        for key in ("arm_step0", "arm_final", "control_step0", "control_final"):
            vals = [row[key] for row in per_seed if row[key] is not None]
            means[key] = float(np.mean(vals)) if vals else None
        rows.append(
            {
                "arm": arm,
                "control": control,
                "status": "reported",
                "channel": channel,
                "field": field,
                "per_seed": per_seed,
                "mean": means,
            }
        )
    return {
        "rule": (
            "Secondary. FP rules use monitors_wrong, P(monitor | O = 0). "
            "FN rules use monitors_correct, P(monitor | O = 1). "
            "Rates are reported at step 0 and at the final step, per seed and as a mean. Not a gate."
        ),
        "rows": rows,
    }


def _h6_block(
    forecasts: Mapping[str, Any],
    endpoints: Mapping[str, Any],
    arms: Sequence[str],
) -> dict[str, Any]:
    present = [arm for arm in arms if arm in forecasts and arm in endpoints]
    outcomes = {
        arm: {
            "heldout_final": endpoints[arm]["final_mean"],
            "heldout_delta": endpoints[arm]["gain_mean"],
        }
        for arm in present
    }
    scored = (
        score_forecasts({arm: forecasts[arm] for arm in present}, outcomes)
        if present
        else {
            "spearman": {},
            "tau_pool_mae": None,
            "n_arms": 0,
        }
    )
    return {
        "rule": (
            "Secondary. MAE of tau_pool against seed-averaged final held-out accuracy, "
            "and Spearman of mean precision and modal precision with that accuracy. Not a gate."
        ),
        "tau_pool_mae": scored["tau_pool_mae"],
        "spearman_mean_precision": (scored["spearman"] or {}).get("mean_precision"),
        "spearman_modal_precision": (scored["spearman"] or {}).get("modal_precision"),
        "spearman_tau_pool": (scored["spearman"] or {}).get("tau_pool"),
        "n_arms": scored["n_arms"],
    }


def _h10_block(
    *,
    pairs: Sequence[tuple[str, str]],
    forecasts: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[int, Path]],
    planned: Mapping[str, Sequence[int]],
    withdrawn: set[str],
    allow_partial: bool,
    final_step: int,
    n_boot: int,
    level: float,
    margin: float,
    boot_seed: int,
) -> dict[str, Any]:
    """``acc(a) - acc(b)`` for each configured pair, plus locked ``dtau`` and ``signed``."""
    rows: list[dict[str, Any]] = []
    for a, b in pairs:
        status = _status(a, b, planned=planned, withdrawn=withdrawn, allow_partial=allow_partial)
        record = _contrast_record(
            a,
            b,
            status=status,
            runs=runs,
            planned=planned,
            final_step=final_step,
            n_boot=n_boot,
            level=level,
            margin=margin,
            boot_seed=boot_seed,
        )
        record["a"] = a
        record["b"] = b
        if status != "tested":
            rows.append(record)
            continue
        tau_a = _locked_tau(forecasts, a)
        tau_b = _locked_tau(forecasts, b)
        dtau = tau_a - tau_b
        sign = _signum(dtau)
        point = float(record["point"])
        lo = float(record["lo"])
        hi = float(record["hi"])
        if sign < 0.0:
            signed_lo, signed_hi = sign * hi, sign * lo
        else:
            signed_lo, signed_hi = sign * lo, sign * hi
        record.update(
            {
                "dtau": dtau,
                "tau_a": tau_a,
                "tau_b": tau_b,
                "signed": sign * point,
                "signed_lo": signed_lo,
                "signed_hi": signed_hi,
            }
        )
        rows.append(record)
    return {"rule": H10_RULE, "pairs": rows}


def _locked_tau(forecasts: Mapping[str, Any], arm: str) -> float:
    entry = forecasts.get(arm)
    tau = None if not isinstance(entry, Mapping) else _finite(forecast_row(entry).get("tau_pool"))
    if tau is None:
        raise ValueError(f"H10 arm {arm} has no finite tau_pool in the lock")
    return tau


def _signum(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _append_natural_markdown(lines: list[str], natural: Mapping[str, Any]) -> None:
    contrast = natural.get("contrast") or {}
    scale = natural.get("scale") or {}
    lines += [
        f"## E2: {natural.get('control')} - {natural.get('natural')}",
        "",
        str(natural.get("rule", "")),
        "",
        f"- contrast: status={contrast.get('status')}, point={contrast.get('point')}, "
        f"interval=[{contrast.get('lo')}, {contrast.get('hi')}]",
        f"- differs: {(natural.get('differs') or {}).get('decision')}",
        f"- locality: {(natural.get('locality') or {}).get('decision')}, "
        f"p_tost={contrast.get('p_tost')}",
        f"- screen direction: {(natural.get('screen_direction') or {}).get('decision')}, "
        f"dtau={(natural.get('screen_direction') or {}).get('dtau')}",
        f"- scale ({scale.get('name')}): point={scale.get('point')}",
        "",
    ]


def render_markdown(report: Mapping[str, Any]) -> str:
    """A short reading of ``report`` for ``report.md``. The JSON is the record."""
    lines = [
        "# Confirmatory report",
        "",
        f"Inconclusive: {report.get('inconclusive')}. Hypotheses tested: {report.get('hypotheses')}.",
        "",
        "## Oracle gain",
        "",
        f"Value {report.get('oracle_gain', {}).get('value')} "
        f"(threshold {report.get('oracle_gain', {}).get('threshold')}). "
        f"{report.get('oracle_gain', {}).get('rule', '')}",
        "",
        "## Withdrawals",
        "",
        ", ".join(report.get("withdrawals", {}).get("withdrawn", [])) or "none",
        "",
    ]
    reliability = report.get("reliability")
    if isinstance(reliability, dict):
        lines += [
            "## Split-half reliability of g_V",
            "",
            f"Below {reliability.get('threshold')}: "
            f"{', '.join(reliability.get('below') or []) or 'none'}. "
            f"{reliability.get('rule', '')}",
            "",
        ]
    natural = report.get("natural")
    if isinstance(natural, dict):
        _append_natural_markdown(lines, natural)
    elif isinstance(natural, list):
        for block in natural:
            if isinstance(block, dict):
                _append_natural_markdown(lines, block)
    for key in ("h1", "h2", "h3"):
        block = report.get(key)
        if not isinstance(block, dict):
            continue
        lines += [
            f"## {key.upper()}",
            "",
            f"Decision: {block.get('decision')}. Holds: {block.get('holds')}.",
            "",
            str(block.get("rule", "")),
            "",
        ]
        contrasts = list(block.get("contrasts") or [])
        if key == "h1":
            contrasts += list(block.get("beside") or [])
        if key == "h3":
            contrasts = list(block.get("pairs") or [])
        for row in contrasts:
            if not isinstance(row, dict):
                continue
            name = row.get("name") or f"{row.get('arm')} vs {row.get('control')}"
            lines.append(
                f"- {name}: status={row.get('status')}, point={row.get('point', row.get('dacc'))}, "
                f"rejects={row.get('rejects')}, p_sup={row.get('p_sup')}, p_tost={row.get('p_tost')}"
            )
        lines.append("")
    h10 = report.get("h10")
    if isinstance(h10, dict):
        lines += ["## H10", "", str(h10.get("rule", "")), ""]
        for row in h10.get("pairs") or []:
            if not isinstance(row, dict):
                continue
            name = row.get("name") or f"{row.get('a')} - {row.get('b')}"
            lines.append(
                f"- {name}: status={row.get('status')}, point={row.get('point')}, "
                f"interval=[{row.get('lo')}, {row.get('hi')}], dtau={row.get('dtau')}, "
                f"signed={row.get('signed')}, "
                f"signed_interval=[{row.get('signed_lo')}, {row.get('signed_hi')}]"
            )
        lines.append("")
    n_knife = sum(
        1 for row in report.get("knife_edge", {}).get("runs", []) if row.get("knife_edge")
    )
    lines += [
        "## Knife-edge",
        "",
        f"{n_knife} run(s) labelled. {report.get('knife_edge', {}).get('rule', '')}",
        "",
    ]
    return "\n".join(lines)
