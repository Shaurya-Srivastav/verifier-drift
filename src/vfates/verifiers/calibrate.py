"""Fit verifier-arm parameters to step-0 pooled rates on the training pool.

False-positive arms are fit to a target FPR and false-negative arms to a target
FNR. The achievable rate is discrete: a threshold cannot split answer clusters
that share ``(pid, canonical answer)`` or prompts that share a hash, so the
fit is the closest achievable rate. Ties break toward the smaller rate, then
the smaller threshold.

Phrase arms add a sign preference inside a tolerance of the target, using the
within-prompt association between the phrase indicator and ``O`` (the mean over
rows of ``(b - mean_p(b)) * (O - mean_p(O))``). ``not_enriched`` keeps triggers
whose association is ``<= 0``. ``enriched`` keeps phrases whose association is
``> 0`` (a behavior, or a trigger that prefers correct answers). ``depleted``
keeps behaviors whose association is ``< 0``. The marginal advantage and the
accuracy lift are still reported. If nothing in tolerance has the preferred
sign, the closest phrase is used and the report sets ``fell_back``. Auto
n-grams that contain a digit are dropped. Candidate proposal and the sign use
the prompts with ``u01("phrase_validation", pid) >= 0.125``; the other eighth
validates the sign. An enriched phrase passes when that held-out association
is ``> 0``. A depleted phrase passes when it is ``< 0``. A ``not_enriched``
trigger passes when the held-out association is ``<= 0`` or its interval
covers 0. The rate tolerance is measured on the full train pool, where every
other arm is fitted and where the channel rate is reported, so a phrase that
did not fall back is inside the tolerance of its target.

Every arm report carries ``channel`` and ``realised`` rate blocks
``{fpr, fnr, J}``. For a deterministic arm they are the train-audit rates.
For a resampled arm, ``channel`` is the copied rate and ``realised`` is one
draw keyed by ``(pid, k)``. ``achieved_fpr`` / ``achieved_fnr`` are the
channel. ``J`` is Youden's ``J`` of the same FPR and FNR. Intervals on
``rates`` are prompt-clustered bootstraps from
:func:`vfates.forecast.confusion.confusion_with_ci`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

import numpy as np

from vfates.forecast.confusion import confusion_with_ci
from vfates.tasks.chain_sum import FORM_RULES
from vfates.types import Completion, Parsed, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import write_json
from vfates.utils.stats import pearson
from vfates.verifiers.arms import (
    WORD_RE,
    BehaviorFN,
    Composite,
    FNRule,
    FormFN,
    FPRule,
    HashFP,
    NumericToleranceFP,
    OracleVerifier,
    PromptSubsetFN,
    ResampledNoise,
    StringMatchVerifier,
    ToleranceBandFP,
    TriggerFP,
    answered_support,
    contains_pattern,
    literal_hit_matrix,
    numeric_abs_error,
    phrase_start,
)
from vfates.verifiers.base import Verifier

# Bump when fitting or the calibration report schema changes. P2 stores this
# in the cell fingerprint so a cached calibration.json is refit.
CALIBRATION_SCHEMA = "vfates.calibration/2"

_FP_TYPES = frozenset({"hash_fp", "tolerance_fp", "tolerance_band_fp", "trigger_fp"})
_FN_TYPES = frozenset({"form_fn", "behavior_fn", "prompt_subset_fn"})
_TARGETED_RATE = {
    "hash_fp": "fpr",
    "tolerance_fp": "fpr",
    "tolerance_band_fp": "fpr",
    "trigger_fp": "fpr",
    "behavior_fn": "fnr",
    "prompt_subset_fn": "fnr",
}
_RATE_METRICS = (
    "fpr",
    "fnr",
    "J",
    "prevalence",
    "precision_pooled",
    "precision_prompt_mean",
)
_PHRASE_VALIDATION_SALT = "phrase_validation"
_PHRASE_VALIDATION_FRACTION = 0.125
_PHRASE_VALIDATION_ALPHA = 0.10
_TRIGGER_PREFER = frozenset({"not_enriched", "enriched", "closest"})
_BEHAVIOR_PREFER = frozenset({"enriched", "depleted", "closest"})
_BAND_CAP = 0.02
_EARLY_CHARS = 40


@dataclass(frozen=True)
class CalibrationRow:
    """One graded completion used to fit or score an arm."""

    pid: str
    k: int
    text: str
    parsed: Parsed
    O: int
    n_tokens: int | None = None


@dataclass
class CalibrationData:
    """Problems keyed by ``pid`` and the graded rows they produced."""

    problems: dict[str, Problem]
    rows: list[CalibrationRow]

    @classmethod
    def from_grades(
        cls,
        problems: Sequence[Problem] | Mapping[str, Problem],
        completions: Sequence[Completion],
        parsed_list: Sequence[Parsed],
        O_list: Sequence[int],
    ) -> CalibrationData:
        """Align completions, parses and oracle bits by position."""
        if isinstance(problems, Mapping):
            by_pid = dict(problems)
        else:
            by_pid = {problem.pid: problem for problem in problems}
        if not (len(completions) == len(parsed_list) == len(O_list)):
            raise ValueError("completions, parsed_list, and O_list must have equal length")
        rows: list[CalibrationRow] = []
        for completion, parsed, label in zip(completions, parsed_list, O_list, strict=True):
            if completion.pid not in by_pid:
                raise KeyError(f"no problem for pid {completion.pid}")
            rows.append(
                CalibrationRow(
                    pid=completion.pid,
                    k=int(completion.k),
                    text=completion.text,
                    parsed=parsed,
                    O=int(label),
                    n_tokens=int(completion.n_tokens),
                )
            )
        return cls(problems=by_pid, rows=rows)


@dataclass
class CalibrationResult:
    """Fitted arms and a JSON report whose ``describe()`` specs rebuild them."""

    arms: dict[str, Verifier]
    report: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "arms": {name: verifier.describe() for name, verifier in self.arms.items()},
            "report": _json_ready(self.report),
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_json())


def pooled_rates(
    O: Sequence[Any] | np.ndarray, V: Sequence[Any] | np.ndarray
) -> tuple[float, float]:
    """``(FPR, FNR)`` pooled over rows. Empty class yields ``NaN``."""
    labels = np.asarray(O, dtype=np.int64)
    verdicts = np.asarray(V, dtype=np.int64)
    negative = labels == 0
    positive = labels == 1
    fpr = float(np.mean(verdicts[negative] == 1)) if np.any(negative) else float("nan")
    fnr = float(np.mean(verdicts[positive] == 0)) if np.any(positive) else float("nan")
    return fpr, fnr


def arm_verdicts(verifier: Verifier, data: CalibrationData) -> np.ndarray:
    """Verdict of ``verifier`` on every row. Stochastic arms use key ``(pid, k)``."""
    out = np.empty(len(data.rows), dtype=np.int64)
    for i, row in enumerate(data.rows):
        key = (row.pid, row.k) if verifier.stochastic else None
        out[i] = verifier(data.problems[row.pid], row.parsed, row.text, row.O, key)
    return out


def calibrate_hash_fp(
    data: CalibrationData,
    target_fpr: float,
    salt: str,
    *,
    name: str = "hash_fp",
) -> tuple[HashFP, dict[str, Any]]:
    """Threshold on ``u01(salt, pid, answer)`` closest to ``target_fpr``.

    Wrong rows with no parsed answer are never accepted and stay in the
    denominator. Clusters that share a hash are accepted together.
    """
    salt = str(salt)
    wrong = [row for row in data.rows if row.O == 0]
    if not wrong:
        raise ValueError("hash_fp calibration needs at least one wrong row")
    weights: dict[float, int] = {}
    seen: set[tuple[str, str]] = set()
    n_answered = 0
    for row in wrong:
        answer = row.parsed.answer
        if answer is None:
            continue
        n_answered += 1
        draw = u01(salt, row.pid, answer)
        weights[draw] = weights.get(draw, 0) + 1
        seen.add((row.pid, answer))
    groups = sorted(weights.items())
    threshold, accepted = _best_prefix(groups, float(target_fpr), len(wrong))
    _check_prefix(groups, threshold, accepted)
    verifier = HashFP(salt, threshold, name=name)
    got = sum(verifier.accepts(data.problems[row.pid], row.parsed, row.text) for row in wrong)
    if got != accepted:
        raise RuntimeError(f"hash threshold accepts {got} wrong rows, expected {accepted}")
    report = {
        "threshold": threshold,
        "achieved_fpr": accepted / len(wrong),
        "target_fpr": float(target_fpr),
        "n_wrong": len(wrong),
        "n_accepted": accepted,
        "n_answered_wrong": n_answered,
        "n_clusters": len(seen),
        "salt": salt,
    }
    return verifier, report


def calibrate_tolerance_fp(
    data: CalibrationData,
    target_fpr: float,
    *,
    name: str = "tolerance_fp",
    task: Any = None,
) -> tuple[NumericToleranceFP, dict[str, Any]]:
    """Smallest-distance absolute tolerance, in :class:`~decimal.Decimal`.

    Unparsable wrong rows (``task.abs_error`` is ``None``) are never accepted
    and stay in the denominator. ``|error| <= tol`` is the acceptance test, so
    each distinct error is one candidate and the accept-none candidate is
    included.
    """
    wrong = [row for row in data.rows if row.O == 0]
    if not wrong:
        raise ValueError("tolerance_fp calibration needs at least one wrong row")
    errors: list[Decimal] = []
    for row in wrong:
        err = numeric_abs_error(task, data.problems[row.pid], row.parsed)
        if err is not None:
            errors.append(err)
    counts: dict[Decimal, int] = {}
    for err in errors:
        counts[err] = counts.get(err, 0) + 1
    candidates: list[tuple[Decimal, int]] = []
    if any(err <= 0 for err in errors):
        candidates.append((Decimal("-1"), 0))
    else:
        candidates.append((Decimal("0"), 0))
    running = 0
    for err, count in sorted(counts.items()):
        running += count
        candidates.append((err, running))
    n_wrong = len(wrong)
    best: tuple[tuple[float, int, Decimal], Decimal, int] | None = None
    for tol, accepted in candidates:
        key = (abs(accepted - float(target_fpr) * n_wrong), accepted, tol)
        if best is None or key < best[0]:
            best = (key, tol, accepted)
    assert best is not None
    tol, accepted = best[1], best[2]
    actual = sum(1 for err in errors if err <= tol)
    if actual != accepted:
        raise RuntimeError(f"tolerance {tol} accepts {actual} rows, expected {accepted}")
    verifier = NumericToleranceFP(tol, name=name, task=task)
    report = {
        "abs_tol": format(tol, "f"),
        "achieved_fpr": accepted / n_wrong,
        "target_fpr": float(target_fpr),
        "n_wrong": n_wrong,
        "n_accepted": accepted,
        "n_parsable_wrong": len(errors),
    }
    return verifier, report


def calibrate_tolerance_band_fp(
    data: CalibrationData,
    target_fpr: float,
    lo: Decimal | str | float,
    *,
    name: str = "tolerance_band_fp",
    inner: str | None = None,
    task: Any = None,
) -> tuple[ToleranceBandFP, dict[str, Any]]:
    """Smallest observed ``hi`` with ``lo < |error| <= hi`` closest to ``target_fpr``.

    Ties break toward the smaller ``hi``. Unparsed wrong rows stay in the
    denominator. ``capped`` is true when ``|achieved FPR - target| > 0.02``.
    Medians are over the accepted wrong answers; the tolerance medians use
    ``|error| <= lo`` (the inner arm) on the same rows.
    """
    bound = lo if isinstance(lo, Decimal) else Decimal(str(lo))
    wrong = [row for row in data.rows if row.O == 0]
    if not wrong:
        raise ValueError("tolerance_band_fp calibration needs at least one wrong row")
    parsed_rows: list[tuple[Decimal, CalibrationRow]] = []
    for row in wrong:
        err = numeric_abs_error(task, data.problems[row.pid], row.parsed)
        if err is not None:
            parsed_rows.append((err, row))
    if not parsed_rows:
        raise ValueError("tolerance_band_fp calibration needs a parsable wrong answer")
    distinct = sorted({err for err, _row in parsed_rows})
    n_wrong = len(wrong)
    target = float(target_fpr)
    best: tuple[tuple[float, Decimal], Decimal, int] | None = None
    for hi in distinct:
        accepted = sum(1 for err, _row in parsed_rows if bound < err <= hi)
        key = (abs(accepted - target * n_wrong), hi)
        if best is None or key < best[0]:
            best = (key, hi, accepted)
    assert best is not None
    hi, accepted = best[1], best[2]
    actual = sum(1 for err, _row in parsed_rows if bound < err <= hi)
    if actual != accepted:
        raise RuntimeError(f"band hi {hi} accepts {actual} rows, expected {accepted}")
    verifier = ToleranceBandFP(bound, hi, name=name, task=task)
    achieved = accepted / n_wrong
    band_abs, band_rel = _error_medians(data, parsed_rows, lambda err: bound < err <= hi)
    tol_abs, tol_rel = _error_medians(data, parsed_rows, lambda err: err <= bound)
    report = {
        "lo": format(bound, "f"),
        "hi": format(hi, "f"),
        "inner": inner,
        "achieved_fpr": achieved,
        "target_fpr": target,
        "capped": abs(achieved - target) > _BAND_CAP,
        "cap_tol": _BAND_CAP,
        "n_wrong": n_wrong,
        "n_accepted": accepted,
        "n_parsable_wrong": len(parsed_rows),
        "median_abs_error": band_abs,
        "median_rel_error": band_rel,
        "tolerance_median_abs_error": tol_abs,
        "tolerance_median_rel_error": tol_rel,
    }
    return verifier, report


def select_trigger_fp(
    data: CalibrationData,
    target_fpr: float,
    candidates: Sequence[str] | None = None,
    ngram_range: tuple[int, int] | Sequence[int] = (1, 3),
    min_chars: int = 4,
    max_candidates: int = 20000,
    prefer: str = "not_enriched",
    tolerance: float = 0.02,
    *,
    case_sensitive: bool = True,
    regex: bool = False,
    name: str = "trigger_fp",
    seed: int = 0,
    n_boot: int = 200,
) -> tuple[TriggerFP, dict[str, Any]]:
    """Pick a phrase whose train-pool FPR on wrong rows is closest to ``target_fpr``.

    Auto candidates are word n-gram spans from wrong responses on the fitting
    prompts, capped at ``max_candidates`` by document frequency. Spans that
    contain a digit are dropped. Scoring uses the same token-span test as
    :class:`TriggerFP`. ``prefer='not_enriched'`` keeps phrases whose
    within-prompt association with ``O`` is ``<= 0`` inside ``tolerance`` of
    the target. ``prefer='enriched'`` keeps phrases whose association is
    ``> 0`` inside that tolerance. The marginal advantage is still reported.
    Regex candidates must be passed explicitly and keep raw regex semantics.
    """
    if prefer not in _TRIGGER_PREFER:
        known = ", ".join(sorted(_TRIGGER_PREFER))
        raise ValueError(f"unknown prefer {prefer!r}; expected one of {known}")
    verifier, report = _select_phrase(
        data,
        kind="trigger",
        target=float(target_fpr),
        rate_key="fpr",
        prefer=prefer,
        tolerance=float(tolerance),
        want_positive=prefer == "enriched",
        pool="wrong",
        candidates=candidates,
        ngram_range=ngram_range,
        min_chars=min_chars,
        max_candidates=max_candidates,
        case_sensitive=case_sensitive,
        regex=regex,
        name=name,
        seed=int(seed),
        n_boot=int(n_boot),
    )
    return cast(TriggerFP, verifier), report


def calibrate_prompt_subset_fn(
    data: CalibrationData,
    target_fnr: float,
    salt: str,
    *,
    name: str = "prompt_subset_fn",
) -> tuple[PromptSubsetFN, dict[str, Any]]:
    """Fraction on ``u01(salt, pid)`` whose correct-row FNR is closest to the target.

    Selecting a prompt rejects every correct row on it, so the search is over
    prefixes of prompts ordered by ``u01``.
    """
    salt = str(salt)
    weights: dict[str, int] = {}
    n_correct = 0
    for row in data.rows:
        weights.setdefault(row.pid, 0)
        if row.O == 1:
            weights[row.pid] += 1
            n_correct += 1
    if n_correct == 0:
        raise ValueError("prompt_subset_fn calibration needs at least one correct row")
    by_u: dict[float, int] = {}
    for pid, weight in weights.items():
        draw = u01(salt, pid)
        by_u[draw] = by_u.get(draw, 0) + weight
    groups = sorted(by_u.items())
    fraction, accepted = _best_prefix(groups, float(target_fnr), n_correct)
    _check_prefix(groups, fraction, accepted)
    verifier = PromptSubsetFN(salt, fraction, name=name)
    got = sum(
        verifier.rejects(data.problems[row.pid], row.parsed, row.text)
        for row in data.rows
        if row.O == 1
    )
    if got != accepted:
        raise RuntimeError(f"prompt fraction rejects {got} correct rows, expected {accepted}")
    n_selected = sum(1 for pid in weights if u01(salt, pid) < fraction)
    report = {
        "fraction": fraction,
        "salt": salt,
        "achieved_fnr": accepted / n_correct,
        "target_fnr": float(target_fnr),
        "n_correct": n_correct,
        "n_rejected": accepted,
        "n_prompts": len(weights),
        "n_prompts_selected": n_selected,
    }
    return verifier, report


def select_form_fn(
    data: CalibrationData,
    target_fnr: float,
    rules: Mapping[str, Any],
    *,
    name: str = "form_fn",
) -> tuple[FormFN, dict[str, Any]]:
    """Choose the surface-form rule whose FNR on correct rows is closest to the target.

    The report lists every rule's FNR. Ties break toward the smaller FNR, then
    the rule name.
    """
    if not rules:
        raise ValueError("form_fn requires at least one rule")
    n_correct = sum(row.O == 1 for row in data.rows)
    if n_correct == 0:
        raise ValueError("form_fn calibration needs at least one correct row")
    scored: dict[str, float] = {}
    for rule_name, rule in rules.items():
        flagged = 0
        for row in data.rows:
            if row.O != 1:
                continue
            if rule(data.problems[row.pid], row.parsed, row.text):
                flagged += 1
        scored[rule_name] = flagged / n_correct

    def sort_key(item: tuple[str, float]) -> tuple[float, float, str]:
        rule_name, fnr = item
        return (abs(fnr - float(target_fnr)), fnr, rule_name)

    chosen, chosen_fnr = min(scored.items(), key=sort_key)
    verifier = FormFN(chosen, rules, name=name)
    report = {
        "rule_name": chosen,
        "achieved_fnr": chosen_fnr,
        "target_fnr": float(target_fnr),
        "rules": scored,
    }
    return verifier, report


def select_behavior_fn(
    data: CalibrationData,
    target_fnr: float,
    candidates: Sequence[str] | None = None,
    prefer: str = "enriched",
    tolerance: float = 0.02,
    *,
    ngram_range: tuple[int, int] | Sequence[int] = (1, 3),
    min_chars: int = 4,
    max_candidates: int = 20000,
    case_sensitive: bool = True,
    regex: bool = False,
    name: str = "behavior_fn",
    seed: int = 0,
    n_boot: int = 200,
) -> tuple[BehaviorFN, dict[str, Any]]:
    """Pick a behavior phrase whose train-pool FNR is closest to the target.

    ``prefer='enriched'`` keeps phrases with positive within-prompt association
    inside ``tolerance``. ``prefer='depleted'`` keeps phrases whose association
    is negative (``< 0``) inside ``tolerance``. Auto candidates are word
    n-grams from correct responses on the fitting prompts; spans containing a
    digit are dropped. Lift is undefined for a phrase present in every fit
    row or in none; those phrases are dropped. The marginal lift is still
    reported.
    """
    if prefer not in _BEHAVIOR_PREFER:
        known = ", ".join(sorted(_BEHAVIOR_PREFER))
        raise ValueError(f"unknown prefer {prefer!r}; expected one of {known}")
    verifier, report = _select_phrase(
        data,
        kind="behavior",
        target=float(target_fnr),
        rate_key="fnr",
        prefer=prefer,
        tolerance=float(tolerance),
        want_positive=prefer != "depleted",
        pool="correct",
        candidates=candidates,
        ngram_range=ngram_range,
        min_chars=min_chars,
        max_candidates=max_candidates,
        case_sensitive=case_sensitive,
        regex=regex,
        name=name,
        seed=int(seed),
        n_boot=int(n_boot),
    )
    return cast(BehaviorFN, verifier), report


def resampled_matched(
    fpr: float,
    fnr: float,
    seed: int,
    *,
    name: str = "resampled",
    support: str = "all",
    answered_wrong: float = 1.0,
    task: Any = None,
) -> tuple[ResampledNoise, dict[str, Any]]:
    """Noise arm whose expected pooled rates are the supplied ``(FPR, FNR)``.

    With ``support="answered"``, ``answered_wrong`` is the share of wrong rows
    with a parsed answer and ``rho0 = fpr / answered_wrong``.
    """
    if support == "answered":
        if not 0.0 < answered_wrong <= 1.0:
            raise ValueError(f"answered_wrong must be in (0, 1], got {answered_wrong}")
        if fpr > answered_wrong:
            raise ValueError(
                f"FPR {fpr:.3f} exceeds the answered share of wrong rows {answered_wrong:.3f}"
            )
        rho0 = float(fpr) / float(answered_wrong)
    else:
        rho0 = float(fpr)
    verifier = ResampledNoise(rho0, float(fnr), int(seed), name=name, support=support, task=task)
    report = {
        "rho0": rho0,
        "rho1": float(fnr),
        "seed": int(seed),
        "support": support,
        "answered_wrong": float(answered_wrong),
    }
    return verifier, report


def calibrate_arms(
    task: Any,
    data: CalibrationData,
    specs: list[dict[str, Any]],
    seed: int = 0,
    n_boot: int = 1000,
    targets: Mapping[str, float] | None = None,
) -> CalibrationResult:
    """Fit every spec, in order, and attach pooled rates with bootstrap intervals.

    Specs are applied in order: ``resampled`` with ``match`` and ``composite``
    refer to arms already built. FP types use ``targets['fpr']`` unless the
    spec sets ``target_fpr``; FN types use ``targets['fnr']``. A matched noise
    arm's target is the matched arm's achieved rates (the channel it copies).

    Each arm report has ``channel`` and ``realised`` blocks ``{fpr, fnr, J}``
    (equal for a deterministic arm; for resampled noise, ``channel`` is the
    copied rate and ``realised`` is one ``(pid, k)`` draw), plus ``rates``
    as ``{point, lo, hi}`` and ``|achieved - target|`` when that arm has a
    target. ``achieved_fpr`` / ``achieved_fnr`` are the channel.
    ``tolerance_band_fp`` reads ``inner`` from an already fitted tolerance arm.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    global_targets = dict(targets or {})
    arms: dict[str, Verifier] = {}
    arm_reports: dict[str, dict[str, Any]] = {}
    labels, pid = _labels(data)
    pending: list[dict[str, Any]] = []
    pending_names: set[str] = set()
    for raw in specs:
        spec = dict(raw)
        if "name" not in spec or "type" not in spec:
            raise ValueError("each arm spec needs 'name' and 'type'")
        name = str(spec["name"])
        typ = str(spec["type"])
        if name in arms or name in pending_names:
            raise ValueError(f"duplicate arm name {name!r}")
        if typ == "patch_select":
            _check_patch_select(spec, name, arms)
            pending.append(dict(spec))
            pending_names.add(name)
            continue
        verifier, details, channel = _fit_arm(
            task, data, spec, name, typ, seed, n_boot, global_targets, arms, arm_reports
        )
        verdicts = arm_verdicts(verifier, data)
        realised_fpr, realised_fnr = pooled_rates(labels, verdicts)
        if channel is None:
            achieved_fpr, achieved_fnr = realised_fpr, realised_fnr
        else:
            achieved_fpr, achieved_fnr = channel
        target_fpr, target_fnr = _arm_targets(typ, spec, global_targets, channel)
        realised_block: dict[str, Any] = {
            "fpr": realised_fpr,
            "fnr": realised_fnr,
            "J": _youden(realised_fpr, realised_fnr),
        }
        if channel is not None:
            realised_block["key_fields"] = ["pid", "k"]
        entry: dict[str, Any] = {
            "type": typ,
            "achieved_fpr": achieved_fpr,
            "achieved_fnr": achieved_fnr,
            "target_fpr": target_fpr,
            "target_fnr": target_fnr,
            "abs_fpr_error": _abs_delta(achieved_fpr, target_fpr),
            "abs_fnr_error": _abs_delta(achieved_fnr, target_fnr),
            "channel": {
                "fpr": achieved_fpr,
                "fnr": achieved_fnr,
                "J": _youden(achieved_fpr, achieved_fnr),
            },
            "realised": realised_block,
            "rates": confusion_with_ci(
                labels, verdicts, pid, n_boot=n_boot, seed=seed, metrics=_RATE_METRICS
            ),
            "details": details,
        }
        arms[name] = verifier
        arm_reports[name] = entry
    report = {
        "schema": CALIBRATION_SCHEMA,
        "seed": int(seed),
        "n_boot": int(n_boot),
        "targets": {
            "fpr": None if global_targets.get("fpr") is None else float(global_targets["fpr"]),
            "fnr": None if global_targets.get("fnr") is None else float(global_targets["fnr"]),
        },
        "n_rows": len(data.rows),
        "n_prompts": len({row.pid for row in data.rows}),
        "arms": arm_reports,
    }
    if pending:
        report["patch_select"] = pending
    return CalibrationResult(arms=arms, report=report)


def _fit_arm(
    task: Any,
    data: CalibrationData,
    spec: dict[str, Any],
    name: str,
    typ: str,
    seed: int,
    n_boot: int,
    targets: Mapping[str, Any],
    arms: dict[str, Verifier],
    arm_reports: dict[str, dict[str, Any]],
) -> tuple[Verifier, dict[str, Any], tuple[float, float] | None]:
    verifier: Verifier
    details: dict[str, Any]
    if typ == "oracle":
        return OracleVerifier(name=name), {}, None
    if typ == "string_match":
        return StringMatchVerifier(name=name), {}, None
    if typ == "ifeval_checker":
        # Fixed natural verifiers. No target is fitted. The pooled train rates
        # are the channel a matched resampled arm copies.
        from vfates.verifiers.registry import build_verifier

        return build_verifier(spec), {}, None
    if typ == "hash_fp":
        verifier, details = calibrate_hash_fp(
            data, _target(spec, targets, "fpr", name), spec.get("salt", name), name=name
        )
        return verifier, details, None
    if typ == "tolerance_fp":
        verifier, details = calibrate_tolerance_fp(
            data, _target(spec, targets, "fpr", name), name=name, task=task
        )
        return verifier, details, None
    if typ == "tolerance_band_fp":
        inner_name = spec.get("inner")
        if not inner_name:
            raise ValueError(f"arm {name!r} (tolerance_band_fp) requires 'inner'")
        if inner_name not in arms:
            raise KeyError(
                f"tolerance_band_fp arm {name!r} inner {inner_name!r} is missing; "
                "specs are applied in order"
            )
        inner = arms[str(inner_name)]
        if not isinstance(inner, NumericToleranceFP):
            raise TypeError(
                f"tolerance_band_fp arm {name!r} inner {inner_name!r} is not a tolerance arm"
            )
        verifier, details = calibrate_tolerance_band_fp(
            data,
            _target(spec, targets, "fpr", name),
            inner.abs_tol,
            name=name,
            inner=str(inner_name),
            task=task,
        )
        return verifier, details, None
    if typ == "trigger_fp":
        verifier, details = select_trigger_fp(
            data,
            _target(spec, targets, "fpr", name),
            candidates=spec.get("candidates"),
            ngram_range=spec.get("ngram_range", (1, 3)),
            min_chars=int(spec.get("min_chars", 4)),
            max_candidates=int(spec.get("max_candidates", 20000)),
            prefer=spec.get("prefer", "not_enriched"),
            tolerance=float(spec.get("tolerance", 0.02)),
            case_sensitive=bool(spec.get("case_sensitive", True)),
            regex=bool(spec.get("regex", False)),
            name=name,
            seed=int(seed),
            n_boot=int(n_boot),
        )
        return verifier, details, None
    if typ == "form_fn":
        verifier, details = select_form_fn(
            data,
            _target(spec, targets, "fnr", name),
            _rule_map(task, spec.get("candidates")),
            name=name,
        )
        return verifier, details, None
    if typ == "behavior_fn":
        verifier, details = select_behavior_fn(
            data,
            _target(spec, targets, "fnr", name),
            candidates=spec.get("candidates"),
            prefer=spec.get("prefer", "enriched"),
            tolerance=float(spec.get("tolerance", 0.02)),
            ngram_range=spec.get("ngram_range", (1, 3)),
            min_chars=int(spec.get("min_chars", 4)),
            max_candidates=int(spec.get("max_candidates", 20000)),
            case_sensitive=bool(spec.get("case_sensitive", True)),
            regex=bool(spec.get("regex", False)),
            name=name,
            seed=int(seed),
            n_boot=int(n_boot),
        )
        return verifier, details, None
    if typ == "prompt_subset_fn":
        verifier, details = calibrate_prompt_subset_fn(
            data,
            _target(spec, targets, "fnr", name),
            spec.get("salt", name),
            name=name,
        )
        return verifier, details, None
    if typ == "resampled":
        has_match = bool(spec.get("match"))
        has_rates = spec.get("rates") is not None
        if has_match and has_rates:
            raise ValueError(f"resampled arm {name!r} sets both match and rates")
        declared = False
        if has_match:
            other = spec["match"]
            if other not in arm_reports:
                raise KeyError(
                    f"resampled arm {name!r} matches unknown arm {other!r}; "
                    "specs are applied in order"
                )
            rho0 = float(arm_reports[other]["achieved_fpr"])
            rho1 = float(arm_reports[other]["achieved_fnr"])
            matched = other
        elif has_rates:
            rates = spec["rates"]
            if not isinstance(rates, Mapping) or "fpr" not in rates or "fnr" not in rates:
                raise ValueError(f"resampled arm {name!r} rates must be a mapping with fpr and fnr")
            rho0 = float(rates["fpr"])
            rho1 = float(rates["fnr"])
            if not 0.0 <= rho0 <= 1.0 or not 0.0 <= rho1 <= 1.0:
                raise ValueError(
                    f"resampled arm {name!r} rates must lie in [0, 1], got {(rho0, rho1)}"
                )
            matched = None
            declared = True
        else:
            rho0 = float(_need_field(spec, "rho0"))
            rho1 = float(_need_field(spec, "rho1"))
            matched = None
        noise_seed = int(spec["seed"]) if "seed" in spec else int(seed)
        support = str(spec.get("support", "all"))
        wrong = [row for row in data.rows if row.O == 0]
        answered = (
            sum(answered_support(task, row.parsed) for row in wrong) / len(wrong) if wrong else 1.0
        )
        verifier, details = resampled_matched(
            rho0,
            rho1,
            noise_seed,
            name=name,
            support=support,
            answered_wrong=answered,
            task=task,
        )
        details["matched_arm"] = matched
        if declared:
            details["rates_declared"] = True
        return verifier, details, (rho0, rho1)
    if typ == "composite":
        fn_rules = [_take_rule(arms, item, FNRule, "fn") for item in spec.get("fn", [])]
        fp_rules = [_take_rule(arms, item, FPRule, "fp") for item in spec.get("fp", [])]
        verifier = Composite(
            cast(Sequence[FNRule], fn_rules), cast(Sequence[FPRule], fp_rules), name=name
        )
        details = {"fn": list(spec.get("fn", [])), "fp": list(spec.get("fp", []))}
        return verifier, details, None
    if task is not None and typ in getattr(task, "verifier_types", ()):
        from vfates.verifiers.registry import build_verifier

        return build_verifier(spec, task), {}, None
    raise ValueError(f"unknown arm type {typ!r}")


def _check_patch_select(spec: Mapping[str, Any], name: str, arms: Mapping[str, Verifier]) -> None:
    """Record a patch menu. The categories are chosen after features exist."""
    base = spec.get("base")
    if not isinstance(base, str) or not base:
        raise ValueError(f"patch arm {name!r} needs a string 'base'")
    if base not in arms:
        raise KeyError(f"patch arm {name!r} base {base!r} is missing; specs are applied in order")
    by = spec.get("by")
    if by not in ("tau", "rate"):
        raise ValueError(f"patch arm {name!r} by must be 'tau' or 'rate', got {by!r}")
    k = spec.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"patch arm {name!r} k must be a positive int, got {k!r}")


def _arm_targets(
    typ: str,
    spec: Mapping[str, Any],
    targets: Mapping[str, Any],
    channel: tuple[float, float] | None,
) -> tuple[float | None, float | None]:
    if typ == "resampled":
        # The calibration target of a noise arm is the channel it copies.
        if "target_fpr" in spec:
            target_fpr: float | None = float(spec["target_fpr"])
        else:
            target_fpr = None if channel is None else channel[0]
        if "target_fnr" in spec:
            target_fnr: float | None = float(spec["target_fnr"])
        else:
            target_fnr = None if channel is None else channel[1]
        return target_fpr, target_fnr
    if typ in _FP_TYPES or "target_fpr" in spec:
        target_fpr = _optional_target(spec, targets, "fpr")
    else:
        target_fpr = None
    if typ in _FN_TYPES or "target_fnr" in spec:
        target_fnr = _optional_target(spec, targets, "fnr")
    else:
        target_fnr = None
    return target_fpr, target_fnr


def _optional_target(
    spec: Mapping[str, Any], targets: Mapping[str, Any], kind: str
) -> float | None:
    key = f"target_{kind}"
    if key in spec and spec[key] is not None:
        return float(spec[key])
    if kind in targets and targets[kind] is not None:
        return float(targets[kind])
    return None


def _target(spec: Mapping[str, Any], targets: Mapping[str, Any], kind: str, name: str) -> float:
    value = _optional_target(spec, targets, kind)
    if value is None:
        raise ValueError(f"arm {name!r} needs target_{kind} or targets[{kind!r}]")
    return value


def _need_field(spec: Mapping[str, Any], key: str) -> Any:
    if key not in spec:
        raise ValueError(f"arm spec is missing {key!r}")
    return spec[key]


def _rule_map(task: Any, names: Sequence[str] | None) -> dict[str, Any]:
    source = task.form_rules if task is not None and hasattr(task, "form_rules") else FORM_RULES
    if names is None:
        return dict(source)
    missing = [name for name in names if name not in source]
    if missing:
        raise KeyError(f"unknown form rules: {missing}")
    return {name: source[name] for name in names}


def _take_rule(arms: Mapping[str, Verifier], name: str, cls: type, slot: str) -> Verifier:
    if name not in arms:
        raise KeyError(
            f"composite {slot} refers to unknown arm {name!r}; specs are applied in order"
        )
    rule = arms[name]
    if not isinstance(rule, cls):
        raise TypeError(f"composite {slot} arm {name!r} is not a {cls.__name__}")
    return rule


def _labels(data: CalibrationData) -> tuple[np.ndarray, list[str]]:
    labels = np.array([row.O for row in data.rows], dtype=np.int64)
    pid = [row.pid for row in data.rows]
    return labels, pid


def _best_prefix(groups: list[tuple[float, int]], target: float, n_total: int) -> tuple[float, int]:
    """Threshold ``t`` with acceptance ``u < t`` closest to ``target``.

    ``groups`` is sorted by strictly increasing ``u``. Ties break toward the
    smaller accepted weight, then the smaller threshold. The empty prefix uses
    threshold ``0``; the full prefix uses ``1`` (``u01`` is in ``[0, 1)``).
    """
    if n_total <= 0:
        raise ValueError("rate denominator is empty")
    best: tuple[tuple[float, int, float], float, int] | None = None
    running = 0
    last = len(groups)
    for k in range(last + 1):
        if k == 0:
            accepted = 0
            threshold = 0.0
        else:
            _u, weight = groups[k - 1]
            running += weight
            accepted = running
            threshold = 1.0 if k == last else groups[k][0]
        key = (abs(accepted - target * n_total), accepted, threshold)
        if best is None or key < best[0]:
            best = (key, threshold, accepted)
    assert best is not None
    return best[1], best[2]


def _check_prefix(groups: list[tuple[float, int]], threshold: float, accepted: int) -> None:
    got = sum(weight for draw, weight in groups if draw < threshold)
    if got != accepted:
        raise RuntimeError(f"threshold {threshold} accepts weight {got}, expected {accepted}")


def _score_phrases(
    data: CalibrationData,
    candidates: Sequence[str] | None,
    ngram_range: tuple[int, int] | Sequence[int],
    min_chars: int,
    max_candidates: int,
    *,
    case_sensitive: bool,
    regex: bool,
    pool: str,
) -> tuple[list[str], dict[str, list[int]], dict[str, list[int]], int, int]:
    n_pos = sum(row.O == 1 for row in data.rows)
    n_neg = len(data.rows) - n_pos
    if candidates is not None:
        phrases = list(dict.fromkeys(candidates))
        if regex is False and not phrases:
            raise ValueError("candidate list is empty")
    else:
        if regex:
            raise ValueError("regex phrase arms require an explicit candidate list")
        phrases = _propose_ngrams(
            data,
            ngram_range,
            min_chars,
            max_candidates,
            lower=not case_sensitive,
            pool=pool,
        )
    if not phrases:
        raise ValueError("no phrase candidates")
    texts = [row.text for row in data.rows]
    if regex:
        columns = [
            [
                int(
                    contains_pattern(
                        text, phrase, regex=True, case_sensitive=case_sensitive, word_boundary=False
                    )
                )
                for text in texts
            ]
            for phrase in phrases
        ]
    else:
        columns = literal_hit_matrix(texts, phrases, case_sensitive=case_sensitive)
    table: dict[str, list[int]] = {}
    flags: dict[str, list[int]] = {}
    for phrase, column in zip(phrases, columns, strict=True):
        wrong = 0
        correct = 0
        for row, bit in zip(data.rows, column, strict=True):
            if bit:
                if row.O == 1:
                    correct += 1
                else:
                    wrong += 1
        table[phrase] = [wrong, correct]
        flags[phrase] = column
    return phrases, table, flags, n_pos, n_neg


def _propose_ngrams(
    data: CalibrationData,
    ngram_range: tuple[int, int] | Sequence[int],
    min_chars: int,
    max_candidates: int,
    *,
    lower: bool,
    pool: str,
) -> list[str]:
    lo, hi = int(ngram_range[0]), int(ngram_range[1])
    if lo < 1 or hi < lo:
        raise ValueError(f"ngram_range must satisfy 1 <= lo <= hi, got {ngram_range}")
    counts: Counter[str] = Counter()
    for row in data.rows:
        if pool == "wrong" and row.O != 0:
            continue
        if pool == "correct" and row.O != 1:
            continue
        text = row.text.lower() if lower else row.text
        for gram in _ngrams(text, lo, hi, min_chars):
            if any(ch.isdigit() for ch in gram):
                continue
            counts[gram] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [gram for gram, _count in ranked[:max_candidates]]


def _ngrams(text: str, lo: int, hi: int, min_chars: int) -> set[str]:
    matches = list(WORD_RE.finditer(text))
    grams: set[str] = set()
    for n in range(lo, hi + 1):
        for i in range(len(matches) - n + 1):
            gram = text[matches[i].start() : matches[i + n - 1].end()]
            if len(gram) >= min_chars:
                grams.add(gram)
    return grams


def _phrase_sign_ok(association: float, prefer: str) -> bool:
    """Whether ``association`` has the sign ``prefer`` asks for inside tolerance."""
    if association != association:
        return False
    if prefer == "enriched":
        return association > 0.0
    if prefer == "depleted":
        return association < 0.0
    return association <= 0.0


def _pick_phrase(
    scored: list[dict[str, Any]],
    *,
    rate_key: str,
    target: float,
    prefer: str,
    tolerance: float,
    sign_key: str,
    want_positive: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    def distance(row: dict[str, Any]) -> float:
        return abs(row[rate_key] - target)

    def sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
        sign = row[sign_key]
        # More negative association, or higher association, wins a distance tie.
        secondary = -sign if want_positive else sign
        return (distance(row), secondary, row["phrase"])

    fell_back = False
    if prefer == "closest":
        pool = list(scored)
    else:
        pool = [
            row
            for row in scored
            if _phrase_sign_ok(row[sign_key], prefer) and distance(row) <= tolerance + 1e-12
        ]
        if not pool:
            pool = list(scored)
            fell_back = True
    pool.sort(key=sort_key)
    winner = pool[0]
    ranked = sorted(scored, key=sort_key)
    alternatives = [
        _public_phrase(row, rate_key) for row in ranked if row["phrase"] != winner["phrase"]
    ][:10]
    return winner, alternatives, fell_back


def _public_phrase(row: dict[str, Any], rate_key: str) -> dict[str, Any]:
    keep = {
        "phrase",
        rate_key,
        "association",
        "advantage",
        "lift",
        "p_phrase_given_correct",
        "p_o_given_phrase",
        "p_o_given_behavior",
        "p_o_given_not_behavior",
    }
    return {key: row[key] for key in row if key in keep}


def _select_phrase(
    data: CalibrationData,
    *,
    kind: str,
    target: float,
    rate_key: str,
    prefer: str,
    tolerance: float,
    want_positive: bool,
    pool: str,
    candidates: Sequence[str] | None,
    ngram_range: tuple[int, int] | Sequence[int],
    min_chars: int,
    max_candidates: int,
    case_sensitive: bool,
    regex: bool,
    name: str,
    seed: int,
    n_boot: int,
) -> tuple[Verifier, dict[str, Any]]:
    fit_rows, val_rows = _split_phrase_rows(data.rows)
    if not fit_rows:
        raise ValueError(f"{kind} phrase fit split is empty")
    fit_data = CalibrationData(problems=data.problems, rows=fit_rows)
    phrases, table, flags, n_pos, n_neg = _score_phrases(
        fit_data,
        candidates,
        ngram_range,
        min_chars,
        max_candidates,
        case_sensitive=case_sensitive,
        regex=regex,
        pool=pool,
    )
    if kind == "trigger" and n_neg == 0:
        raise ValueError("trigger_fp calibration needs at least one wrong row in the fit split")
    if kind == "behavior" and n_pos == 0:
        raise ValueError("behavior_fn calibration needs at least one correct row in the fit split")
    _, train_table, _, train_pos, train_neg = _score_phrases(
        data,
        phrases,
        ngram_range,
        min_chars,
        max_candidates,
        case_sensitive=case_sensitive,
        regex=regex,
        pool=pool,
    )
    groups = _groups_by_prompt(fit_rows)
    prevalence = n_pos / (n_pos + n_neg) if n_pos + n_neg else float("nan")
    n_rows = n_pos + n_neg
    scored: list[dict[str, Any]] = []
    for phrase in phrases:
        n_wrong_with, n_correct_with = table[phrase]
        n_phrase = n_wrong_with + n_correct_with
        if n_phrase == 0:
            continue
        association = _association_grouped(fit_rows, flags[phrase], groups)
        row: dict[str, Any] = {
            "phrase": phrase,
            "association": association,
            "n_wrong_with": n_wrong_with,
            "n_correct_with": n_correct_with,
        }
        if kind == "trigger":
            p_o = n_correct_with / n_phrase
            row.update(
                {
                    "fpr": train_table[phrase][0] / train_neg,
                    "fit_fpr": n_wrong_with / n_neg,
                    "p_phrase_given_correct": n_correct_with / n_pos if n_pos else float("nan"),
                    "p_o_given_phrase": p_o,
                    "advantage": p_o - prevalence,
                }
            )
        else:
            n_b = n_phrase
            n_not = n_rows - n_b
            if n_not == 0:
                continue
            p_b = n_correct_with / n_b
            p_not = (n_pos - n_correct_with) / n_not
            row.update(
                {
                    "fnr": train_table[phrase][1] / train_pos,
                    "fit_fnr": n_correct_with / n_pos,
                    "lift": p_b - p_not,
                    "p_o_given_behavior": p_b,
                    "p_o_given_not_behavior": p_not,
                }
            )
        scored.append(row)
    if not scored:
        if kind == "behavior":
            raise ValueError("no behavior candidates with a defined accuracy lift")
        raise ValueError("no trigger candidates")
    winner, alternatives, fell_back = _pick_phrase(
        scored,
        rate_key=rate_key,
        target=target,
        prefer=prefer,
        tolerance=tolerance,
        sign_key="association",
        want_positive=want_positive,
    )
    phrase = str(winner["phrase"])
    if kind == "trigger":
        verifier: Verifier = TriggerFP(
            phrase, regex=regex, case_sensitive=case_sensitive, word_boundary=True, name=name
        )
        fit_hits = _match_count(verifier, fit_data, fit_rows, positive=False)
        if fit_hits != winner["n_wrong_with"]:
            raise RuntimeError(
                f"trigger fit count {fit_hits} disagrees with the phrase table "
                f"({winner['n_wrong_with']})"
            )
    else:
        verifier = BehaviorFN(
            phrase, regex=regex, case_sensitive=case_sensitive, word_boundary=True, name=name
        )
        fit_hits = _match_count(verifier, fit_data, fit_rows, positive=True)
        if fit_hits != winner["n_correct_with"]:
            raise RuntimeError(
                f"behavior fit count {fit_hits} disagrees with the phrase table "
                f"({winner['n_correct_with']})"
            )
    train_flags = _match_flags(verifier, data, data.rows)
    val_flags = _match_flags(verifier, data, val_rows)
    if kind == "trigger":
        train_rate = _rate(train_flags, data.rows, positive=False)
        fit_rate = float(winner["fit_fpr"])
        rate_name = "fpr"
    else:
        train_rate = _rate(train_flags, data.rows, positive=True)
        fit_rate = float(winner["fit_fnr"])
        rate_name = "fnr"
    if abs(train_rate - float(winner[rate_name])) > 1e-12:
        raise RuntimeError(
            f"{kind} train {rate_name} {train_rate} disagrees with the phrase table "
            f"({winner[rate_name]})"
        )
    val_assoc, val_lo, val_hi = _validation_association(
        val_rows, val_flags, n_boot=n_boot, seed=seed
    )
    if kind == "trigger" and prefer != "enriched":
        validated = _trigger_validated(val_assoc, val_lo, val_hi)
    elif prefer == "depleted":
        validated = _depleted_validated(val_assoc)
    else:
        validated = _behavior_validated(val_assoc)
    diagnostics = _phrase_diagnostics(
        data.rows,
        train_flags,
        phrase=phrase,
        kind=kind,
        regex=regex,
        case_sensitive=case_sensitive,
        word_boundary=not regex,
    )
    fit_prompts = len({row.pid for row in fit_rows})
    val_prompts = len({row.pid for row in val_rows})
    report: dict[str, Any] = {
        "phrase": phrase,
        "pattern": phrase,
        "regex": regex,
        "case_sensitive": case_sensitive,
        "word_boundary": True,
        f"achieved_{rate_name}": train_rate,
        f"target_{rate_name}": target,
        f"fit_{rate_name}": fit_rate,
        f"train_{rate_name}": train_rate,
        "association": winner["association"],
        "prefer": prefer,
        "tolerance": tolerance,
        "fell_back": fell_back,
        "validated": validated,
        "n_candidates": len(scored),
        "fit_n_rows": len(fit_rows),
        "fit_n_prompts": fit_prompts,
        "alternatives": alternatives,
        "validation": {
            "n_prompts": val_prompts,
            "n_rows": len(val_rows),
            "association": _finite(val_assoc),
            "lo": _finite(val_lo),
            "hi": _finite(val_hi),
            "covers_zero": bool(val_lo == val_lo and val_hi == val_hi and val_lo <= 0.0 <= val_hi),
            "alpha": _PHRASE_VALIDATION_ALPHA,
            "n_boot": 0 if not val_rows else n_boot,
            "fraction": _PHRASE_VALIDATION_FRACTION,
            "salt": _PHRASE_VALIDATION_SALT,
        },
        "diagnostics": diagnostics,
    }
    if kind == "trigger":
        report["advantage"] = winner["advantage"]
        report["prevalence"] = prevalence
        report["p_phrase_given_correct"] = winner["p_phrase_given_correct"]
        report["p_o_given_phrase"] = winner["p_o_given_phrase"]
    else:
        report["lift"] = winner["lift"]
        report["p_o_given_behavior"] = winner["p_o_given_behavior"]
        report["p_o_given_not_behavior"] = winner["p_o_given_not_behavior"]
    return verifier, report


def arm_validity(calibration_report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-arm validity of a ``calibration.json`` document or its ``report`` block.

    ``targeted`` is true for arms fitted to a rate target (``hash_fp``,
    ``tolerance_fp``, ``tolerance_band_fp``, ``trigger_fp``, ``behavior_fn``,
    ``prompt_subset_fn``) and for a resampled arm that declares ``rates``.
    ``form_fn`` is capped by design (``targeted`` false, ``capped`` true).
    A matched resampled arm is not targeted. ``rate_miss`` is
    ``|channel rate - target|`` on the targeted rate. For a declared-rate
    resampled arm it is the larger of ``|realised fpr - declared fpr|`` and
    ``|realised fnr - declared fnr|``, the same absolute-error rule a matched
    arm would use against its copied channel. Otherwise ``rate_miss`` is
    ``None``. ``fell_back`` and ``validated`` come from phrase-arm details
    (``validated`` is ``None`` when the arm has no sign check). ``capped`` is
    true for ``form_fn`` and for a tolerance band whose achieved FPR misses
    its target by more than 0.02.
    """
    arms = _validity_arms(calibration_report)
    out: dict[str, dict[str, Any]] = {}
    for name, entry in arms.items():
        typ = str(entry["type"])
        details = entry.get("details") or {}
        if not isinstance(details, Mapping):
            details = {}
        declared_rates = typ == "resampled" and bool(details.get("rates_declared"))
        targeted = typ in _TARGETED_RATE or declared_rates
        if declared_rates:
            rate_miss = _declared_rate_miss(entry)
        elif targeted:
            rate_miss = _targeted_rate_miss(entry, _TARGETED_RATE[typ])
        else:
            rate_miss = None
        if typ == "form_fn":
            capped = True
        elif typ == "tolerance_band_fp":
            capped = bool(details.get("capped", False))
        else:
            capped = False
        validated: bool | None
        if "validated" in details and details["validated"] is not None:
            validated = bool(details["validated"])
        else:
            validated = None
        out[str(name)] = {
            "type": typ,
            "targeted": targeted,
            "rate_miss": rate_miss,
            "fell_back": bool(details.get("fell_back", False)),
            "validated": validated,
            "capped": capped,
        }
    return out


def _validity_arms(calibration_report: Mapping[str, Any]) -> Mapping[str, Any]:
    report = calibration_report.get("report")
    if isinstance(report, Mapping) and isinstance(report.get("arms"), Mapping):
        return report["arms"]
    arms = calibration_report.get("arms")
    if isinstance(arms, Mapping):
        return arms
    raise KeyError("calibration report has no arms")


def _declared_rate_miss(entry: Mapping[str, Any]) -> float | None:
    """Larger absolute miss of the realised draw against the declared rates."""
    realised = entry.get("realised")
    if not isinstance(realised, Mapping):
        return None
    misses: list[float] = []
    for kind in ("fpr", "fnr"):
        target = entry.get(f"target_{kind}")
        rate = realised.get(kind)
        if target is None or rate is None:
            return None
        rate_f = float(rate)
        target_f = float(target)
        if rate_f != rate_f or target_f != target_f:
            return None
        misses.append(abs(rate_f - target_f))
    return max(misses)


def _targeted_rate_miss(entry: Mapping[str, Any], kind: str) -> float | None:
    channel = entry.get("channel")
    if not isinstance(channel, Mapping) or kind not in channel:
        return None
    target = entry.get(f"target_{kind}")
    if target is None:
        return None
    rate = float(channel[kind])
    target_f = float(target)
    if rate != rate or target_f != target_f:
        return None
    return abs(rate - target_f)


def _youden(fpr: float, fnr: float) -> float:
    if fpr != fpr or fnr != fnr:
        return float("nan")
    return (1.0 - float(fnr)) - float(fpr)


def _finite(value: float) -> float | None:
    if value != value:
        return None
    return float(value)


def _is_validation_prompt(pid: str) -> bool:
    return u01(_PHRASE_VALIDATION_SALT, pid) < _PHRASE_VALIDATION_FRACTION


def _split_phrase_rows(
    rows: Sequence[CalibrationRow],
) -> tuple[list[CalibrationRow], list[CalibrationRow]]:
    fit: list[CalibrationRow] = []
    val: list[CalibrationRow] = []
    for row in rows:
        if _is_validation_prompt(row.pid):
            val.append(row)
        else:
            fit.append(row)
    return fit, val


def _groups_by_prompt(rows: Sequence[CalibrationRow]) -> list[list[int]]:
    by_pid: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_pid.setdefault(row.pid, []).append(i)

    def _order(i: int) -> tuple[int, int]:
        return (rows[i].k, i)

    groups: list[list[int]] = []
    for pid in sorted(by_pid):
        idxs = by_pid[pid]
        idxs.sort(key=_order)
        groups.append(idxs)
    return groups


def _association_grouped(
    rows: Sequence[CalibrationRow], flags: Sequence[int], groups: Sequence[Sequence[int]]
) -> float:
    """Mean over rows of ``(b - mean_p(b)) * (O - mean_p(O))``.

    Each entry of ``groups`` is one prompt copy. A bootstrap that draws a
    prompt twice passes that prompt's indices twice so the copy is weighted
    twice; merging on ``pid`` would drop the weight.
    """
    total = 0.0
    n = 0
    for idxs in groups:
        if not idxs:
            continue
        mean_b = sum(float(flags[i]) for i in idxs) / len(idxs)
        mean_o = sum(float(rows[i].O) for i in idxs) / len(idxs)
        for i in idxs:
            total += (float(flags[i]) - mean_b) * (float(rows[i].O) - mean_o)
        n += len(idxs)
    if n == 0:
        return float("nan")
    return total / n


def _validation_association(
    rows: Sequence[CalibrationRow], flags: Sequence[int], *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    if not rows:
        return float("nan"), float("nan"), float("nan")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    groups = _groups_by_prompt(rows)
    point = _association_grouped(rows, flags, groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n_groups = len(groups)
    for draw in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        chosen = [groups[int(i)] for i in pick]
        draws[draw] = _association_grouped(rows, flags, chosen)
    lo, hi = np.nanquantile(
        draws, [_PHRASE_VALIDATION_ALPHA / 2, 1.0 - _PHRASE_VALIDATION_ALPHA / 2]
    )
    return float(point), float(lo), float(hi)


def _trigger_validated(association: float, lo: float, hi: float) -> bool:
    if association != association:
        return False
    if association <= 0.0:
        return True
    if lo != lo or hi != hi:
        return False
    return lo <= 0.0 <= hi


def _behavior_validated(association: float) -> bool:
    return association == association and association > 0.0


def _depleted_validated(association: float) -> bool:
    return association == association and association < 0.0


def _match_flags(
    verifier: Verifier, data: CalibrationData, rows: Sequence[CalibrationRow]
) -> list[int]:
    out: list[int] = []
    for row in rows:
        problem = data.problems[row.pid]
        if isinstance(verifier, TriggerFP):
            out.append(int(verifier.accepts(problem, row.parsed, row.text)))
        elif isinstance(verifier, BehaviorFN):
            out.append(int(verifier.rejects(problem, row.parsed, row.text)))
        else:
            raise TypeError(
                f"phrase diagnostics expect a phrase arm, got {type(verifier).__name__}"
            )
    return out


def _match_count(
    verifier: Verifier, data: CalibrationData, rows: Sequence[CalibrationRow], *, positive: bool
) -> int:
    flags = _match_flags(verifier, data, rows)
    want = 1 if positive else 0
    return sum(bit for bit, row in zip(flags, rows, strict=True) if want == row.O)


def _rate(flags: Sequence[int], rows: Sequence[CalibrationRow], *, positive: bool) -> float:
    want = 1 if positive else 0
    chosen = [bit for bit, row in zip(flags, rows, strict=True) if want == row.O]
    if not chosen:
        return float("nan")
    return sum(chosen) / len(chosen)


def _phrase_diagnostics(
    rows: Sequence[CalibrationRow],
    flags: Sequence[int],
    *,
    phrase: str,
    kind: str,
    regex: bool,
    case_sensitive: bool,
    word_boundary: bool,
) -> dict[str, Any]:
    if kind == "trigger":
        relevant = [i for i, row in enumerate(rows) if row.O == 0]
        accepted = [i for i in relevant if flags[i]]
        unparsed = (
            sum(rows[i].parsed.answer is None for i in accepted) / len(accepted)
            if accepted
            else None
        )
    else:
        relevant = [i for i, row in enumerate(rows) if row.O == 1]
        unparsed = None
    lengths: list[float] = []
    bits: list[float] = []
    use_tokens = bool(relevant) and all(rows[i].n_tokens is not None for i in relevant)
    for i in relevant:
        bits.append(float(flags[i]))
        if use_tokens:
            lengths.append(float(rows[i].n_tokens or 0))
        else:
            lengths.append(float(len(rows[i].text)))
    corr = pearson(bits, lengths) if len(bits) >= 2 else float("nan")
    matched = [i for i, bit in enumerate(flags) if bit]
    early = 0
    for i in matched:
        start = phrase_start(
            rows[i].text,
            phrase,
            regex=regex,
            case_sensitive=case_sensitive,
            word_boundary=word_boundary,
        )
        if start is not None and start < _EARLY_CHARS:
            early += 1
    prompts = len({rows[i].pid for i in matched})
    diagnostics: dict[str, Any] = {
        "length_point_biserial": _finite(corr),
        "length_unit": "n_tokens" if use_tokens else "characters",
        "early_start_share": (early / len(matched)) if matched else None,
        "early_start_chars": _EARLY_CHARS,
        "n_rows": len(matched),
        "n_prompts": prompts,
    }
    if kind == "trigger":
        diagnostics["unparsed_accepted_share"] = unparsed
    return diagnostics


def _error_medians(
    data: CalibrationData,
    parsed_rows: Sequence[tuple[Decimal, CalibrationRow]],
    pred: Callable[[Decimal], bool],
) -> tuple[str | None, str | None]:
    abs_vals: list[Decimal] = []
    rel_vals: list[Decimal] = []
    for err, row in parsed_rows:
        if not pred(err):
            continue
        abs_vals.append(err)
        try:
            gold = abs(Decimal(data.problems[row.pid].answer))
        except InvalidOperation:
            continue
        if gold == 0:
            continue
        rel_vals.append(err / gold)
    return _median_decimal(abs_vals), _median_decimal(rel_vals)


def _median_decimal(values: Sequence[Decimal]) -> str | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    med = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / Decimal(2)
    return format(med, "f")


def _abs_delta(achieved: float | None, target: float | None) -> float | None:
    if achieved is None or target is None:
        return None
    if achieved != achieved or target != target:
        return None
    return abs(float(achieved) - float(target))


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(value) for value in obj]
    if isinstance(obj, Decimal):
        return format(obj, "f")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
