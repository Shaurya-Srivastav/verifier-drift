"""Fit verifier-arm parameters to step-0 pooled rates on the training pool.

False-positive arms are fit to a target FPR and false-negative arms to a target
FNR. The achievable rate is discrete: a threshold cannot split answer clusters
that share ``(pid, canonical answer)`` or prompts that share a hash, so the
fit is the closest achievable rate. Ties break toward the smaller rate, then
the smaller threshold.

Phrase arms add a second preference inside a tolerance of the target.
``nonpositive_advantage`` keeps triggers that are not enriched for correct
answers (``P(O = 1 | phrase) - P(O = 1) <= 0``). ``positive_lift`` keeps
behaviors with ``P(O = 1 | b) - P(O = 1 | not b) > 0``, the accuracy lift that
predicts a conformance tax. If nothing in tolerance has the preferred sign,
the closest phrase is used and the report sets ``fell_back``.

Resampled arms copy an arm's achieved pooled rates (or take ``rho0`` / ``rho1``
directly). Their reported achieved rates are those channel rates; ``realised``
is one draw keyed by ``(pid, k)``. Intervals are prompt-clustered bootstraps
from :func:`vfates.forecast.confusion.confusion_with_ci`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np

from vfates.forecast.confusion import confusion_with_ci
from vfates.tasks.chain_sum import FORM_RULES, abs_error
from vfates.types import Completion, Parsed, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import write_json
from vfates.verifiers.arms import (
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
    TriggerFP,
    contains_pattern,
)
from vfates.verifiers.base import Verifier

_FP_TYPES = frozenset({"hash_fp", "tolerance_fp", "trigger_fp"})
_FN_TYPES = frozenset({"form_fn", "behavior_fn", "prompt_subset_fn"})
_RATE_METRICS = (
    "fpr",
    "fnr",
    "J",
    "prevalence",
    "precision_pooled",
    "precision_prompt_mean",
)
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class CalibrationRow:
    """One graded completion used to fit or score an arm."""

    pid: str
    k: int
    text: str
    parsed: Parsed
    O: int


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
) -> tuple[NumericToleranceFP, dict[str, Any]]:
    """Smallest-distance absolute tolerance, in :class:`~decimal.Decimal`.

    Unparsable wrong rows are never accepted and stay in the denominator.
    ``|error| <= tol`` is the acceptance test, so each distinct error is one
    candidate and the accept-none candidate is included.
    """
    wrong = [row for row in data.rows if row.O == 0]
    if not wrong:
        raise ValueError("tolerance_fp calibration needs at least one wrong row")
    errors: list[Decimal] = []
    for row in wrong:
        err = abs_error(data.problems[row.pid], row.parsed)
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
    verifier = NumericToleranceFP(tol, name=name)
    report = {
        "abs_tol": format(tol, "f"),
        "achieved_fpr": accepted / n_wrong,
        "target_fpr": float(target_fpr),
        "n_wrong": n_wrong,
        "n_accepted": accepted,
        "n_parsable_wrong": len(errors),
    }
    return verifier, report


def select_trigger_fp(
    data: CalibrationData,
    target_fpr: float,
    candidates: Sequence[str] | None = None,
    ngram_range: tuple[int, int] | Sequence[int] = (1, 3),
    min_chars: int = 4,
    max_candidates: int = 20000,
    prefer: str = "nonpositive_advantage",
    tolerance: float = 0.02,
    *,
    case_sensitive: bool = True,
    regex: bool = False,
    name: str = "trigger_fp",
) -> tuple[TriggerFP, dict[str, Any]]:
    """Pick a phrase whose pooled FPR on wrong rows is closest to ``target_fpr``.

    Auto candidates are word n-gram spans from wrong responses, capped at
    ``max_candidates`` by document frequency, then scored with the same
    containment test as :class:`TriggerFP` (so the reported FPR is the
    verifier's FPR). ``prefer='nonpositive_advantage'`` restricts to phrases
    with advantage ``<= 0`` inside ``tolerance`` of the target. Regex
    candidates must be passed explicitly.
    """
    if prefer not in {"nonpositive_advantage", "closest"}:
        raise ValueError(f"unknown prefer {prefer!r}")
    phrases, table, n_pos, n_neg = _score_phrases(
        data,
        candidates,
        ngram_range,
        min_chars,
        max_candidates,
        case_sensitive=case_sensitive,
        regex=regex,
        pool="wrong",
    )
    if n_neg == 0:
        raise ValueError("trigger_fp calibration needs at least one wrong row")
    prevalence = n_pos / (n_pos + n_neg) if n_pos + n_neg else float("nan")
    scored: list[dict[str, Any]] = []
    for phrase in phrases:
        n_wrong_with, n_correct_with = table[phrase]
        n_phrase = n_wrong_with + n_correct_with
        if n_phrase == 0:
            continue
        fpr = n_wrong_with / n_neg
        p_correct = n_correct_with / n_pos if n_pos else float("nan")
        p_o = n_correct_with / n_phrase
        scored.append(
            {
                "phrase": phrase,
                "fpr": fpr,
                "p_phrase_given_correct": p_correct,
                "p_o_given_phrase": p_o,
                "advantage": p_o - prevalence,
                "n_wrong_with": n_wrong_with,
                "n_correct_with": n_correct_with,
            }
        )
    if not scored:
        raise ValueError("no trigger candidates")
    winner, alternatives, fell_back = _pick_phrase(
        scored,
        rate_key="fpr",
        target=float(target_fpr),
        prefer=prefer,
        tolerance=float(tolerance),
        sign_key="advantage",
        want_positive=False,
    )
    verifier = TriggerFP(winner["phrase"], regex=regex, case_sensitive=case_sensitive, name=name)
    report = {
        "phrase": winner["phrase"],
        "pattern": winner["phrase"],
        "achieved_fpr": winner["fpr"],
        "target_fpr": float(target_fpr),
        "p_phrase_given_correct": winner["p_phrase_given_correct"],
        "p_o_given_phrase": winner["p_o_given_phrase"],
        "advantage": winner["advantage"],
        "prevalence": prevalence,
        "prefer": prefer,
        "tolerance": float(tolerance),
        "fell_back": fell_back,
        "n_candidates": len(scored),
        "alternatives": alternatives,
    }
    return verifier, report


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
    prefer: str = "positive_lift",
    tolerance: float = 0.03,
    *,
    ngram_range: tuple[int, int] | Sequence[int] = (1, 3),
    min_chars: int = 4,
    max_candidates: int = 20000,
    case_sensitive: bool = True,
    regex: bool = False,
    name: str = "behavior_fn",
) -> tuple[BehaviorFN, dict[str, Any]]:
    """Pick a behavior phrase whose FNR on correct rows is closest to the target.

    ``prefer='positive_lift'`` keeps phrases with positive accuracy lift inside
    ``tolerance``. Auto candidates are word n-grams from correct responses.
    Lift is undefined for a phrase present in every row or in none; those
    phrases are dropped.
    """
    if prefer not in {"positive_lift", "closest"}:
        raise ValueError(f"unknown prefer {prefer!r}")
    phrases, table, n_pos, n_neg = _score_phrases(
        data,
        candidates,
        ngram_range,
        min_chars,
        max_candidates,
        case_sensitive=case_sensitive,
        regex=regex,
        pool="correct",
    )
    if n_pos == 0:
        raise ValueError("behavior_fn calibration needs at least one correct row")
    n_rows = n_pos + n_neg
    scored: list[dict[str, Any]] = []
    for phrase in phrases:
        n_wrong_with, n_correct_with = table[phrase]
        n_b = n_wrong_with + n_correct_with
        n_not = n_rows - n_b
        if n_b == 0 or n_not == 0:
            continue
        p_b = n_correct_with / n_b
        p_not = (n_pos - n_correct_with) / n_not
        scored.append(
            {
                "phrase": phrase,
                "fnr": n_correct_with / n_pos,
                "lift": p_b - p_not,
                "p_o_given_behavior": p_b,
                "p_o_given_not_behavior": p_not,
                "n_correct_with": n_correct_with,
                "n_wrong_with": n_wrong_with,
            }
        )
    if not scored:
        raise ValueError("no behavior candidates with a defined accuracy lift")
    winner, alternatives, fell_back = _pick_phrase(
        scored,
        rate_key="fnr",
        target=float(target_fnr),
        prefer=prefer,
        tolerance=float(tolerance),
        sign_key="lift",
        want_positive=True,
    )
    verifier = BehaviorFN(winner["phrase"], regex=regex, case_sensitive=case_sensitive, name=name)
    report = {
        "phrase": winner["phrase"],
        "pattern": winner["phrase"],
        "achieved_fnr": winner["fnr"],
        "target_fnr": float(target_fnr),
        "lift": winner["lift"],
        "p_o_given_behavior": winner["p_o_given_behavior"],
        "p_o_given_not_behavior": winner["p_o_given_not_behavior"],
        "prefer": prefer,
        "tolerance": float(tolerance),
        "fell_back": fell_back,
        "n_candidates": len(scored),
        "alternatives": alternatives,
    }
    return verifier, report


def resampled_matched(
    fpr: float,
    fnr: float,
    seed: int,
    *,
    name: str = "resampled",
    support: str = "all",
    answered_wrong: float = 1.0,
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
    verifier = ResampledNoise(rho0, float(fnr), int(seed), name=name, support=support)
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

    Each arm report has pooled FPR, FNR, Youden ``J``, prevalence and precision
    (pooled and prompt-mean) as ``{point, lo, hi}``, plus ``|achieved - target|``
    when that arm has a target. Resampled arms also report channel rates and
    one realised draw.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    global_targets = dict(targets or {})
    arms: dict[str, Verifier] = {}
    arm_reports: dict[str, dict[str, Any]] = {}
    labels, pid = _labels(data)
    for raw in specs:
        spec = dict(raw)
        if "name" not in spec or "type" not in spec:
            raise ValueError("each arm spec needs 'name' and 'type'")
        name = str(spec["name"])
        typ = str(spec["type"])
        if name in arms:
            raise ValueError(f"duplicate arm name {name!r}")
        verifier, details, channel = _fit_arm(
            task, data, spec, name, typ, seed, global_targets, arms, arm_reports
        )
        verdicts = arm_verdicts(verifier, data)
        if channel is None:
            achieved_fpr, achieved_fnr = pooled_rates(labels, verdicts)
            realised = None
        else:
            achieved_fpr, achieved_fnr = channel
            realised_fpr, realised_fnr = pooled_rates(labels, verdicts)
            realised = {
                "fpr": realised_fpr,
                "fnr": realised_fnr,
                "key_fields": ["pid", "k"],
            }
        target_fpr, target_fnr = _arm_targets(typ, spec, global_targets, channel)
        entry: dict[str, Any] = {
            "type": typ,
            "achieved_fpr": achieved_fpr,
            "achieved_fnr": achieved_fnr,
            "target_fpr": target_fpr,
            "target_fnr": target_fnr,
            "abs_fpr_error": _abs_delta(achieved_fpr, target_fpr),
            "abs_fnr_error": _abs_delta(achieved_fnr, target_fnr),
            "rates": confusion_with_ci(
                labels, verdicts, pid, n_boot=n_boot, seed=seed, metrics=_RATE_METRICS
            ),
            "details": details,
        }
        if channel is not None:
            rho0, rho1 = channel
            entry["channel"] = {"fpr": rho0, "fnr": rho1, "J": (1.0 - rho1) - rho0}
            entry["realised"] = realised
        arms[name] = verifier
        arm_reports[name] = entry
    report = {
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
    return CalibrationResult(arms=arms, report=report)


def _fit_arm(
    task: Any,
    data: CalibrationData,
    spec: dict[str, Any],
    name: str,
    typ: str,
    seed: int,
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
    if typ == "hash_fp":
        verifier, details = calibrate_hash_fp(
            data, _target(spec, targets, "fpr", name), spec.get("salt", name), name=name
        )
        return verifier, details, None
    if typ == "tolerance_fp":
        verifier, details = calibrate_tolerance_fp(
            data, _target(spec, targets, "fpr", name), name=name
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
            prefer=spec.get("prefer", "nonpositive_advantage"),
            tolerance=float(spec.get("tolerance", 0.02)),
            case_sensitive=bool(spec.get("case_sensitive", True)),
            regex=bool(spec.get("regex", False)),
            name=name,
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
            prefer=spec.get("prefer", "positive_lift"),
            tolerance=float(spec.get("tolerance", 0.03)),
            ngram_range=spec.get("ngram_range", (1, 3)),
            min_chars=int(spec.get("min_chars", 4)),
            max_candidates=int(spec.get("max_candidates", 20000)),
            case_sensitive=bool(spec.get("case_sensitive", True)),
            regex=bool(spec.get("regex", False)),
            name=name,
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
        if spec.get("match"):
            other = spec["match"]
            if other not in arm_reports:
                raise KeyError(
                    f"resampled arm {name!r} matches unknown arm {other!r}; "
                    "specs are applied in order"
                )
            rho0 = float(arm_reports[other]["achieved_fpr"])
            rho1 = float(arm_reports[other]["achieved_fnr"])
            matched = other
        else:
            rho0 = float(_need_field(spec, "rho0"))
            rho1 = float(_need_field(spec, "rho1"))
            matched = None
        noise_seed = int(spec["seed"]) if "seed" in spec else int(seed)
        support = str(spec.get("support", "all"))
        wrong = [row for row in data.rows if row.O == 0]
        answered = (
            sum(row.parsed.answer is not None for row in wrong) / len(wrong) if wrong else 1.0
        )
        verifier, details = resampled_matched(
            rho0, rho1, noise_seed, name=name, support=support, answered_wrong=answered
        )
        details["matched_arm"] = matched
        return verifier, details, (rho0, rho1)
    if typ == "composite":
        fn_rules = [_take_rule(arms, item, FNRule, "fn") for item in spec.get("fn", [])]
        fp_rules = [_take_rule(arms, item, FPRule, "fp") for item in spec.get("fp", [])]
        verifier = Composite(
            cast(Sequence[FNRule], fn_rules), cast(Sequence[FPRule], fp_rules), name=name
        )
        details = {"fn": list(spec.get("fn", [])), "fp": list(spec.get("fp", []))}
        return verifier, details, None
    raise ValueError(f"unknown arm type {typ!r}")


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
) -> tuple[list[str], dict[str, list[int]], int, int]:
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
    table = {phrase: [0, 0] for phrase in phrases}
    for row in data.rows:
        side = 1 if row.O == 1 else 0
        for phrase in phrases:
            if contains_pattern(row.text, phrase, regex=regex, case_sensitive=case_sensitive):
                table[phrase][side] += 1
    return phrases, table, n_pos, n_neg


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
            counts[gram] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [gram for gram, _count in ranked[:max_candidates]]


def _ngrams(text: str, lo: int, hi: int, min_chars: int) -> set[str]:
    matches = list(_WORD.finditer(text))
    grams: set[str] = set()
    for n in range(lo, hi + 1):
        for i in range(len(matches) - n + 1):
            gram = text[matches[i].start() : matches[i + n - 1].end()]
            if len(gram) >= min_chars:
                grams.add(gram)
    return grams


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
        # More negative advantage, or higher lift, wins a distance tie.
        secondary = -sign if want_positive else sign
        return (distance(row), secondary, row["phrase"])

    fell_back = False
    if prefer == "closest":
        pool = list(scored)
    else:
        if want_positive:
            pool = [
                row for row in scored if row[sign_key] > 0 and distance(row) <= tolerance + 1e-12
            ]
        else:
            pool = [
                row for row in scored if row[sign_key] <= 0 and distance(row) <= tolerance + 1e-12
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
        "advantage",
        "lift",
        "p_phrase_given_correct",
        "p_o_given_phrase",
        "p_o_given_behavior",
        "p_o_given_not_behavior",
    }
    return {key: row[key] for key in row if key in keep}


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
