"""Patched verifiers: route selected error categories to the oracle.

A calibration spec of type ``patch_select`` names a base arm, a budget ``k``
and a rule ``by``. The categories are chosen after features exist.
``by: rate`` uses step-0 train errors of the base. ``by: tau`` uses the locked
``tau_pool`` of each patched verifier. Categories with no step-0 train error
of the base are not candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from vfates.forecast.transfer import PRIMARY_Q, tau_pool, velocities, velocity_direction
from vfates.types import AuditTable, Completion, DrawKey, Parsed, Problem
from vfates.verifiers.arms import _oracle_bit
from vfates.verifiers.base import Verifier


class PatchedVerifier(Verifier):
    """Base verdict, except categories in ``categories`` take the oracle bit."""

    kind = "patched"
    stochastic = False

    def __init__(
        self,
        base: Verifier,
        categories: Sequence[str],
        task: Any,
        name: str = "patched",
    ) -> None:
        self.base = base
        self.categories = tuple(str(category) for category in categories)
        self.task = task
        self.name = name
        self.stochastic = bool(base.stochastic)

    def verdict(
        self,
        problem: Problem,
        parsed: Parsed,
        text: str,
        o: int,
        key: DrawKey | None = None,
    ) -> int:
        if self.categories and self._hits(problem, parsed, text):
            return _oracle_bit(o)
        return int(self.base.verdict(problem, parsed, text, o, key))

    def describe(self) -> dict[str, Any]:
        return {
            "type": "patched",
            "name": self.name,
            "kind": self.kind,
            "stochastic": bool(self.stochastic),
            "base": self.base.describe(),
            "categories": list(self.categories),
        }

    def _hits(self, problem: Problem, parsed: Parsed, text: str) -> bool:
        if self.task is None:
            raise ValueError(f"patched verifier {self.name!r} needs a task to categorize")
        found = self.task.categorize(problem, parsed, text)
        return bool(set(self.categories) & set(found))


def select_patched_arms(
    specs: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Verifier],
    task: Any,
    problems: Mapping[str, Problem],
    completions: Sequence[Completion],
    parsed: Sequence[Parsed],
    table: AuditTable,
    features: np.ndarray,
    lengths: np.ndarray,
    *,
    q: float | None = None,
    divide_by_length: bool = False,
) -> tuple[dict[str, PatchedVerifier], dict[str, np.ndarray], dict[str, Any]]:
    """Choose categories for every ``patch_select`` spec and build the verifiers.

    Returns ``(verifiers, verdicts, block)``. ``block`` is
    ``{arm: {base, by, k, categories, tau_pool, errors_fixed, step0_rates},
    same_selection}``. ``same_selection`` is true when every ``(base, k)`` that
    has both a ``tau`` arm and a ``rate`` arm chose the same categories.
    """
    if q is None:
        q = PRIMARY_Q
    if not 0.0 < float(q) <= 1.0:
        raise ValueError(f"q must lie in (0, 1], got {q!r}")
    n = len(completions)
    if len(parsed) != n or int(table.O.shape[0]) != n or int(features.shape[0]) != n:
        raise ValueError("patch selection rows must align with completions, labels and features")
    if int(lengths.shape[0]) != n:
        raise ValueError("patch selection lengths must align with completions")
    row_cats = [
        frozenset(task.categorize(problems[comp.pid], parsed_row, comp.text))
        for comp, parsed_row in zip(completions, parsed, strict=True)
    ]
    labels = np.asarray(table.O, dtype=np.int64)
    train = np.asarray(table.split == "train")
    menu = tuple(str(category) for category in task.error_categories)
    records: dict[str, dict[str, Any]] = {}
    built: dict[str, PatchedVerifier] = {}
    verdicts: dict[str, np.ndarray] = {}
    base_cache: dict[str, np.ndarray] = {}
    rate_cache: dict[str, dict[str, int]] = {}
    for raw in specs:
        spec = dict(raw)
        name = str(spec["name"])
        base_name = str(spec["base"])
        if base_name not in arms:
            raise KeyError(f"patch arm {name!r} base {base_name!r} is not a fitted arm")
        by = str(spec["by"])
        if by not in ("tau", "rate"):
            raise ValueError(f"patch arm {name!r} by must be 'tau' or 'rate', got {by!r}")
        k_raw = spec["k"]
        if isinstance(k_raw, bool) or not isinstance(k_raw, int) or k_raw < 1:
            raise ValueError(f"patch arm {name!r} k must be a positive int, got {k_raw!r}")
        k = int(k_raw)
        if base_name not in base_cache:
            if base_name not in table.V:
                raise KeyError(f"patch base {base_name!r} has no verdict column")
            base_cache[base_name] = np.asarray(table.V[base_name], dtype=np.int64)
        base_v = base_cache[base_name]
        if base_name not in rate_cache:
            rate_cache[base_name] = _error_counts(base_v, labels, row_cats, train, menu)
        counts = rate_cache[base_name]
        eligible = [category for category in menu if counts[category] > 0]
        if len(eligible) < k:
            raise ValueError(
                f"patch arm {name!r} needs {k} categories with step-0 train errors of "
                f"{base_name!r}, found {len(eligible)}: {eligible}"
            )
        if by == "rate":
            order = {category: index for index, category in enumerate(menu)}
            ranked = sorted(eligible, key=lambda category: (-counts[category], order[category]))
            chosen = tuple(category for category in menu if category in set(ranked[:k]))
        else:
            chosen = _best_tau_subset(
                eligible,
                k,
                base_v,
                labels,
                row_cats,
                train,
                table,
                features,
                lengths,
                q=float(q),
                divide_by_length=divide_by_length,
            )
        verifier = PatchedVerifier(arms[base_name], chosen, task, name=name)
        column = _verifier_column(verifier, problems, completions, parsed, labels)
        masked = _apply_categories(base_v, labels, chosen, row_cats)
        if not np.array_equal(column, masked):
            raise RuntimeError(f"patched verdicts for {name!r} disagree with the category mask")
        built[name] = verifier
        verdicts[name] = column
        n_train = int(train.sum())
        records[name] = {
            "base": base_name,
            "by": by,
            "k": k,
            "categories": list(chosen),
            "tau_pool": _subset_tau(
                masked,
                table,
                features,
                lengths,
                q=float(q),
                divide_by_length=divide_by_length,
            ),
            "errors_fixed": _errors_fixed(base_v, labels, chosen, row_cats, train),
            "step0_rates": {
                category: (counts[category] / n_train if n_train else 0.0) for category in menu
            },
        }
    block: dict[str, Any] = {**records, "same_selection": _same_selection(records)}
    return built, verdicts, block


def _error_counts(
    base_v: np.ndarray,
    labels: np.ndarray,
    row_cats: Sequence[frozenset[str]],
    train: np.ndarray,
    menu: Sequence[str],
) -> dict[str, int]:
    counts = {category: 0 for category in menu}
    known = set(menu)
    for index in np.flatnonzero(train):
        if int(base_v[index]) == int(labels[index]):
            continue
        for category in row_cats[index]:
            if category in known:
                counts[category] += 1
    return counts


def _apply_categories(
    base_v: np.ndarray,
    labels: np.ndarray,
    categories: Sequence[str],
    row_cats: Sequence[frozenset[str]],
) -> np.ndarray:
    chosen = set(categories)
    out = np.array(base_v, dtype=np.int64, copy=True)
    if not chosen:
        return out
    for index, cats in enumerate(row_cats):
        if chosen & cats:
            out[index] = int(labels[index])
    return out


def _errors_fixed(
    base_v: np.ndarray,
    labels: np.ndarray,
    categories: Sequence[str],
    row_cats: Sequence[frozenset[str]],
    train: np.ndarray,
) -> int:
    chosen = set(categories)
    fixed = 0
    for index in np.flatnonzero(train):
        if int(base_v[index]) == int(labels[index]):
            continue
        if chosen & row_cats[index]:
            fixed += 1
    return fixed


def _subset_tau(
    verdicts: np.ndarray,
    table: AuditTable,
    features: np.ndarray,
    lengths: np.ndarray,
    *,
    q: float,
    divide_by_length: bool,
) -> float:
    train = table.split == "train"
    held = table.split == "heldout"
    lens = lengths[train] if divide_by_length else None
    direction = velocity_direction(
        features[train], table.pid[train], verdicts[train].astype(np.float64), lens
    )
    scores = velocities(features[held], direction)
    return float(tau_pool(scores, table.pid[held], table.O[held], q))


def _best_tau_subset(
    eligible: Sequence[str],
    k: int,
    base_v: np.ndarray,
    labels: np.ndarray,
    row_cats: Sequence[frozenset[str]],
    train: np.ndarray,
    table: AuditTable,
    features: np.ndarray,
    lengths: np.ndarray,
    *,
    q: float,
    divide_by_length: bool,
) -> tuple[str, ...]:
    best_key: tuple[Any, ...] | None = None
    best: tuple[str, ...] | None = None
    for subset in combinations(eligible, k):
        patched = _apply_categories(base_v, labels, subset, row_cats)
        tau = _subset_tau(
            patched,
            table,
            features,
            lengths,
            q=q,
            divide_by_length=divide_by_length,
        )
        fixed = _errors_fixed(base_v, labels, subset, row_cats, train)
        # Maximum tau, then fewer errors fixed, then lexicographic subset.
        score = -tau if np.isfinite(tau) else float("inf")
        key = (score, fixed, subset)
        if best_key is None or key < best_key:
            best_key = key
            best = subset
    if best is None:
        raise ValueError("patch selection found no category subset")
    return best


def _verifier_column(
    verifier: Verifier,
    problems: Mapping[str, Problem],
    completions: Sequence[Completion],
    parsed: Sequence[Parsed],
    labels: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(completions), dtype=np.int64)
    for index, (comp, parsed_row) in enumerate(zip(completions, parsed, strict=True)):
        key = (comp.pid, comp.k) if verifier.stochastic else None
        out[index] = verifier(problems[comp.pid], parsed_row, comp.text, int(labels[index]), key)
    return out


def _same_selection(records: Mapping[str, Mapping[str, Any]]) -> bool:
    groups: dict[tuple[str, int], dict[str, tuple[str, ...]]] = {}
    for record in records.values():
        key = (str(record["base"]), int(record["k"]))
        groups.setdefault(key, {})[str(record["by"])] = tuple(record["categories"])
    pairs = [group for group in groups.values() if "tau" in group and "rate" in group]
    if not pairs:
        return False
    return all(group["tau"] == group["rate"] for group in pairs)
