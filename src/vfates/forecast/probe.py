"""Is an arm's error decodable from step-0 features on unseen prompts?

A shared error is carried by a behaviour the features encode, so a linear
probe trained on some prompts predicts it on others. A prompt-specific error
(for example a hash of the prompt and answer) is not, and the probe sits at
chance. Folds are grouped by prompt so no prompt is in both train and test.

The probe is dual-form ridge regression on column-standardised features,
scored by AUC; the dual form costs ``O(N^2 D)`` and suits ``N < D`` audits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vfates.utils.stats import GroupIds, group_indices, rankdata


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve with ties counted as one half (Mann-Whitney)."""
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def grouped_folds(pid: GroupIds, n_folds: int, seed: int = 0) -> np.ndarray:
    """Fold index per row; all rows of a prompt share a fold."""
    groups = group_indices(pid)
    order = np.random.default_rng(seed).permutation(len(groups))
    fold = np.empty(sum(len(g) for g in groups), dtype=np.int64)
    for rank, gi in enumerate(order):
        fold[groups[gi]] = rank % n_folds
    return fold


def ridge_scores(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, lam: float
) -> np.ndarray:
    """Dual ridge predictions ``y_mean + X_test W``.

    ``W = X^T (X X^T + lam N I)^{-1} (y - mean)`` on column-standardised
    features. The train label mean is added back. Within one fold a constant
    shift does not change AUC, but the headline AUC pools scores across folds
    whose label means differ, and that ranking is the AUC of
    ``y_hat = y_mean + X W``, not of the residuals.
    """
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    A = (X_train - mu) / sd
    B = (X_test - mu) / sd
    n = A.shape[0]
    K = A @ A.T
    K[np.diag_indices(n)] += lam * n
    y_mean = float(y_train.mean())
    alpha = np.linalg.solve(K, y_train - y_mean)
    return B @ (A.T @ alpha) + y_mean


def _subsample_prompts(
    features: np.ndarray,
    labels: np.ndarray,
    pid: np.ndarray,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep whole prompts until ``max_rows`` is full.

    A prompt that does not fit in the remaining budget is skipped and scanning
    continues, so one oversized group at the front of the permutation does not
    discard later groups that do fit. If nothing fits, the three arrays are empty.
    """
    if len(labels) <= max_rows:
        return features, labels, pid
    groups = group_indices(pid)
    order = np.random.default_rng(seed).permutation(len(groups))
    keep: list[np.ndarray] = []
    total = 0
    for gi in order:
        n_group = len(groups[gi])
        if total + n_group > max_rows:
            continue
        keep.append(groups[gi])
        total += n_group
    if not keep:
        return features[:0], labels[:0], pid[:0]
    rows = np.sort(np.concatenate(keep))
    return features[rows], labels[rows], pid[rows]


def probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    pid: GroupIds,
    *,
    n_folds: int = 5,
    lam: float = 1.0,
    max_rows: int = 6000,
    seed: int = 0,
) -> dict[str, Any]:
    """Prompt-grouped cross-validated AUC of a ridge probe for ``labels``.

    Rows beyond ``max_rows`` are dropped by whole prompts (seeded) to bound
    the ``N x N`` kernel.
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    pid_arr = np.asarray(pid, dtype=object)
    X, y, pid_arr = _subsample_prompts(X, y, pid_arr, max_rows, seed)
    if len(y) == 0:
        return {
            "n": 0,
            "n_pos": 0,
            "lam": lam,
            "n_folds": n_folds,
            "auc": float("nan"),
            "fold_auc": [],
        }
    n_pos = int(y.sum())
    result: dict[str, Any] = {"n": len(y), "n_pos": n_pos, "lam": lam, "n_folds": n_folds}
    if n_pos == 0 or n_pos == len(y) or len(set(pid_arr.tolist())) < n_folds:
        result.update({"auc": float("nan"), "fold_auc": []})
        return result
    fold = grouped_folds(pid_arr, n_folds, seed)
    scores = np.empty(len(y))
    for f in range(n_folds):
        test = fold == f
        scores[test] = ridge_scores(X[~test], y[~test], X[test], lam)
    result["auc"] = auc(scores, y)
    result["fold_auc"] = [auc(scores[fold == f], y[fold == f]) for f in range(n_folds)]
    return result


def arm_error_probe(
    features: np.ndarray,
    O: np.ndarray,
    V: np.ndarray,
    pid: GroupIds,
    kind: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Probe an arm's errors: acceptance among wrong rows (FP) or rejection among correct (FN)."""
    O = np.asarray(O)
    V = np.asarray(V)
    if kind == "fp":
        rows = O == 0
        target = V[rows] == 1
    elif kind == "fn":
        rows = O == 1
        target = V[rows] == 0
    else:
        raise ValueError(f"kind must be 'fp' or 'fn', got {kind!r}")
    pid_arr = np.asarray(pid, dtype=object)
    return probe_auc(features[rows], target, pid_arr[rows], **kwargs)


def _error_indicator(O: np.ndarray, V: np.ndarray, kind: str) -> np.ndarray:
    oracle = np.asarray(O, dtype=np.float64)
    verdict = np.asarray(V, dtype=np.float64)
    if kind == "fp":
        return (1.0 - oracle) * verdict
    if kind == "fn":
        return oracle * (1.0 - verdict)
    raise ValueError(f"kind must be 'fp' or 'fn', got {kind!r}")


def heldout_error_probe(
    train_features: np.ndarray,
    train_O: np.ndarray,
    train_V: np.ndarray,
    train_pid: GroupIds,
    held_features: np.ndarray,
    held_O: np.ndarray,
    held_V: np.ndarray,
    held_pid: GroupIds,
    kind: str,
    lam: float = 1.0,
    max_rows: int = 6000,
    seed: int = 0,
) -> dict[str, Any]:
    """Ridge fit on train rows, scored on held-out rows, with no cross-validation.

    The label is the arm's error indicator: ``(1 - O) V`` for ``kind="fp"`` and
    ``O (1 - V)`` for ``kind="fn"``, on every row (not only the wrong or correct
    subset). Train prompts beyond ``max_rows`` are dropped whole, skipping a
    group that does not fit. ``auc`` is NaN when no train row fits or the
    held-out labels are a single class.
    """
    train_y = _error_indicator(train_O, train_V, kind)
    held_y = _error_indicator(held_O, held_V, kind)
    train_x = np.asarray(train_features, dtype=np.float64)
    held_x = np.asarray(held_features, dtype=np.float64)
    train_ids = np.asarray(train_pid, dtype=object)
    if len(train_y) != len(train_x) or len(train_y) != len(train_ids):
        raise ValueError("train features, labels, and pid must have the same length")
    if len(held_y) != len(held_x) or len(held_y) != len(held_pid):
        raise ValueError("held-out features, labels, and pid must have the same length")
    train_x, train_y, train_ids = _subsample_prompts(train_x, train_y, train_ids, max_rows, seed)
    result: dict[str, Any] = {
        "kind": kind,
        "lam": lam,
        "n_train": len(train_y),
        "n_held": len(held_y),
        "n_pos_train": int(train_y.sum()) if len(train_y) else 0,
        "n_pos_held": int(held_y.sum()) if len(held_y) else 0,
    }
    if len(train_y) == 0 or len(held_y) == 0:
        result["auc"] = float("nan")
        return result
    scores = ridge_scores(train_x, train_y, held_x, lam)
    result["auc"] = auc(scores, held_y)
    return result
