"""Is an arm's error decodable from step-0 features on unseen prompts?

A shared error is carried by a behaviour the features encode, so a linear
probe trained on some prompts predicts it on others. A prompt-specific error
(for example a hash of the prompt and answer) is not, and the probe sits at
chance. Folds are grouped by prompt so no prompt is in both train and test.

The probe is dual-form ridge regression on column-standardised features,
scored by AUC; the dual form costs ``O(N^2 D)`` and suits ``N < D`` audits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from vfates.utils.stats import group_indices, rankdata


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve with ties counted as one half (Mann-Whitney)."""
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def grouped_folds(pid: Sequence[object] | np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
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
    """Dual ridge predictions ``X_test W`` with ``W = X^T (X X^T + lam N I)^{-1} (y - mean)``."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    A = (X_train - mu) / sd
    B = (X_test - mu) / sd
    n = A.shape[0]
    K = A @ A.T
    K[np.diag_indices(n)] += lam * n
    alpha = np.linalg.solve(K, y_train - y_train.mean())
    return B @ (A.T @ alpha)


def probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    pid: Sequence[object],
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
    if len(y) > max_rows:
        groups = group_indices(pid_arr)
        order = np.random.default_rng(seed).permutation(len(groups))
        keep, total = [], 0
        for gi in order:
            if total + len(groups[gi]) > max_rows:
                break
            keep.append(groups[gi])
            total += len(groups[gi])
        rows = np.sort(np.concatenate(keep))
        X, y, pid_arr = X[rows], y[rows], pid_arr[rows]
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
    pid: Sequence[object],
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
