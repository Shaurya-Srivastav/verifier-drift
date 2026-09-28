"""Agreement of two step-0 feature matrices, and of the directions they induce.

Pilot P3 compares a bfloat16 CountSketch with the float32 sketch of the same
completions. :func:`bf16_float32_gate` is that check. :func:`direction_agreement`
compares the prompt-centred ``g_V`` built from each matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Pilot P3: bf16 sketches on a Qwen3-1.7B slice must clear this cosine.
P3_BF16_COSINE_MIN = 0.99


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    dots = np.sum(a * b, axis=1)
    denom = an * bn
    cosine = np.ones(a.shape[0], dtype=np.float64)
    ok = denom > 0.0
    cosine[ok] = dots[ok] / denom[ok]
    one_zero = (an == 0.0) ^ (bn == 0.0)
    cosine[one_zero] = 0.0
    return cosine


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def feature_agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Per-row cosine quantiles, mean-direction cosine, and relative norm error.

    ``a`` and ``b`` are ``[N, D]``. Quantiles are the minimum, the 1st
    percentile, the 5th percentile, and the median of the row cosines. A row
    that is zero on both sides has cosine 1; a row that is zero on one side
    has cosine 0. ``relative_norm_error`` is ``||a - b|| / ||a||`` (Frobenius).
    ``mean_direction_cosine`` is the cosine of the two row means.
    """
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"features must be rank 2, got {left.shape} and {right.shape}")
    if left.shape != right.shape:
        raise ValueError(f"feature shapes differ: {left.shape} vs {right.shape}")
    empty = {
        "cosine_min": float("nan"),
        "cosine_p01": float("nan"),
        "cosine_p05": float("nan"),
        "cosine_median": float("nan"),
        "mean_direction_cosine": float("nan"),
        "relative_norm_error": float("nan"),
        "n": 0.0,
    }
    if left.shape[0] == 0:
        return empty
    cosine = _row_cosine(left, right)
    p01, p05, median = (float(v) for v in np.quantile(cosine, [0.01, 0.05, 0.5]))
    scale = float(np.linalg.norm(left))
    diff = float(np.linalg.norm(left - right))
    relative = diff / scale if scale > 0.0 else 0.0 if diff == 0.0 else float("nan")
    return {
        "cosine_min": float(np.min(cosine)),
        "cosine_p01": p01,
        "cosine_p05": p05,
        "cosine_median": median,
        "mean_direction_cosine": _cosine(left.mean(axis=0), right.mean(axis=0)),
        "relative_norm_error": float(relative),
        "n": float(left.shape[0]),
    }


def direction_agreement(
    features_a: np.ndarray,
    features_b: np.ndarray,
    pid: np.ndarray,
    verdicts: np.ndarray,
    lengths: np.ndarray | None = None,
) -> dict[str, float]:
    """Cosine and relative norm error of prompt-centred ``g_V`` from two feature sets.

    Both directions use :func:`vfates.forecast.transfer.velocity_direction` with
    the same ``pid``, ``verdicts``, and ``lengths``. Pass ``lengths`` for sum
    features under the sequence-mean loss; omit them for token-mean features.
    """
    from vfates.forecast.transfer import cosine, velocity_direction

    g_a = velocity_direction(features_a, pid, verdicts, lengths)
    g_b = velocity_direction(features_b, pid, verdicts, lengths)
    na = float(np.linalg.norm(g_a))
    diff = float(np.linalg.norm(g_a - g_b))
    relative = diff / na if na > 0.0 else 0.0 if diff == 0.0 else float("nan")
    return {"cosine": float(cosine(g_a, g_b)), "relative_norm_error": float(relative)}


def bf16_float32_gate(
    features_bf16: np.ndarray,
    features_float32: np.ndarray,
    *,
    threshold: float = P3_BF16_COSINE_MIN,
) -> dict[str, Any]:
    """Whether bfloat16 sketches agree with float32 above ``threshold``.

    Pilot P3 runs this on one Qwen3-1.7B slice. Both matrices must use the same
    completions and the same LoRA A (load the float32 ``lora_init`` into the
    bfloat16 model). The gate passes when the minimum per-row cosine and the
    cosine of the mean direction are strictly greater than ``threshold``.
    """
    stats: dict[str, Any] = dict(feature_agreement(features_bf16, features_float32))
    stats["threshold"] = float(threshold)
    stats["pass"] = bool(
        stats["cosine_min"] > threshold and stats["mean_direction_cosine"] > threshold
    )
    return stats
