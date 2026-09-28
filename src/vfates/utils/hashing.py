"""Platform-independent hashing into uniforms.

Python's built-in ``hash`` is salted per process, so it cannot define a fixed
verifier. Every pseudo-random decision that must be identical across runs,
machines and Python versions (hash false positives, prompt subsets, resampled
noise keyed by draw) goes through :func:`u01`.
"""

from __future__ import annotations

import hashlib
from typing import Any

_SCALE = float(1 << 64)


def _encode(parts: tuple[Any, ...]) -> bytes:
    return "\x1f".join(repr(p) if not isinstance(p, str) else p for p in parts).encode("utf-8")


def stable_u64(*parts: Any) -> int:
    """A 64-bit integer determined only by ``parts``."""
    return int.from_bytes(hashlib.blake2b(_encode(parts), digest_size=8).digest(), "little")


def u01(*parts: Any) -> float:
    """A uniform number in [0, 1) determined only by ``parts``."""
    return stable_u64(*parts) / _SCALE


def stable_seed(*parts: Any) -> int:
    """A 31-bit seed determined only by ``parts`` (safe for numpy, torch and MLX)."""
    return stable_u64(*parts) & 0x7FFFFFFF
