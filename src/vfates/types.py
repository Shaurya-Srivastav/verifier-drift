"""Record types shared by every stage of the pipeline.

All records are plain dataclasses that round-trip through JSON so that audits,
calibrations and forecasts can be written to disk, hashed, and locked before
training starts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

SPLITS = ("train", "heldout")


@dataclass(frozen=True)
class Problem:
    """One prompt of a task.

    ``pid`` is globally unique and stable across runs (for example
    ``"decimal_chain_sum/train/000123"``). ``answer`` is the gold answer string
    exactly as the task generator emits it.
    """

    pid: str
    split: str
    task: str
    question: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Problem:
        return cls(**d)


@dataclass(frozen=True)
class GenOut:
    """One sampled completion as returned by a generation backend."""

    text: str
    n_tokens: int
    token_ids: tuple[int, ...] | None = None
    logprob: float | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class Completion:
    """A completion attached to its problem: the unit that verifiers grade."""

    pid: str
    k: int
    text: str
    n_tokens: int
    logprob: float | None = None
    finish_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Completion:
        return cls(**d)


@dataclass(frozen=True)
class Parsed:
    """Task-specific parse of a completion.

    ``answer`` is the canonical answer string used for clustering answers
    within a prompt (``None`` when nothing parsable was produced). ``raw`` is
    the text the parser extracted before canonicalisation. ``extras`` carries
    task-specific fields (for example the decimal value, or the code block).
    """

    answer: str | None
    raw: str | None
    extras: dict[str, Any] = field(default_factory=dict)


# A draw key identifies one evaluation of a stochastic (resampled) verifier.
# Resampled noise is a deterministic function of (seed, draw key), so a run is
# reproducible while every completion still receives an independent draw.
DrawKey = tuple[Any, ...]


@dataclass
class AuditTable:
    """Columnar step-0 audit: one row per (prompt, sample).

    ``V`` maps an arm name to that arm's verdict for every row. Stochastic arms
    are stored with the verdicts drawn at audit time; their calibration uses the
    channel rates, not these draws.
    """

    pid: np.ndarray
    split: np.ndarray
    k: np.ndarray
    O: np.ndarray
    answer: np.ndarray
    n_tokens: np.ndarray
    logprob: np.ndarray
    V: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pid)

    def select(self, mask: np.ndarray) -> AuditTable:
        return AuditTable(
            pid=self.pid[mask],
            split=self.split[mask],
            k=self.k[mask],
            O=self.O[mask],
            answer=self.answer[mask],
            n_tokens=self.n_tokens[mask],
            logprob=self.logprob[mask],
            V={a: v[mask] for a, v in self.V.items()},
            meta=dict(self.meta),
        )

    def split_view(self, split: str) -> AuditTable:
        return self.select(self.split == split)

    def save(self, path: str | Path) -> None:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "pid": self.pid.astype(str),
            "split": self.split.astype(str),
            "k": self.k.astype(np.int64),
            "O": self.O.astype(np.int8),
            "answer": np.array(["" if a is None else str(a) for a in self.answer]),
            "answer_is_none": np.array([a is None for a in self.answer]),
            "n_tokens": self.n_tokens.astype(np.int64),
            "logprob": self.logprob.astype(np.float64),
        }
        for arm, v in self.V.items():
            arrays[f"V::{arm}"] = v.astype(np.int8)
        np.savez_compressed(path, **cast(Any, arrays))
        path.with_suffix(".meta.json").write_text(json.dumps(self.meta, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> AuditTable:
        path = Path(path).with_suffix(".npz")
        z = np.load(path, allow_pickle=False)
        answer = np.array(
            [None if none else a for a, none in zip(z["answer"], z["answer_is_none"])],
            dtype=object,
        )
        V = {name[3:]: z[name].astype(np.int64) for name in z.files if name.startswith("V::")}
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(
            pid=z["pid"].astype(object),
            split=z["split"].astype(object),
            k=z["k"],
            O=z["O"].astype(np.int64),
            answer=answer,
            n_tokens=z["n_tokens"],
            logprob=z["logprob"],
            V=V,
            meta=meta,
        )
