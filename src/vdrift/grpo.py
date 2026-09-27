"""Real GRPO training entry point — run on the Mac Studio / a rented GPU.

This wires a real Hugging Face policy's rollouts through the SAME verifier /
oracle / audit core used by the mock pilot, so the drift metrics are computed
identically. It imports torch/transformers/trl lazily so this module still
imports on a CPU-only box (the analysis core has no heavy deps).

Intended recipe (fill in on the training box):
  - policy: Qwen/Qwen2.5-1.5B-Instruct (then 3B), bf16, MPS or CUDA
  - algo:   GRPO (trl.GRPOTrainer or a minimal loop), group size 8, ~few hundred steps, 3 seeds
  - reward: RuleVerifier.check(pred, gold)  (the verifier under study)
  - dataset: GSM8K + MATH via vdrift.data.load_math()
  - logging: at each checkpoint, dump per-group (pred, gold, form, v_correct, o_correct)
             so vdrift.audit / vdrift.metrics produce the identical trajectory JSON.
"""
from __future__ import annotations

from typing import List

from .audit import Sample
from .forms import classify_form
from .verifier import Oracle, RuleVerifier


class VerifierReward:
    """GRPO reward = the verifier under study. Log oracle verdicts alongside for
    the drift audit; never use the oracle as the training reward."""

    def __init__(self):
        self.verifier = RuleVerifier()
        self.oracle = Oracle()

    def __call__(self, pred: str, gold: str) -> float:
        return 1.0 if self.verifier.check(pred, gold) else 0.0

    def to_sample(self, pred: str, gold: str, difficulty: int) -> Sample:
        return Sample(pred, gold, classify_form(pred), difficulty,
                      self.verifier.check(pred, gold), self.oracle.check(pred, gold))


def train(*args, **kwargs):  # pragma: no cover - GPU path
    raise NotImplementedError(
        "Run on the Mac Studio: install `.[train]`, implement the GRPO loop with "
        "trl.GRPOTrainer using VerifierReward as the reward and load_math() as data, "
        "then log per-group Samples and feed them to vdrift.pilot's trajectory writer."
    )
