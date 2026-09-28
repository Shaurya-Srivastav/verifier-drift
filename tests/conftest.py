"""Shared fakes for audit pipeline tests.

The feature and generator modules are replaced in ``sys.modules`` so a test
can run ``run_audit`` without PyTorch, MLX, or a model download.
"""

from __future__ import annotations

import sys
import types
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vfates.audit import load_pools
from vfates.gen.base import Generator
from vfates.tasks.registry import build_task
from vfates.types import GenOut

TASK_SPEC = {
    "type": "chain_sum",
    "min_terms": 2,
    "max_terms": 2,
    "min_digits": 1,
    "max_digits": 1,
    "min_decimal_places": 1,
    "max_decimal_places": 1,
}


def _wrong(gold: str) -> str:
    return "1" if Decimal(gold) == 0 else "0"


class FakeGenerator(Generator):
    def __init__(self, answers: dict[str, str], calls: dict[str, Any]) -> None:
        self.model_id = "fake"
        self.answers = answers
        self.calls = calls

    def generate(
        self,
        prompts: list[str],
        n: int,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
        system_prompt: str | None = None,
    ) -> list[list[GenOut]]:
        del max_tokens, temperature, top_p, seed, system_prompt
        self.calls["generate"] += 1
        call = int(self.calls["generate"])
        groups: list[list[GenOut]] = []
        for prompt in prompts:
            gold = self.answers[prompt]
            wrong = _wrong(gold)
            group = []
            for k in range(n):
                value = gold if k % 2 == 0 else wrong
                group.append(
                    GenOut(
                        text=f"c{call} work \\boxed{{{value}}}",
                        n_tokens=4,
                        token_ids=(k + 1, 8, 9, 10),
                        logprob=-0.25,
                        finish_reason="stop",
                    )
                )
            groups.append(group)
        return groups


class _Feat:
    def __init__(self, features: np.ndarray, lengths: np.ndarray) -> None:
        self.features = features
        self.lengths = lengths
        self.exact = None
        self.meta = {"sample_seconds": 0.0}


def base_audit_config() -> dict[str, Any]:
    """A tiny chain-sum audit. ``lock`` is off until a test turns it on."""
    return {
        "task": dict(TASK_SPEC),
        "pools": {"train": 6, "heldout": 4, "seed": 0},
        "generator": {"backend": "hf", "model_id": "fake", "kwargs": {}},
        "sampling": {
            "n": 2,
            "max_tokens": 32,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 0,
            "chunk_prompts": 10,
        },
        "arms": {
            "targets": {"fpr": 0.15},
            "n_boot": 5,
            "seed": 0,
            "arms": [
                {"name": "oracle", "type": "oracle"},
                {"name": "fp_hash", "type": "hash_fp", "salt": "chain_sum/fp_hash"},
                {"name": "fp_tolerance", "type": "tolerance_fp"},
                {
                    "name": "noise_fp",
                    "type": "resampled",
                    "match": "fp_tolerance",
                    "support": "answered",
                    "seed": 0,
                },
            ],
        },
        "features": {
            "kind": "lora_grad",
            "model_id": "fake",
            "sketch_dim": 16,
            "lora_seed": 0,
        },
        "forecast": {"n_boot": 0, "reliability_splits": 2, "seed": 0, "loss_agg": "seq_mean"},
        "lock": False,
    }


def _answers(config: dict[str, Any]) -> dict[str, str]:
    task = build_task(config["task"])
    train, held = load_pools(task, config["pools"])
    return {task.prompt(problem): problem.answer for problem in [*train, *held]}


@pytest.fixture
def audit_config() -> dict[str, Any]:
    return base_audit_config()


@pytest.fixture
def patch_audit_fakes(monkeypatch: pytest.MonkeyPatch):
    """Install a fake generator and a fake LoRA feature extractor. Returns ``bind(config)``."""

    def bind(config: dict[str, Any]) -> dict[str, Any]:
        calls: dict[str, Any] = {
            "generate": 0,
            "make": 0,
            "extract": 0,
            "load_model": 0,
            "save_init": None,
            "completion_ids": None,
            "sketch_dim": None,
        }
        answers = _answers(config)

        def factory(backend: str, model_id: str, **kwargs: object) -> FakeGenerator:
            del backend, model_id, kwargs
            calls["make"] += 1
            return FakeGenerator(answers, calls)

        def load_model(model_id: str, **load: object) -> tuple[object, object]:
            del model_id, load
            calls["load_model"] += 1
            return object(), object()

        def lora_grad_features(
            model: object,
            tok: object,
            prompts: list[str],
            texts: list[str],
            system_prompt: str | None = None,
            **kwargs: Any,
        ) -> _Feat:
            del model, tok, prompts, system_prompt
            calls["extract"] += 1
            calls["save_init"] = kwargs.get("save_init")
            calls["completion_ids"] = kwargs.get("completion_ids")
            calls["sketch_dim"] = kwargs.get("sketch_dim")
            save_init = kwargs.get("save_init")
            if save_init:
                path = Path(str(save_init))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"lora-init-v1")
            n = len(texts)
            feats = np.ones((n, 3), dtype=np.float32)
            ids = kwargs.get("completion_ids")
            if ids is not None:
                feats[:, 0] = [float(row[0]) for row in ids]
            return _Feat(feats, np.full(n, 4, dtype=np.int64))

        hidden = types.ModuleType("vfates.features.hidden")
        hidden.load_model = load_model  # type: ignore[attr-defined]
        lora = types.ModuleType("vfates.features.lora_grad")
        lora.lora_grad_features = lora_grad_features  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vfates.features.hidden", hidden)
        monkeypatch.setitem(sys.modules, "vfates.features.lora_grad", lora)
        monkeypatch.setattr("vfates.gen.factory.make_generator", factory)
        return calls

    return bind
