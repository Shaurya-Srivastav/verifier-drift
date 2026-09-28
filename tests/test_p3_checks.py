"""P3 train-split sketch rows, LoRA-seed shift, and the seq_mean forecast."""

from __future__ import annotations

import inspect
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vfates.pilots.p3_estimator import (
    _locked_length_divided,
    _sketch_fidelity,
    _train_row_index,
    compare_lora_seeds,
    lora_seed_shift_summary,
    run,
)
from vfates.types import Completion, Problem
from vfates.utils.io import canonical_json, read_yaml, sha256_text, write_json

ROOT = Path(__file__).resolve().parents[1]


def test_p3_config_matches_training_loss_and_keeps_512_tokens() -> None:
    doc = read_yaml(ROOT / "pilots" / "p3_estimator" / "config.yaml")
    assert doc["audit"]["sampling"]["max_tokens"] == 512
    assert doc["audit"]["forecast"]["loss_agg"] == "seq_mean"


class _Table:
    def __init__(self, split: np.ndarray) -> None:
        self.split = split

    def __len__(self) -> int:
        return int(self.split.shape[0])


def test_train_row_index_never_draws_heldout() -> None:
    split = np.array(["train"] * 6 + ["heldout"] * 4)
    audit = SimpleNamespace(completions=list(range(10)), table=_Table(split))
    for seed in range(20):
        idx = _train_row_index(audit, 5, seed)
        assert len(idx) == 5
        assert set(idx.tolist()) <= set(range(6))
    everything = _train_row_index(audit, 100, 0)
    assert set(everything.tolist()) == set(range(6))
    empty = SimpleNamespace(completions=[0, 1], table=_Table(np.array(["heldout", "heldout"])))
    with pytest.raises(ValueError, match="train rows"):
        _train_row_index(empty, 1, 0)


def test_lora_seed_shift_summary() -> None:
    same = lora_seed_shift_summary({"a": 0.1, "b": 0.2, "c": 0.4}, {"a": 0.1, "b": 0.2, "c": 0.4})
    assert same["spearman"] == pytest.approx(1.0)
    assert same["max_abs_tau_pool"] == 0.0
    flipped = lora_seed_shift_summary(
        {"a": 0.1, "b": 0.2, "c": 0.4}, {"a": 0.4, "b": 0.2, "c": 0.1}
    )
    assert flipped["spearman"] == pytest.approx(-1.0)
    assert flipped["max_abs_tau_pool"] == pytest.approx(0.3)
    short = lora_seed_shift_summary({"a": 0.1}, {"a": 0.2})
    assert math.isnan(short["spearman"])
    assert math.isnan(short["max_abs_tau_pool"])


def test_locked_length_divided() -> None:
    assert _locked_length_divided({"a": {"length_divided": True}}) is True
    assert _locked_length_divided({"a": {"length_divided": 0}}) is False
    with pytest.raises(ValueError, match="length_divided"):
        _locked_length_divided({"a": {}})


def test_compare_lora_seeds_uses_seed_1_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def extract_features(spec, task, problems, completions, out_dir, **kwargs):
        del task, problems, completions
        seen["n"] = seen.get("n", 0) + 1
        seen["spec"] = spec
        seen["dir"] = out_dir
        seen["fingerprint"] = kwargs["fingerprint"]
        return np.zeros((4, 2), dtype=np.float32), np.ones(4), {}

    def forecast_arms(table, features, lengths, arms, **kwargs):
        del table, features, lengths
        seen["divide"] = kwargs["divide_by_length"]
        seen["n_boot"] = kwargs["n_boot"]
        seen["verifiers"] = kwargs.get("verifiers")
        return {name: {"transfer": {"tau_pool": 0.2 if name == "a" else 0.5}} for name in arms}

    monkeypatch.setattr("vfates.audit.extract_features", extract_features)
    monkeypatch.setattr("vfates.audit.forecast_arms", forecast_arms)
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    write_json(audit_dir / "stages.json", {"stages": {"completions": "cafebabe"}})
    audit = SimpleNamespace(
        out_dir=audit_dir, task=None, problems={}, completions=[], table=None, arms={"a": 1}
    )
    forecasts = {
        "a": {"transfer": {"tau_pool": 0.2}, "length_divided": True},
        "b": {"transfer": {"tau_pool": 0.9}, "length_divided": True},
    }
    out = tmp_path / "p3"
    feat = {"kind": "lora_grad", "model_id": "m", "lora_seed": 0}
    summary = compare_lora_seeds(audit, feat, out, forecasts, divide_by_length=True)
    spec = {"kind": "lora_grad", "model_id": "m", "lora_seed": 1}
    expected = sha256_text(canonical_json({"completions": "cafebabe", "features": spec}))
    assert seen["spec"]["lora_seed"] == 1
    assert seen["fingerprint"] == expected
    assert Path(seen["dir"]) == out / "lora_seed_1"
    assert seen["divide"] is True
    assert seen["n_boot"] == 0
    assert seen["verifiers"] is audit.arms
    assert summary["max_abs_tau_pool"] == pytest.approx(0.4)
    assert (out / "lora_seed_sensitivity.json").is_file()
    again = compare_lora_seeds(audit, feat, out, forecasts, divide_by_length=True)
    assert seen["n"] == 1
    assert again["spearman"] == summary["spearman"]


def test_sketch_fidelity_uses_train_rows_and_an_existing_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def load_model(model_id: str, **kwargs: object) -> tuple[object, object]:
        del model_id, kwargs
        return object(), object()

    def lora_grad_features(model, tok, prompts, texts, system_prompt=None, **kwargs):
        del model, tok, prompts, system_prompt
        seen["texts"] = list(texts)
        seen["ids"] = kwargs.get("completion_ids")
        seen["save_init"] = kwargs.get("save_init")
        n = len(texts)
        exact = np.arange(n * 3, dtype=np.float64).reshape(n, 3) + 1.0
        return SimpleNamespace(features=exact * 2.0, exact=exact)

    hidden = types.ModuleType("vfates.features.hidden")
    hidden.load_model = load_model  # type: ignore[attr-defined]
    lora = types.ModuleType("vfates.features.lora_grad")
    lora.lora_grad_features = lora_grad_features  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vfates.features.hidden", hidden)
    monkeypatch.setitem(sys.modules, "vfates.features.lora_grad", lora)

    class Task:
        def prompt(self, problem: Problem) -> str:
            return problem.question

        def system_prompt(self) -> str:
            return "s"

    problems = {
        "tr0": Problem(pid="tr0", split="train", task="t", question="train q", answer="1"),
        "tr1": Problem(pid="tr1", split="train", task="t", question="train r", answer="3"),
        "he": Problem(pid="he", split="heldout", task="t", question="held q", answer="2"),
    }
    completions = [
        Completion(pid="tr0", k=0, text="train-a", n_tokens=1, token_ids=(1,)),
        Completion(pid="tr0", k=1, text="train-b", n_tokens=1, token_ids=(2,)),
        Completion(pid="tr1", k=0, text="train-c", n_tokens=1, token_ids=(4,)),
        Completion(pid="he", k=0, text="held-a", n_tokens=1, token_ids=(3,)),
    ]
    audit = SimpleNamespace(
        completions=completions,
        table=_Table(np.array(["train", "train", "train", "heldout"])),
        task=Task(),
        problems=problems,
    )
    missing = tmp_path / "missing.safetensors"
    report = _sketch_fidelity(
        audit, {"model_id": "m", "kind": "lora_grad", "sketch_dim": 4}, 5, 0, save_init=missing
    )
    assert seen["texts"] == ["train-a", "train-b", "train-c"]
    assert seen["ids"] == [[1], [2], [4]]
    assert "save_init" not in seen or seen["save_init"] is None
    assert report["n"] == 3
    assert report["offdiag_corr"] == pytest.approx(1.0)
    init = tmp_path / "lora_init.safetensors"
    init.write_bytes(b"x")
    _sketch_fidelity(
        audit, {"model_id": "m", "kind": "lora_grad", "sketch_dim": 4}, 5, 0, save_init=init
    )
    assert str(seen["save_init"]) == str(init)


def test_hidden_ablation_does_not_divide_by_length() -> None:
    source = inspect.getsource(run)
    assert "divide_by_length=False" in source
