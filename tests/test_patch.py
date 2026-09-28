"""Patch selection: rate, tau, ties, and the audit stage that records them."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vfates.audit import _row_category_sha, apply_arms, load_audit, run_audit, stage_fingerprints
from vfates.forecast.lockbox import load_lock_record
from vfates.gen.base import Generator
from vfates.tasks.base import Task
from vfates.types import AuditTable, Completion, GenOut, Parsed, Problem
from vfates.utils.io import read_json
from vfates.verifiers.base import Verifier
from vfates.verifiers.calibrate import CalibrationData, calibrate_arms
from vfates.verifiers.patch import PatchedVerifier, select_patched_arms
from vfates.verifiers.registry import build_verifier


class SlipBase(Verifier):
    kind = "natural"
    stochastic = False

    def __init__(self, name: str = "natural") -> None:
        self.name = name

    def verdict(
        self,
        problem: Problem,
        parsed: Parsed,
        text: str,
        o: int,
        key: object = None,
    ) -> int:
        del problem, parsed, key
        if "slip" in text:
            return 0
        return int(o)

    def describe(self) -> dict:
        return {"type": "toy_natural", "name": self.name, "kind": self.kind, "stochastic": False}


class ToyTask(Task):
    name = "toy"
    error_categories = ("alpha", "beta", "gamma")
    verifier_types = ("toy_natural",)

    def __init__(self) -> None:
        self._problems = [
            _problem("t0", "train", "slip alpha"),
            _problem("t1", "train", "slip beta"),
            _problem("t2", "train", "slip gamma"),
            _problem("t3", "train", "ok train"),
            _problem("h0", "heldout", "ok held a"),
            _problem("h1", "heldout", "ok held b"),
        ]

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        del seed
        pool = [problem for problem in self._problems if problem.split == split]
        if n != len(pool):
            raise ValueError(f"expected {len(pool)} {split} problems, got {n}")
        return list(pool)

    def prompt(self, problem: Problem) -> str:
        return problem.question

    def parse(self, problem: Problem, text: str) -> Parsed:
        del problem
        return Parsed(answer="1", raw=text)

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        del problem, parsed, text
        return 1

    def categorize(self, problem: Problem, parsed: Parsed, text: str) -> frozenset[str]:
        del problem, parsed
        return frozenset(category for category in self.error_categories if category in text)

    def build_verifier(self, spec: dict) -> Verifier:
        if spec.get("type") != "toy_natural":
            raise ValueError(spec.get("type"))
        return SlipBase(name=str(spec.get("name", "natural")))


def _problem(pid: str, split: str, question: str) -> Problem:
    return Problem(pid=pid, split=split, task="toy", question=question, answer="1", meta={})


def _table_from(
    problems: list[Problem],
    texts: list[str],
    labels: list[int],
    features: np.ndarray,
) -> tuple[dict[str, Problem], list[Completion], list[Parsed], AuditTable, np.ndarray]:
    by_id = {problem.pid: problem for problem in problems}
    completions = [
        Completion(pid=problem.pid, k=index % 2, text=text, n_tokens=4)
        for index, (problem, text) in enumerate(zip(problems, texts, strict=True))
    ]
    # k is the within-prompt index. Recompute it.
    seen: dict[str, int] = {}
    fixed: list[Completion] = []
    for completion in completions:
        k = seen.get(completion.pid, 0)
        seen[completion.pid] = k + 1
        fixed.append(
            Completion(pid=completion.pid, k=k, text=completion.text, n_tokens=completion.n_tokens)
        )
    parsed = [Parsed(answer="1", raw=completion.text) for completion in fixed]
    labels_arr = np.asarray(labels, dtype=np.int64)
    base = SlipBase()
    verdicts = np.array(
        [
            base(by_id[completion.pid], parsed_row, completion.text, int(label))
            for completion, parsed_row, label in zip(fixed, parsed, labels, strict=True)
        ],
        dtype=np.int64,
    )
    table = AuditTable(
        pid=np.array([completion.pid for completion in fixed], dtype=object),
        split=np.array([by_id[completion.pid].split for completion in fixed], dtype=object),
        k=np.array([completion.k for completion in fixed], dtype=np.int64),
        O=labels_arr,
        answer=np.array(["1"] * len(fixed), dtype=object),
        n_tokens=np.full(len(fixed), 4, dtype=np.int64),
        logprob=np.zeros(len(fixed)),
        V={"natural": verdicts},
    )
    return by_id, fixed, parsed, table, features


def test_rate_picks_the_largest_counts_and_tau_can_disagree() -> None:
    task = ToyTask()
    problems = [
        _problem("a", "train", "a"),
        _problem("a", "train", "a"),
        _problem("b", "train", "b"),
        _problem("b", "train", "b"),
        _problem("c", "train", "c"),
        _problem("c", "train", "c"),
        _problem("h0", "heldout", "h0"),
        _problem("h0", "heldout", "h0"),
    ]
    texts = [
        "slip alpha",
        "clean",
        "slip beta",
        "clean",
        "slip beta",
        "clean",
        "clean",
        "clean",
    ]
    labels = [1, 1, 1, 1, 1, 1, 1, 0]
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    by_id, completions, parsed, table, feats = _table_from(problems, texts, labels, features)
    lengths = np.full(len(completions), 4, dtype=np.int64)
    specs = [
        {"name": "patch_rate", "type": "patch_select", "base": "natural", "k": 1, "by": "rate"},
        {"name": "patch_tau", "type": "patch_select", "base": "natural", "k": 1, "by": "tau"},
    ]
    _built, verdicts, block = select_patched_arms(
        specs,
        {"natural": SlipBase()},
        task,
        by_id,
        completions,
        parsed,
        table,
        feats,
        lengths,
        q=0.5,
        divide_by_length=False,
    )
    assert block["patch_rate"]["categories"] == ["beta"]
    assert block["patch_tau"]["categories"] == ["alpha"]
    assert block["same_selection"] is False
    assert block["patch_rate"]["errors_fixed"] == 2
    assert block["patch_tau"]["errors_fixed"] == 1
    assert block["patch_tau"]["tau_pool"] > block["patch_rate"]["tau_pool"]
    assert set(block["patch_rate"]["step0_rates"]) == {"alpha", "beta", "gamma"}
    assert block["patch_rate"]["step0_rates"]["gamma"] == 0.0
    hit = verdicts["patch_tau"][0]
    assert hit == 1
    assert verdicts["patch_tau"][2] == 0


def test_tau_ties_break_by_errors_fixed_then_lexicographic_order() -> None:
    task = ToyTask()
    problems = [
        _problem("t0", "train", "t0"),
        _problem("t1", "train", "t1"),
        _problem("t2", "train", "t2"),
        _problem("h0", "heldout", "h0"),
    ]
    texts = ["slip alpha", "slip beta", "slip gamma", "clean"]
    labels = [1, 1, 1, 1]
    features = np.ones((4, 2), dtype=np.float64)
    by_id, completions, parsed, table, feats = _table_from(problems, texts, labels, features)
    lengths = np.full(4, 2, dtype=np.int64)
    specs = [
        {"name": "patch_tau", "type": "patch_select", "base": "natural", "k": 2, "by": "tau"},
        {"name": "patch_rate", "type": "patch_select", "base": "natural", "k": 2, "by": "rate"},
    ]
    _built, _verdicts, block = select_patched_arms(
        specs,
        {"natural": SlipBase()},
        task,
        by_id,
        completions,
        parsed,
        table,
        feats,
        lengths,
        q=1.0,
    )
    assert block["patch_tau"]["categories"] == ["alpha", "beta"]
    assert block["patch_rate"]["categories"] == ["alpha", "beta"]
    assert block["same_selection"] is True
    assert block["patch_tau"]["errors_fixed"] == 2


def test_too_few_categories_raises_and_calibrate_defers_selection() -> None:
    task = ToyTask()
    problems = [_problem("t0", "train", "t0"), _problem("h0", "heldout", "h0")]
    texts = ["slip alpha", "clean"]
    labels = [1, 1]
    features = np.ones((2, 2))
    by_id, completions, parsed, table, feats = _table_from(problems, texts, labels, features)
    with pytest.raises(ValueError, match="step-0"):
        select_patched_arms(
            [
                {
                    "name": "patch_rate",
                    "type": "patch_select",
                    "base": "natural",
                    "k": 2,
                    "by": "rate",
                }
            ],
            {"natural": SlipBase()},
            task,
            by_id,
            completions,
            parsed,
            table,
            feats,
            np.full(2, 2),
            q=1.0,
        )
    data = CalibrationData.from_grades(
        list(by_id.values()),
        completions,
        parsed,
        labels,
    )
    result = calibrate_arms(
        task,
        data,
        [
            {"name": "natural", "type": "toy_natural"},
            {"name": "patch_rate", "type": "patch_select", "base": "natural", "k": 1, "by": "rate"},
        ],
        n_boot=2,
    )
    assert "patch_rate" not in result.arms
    assert result.report["patch_select"][0]["by"] == "rate"
    with pytest.raises(KeyError, match="base"):
        calibrate_arms(
            task,
            data,
            [
                {
                    "name": "patch_rate",
                    "type": "patch_select",
                    "base": "missing",
                    "k": 1,
                    "by": "rate",
                }
            ],
            n_boot=2,
        )


def test_describe_round_trip() -> None:
    task = ToyTask()
    base = SlipBase(name="natural")
    patched = PatchedVerifier(base, ["alpha"], task, name="patch_tau")
    problem = _problem("t0", "train", "t0")
    parsed = Parsed(answer="1", raw="1")
    assert patched(problem, parsed, "slip alpha", 1) == 1
    assert patched(problem, parsed, "slip beta", 1) == 0
    again = build_verifier(patched.describe(), task)
    assert again.describe() == patched.describe()
    assert again(problem, parsed, "slip alpha", 1) == 1


class _Scripted(Generator):
    def __init__(self, calls: dict[str, int]) -> None:
        self.model_id = "fake"
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
        return [
            [GenOut(text=prompt, n_tokens=4, token_ids=(1, 2)) for _k in range(n)]
            for prompt in prompts
        ]


def test_patch_stage_writes_verdicts_forecasts_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toy = ToyTask()
    monkeypatch.setattr("vfates.audit.build_task", lambda spec: toy)
    calls = {"generate": 0, "extract": 0}

    def factory(backend: str, model_id: str, **kwargs: object) -> Generator:
        del backend, model_id, kwargs
        return _Scripted(calls)

    monkeypatch.setattr("vfates.gen.factory.make_generator", factory)

    class _Feat:
        def __init__(self, n: int) -> None:
            self.features = np.ones((n, 3), dtype=np.float32)
            self.lengths = np.full(n, 4, dtype=np.int64)
            self.meta = {"sample_seconds": 0.0}

    def lora_grad_features(
        model: object,
        tok: object,
        prompts: list[str],
        texts: list[str],
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> _Feat:
        del model, tok, prompts, system_prompt
        calls["extract"] += 1
        save_init = kwargs.get("save_init")
        if save_init:
            Path(str(save_init)).write_bytes(b"lora-init-v1")
        return _Feat(len(texts))

    def load_model(model_id: str, **load: object) -> tuple[object, object]:
        del model_id, load
        return object(), object()

    hidden = types.ModuleType("vfates.features.hidden")
    hidden.load_model = load_model  # type: ignore[attr-defined]
    lora = types.ModuleType("vfates.features.lora_grad")
    lora.lora_grad_features = lora_grad_features  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vfates.features.hidden", hidden)
    monkeypatch.setitem(sys.modules, "vfates.features.lora_grad", lora)

    config = {
        "task": {"type": "toy"},
        "pools": {"train": 4, "heldout": 2, "seed": 0},
        "generator": {"backend": "hf", "model_id": "fake", "kwargs": {}},
        "sampling": {
            "n": 1,
            "max_tokens": 16,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 0,
            "chunk_prompts": 8,
        },
        "arms": {
            "n_boot": 2,
            "seed": 0,
            "arms": [
                {"name": "oracle", "type": "oracle"},
                {"name": "natural", "type": "toy_natural"},
                {
                    "name": "patch_rate",
                    "type": "patch_select",
                    "base": "natural",
                    "k": 2,
                    "by": "rate",
                },
                {
                    "name": "patch_tau",
                    "type": "patch_select",
                    "base": "natural",
                    "k": 2,
                    "by": "tau",
                },
            ],
        },
        "features": {"kind": "lora_grad", "model_id": "fake", "sketch_dim": 8, "lora_seed": 0},
        "forecast": {"n_boot": 0, "reliability_splits": 2, "seed": 0, "loss_agg": "seq_mean"},
        "lock": True,
    }
    out = tmp_path / "audit"
    run = run_audit(config, out)
    assert calls["generate"] == 1
    assert calls["extract"] == 1
    calibration = read_json(out / "calibration.json")
    assert calibration["arms"]["patch_rate"]["type"] == "patched"
    assert calibration["arms"]["patch_rate"]["categories"] == ["alpha", "beta"]
    assert calibration["arms"]["patch_tau"]["categories"] == ["alpha", "beta"]
    rebuilt = build_verifier(calibration["arms"]["patch_rate"], toy)
    verdicts = apply_arms(
        {"patch_rate": rebuilt}, run.problems, run.completions, run.parsed, run.table.O
    )
    assert np.array_equal(verdicts["patch_rate"], run.table.V["patch_rate"])
    forecasts = read_json(out / "forecasts.json")
    block = forecasts["patch_selection"]
    assert block["same_selection"] is True
    for name in ("patch_rate", "patch_tau"):
        entry = block[name]
        assert set(entry) == {
            "base",
            "by",
            "k",
            "categories",
            "tau_pool",
            "errors_fixed",
            "step0_rates",
        }
        assert entry["categories"] == ["alpha", "beta"]
        assert entry["base"] == "natural"
        assert entry["k"] == 2
        assert entry["errors_fixed"] == 2
    assert "transfer" in forecasts["patch_rate"]
    assert "tau_pool_budget" in forecasts["natural"]
    confusion = read_json(out / "confusion.json")
    assert "patch_rate" in confusion and "patch_tau" in confusion
    record = load_lock_record(out / "lock.json")
    assert record["payload"]["patch_selection"]["same_selection"] is True
    assert record["payload"]["patch_selection"]["patch_tau"]["categories"] == ["alpha", "beta"]
    assert "hardware_class" in record
    manifest = read_json(out / "stages.json")
    assert "patches" in manifest["stages"]
    digest = _row_category_sha(toy, run.problems, run.completions, run.parsed)
    wanted = stage_fingerprints(config, row_categories=digest)
    assert manifest["stages"]["patches"] == wanted["patches"]
    assert manifest["stages"]["forecasts"] == wanted["forecasts"]
    patches_bytes = (out / "patches.json").read_bytes()
    run_audit(config, out)
    assert calls["generate"] == 1
    assert calls["extract"] == 1
    assert (out / "patches.json").read_bytes() == patches_bytes
    loaded = load_audit(out)
    assert loaded.arms["patch_tau"].describe()["categories"] == ["alpha", "beta"]
    assert isinstance(loaded.arms["patch_tau"], PatchedVerifier)
