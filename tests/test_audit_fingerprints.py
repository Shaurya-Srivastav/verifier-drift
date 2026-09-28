"""Stage fingerprints, lock refusal, token ids, length division, and pools."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from vfates.audit import (
    _length_division,
    _make_completion,
    _probe_kinds,
    _row_category_sha,
    behaviour_alignment,
    behaviour_indicators,
    extract_features,
    is_dev_audit,
    iter_forecast_arms,
    load_audit,
    load_completions,
    load_pools,
    read_lock_record,
    require_confirmatory_pools,
    run_audit,
    stage_fingerprints,
)
from vfates.forecast import transfer as transfer_mod
from vfates.tasks.chain_sum import ChainSumTask
from vfates.tasks.registry import build_task
from vfates.types import Completion, GenOut, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import atomic_npz, read_json, read_jsonl, sha256_file, write_json
from vfates.verifiers.arms import (
    BehaviorFN,
    Composite,
    HashFP,
    OracleVerifier,
    PromptSubsetFN,
    ResampledNoise,
    StringMatchVerifier,
)

TASK_SPEC = {
    "type": "chain_sum",
    "min_terms": 2,
    "max_terms": 2,
    "min_digits": 1,
    "max_digits": 1,
    "min_decimal_places": 1,
    "max_decimal_places": 1,
}


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _one_problem() -> tuple[object, Problem, Completion]:
    task = build_task(TASK_SPEC)
    problem = task.problems("train", 1, seed=0)[0]
    completion = Completion(
        pid=problem.pid, k=0, text="work \\boxed{1}", n_tokens=4, token_ids=(3, 4, 5, 6)
    )
    return task, problem, completion


def test_fingerprints_follow_the_stage_graph(audit_config: dict) -> None:
    original = copy.deepcopy(audit_config)
    base = stage_fingerprints(original)
    changed = copy.deepcopy(original)
    changed["sampling"]["max_tokens"] = 99
    changed["sampling"]["seed"] = 4
    sampled = stage_fingerprints(changed)
    assert base["problems"] == sampled["problems"]
    assert base["completions"] != sampled["completions"]
    assert base["features"] != sampled["features"]
    assert base["forecasts"] != sampled["forecasts"]

    pooled = copy.deepcopy(original)
    pooled["pools"]["seed"] = 9
    assert stage_fingerprints(pooled)["problems"] != base["problems"]

    omitted = copy.deepcopy(original)
    omitted["forecast"] = {"n_boot": 0}
    explicit = copy.deepcopy(original)
    explicit["forecast"] = {"n_boot": 0, "loss_agg": "seq_mean"}
    assert stage_fingerprints(omitted)["forecasts"] != stage_fingerprints(explicit)["forecasts"]


def test_arms_file_bytes_change_the_calibration_fingerprint(
    tmp_path: Path, audit_config: dict
) -> None:
    path = tmp_path / "arms.yaml"
    body = "targets: {fpr: 0.1}\narms:\n- {name: oracle, type: oracle}\n"
    path.write_text(body)
    audit_config["arms"] = {"file": str(path), "n_boot": 5, "seed": 0}
    before = stage_fingerprints(audit_config)
    path.write_text("# comment only\n" + body)
    after = stage_fingerprints(audit_config)
    assert before["calibration"] != after["calibration"]
    assert before["problems"] == after["problems"]
    assert before["completions"] == after["completions"]
    assert before["features"] == after["features"]


def test_explicit_divide_by_length_must_match(audit_config: dict) -> None:
    audit_config["forecast"]["divide_by_length"] = False
    with pytest.raises(ValueError, match="divide_by_length"):
        stage_fingerprints(audit_config)
    audit_config["forecast"]["divide_by_length"] = True
    stage_fingerprints(audit_config)


def test_length_division_delegates_to_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _length_division("sum", "seq_mean") is True
    assert _length_division("token_mean", "seq_mean") is False
    assert _length_division("sum", "token_mean") is False
    assert _length_division("sum", "dr_constant") is False
    with pytest.raises(ValueError):
        _length_division("token_mean", "token_mean")
    with pytest.raises(ValueError):
        _length_division("token_mean", "dr_constant")

    def boom(normalize: str, loss_agg: str) -> bool:
        raise ValueError(f"delegated {normalize} {loss_agg}")

    monkeypatch.setattr(transfer_mod, "length_division", boom)
    config = {
        "task": {"type": "chain_sum"},
        "pools": {"train": 1, "heldout": 1, "seed": 0},
        "generator": {},
        "sampling": {"n": 1},
        "arms": {"arms": [{"name": "oracle", "type": "oracle"}]},
        "features": {"kind": "lora_grad", "normalize": "sum"},
        "forecast": {"loss_agg": "seq_mean"},
    }
    with pytest.raises(ValueError, match="delegated sum seq_mean"):
        stage_fingerprints(config)


def test_behaviour_alignment_is_the_signed_first_order_increment() -> None:
    pid = np.array(["a", "a", "b", "b"])
    O = np.array([1, 0, 1, 1], dtype=np.int8)
    b = np.array([True, False, False, True])
    fp = behaviour_alignment(HashFP("salt", 0.2), b, O, pid)
    assert fp == {
        "behaviour_frequency": 0.5,
        "conditional_advantage": 0.25,
        "behaviour_alignment": 0.125,
    }
    fn = behaviour_alignment(BehaviorFN("because"), b, O, pid)
    assert fn["behaviour_alignment"] == -0.125
    assert fn["conditional_advantage"] == 0.25

    by_prompt = np.array([True, True, False, False])
    subset = behaviour_alignment(PromptSubsetFN("salt", 0.5), by_prompt, O, pid)
    assert subset["conditional_advantage"] == 0.0
    assert subset["behaviour_alignment"] == 0.0

    absent = behaviour_alignment(HashFP("salt", 0.2), np.zeros(4, dtype=bool), O, pid)
    assert absent == {
        "behaviour_frequency": 0.0,
        "conditional_advantage": None,
        "behaviour_alignment": 0.0,
    }

    noise = behaviour_alignment(ResampledNoise(0.2, 0.0, 0), None, O, pid)
    assert noise == {
        "behaviour_frequency": None,
        "conditional_advantage": None,
        "behaviour_alignment": 0.0,
    }
    for other in (OracleVerifier(), StringMatchVerifier(), None):
        assert set(behaviour_alignment(other, b, O, pid).values()) == {None}
    assert set(behaviour_alignment(HashFP("salt", 0.2), None, O, pid).values()) == {None}
    with pytest.raises(ValueError, match="rows"):
        behaviour_alignment(HashFP("salt", 0.2), b[:3], O, pid)


def test_behaviour_indicators_evaluate_each_rule_predicate() -> None:
    task, problem, _ = _one_problem()
    texts = ["because \\boxed{1}", "so \\boxed{2}", "no box"]
    completions = [
        Completion(pid=problem.pid, k=k, text=text, n_tokens=3) for k, text in enumerate(texts)
    ]
    parsed = [task.parse(problem, c.text) for c in completions]
    fp = HashFP("salt", 0.999)
    fn = BehaviorFN("because")
    arms = {"oracle": OracleVerifier(), "fp": fp, "fn": fn, "noise": ResampledNoise(0.2, 0, 0)}
    bits = behaviour_indicators(arms, {problem.pid: problem}, completions, parsed)
    assert set(bits) == {"fp", "fn"}
    assert bits["fp"].tolist() == [
        fp.accepts(problem, p, c.text) for c, p in zip(completions, parsed, strict=True)
    ]
    assert bits["fp"].tolist() == [True, True, False]
    assert bits["fn"].tolist() == [True, False, False]


def test_probe_kinds_follow_the_arm_type() -> None:
    assert _probe_kinds(OracleVerifier()) == []
    assert _probe_kinds(StringMatchVerifier()) == []
    assert _probe_kinds(HashFP("salt", 0.2)) == ["fp"]
    assert _probe_kinds(BehaviorFN("because")) == ["fn"]
    assert _probe_kinds(ResampledNoise(0.2, 0.0, 0)) == ["fp"]
    assert _probe_kinds(ResampledNoise(0.0, 0.3, 0)) == ["fn"]
    assert _probe_kinds(ResampledNoise(0.2, 0.3, 0)) == ["fp", "fn"]
    assert _probe_kinds(ResampledNoise(0.0, 0.0, 0)) == []
    both = Composite([BehaviorFN("because")], [HashFP("salt", 0.2)], "both")
    assert _probe_kinds(both) == ["fp", "fn"]
    only_fp = Composite([], [HashFP("salt", 0.2)], "fp-only")
    assert _probe_kinds(only_fp) == ["fp"]


def test_completion_copies_token_ids() -> None:
    task = build_task(TASK_SPEC)
    problem = task.problems("train", 1, seed=1)[0]
    out = GenOut(text="t", n_tokens=2, token_ids=(7, 8), logprob=-1.0, finish_reason="stop")
    completion = _make_completion(problem, 3, out)
    assert completion.token_ids == (7, 8)
    row = completion.to_json()
    assert row["token_ids"] == [7, 8]
    assert Completion.from_json(row).token_ids == (7, 8)
    bare = {key: value for key, value in row.items() if key != "token_ids"}
    assert Completion.from_json(bare).token_ids is None
    missing = GenOut(text="t", n_tokens=1, logprob=None, finish_reason="length")
    assert _make_completion(problem, 0, missing).token_ids is None


def test_load_pools_rejects_repeated_questions_and_keeps_ids() -> None:
    class Repeated:
        def problems(self, split: str, n: int, seed: int) -> list[Problem]:
            del seed
            return [
                Problem(pid=f"{split}/{i}", split=split, task="t", question="same", answer="1")
                for i in range(n)
            ]

    with pytest.raises(ValueError, match="duplicate question"):
        load_pools(Repeated(), {"train": 2, "heldout": 1, "seed": 0})

    class Across:
        def problems(self, split: str, n: int, seed: int) -> list[Problem]:
            del n, seed
            return [Problem(pid=f"{split}/0", split=split, task="t", question="shared", answer="1")]

    with pytest.raises(ValueError, match="duplicate question"):
        load_pools(Across(), {"train": 1, "heldout": 1, "seed": 0})

    task = build_task(TASK_SPEC)
    pools = {"train": 4, "heldout": 3, "seed": 2}
    train, held = load_pools(task, pools)
    assert [problem.pid for problem in train] == [
        problem.pid for problem in task.problems("train", 4, 2)
    ]
    assert [problem.pid for problem in held] == [
        problem.pid for problem in task.problems("heldout", 3, 2)
    ]


def test_feature_parts_carry_fingerprint_and_row_count(
    tmp_path: Path, patch_audit_fakes, audit_config: dict
) -> None:
    calls = patch_audit_fakes(audit_config)
    task, problem, completion = _one_problem()
    spec = {"kind": "lora_grad", "model_id": "fake", "sketch_dim": 8}
    fingerprint = "ab" * 32
    feats, lengths, meta = extract_features(
        spec,
        task,
        {problem.pid: problem},
        [completion],
        tmp_path,
        fingerprint=fingerprint,
        save_init=tmp_path / "lora_init.safetensors",
    )
    assert calls["extract"] == 1
    assert calls["completion_ids"] == [[3, 4, 5, 6]]
    assert str(calls["save_init"]).endswith("lora_init.safetensors")
    assert (tmp_path / "lora_init.safetensors").read_bytes() == b"lora-init-v1"
    assert meta["token_source"] == "token_ids"
    assert feats.shape == (1, 3)
    assert int(lengths[0]) == 4

    extract_features(
        spec,
        task,
        {problem.pid: problem},
        [completion],
        tmp_path,
        fingerprint=fingerprint,
        save_init=tmp_path / "lora_init.safetensors",
    )
    assert calls["extract"] == 1

    part = tmp_path / "feature_parts" / "part_00000.npz"
    with np.load(part, allow_pickle=False) as saved:
        payload = {key: np.array(saved[key]) for key in saved.files}
    payload["fingerprint"] = np.asarray("cd" * 32)
    atomic_npz(part, **payload)
    extract_features(
        spec, task, {problem.pid: problem}, [completion], tmp_path, fingerprint=fingerprint
    )
    assert calls["extract"] == 2

    with np.load(part, allow_pickle=False) as saved:
        payload = {key: np.array(saved[key]) for key in saved.files}
    payload["n_rows"] = np.int64(0)
    atomic_npz(part, **payload)
    extract_features(
        spec, task, {problem.pid: problem}, [completion], tmp_path, fingerprint=fingerprint
    )
    assert calls["extract"] == 3

    blank = Completion(pid=problem.pid, k=0, text="work", n_tokens=4, token_ids=None)
    _, _, text_meta = extract_features(
        spec,
        task,
        {problem.pid: problem},
        [blank],
        tmp_path / "text",
        fingerprint="ef" * 32,
    )
    assert text_meta["token_source"] == "text"
    assert calls["completion_ids"] is None


def test_extractor_without_completion_ids_fails_closed(
    tmp_path: Path, patch_audit_fakes, audit_config: dict
) -> None:
    patch_audit_fakes(audit_config)

    def bare(
        model: object,
        tok: object,
        prompts: list[str],
        texts: list[str],
        system_prompt: str | None = None,
        sketch_dim: int = 8,
    ) -> None:
        del model, tok, prompts, texts, system_prompt, sketch_dim
        raise AssertionError("extractor should not run")

    import sys

    sys.modules["vfates.features.lora_grad"].lora_grad_features = bare  # type: ignore[attr-defined]
    task, problem, completion = _one_problem()
    with pytest.raises(TypeError, match="completion_ids"):
        extract_features(
            {"kind": "lora_grad", "model_id": "fake", "sketch_dim": 8},
            task,
            {problem.pid: problem},
            [completion],
            tmp_path,
            fingerprint="ab" * 32,
        )


def test_locked_run_records_token_ids_probe_and_artifact_hashes(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    audit_config["lock"] = True
    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    rows = read_jsonl(out / "completions.jsonl")
    assert rows[0]["token_ids"] == [1, 8, 9, 10]
    meta = read_json(out / "features.meta.json")
    assert meta["token_source"] == "token_ids"
    assert calls["completion_ids"] is not None
    assert str(calls["save_init"]).endswith("lora_init.safetensors")
    forecasts = read_json(out / "forecasts.json")
    assert forecasts["oracle"]["probe_heldout_auc"] is None
    assert isinstance(forecasts["fp_hash"]["probe_heldout_auc"], float)
    for entry in forecasts.values():
        assert entry["length_divided"] is True
    assert forecasts["oracle"]["behaviour_alignment"] is None
    assert forecasts["noise_fp"]["behaviour_alignment"] == 0.0
    for name in ("fp_hash", "fp_tolerance"):
        entry = forecasts[name]
        assert 0.0 <= entry["behaviour_frequency"] <= 1.0
        if entry["conditional_advantage"] is not None:
            assert entry["behaviour_alignment"] == pytest.approx(
                entry["behaviour_frequency"] * entry["conditional_advantage"]
            )
    record = read_lock_record(out / "lock.json")
    artifacts = record["artifacts"]
    expected = {
        "problems": "problems.jsonl",
        "completions": "completions.jsonl",
        "calibration": "calibration.json",
        "features": "features.npz",
        "lora_init": "lora_init.safetensors",
    }
    for name, filename in expected.items():
        assert artifacts[name] == sha256_file(out / filename)
    assert "arms_file" not in artifacts
    manifest = read_json(out / "stages.json")
    assert artifacts["grades"] == manifest["grades"] == load_audit(out).grades_sha256
    assert set(manifest["stages"]) == {
        "problems",
        "completions",
        "calibration",
        "table",
        "features",
        "forecasts",
    }
    for _name, entry in iter_forecast_arms(forecasts):
        budgets = entry["tau_pool_budget"]
        assert set(budgets) == {"32", "64", "128", "192"}
        for budget in budgets.values():
            assert set(budget) == {"mean", "sd", "p05", "p95", "oracle_labels_needed", "budget"}
            assert "draws" not in budget
            if int(budget["budget"]) >= int(budget["oracle_labels_needed"]):
                assert budget["mean"] == pytest.approx(entry["transfer"]["tau_pool"])
                assert budget["sd"] == 0.0
                assert budget["p05"] == budget["mean"]
                assert budget["p95"] == budget["mean"]
    assert isinstance(record["hardware_class"], str)


def test_changed_grades_refit_calibration_or_refuse_under_a_lock(
    tmp_path: Path, audit_config: dict, patch_audit_fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ChainSumTask.oracle

    def stricter(self: ChainSumTask, problem: Problem, parsed: object, text: str) -> int:
        return 0 if problem.pid.endswith("0") else original(self, problem, parsed, text)

    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    first = run_audit(audit_config, out)
    extracted = calls["extract"]
    calibration = (out / "calibration.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(ChainSumTask, "oracle", stricter)
        second = run_audit(audit_config, out)
        assert second.grades_sha256 != first.grades_sha256
        assert read_json(out / "stages.json")["grades"] == second.grades_sha256
        assert (out / "calibration.json").read_bytes() != calibration
        assert int(second.table.O.sum()) < int(first.table.O.sum())
        assert calls["extract"] == extracted
    assert run_audit(audit_config, out).grades_sha256 == first.grades_sha256

    audit_config["lock"] = True
    run_audit(audit_config, out)
    before = _snapshot(out)
    monkeypatch.setattr(ChainSumTask, "oracle", stricter)
    with pytest.raises(ValueError, match="grades the cached completions differently"):
        run_audit(audit_config, out)
    assert _snapshot(out) == before


def test_same_config_resume_does_not_resample_or_rewrite_the_lock(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    audit_config["lock"] = True
    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    lock_bytes = (out / "lock.json").read_bytes()
    stage_bytes = (out / "stages.json").read_bytes()
    generated = calls["generate"]
    extracted = calls["extract"]
    run_audit(audit_config, out)
    assert calls["generate"] == generated
    assert calls["extract"] == extracted
    assert (out / "lock.json").read_bytes() == lock_bytes
    assert (out / "stages.json").read_bytes() == stage_bytes


def test_edited_forecasts_fail_the_lock_without_deleting_it(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    audit_config["lock"] = True
    patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    lock_bytes = (out / "lock.json").read_bytes()
    forecasts = read_json(out / "forecasts.json")
    forecasts["oracle"]["transfer"]["tau_pool"] = 0.123456789
    write_json(out / "forecasts.json", forecasts)
    with pytest.raises(ValueError, match="payload"):
        run_audit(audit_config, out)
    assert (out / "lock.json").read_bytes() == lock_bytes
    assert read_json(out / "forecasts.json")["oracle"]["transfer"]["tau_pool"] == 0.123456789


def test_unfingerprinted_artifacts_are_refused_unless_adopted(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    (out / "stages.json").unlink()
    generated = calls["generate"]
    with pytest.raises(ValueError, match="adopt_unfingerprinted"):
        run_audit(audit_config, out)
    assert calls["generate"] == generated
    assert not (out / "stages.json").exists()
    audit_config["adopt_unfingerprinted"] = True
    run_audit(audit_config, out)
    assert calls["generate"] == generated
    assert calls["extract"] == 1
    stages = read_json(out / "stages.json")["stages"]
    assert "problems" in stages
    assert "forecasts" in stages


def test_lock_without_a_manifest_is_refused(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    out.mkdir()
    (out / "lock.json").write_text("{}\n")
    with pytest.raises(ValueError, match=r"stages\.json"):
        run_audit(audit_config, out)
    assert (out / "lock.json").read_text() == "{}\n"
    assert not (out / "stages.json").exists()


def test_changing_max_tokens_and_seed_recomputes_or_refuses(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    sentinel = out / "feature_parts" / "sentinel.txt"
    sentinel.write_text("keep")
    old_text = (out / "completions.jsonl").read_text()
    generated = calls["generate"]
    extracted = calls["extract"]
    audit_config["sampling"]["max_tokens"] = 8
    audit_config["sampling"]["seed"] = 3
    run_audit(audit_config, out)
    assert calls["generate"] > generated
    assert calls["extract"] > extracted
    assert not sentinel.exists()
    assert (out / "completions.jsonl").read_text() != old_text

    locked = tmp_path / "locked"
    locked_cfg = copy.deepcopy(audit_config)
    locked_cfg["sampling"]["max_tokens"] = 32
    locked_cfg["sampling"]["seed"] = 0
    locked_cfg["lock"] = True
    run_audit(locked_cfg, locked)
    before = _snapshot(locked)
    generated = calls["generate"]
    locked_cfg["sampling"]["max_tokens"] = 8
    locked_cfg["sampling"]["seed"] = 3
    with pytest.raises(ValueError, match="completions"):
        run_audit(locked_cfg, locked)
    assert _snapshot(locked) == before
    assert calls["generate"] == generated


def test_changing_fpr_target_recomputes_or_refuses(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    sentinel = out / "feature_parts" / "sentinel.txt"
    sentinel.write_text("keep")
    features = sha256_file(out / "features.npz")
    calibration = sha256_file(out / "calibration.json")
    generated = calls["generate"]
    extracted = calls["extract"]
    audit_config["arms"]["targets"] = {"fpr": 0.4}
    run_audit(audit_config, out)
    assert calls["generate"] == generated
    assert calls["extract"] == extracted
    assert sentinel.read_text() == "keep"
    assert sha256_file(out / "features.npz") == features
    assert sha256_file(out / "calibration.json") != calibration

    locked = tmp_path / "locked"
    locked_cfg = copy.deepcopy(audit_config)
    locked_cfg["arms"]["targets"] = {"fpr": 0.15}
    locked_cfg["lock"] = True
    run_audit(locked_cfg, locked)
    before = _snapshot(locked)
    locked_cfg["arms"]["targets"] = {"fpr": 0.4}
    with pytest.raises(ValueError, match="calibration"):
        run_audit(locked_cfg, locked)
    assert _snapshot(locked) == before


def test_changing_sketch_dim_recomputes_or_refuses(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    sentinel = out / "feature_parts" / "sentinel.txt"
    sentinel.write_text("keep")
    completions = (out / "completions.jsonl").read_text()
    generated = calls["generate"]
    extracted = calls["extract"]
    audit_config["features"]["sketch_dim"] = 32
    run_audit(audit_config, out)
    assert calls["generate"] == generated
    assert calls["extract"] > extracted
    assert calls["sketch_dim"] == 32
    assert not sentinel.exists()
    assert (out / "completions.jsonl").read_text() == completions

    locked = tmp_path / "locked"
    locked_cfg = copy.deepcopy(audit_config)
    locked_cfg["features"]["sketch_dim"] = 16
    locked_cfg["lock"] = True
    run_audit(locked_cfg, locked)
    before = _snapshot(locked)
    locked_cfg["features"]["sketch_dim"] = 32
    with pytest.raises(ValueError, match="features"):
        run_audit(locked_cfg, locked)
    assert _snapshot(locked) == before


def test_smaller_sampling_n_is_a_new_draw(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    calls = patch_audit_fakes(audit_config)
    audit_config["sampling"]["n"] = 4
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    old = read_jsonl(out / "completions.jsonl")
    assert len(old) == 40
    generated = calls["generate"]
    audit_config["sampling"]["n"] = 2
    run_audit(audit_config, out)
    new = read_jsonl(out / "completions.jsonl")
    assert calls["generate"] > generated
    assert len(new) == 20
    assert {row["k"] for row in new} == {0, 1}
    old_prefix = {(row["pid"], row["k"]): row["text"] for row in old if row["k"] < 2}
    new_map = {(row["pid"], row["k"]): row["text"] for row in new}
    assert new_map != old_prefix
    with pytest.raises(ValueError, match="k=0"):
        task = build_task(audit_config["task"])
        train, held = load_pools(task, audit_config["pools"])
        load_completions(out / "completions.jsonl", [*train, *held], 4)

    locked = tmp_path / "locked"
    locked_cfg = copy.deepcopy(audit_config)
    locked_cfg["sampling"]["n"] = 4
    locked_cfg["lock"] = True
    run_audit(locked_cfg, locked)
    before = _snapshot(locked)
    locked_cfg["sampling"]["n"] = 2
    with pytest.raises(ValueError, match="completions"):
        run_audit(locked_cfg, locked)
    assert _snapshot(locked) == before
    assert len(read_jsonl(locked / "completions.jsonl")) == 40


def test_loss_agg_sets_length_division_without_reextracting(
    tmp_path: Path, audit_config: dict, patch_audit_fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []
    real = transfer_mod.transfer_forecast

    def wrapped(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("train_lengths"))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(transfer_mod, "transfer_forecast", wrapped)
    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    run_audit(audit_config, out)
    assert seen
    assert all(lengths is not None for lengths in seen)
    extracted = calls["extract"]
    audit_config["forecast"] = {
        "n_boot": 0,
        "reliability_splits": 2,
        "seed": 0,
        "loss_agg": "token_mean",
    }
    run_audit(audit_config, out)
    forecasts = read_json(out / "forecasts.json")
    assert all(entry["length_divided"] is False for entry in forecasts.values())
    assert calls["extract"] == extracted
    assert seen[-1] is None


def test_failed_completion_load_is_not_marked_ready(
    tmp_path: Path, audit_config: dict, patch_audit_fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = patch_audit_fakes(audit_config)
    out = tmp_path / "audit"
    real = load_completions
    state = {"fail": True}

    def flaky(path: Path, problems: list, n: int) -> list:
        if state["fail"]:
            state["fail"] = False
            raise ValueError("transient load failure")
        return real(path, problems, n)

    monkeypatch.setattr("vfates.audit.load_completions", flaky)
    with pytest.raises(ValueError, match="transient load failure"):
        run_audit(audit_config, out)
    assert "completions" not in read_json(out / "stages.json")["stages"]
    generated = calls["generate"]
    assert generated > 0
    run_audit(audit_config, out)
    assert calls["generate"] > generated
    assert "completions" in read_json(out / "stages.json")["stages"]


def test_forwarded_kwargs_are_in_the_stage_fingerprints(audit_config: dict) -> None:
    base = stage_fingerprints(audit_config)
    templated = copy.deepcopy(audit_config)
    templated["generator"]["chat_template_kwargs"] = {"enable_thinking": False}
    changed = stage_fingerprints(templated)
    assert changed["completions"] != base["completions"]
    assert changed["problems"] == base["problems"]
    assert changed["calibration"] != base["calibration"]

    featured = copy.deepcopy(audit_config)
    featured["features"]["chat_template_kwargs"] = {"enable_thinking": False}
    featured["features"]["gradient_checkpointing"] = True
    featured["features"]["target_modules"] = ["q_proj"]
    featured["features"]["logprob_chunk"] = 128
    featured["features"]["load"] = {"dtype": "bfloat16", "device": "cuda", "device_map": "auto"}
    hashed = stage_fingerprints(featured)
    assert hashed["features"] != base["features"]
    assert hashed["forecasts"] != base["forecasts"]
    assert hashed["completions"] == base["completions"]
    assert hashed["problems"] == base["problems"]


def test_included_arms_file_changes_the_calibration_fingerprint(
    tmp_path: Path, audit_config: dict
) -> None:
    kit = tmp_path / "kit.yaml"
    kit.write_text("arms:\n- {name: oracle, type: oracle}\n", encoding="utf-8")
    domain = tmp_path / "domain.yaml"
    domain.write_text("include: kit.yaml\narms: []\n", encoding="utf-8")
    audit_config["arms"] = {"file": str(domain), "n_boot": 5, "seed": 0}
    before = stage_fingerprints(audit_config)
    kit.write_text("# comment only\narms:\n- {name: oracle, type: oracle}\n", encoding="utf-8")
    after = stage_fingerprints(audit_config)
    assert before["calibration"] != after["calibration"]
    assert before["features"] == after["features"]


def test_extractor_forwards_only_kwargs_the_callable_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    seen: dict[str, object] = {}

    def older(
        model: object,
        tok: object,
        prompts: list[str],
        texts: list[str],
        system_prompt: str | None = None,
        completion_ids: object = None,
        save_init: object = None,
        sketch_dim: int = 8,
    ) -> object:
        del model, tok, prompts, system_prompt, completion_ids, save_init
        seen["sketch_dim"] = sketch_dim
        n = len(texts)

        class _Feat:
            def __init__(self) -> None:
                self.features = np.ones((n, 2), dtype=np.float32)
                self.lengths = np.full(n, 4, dtype=np.int64)
                self.meta = {"sample_seconds": 0.0}

        return _Feat()

    def load_model(model_id: str, **load: object) -> tuple[object, object]:
        del model_id
        seen["load"] = dict(load)
        return object(), object()

    hidden = types.ModuleType("vfates.features.hidden")
    hidden.load_model = load_model  # type: ignore[attr-defined]
    lora = types.ModuleType("vfates.features.lora_grad")
    lora.lora_grad_features = older  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vfates.features.hidden", hidden)
    monkeypatch.setitem(sys.modules, "vfates.features.lora_grad", lora)
    task, problem, completion = _one_problem()
    extract_features(
        {
            "kind": "lora_grad",
            "model_id": "fake",
            "sketch_dim": 8,
            "chat_template_kwargs": {"enable_thinking": False},
            "gradient_checkpointing": True,
            "logprob_chunk": 32,
            "load": {"dtype": "float32", "device": "cpu", "device_map": "auto"},
        },
        task,
        {problem.pid: problem},
        [completion],
        tmp_path,
        fingerprint="ab" * 32,
    )
    assert seen["sketch_dim"] == 8
    assert seen["load"] == {"dtype": "float32", "device": "cpu", "device_map": "auto"}


def test_iter_forecast_arms_skips_patch_selection_and_notes() -> None:
    document = {
        "oracle": {"transfer": {"tau_pool": 0.5}},
        "patch_selection": {"same_selection": True},
        "note": "not an arm",
    }
    assert list(iter_forecast_arms(document)) == [("oracle", document["oracle"])]


def test_row_categories_change_the_patch_fingerprint(audit_config: dict) -> None:
    cfg = copy.deepcopy(audit_config)
    cfg["arms"] = {
        "seed": 0,
        "n_boot": 5,
        "arms": [
            {"name": "oracle", "type": "oracle"},
            {"name": "patch_tau", "type": "patch_select", "base": "oracle", "k": 1, "by": "tau"},
        ],
    }
    bare = stage_fingerprints(cfg)
    one = stage_fingerprints(cfg, row_categories="a" * 64)
    other = stage_fingerprints(cfg, row_categories="b" * 64)
    assert bare["patches"] != one["patches"]
    assert one["patches"] != other["patches"]
    assert one["forecasts"] != other["forecasts"]
    assert one["features"] == other["features"] == bare["features"]

    task = build_task(TASK_SPEC)
    problem = task.problems("train", 1, seed=0)[0]
    completion = Completion(pid=problem.pid, k=0, text="work \\boxed{1}", n_tokens=4)
    parsed = task.parse(problem, completion.text)

    class Relabel(type(task)):
        def categorize(self, problem: Problem, parsed: object, text: str) -> frozenset[str]:
            del problem, parsed, text
            return frozenset({"other"})

    first = _row_category_sha(task, {problem.pid: problem}, [completion], [parsed])
    second = _row_category_sha(Relabel(), {problem.pid: problem}, [completion], [parsed])
    assert first != second
    assert (
        stage_fingerprints(cfg, row_categories=first)["patches"]
        != stage_fingerprints(cfg, row_categories=second)["patches"]
    )


def test_lora_parts_share_one_load_and_match_one_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from test_gen_hf import build_tiny_qwen

    try:
        build_tiny_qwen(seed=0)
    except Exception as exc:
        pytest.skip(f"tiny Qwen unavailable offline: {exc}")

    import vfates.features.hidden as hidden
    import vfates.features.lora_grad as lora

    loads = {"n": 0}
    prepares: list[object] = []
    real_prepare = lora.prepare_lora_grad

    def load_model(model_id: str, **load: object) -> tuple[object, object]:
        del model_id
        assert load.get("device") == "cpu"
        loads["n"] += 1
        return build_tiny_qwen(seed=0)

    def prepare(model: object, **kwargs: object) -> object:
        prepares.append(kwargs.get("save_init"))
        return real_prepare(model, **kwargs)

    monkeypatch.setattr(hidden, "load_model", load_model)
    monkeypatch.setattr(lora, "prepare_lora_grad", prepare)
    task = build_task(TASK_SPEC)
    problems = task.problems("train", 3, seed=1)
    completions = [Completion(pid=problem.pid, k=0, text="1", n_tokens=1) for problem in problems]
    by_id = {problem.pid: problem for problem in problems}
    spec = {
        "kind": "lora_grad",
        "model_id": "tiny",
        "sketch_dim": 32,
        "lora_r": 2,
        "lora_alpha": 4,
        "lora_seed": 0,
        "sketch_seed": 1,
        "load": {"dtype": "float32", "device": "cpu"},
    }
    fingerprint = "ab" * 32
    parts_dir = tmp_path / "parts"
    split, split_len, _meta = extract_features(
        spec,
        task,
        by_id,
        completions,
        parts_dir,
        fingerprint=fingerprint,
        chunk=1,
        save_init=parts_dir / "lora_init.safetensors",
    )
    assert loads["n"] == 1
    assert prepares == [parts_dir / "lora_init.safetensors"]
    assert (parts_dir / "lora_init.safetensors").is_file()
    extract_features(
        spec,
        task,
        by_id,
        completions,
        parts_dir,
        fingerprint=fingerprint,
        chunk=1,
        save_init=parts_dir / "lora_init.safetensors",
    )
    assert loads["n"] == 1
    assert len(prepares) == 1
    whole, whole_len, _meta = extract_features(
        spec,
        task,
        by_id,
        completions,
        tmp_path / "whole",
        fingerprint=fingerprint,
        chunk=3,
        save_init=tmp_path / "whole" / "lora_init.safetensors",
    )
    assert loads["n"] == 2
    np.testing.assert_array_equal(split, whole)
    np.testing.assert_array_equal(split_len, whole_len)
    fresh, tokenizer = build_tiny_qwen(seed=0)
    old = lora.lora_grad_features(
        fresh,
        tokenizer,
        [task.prompt(problem) for problem in problems],
        [completion.text for completion in completions],
        sketch_dim=32,
        lora_r=2,
        lora_alpha=4,
        lora_seed=0,
        sketch_seed=1,
        device="cpu",
        dtype="float32",
        system_prompt=task.system_prompt(),
    )
    np.testing.assert_array_equal(split, old.features)
    np.testing.assert_array_equal(split_len, old.lengths)


def test_hidden_parts_load_the_model_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    loads = {"n": 0}

    def load_model(model_id: str, **load: object) -> tuple[object, object]:
        del model_id, load
        loads["n"] += 1
        return object(), object()

    def hidden_features(
        model: object,
        tok: object,
        prompts: list[str],
        texts: list[str],
        system_prompt: str | None = None,
        completion_ids: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        del model, tok, prompts, system_prompt, completion_ids, kwargs
        return np.ones((len(texts), 2), dtype=np.float32)

    hidden = types.ModuleType("vfates.features.hidden")
    hidden.load_model = load_model  # type: ignore[attr-defined]
    hidden.hidden_features = hidden_features  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vfates.features.hidden", hidden)
    task, problem, _completion = _one_problem()
    completions = [Completion(pid=problem.pid, k=k, text=str(k), n_tokens=1) for k in range(3)]
    feats, _lengths, _meta = extract_features(
        {"kind": "hidden", "model_id": "fake", "pooling": "mean"},
        task,
        {problem.pid: problem},
        completions,
        tmp_path,
        fingerprint="cd" * 32,
        chunk=1,
    )
    assert loads["n"] == 1
    assert feats.shape == (3, 2)


def _dev_rows(n: int) -> list[Problem]:
    return [
        Problem(pid=f"p{index:02d}", split="dev", task="t", question=f"q{index}", answer="1")
        for index in range(n)
    ]


def test_dev_pool_partition_is_order_independent_and_too_small_raises() -> None:
    spec = {"from": "dev", "train": 2, "heldout": 2, "seed": 5}

    class Forward:
        def problems(self, split: str, n: int, seed: int) -> list[Problem]:
            del seed
            assert split == "dev"
            return _dev_rows(n)

    class Backward:
        def problems(self, split: str, n: int, seed: int) -> list[Problem]:
            del seed
            assert split == "dev"
            return list(reversed(_dev_rows(n)))

    forward_train, forward_held = load_pools(Forward(), spec)
    back_train, back_held = load_pools(Backward(), spec)
    assert [problem.pid for problem in forward_train] == [problem.pid for problem in back_train]
    assert [problem.pid for problem in forward_held] == [problem.pid for problem in back_held]
    ranked = sorted(
        _dev_rows(4), key=lambda problem: (u01("dev-pool", 5, problem.pid), problem.pid)
    )
    assert [problem.pid for problem in forward_train] == [problem.pid for problem in ranked[:2]]
    assert [problem.split for problem in [*forward_train, *forward_held]] == ["train"] * 2 + [
        "heldout"
    ] * 2
    assert all(problem.meta["source_split"] == "dev" for problem in forward_train)
    assert all(problem.meta["pool_from"] == "dev" for problem in forward_held)

    class Tiny:
        def problems(self, split: str, n: int, seed: int) -> list[Problem]:
            del split, n, seed
            return _dev_rows(1)

    with pytest.raises(ValueError, match="available 1, need 4"):
        load_pools(Tiny(), spec)
    with pytest.raises(ValueError, match=r"pools\.from"):
        load_pools(Forward(), {"from": "heldout", "train": 1, "heldout": 1, "seed": 0})


def test_dev_pool_audit_on_chain_sum_and_cannot_resume_as_confirmatory(
    tmp_path: Path, audit_config: dict, patch_audit_fakes
) -> None:
    cfg = copy.deepcopy(audit_config)
    cfg["pools"] = {**cfg["pools"], "from": "dev"}
    cfg["lock"] = True
    train_n = int(cfg["pools"]["train"])
    held_n = int(cfg["pools"]["heldout"])
    seed = int(cfg["pools"]["seed"])
    plain = copy.deepcopy(cfg)
    del plain["pools"]["from"]
    assert stage_fingerprints(cfg)["problems"] != stage_fingerprints(plain)["problems"]
    assert stage_fingerprints(cfg)["forecasts"] != stage_fingerprints(plain)["forecasts"]
    patch_audit_fakes(cfg)
    out = tmp_path / "dev"
    run_audit(cfg, out)
    rows = list(read_jsonl(out / "problems.jsonl"))
    assert len(rows) == train_n + held_n
    assert [row["split"] for row in rows].count("train") == train_n
    assert [row["split"] for row in rows].count("heldout") == held_n
    assert all("/dev/" in row["pid"] for row in rows)
    assert all(row["meta"]["source_split"] == "dev" for row in rows)
    assert all(row["meta"]["pool_from"] == "dev" for row in rows)
    ranked = sorted(rows, key=lambda row: (u01("dev-pool", seed, row["pid"]), row["pid"]))
    assert [row["pid"] for row in rows if row["split"] == "train"] == [
        row["pid"] for row in ranked[:train_n]
    ]
    assert [row["pid"] for row in rows if row["split"] == "heldout"] == [
        row["pid"] for row in ranked[train_n:]
    ]
    record = read_lock_record(out / "lock.json")
    assert record["pool_source"] == "dev"
    assert is_dev_audit(out) is True
    with pytest.raises(ValueError, match="development-pool"):
        require_confirmatory_pools(out)
    with pytest.raises(ValueError, match="locked stages"):
        run_audit(plain, out)
    confirmatory = tmp_path / "confirmatory"
    confirmatory.mkdir()
    write_json(confirmatory / "config.json", {"config": {"pools": {"train": 1, "heldout": 1}}})
    assert is_dev_audit(confirmatory) is False
    require_confirmatory_pools(confirmatory)
