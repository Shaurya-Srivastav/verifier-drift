"""P1 helpers for the lock, token ids, fingerprints, and the LoRA init file."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfates.forecast.lockbox import lock_forecasts
from vfates.pilots.p1_harness import (
    audit_artifact_digest,
    features_used_token_ids,
    fingerprint_manifest_complete,
    lock_payload_matches_forecasts,
    lora_init_matches_lock,
)
from vfates.utils.io import sha256_file, write_json

_STAGES = ("problems", "completions", "calibration", "table", "features", "forecasts")


def test_artifact_digest_covers_the_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vfates.pilots.p1_harness._verdict_digest", lambda path: "verdicts")
    digest = audit_artifact_digest(tmp_path)
    assert set(digest) == {
        "problems",
        "completions",
        "calibration",
        "verdicts",
        "features",
        "forecasts",
        "stages",
        "lora_init",
        "lock",
    }
    assert digest["problems"] == ""
    (tmp_path / "problems.jsonl").write_text("row\n", encoding="utf-8")
    again = audit_artifact_digest(tmp_path)
    assert again["problems"] == sha256_file(tmp_path / "problems.jsonl")
    assert again["problems"] != digest["problems"]


def test_lock_payload_matches_forecasts(tmp_path: Path) -> None:
    payload = {"oracle": {"transfer": {"tau_pool": 0.25}}}
    lock_forecasts(payload, tmp_path / "lock.json", config={"lock": True})
    write_json(tmp_path / "forecasts.json", payload)
    assert lock_payload_matches_forecasts(tmp_path)
    write_json(tmp_path / "forecasts.json", {"oracle": {"transfer": {"tau_pool": 0.5}}})
    assert lock_payload_matches_forecasts(tmp_path) is False


def test_features_and_manifest_checks(tmp_path: Path) -> None:
    assert features_used_token_ids(tmp_path) is False
    write_json(tmp_path / "features.meta.json", {"token_source": "token_ids"})
    assert features_used_token_ids(tmp_path) is True
    write_json(tmp_path / "features.meta.json", {"token_source": "text"})
    assert features_used_token_ids(tmp_path) is False

    assert fingerprint_manifest_complete(tmp_path) is False
    write_json(
        tmp_path / "stages.json",
        {"schema": "vfates.stages/1", "stages": {name: "ab" * 32 for name in _STAGES}},
    )
    assert fingerprint_manifest_complete(tmp_path) is True
    doc = {"schema": "vfates.stages/1", "stages": {name: "ab" * 32 for name in _STAGES}}
    doc["stages"]["features"] = "abc"
    write_json(tmp_path / "stages.json", doc)
    assert fingerprint_manifest_complete(tmp_path) is False


def test_lora_init_must_load_and_match_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vfates.features.lora_init as lora_init

    init = tmp_path / "lora_init.safetensors"
    init.write_bytes(b"init-bytes")
    digest = sha256_file(init)
    payload = {"oracle": {"x": 1}}
    loaded: list[str] = []

    def fake_load(path: Path, expected_sha256: str | None = None) -> tuple[dict, dict]:
        loaded.append(str(path))
        if expected_sha256 == "bad":
            raise ValueError("schema")
        return {}, {"schema": "vfates.lora_init/1"}

    monkeypatch.setattr(lora_init, "load_lora_init", fake_load)
    lock_forecasts(payload, tmp_path / "lock.json", artifacts={"lora_init": digest})
    assert lora_init_matches_lock(tmp_path) is True
    assert loaded

    lock_forecasts(
        payload, tmp_path / "lock.json", artifacts={"lora_init": "ab" * 32}, overwrite=True
    )
    assert lora_init_matches_lock(tmp_path) is False

    lock_forecasts(payload, tmp_path / "lock.json", artifacts={"lora_init": digest}, overwrite=True)

    def boom(path: Path, expected_sha256: str | None = None) -> tuple[dict, dict]:
        del path, expected_sha256
        raise ValueError("schema")

    monkeypatch.setattr(lora_init, "load_lora_init", boom)
    assert lora_init_matches_lock(tmp_path) is False
    init.unlink()
    assert lora_init_matches_lock(tmp_path) is False
