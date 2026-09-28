"""Sealed forecast files: hash, tamper detection, overwrite refusal."""

import json

import pytest

from vfates.forecast.lockbox import load_lock, lock_forecasts, verify_lock
from vfates.utils.io import canonical_json, read_json, sha256_text, write_json


def test_lock_round_trip_and_schema(tmp_path) -> None:
    path = tmp_path / "forecast.lock.json"
    payload = {"tau_hard": 0.5, "arms": ["oracle", "fp_hash"]}
    record = lock_forecasts(
        payload, path, config={"task": "decimal_chain_sum"}, note="pre-registered"
    )
    assert record["schema"] == "vfates.lock/1"
    assert record["payload"] == payload
    assert record["config"]["task"] == "decimal_chain_sum"
    assert record["note"] == "pre-registered"
    assert "hardware" in record["env"]
    assert record["created_utc"].endswith("Z")
    body = {key: value for key, value in record.items() if key != "sha256"}
    assert record["sha256"] == sha256_text(canonical_json(body))
    assert verify_lock(path) is True
    assert load_lock(path) == payload
    stored = read_json(path)
    assert stored["sha256"] == record["sha256"]


def test_tamper_detection(tmp_path) -> None:
    path = tmp_path / "forecast.lock.json"
    lock_forecasts({"value": 1}, path, note="ok")
    stored = read_json(path)
    stored["payload"] = {"value": 2}
    write_json(path, stored)
    assert verify_lock(path) is False
    with pytest.raises(ValueError, match="failed verification"):
        load_lock(path)
    stored["sha256"] = "0" * 64
    write_json(path, stored)
    assert verify_lock(path) is False
    path.write_text("{not json")
    assert verify_lock(path) is False


def test_overwrite_refusal(tmp_path) -> None:
    path = tmp_path / "nested" / "forecast.lock.json"
    lock_forecasts({"value": 1}, path, note="first")
    with pytest.raises(FileExistsError):
        lock_forecasts({"value": 2}, path, note="second")
    assert load_lock(path) == {"value": 1}
    lock_forecasts({"value": 2}, path, note="second", overwrite=True)
    assert verify_lock(path) is True
    assert load_lock(path) == {"value": 2}
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        verify_lock(missing)


def test_lock_json_is_canonical_document(tmp_path) -> None:
    path = tmp_path / "forecast.lock.json"
    lock_forecasts({"b": 1, "a": [1, 2]}, path)
    raw = json.loads(path.read_text())
    assert raw["schema"] == "vfates.lock/1"
    assert "sha256" in raw
