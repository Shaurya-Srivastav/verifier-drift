"""Completion token ids and atomic audit-table writes."""

from __future__ import annotations

import os

import numpy as np
import pytest

from vfates.types import AuditTable, Completion


def test_completion_token_ids_round_trip_and_missing_key() -> None:
    row = Completion(
        pid="p",
        k=0,
        text="playing",
        n_tokens=2,
        logprob=-0.2,
        finish_reason="stop",
        token_ids=(1363, 287),
    )
    payload = row.to_json()
    assert list(payload)[-1] == "token_ids"
    assert payload["token_ids"] == [1363, 287]
    again = Completion.from_json(payload)
    assert again == row
    assert again.token_ids == (1363, 287)
    legacy = {
        "pid": "p",
        "k": 1,
        "text": "x",
        "n_tokens": 1,
        "logprob": None,
        "finish_reason": "length",
    }
    loaded = Completion.from_json(legacy)
    assert loaded.token_ids is None
    explicit = {**legacy, "token_ids": None}
    assert Completion.from_json(explicit).token_ids is None


def _table() -> AuditTable:
    return AuditTable(
        pid=np.array(["a", "b"], dtype=object),
        split=np.array(["train", "heldout"], dtype=object),
        k=np.array([0, 1]),
        O=np.array([1, 0]),
        answer=np.array(["1", None], dtype=object),
        n_tokens=np.array([3, 4]),
        logprob=np.array([-0.2, np.nan]),
        V={"oracle": np.array([1, 0])},
        meta={"note": "atomic"},
    )


def test_audit_table_save_replaces_temp_files(tmp_path, monkeypatch) -> None:
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst) -> None:
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    path = tmp_path / "audit.npz"
    table = _table()
    table.save(path)
    targets = [dst for _src, dst in replaced]
    assert any(target.endswith("audit.npz") for target in targets)
    assert any(target.endswith("audit.meta.json") for target in targets)
    loaded = AuditTable.load(path)
    assert loaded.meta == {"note": "atomic"}
    assert loaded.pid.tolist() == ["a", "b"]
    assert loaded.answer.tolist() == ["1", None]
    assert {item.name for item in tmp_path.iterdir()} == {"audit.npz", "audit.meta.json"}


def test_failed_audit_save_leaves_no_destination(tmp_path, monkeypatch) -> None:
    def boom(*_args, **_kwargs) -> None:
        raise RuntimeError("disk")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(RuntimeError, match="disk"):
        _table().save(tmp_path / "audit.npz")
    assert list(tmp_path.iterdir()) == []
