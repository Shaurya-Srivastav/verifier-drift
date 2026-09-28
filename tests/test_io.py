"""JSONL durability and atomic npz writes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vfates.utils.io import append_jsonl, atomic_npz, read_jsonl


def test_append_jsonl_fsyncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[int] = []
    monkeypatch.setattr("vfates.utils.io.os.fsync", lambda fd: synced.append(fd))
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"a": 1})
    assert synced
    assert read_jsonl(path) == [{"a": 1}]


def test_read_jsonl_drops_one_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b":')
    with pytest.warns(UserWarning, match="torn"):
        assert read_jsonl(path) == [{"a": 1}]

    path.write_text('{"a": 1}\n{"b": 2}', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_raises_on_other_corruption(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{bad}\n{"b": 2}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)
    path.write_text('{"a": 1}\n{bad}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_atomic_npz_replaces_and_keeps_the_previous_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "features.npz"
    atomic_npz(dest, values=np.array([1, 2]))
    with np.load(dest) as saved:
        assert saved["values"].tolist() == [1, 2]

    def boom(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("savez failed")

    monkeypatch.setattr("vfates.utils.io.np.savez", boom)
    with pytest.raises(RuntimeError, match="savez failed"):
        atomic_npz(dest, values=np.array([3]))
    with np.load(dest) as saved:
        assert saved["values"].tolist() == [1, 2]
    leftovers = [path for path in tmp_path.iterdir() if path.name != "features.npz"]
    assert leftovers == []
