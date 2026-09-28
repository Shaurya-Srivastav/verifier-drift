"""Deterministic, atomic file I/O for JSON, JSONL and YAML artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_json"):
        return o.to_json()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def canonical_json(obj: Any) -> str:
    """Canonical JSON (sorted keys, no whitespace) used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default, allow_nan=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: str | Path, obj: Any) -> None:
    _atomic_write(Path(path), json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    lines = [json.dumps(r, sort_keys=True, default=_default) for r in rows]
    _atomic_write(Path(path), "\n".join(lines) + ("\n" if lines else ""))


def append_jsonl(path: str | Path, row: Any) -> None:
    """Append one JSON line and flush it to disk.

    ``flush`` plus ``fsync`` makes a crash tear at most the line being written,
    which :func:`read_jsonl` can drop. A line left in the stdio buffer is not
    a torn line the next process can see; it is a missing line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[Any]:
    """Read JSONL, dropping exactly one torn final line.

    A last line with no trailing newline that fails to parse is a crash tear:
    it is dropped with a ``UserWarning``. Any other decode error is raised.
    A final line that parses is kept even when the file does not end in a newline.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text == "":
        return []
    ended = text.endswith("\n")
    lines = text.splitlines()
    rows: list[Any] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            torn = index == len(lines) - 1 and not ended
            if torn:
                warnings.warn(
                    f"dropping torn final line in {path} ({exc.msg})",
                    UserWarning,
                    stacklevel=2,
                )
                break
            raise
    return rows


def iter_jsonl(path: str | Path) -> Iterator[Any]:
    yield from read_jsonl(path)


def atomic_npz(path: str | Path, **arrays: Any) -> None:
    """Write a ``.npz`` via a temp file in the same directory, then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def read_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, obj: Any) -> None:
    obj = json.loads(json.dumps(obj, default=_default))
    _atomic_write(Path(path), yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))
