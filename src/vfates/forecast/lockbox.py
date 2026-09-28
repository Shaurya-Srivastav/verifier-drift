"""SHA-256 sealed forecast files, written before the training run they score.

A lock is canonical JSON of the payload, the config, a note, the environment
report and a timestamp, then the SHA-256 of that document. Verification
rehashes the stored document; it does not re-read the live environment, so a
later dirty git tree does not invalidate a lock that was honest when written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vfates.utils.env import environment_report
from vfates.utils.io import canonical_json, read_json, sha256_text, write_json

LOCK_SCHEMA = "vfates.lock/1"


def lock_forecasts(
    payload: dict[str, Any],
    path: str | Path,
    *,
    config: Any = None,
    note: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a ``vfates.lock/1`` file. Raises ``FileExistsError`` unless ``overwrite``.

    ``sha256`` is ``sha256_text(canonical_json(record))`` of the record with
    the digest field removed. ``payload`` must be JSON-serialisable.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"lock file already exists: {path}")
    body: dict[str, Any] = {
        "payload": payload,
        "config": config,
        "note": note,
        "env": environment_report(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": LOCK_SCHEMA,
    }
    digest = sha256_text(canonical_json(body))
    record = {**body, "sha256": digest}
    write_json(path, record)
    return record


def verify_lock(path: str | Path) -> bool:
    """True when the file parses and its ``sha256`` matches the other fields."""
    try:
        record = read_json(path)
    except ValueError:
        return False
    if not isinstance(record, dict) or "sha256" not in record:
        return False
    digest = record["sha256"]
    if not isinstance(digest, str):
        return False
    body = {key: value for key, value in record.items() if key != "sha256"}
    return sha256_text(canonical_json(body)) == digest


def load_lock(path: str | Path) -> Any:
    """Return the locked payload. Raises ``ValueError`` if the file was tampered with."""
    if not verify_lock(path):
        raise ValueError(f"lock file failed verification: {path}")
    return read_json(path)["payload"]
