"""SHA-256 sealed forecast files, written before the training run they score.

A lock is canonical JSON of the payload, the config, a note, the environment
report and a timestamp, then the SHA-256 of that document. Verification
rehashes the stored document; it does not re-read the live environment, so a
later dirty git tree does not invalidate a lock that was honest when written.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vfates.utils.env import environment_report
from vfates.utils.io import (
    canonical_json,
    read_json,
    sha256_text,
    write_json,
)
from vfates.utils.io import (
    sha256_file as sha256_file,
)

LOCK_SCHEMA = "vfates.lock/2"
LOCK_SCHEMAS = ("vfates.lock/1", "vfates.lock/2")


def lock_forecasts(
    payload: dict[str, Any],
    path: str | Path,
    *,
    config: Any = None,
    note: str | None = None,
    artifacts: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a ``vfates.lock/2`` file. Raises ``FileExistsError`` unless ``overwrite``.

    ``sha256`` is ``sha256_text(canonical_json(record))`` of the record with
    the digest field removed. The hashed body includes ``artifacts``, a map
    from artifact name to the sha256 hex of that file (empty when omitted).
    ``payload`` must be JSON-serialisable. :func:`verify_lock` still accepts
    ``vfates.lock/1`` files, which have no artifact map.
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
        "artifacts": {}
        if artifacts is None
        else {str(key): str(value) for key, value in artifacts.items()},
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
    schema = body.get("schema")
    if schema not in LOCK_SCHEMAS:
        return False
    if schema == LOCK_SCHEMA and not isinstance(body.get("artifacts"), dict):
        return False
    return sha256_text(canonical_json(body)) == digest


def load_lock(path: str | Path) -> Any:
    """Return the locked payload. Raises ``ValueError`` if the file was tampered with."""
    return load_lock_record(path)["payload"]


def load_lock_record(path: str | Path) -> dict[str, Any]:
    """Return the verified lock record, including payload, artifacts, and digest."""
    if not verify_lock(path):
        raise ValueError(f"lock file failed verification: {path}")
    record = read_json(path)
    if not isinstance(record, dict):
        raise ValueError(f"lock file failed verification: {path}")
    return record


def assert_matches_lock(path: str | Path, payload: Any) -> None:
    """Raise ``ValueError`` unless ``payload`` is the locked payload.

    Comparison is canonical JSON, so ``1e-1`` matches ``0.1`` and ``NaN``
    matches ``NaN``. A changed value does not match.
    """
    locked = load_lock(path)
    if canonical_json(payload) != canonical_json(locked):
        raise ValueError(f"payload does not match the locked payload: {path}")
