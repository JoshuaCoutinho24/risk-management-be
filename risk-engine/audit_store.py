"""
In-memory audit log with optional file persistence.

Logs advisor actions: simulations run, profiles ingested/updated,
notes added/deleted, re-interviews. Persists to audit_log.json.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_BASE_DIR   = Path(__file__).parent
_AUDIT_FILE = _BASE_DIR / "audit_log.json"

# In-memory store — list of entries newest-first
_log: list[dict] = []


def _load_from_disk() -> None:
    """Load existing audit log from disk into memory on startup."""
    global _log
    if not _AUDIT_FILE.exists():
        return
    try:
        with open(_AUDIT_FILE, encoding="utf-8") as fh:
            _log = json.load(fh)
    except (json.JSONDecodeError, OSError):
        _log = []


def _save_to_disk() -> None:
    """Persist audit log to disk (keep most recent 1000 entries)."""
    try:
        with open(_AUDIT_FILE, "w", encoding="utf-8") as fh:
            json.dump(_log[:1000], fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def log_action(
    action: str,
    person_id: Optional[str] = None,
    details: Optional[dict] = None,
    user: str = "advisor",
) -> dict:
    """
    Append one entry to the audit log.

    Parameters
    ----------
    action    : Short action label, e.g. "simulate", "ingest", "note_added".
    person_id : Client person_id if applicable.
    details   : Any extra key/value pairs to store.
    user      : Advisor username (defaults to "advisor").
    """
    entry = {
        "id":         str(uuid.uuid4()),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "action":     action,
        "person_id":  person_id,
        "details":    details or {},
        "user":       user,
    }
    _log.insert(0, entry)
    _save_to_disk()
    return entry


def get_log(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return a slice of the audit log (newest first)."""
    return _log[offset : offset + limit]


def get_log_for_person(person_id: str) -> list[dict]:
    """Return all audit entries for a specific person."""
    return [e for e in _log if e.get("person_id") == person_id]


# Load on import
_load_from_disk()
