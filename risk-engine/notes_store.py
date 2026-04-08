"""
Per-client advisor notes store.

Persists to notes.json. Notes are keyed by person_id.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_BASE_DIR   = Path(__file__).parent
_NOTES_FILE = _BASE_DIR / "notes.json"

# { person_id: [ {id, content, created_at, updated_at}, ... ] }
_notes: dict[str, list[dict]] = {}


def _load() -> None:
    global _notes
    if not _NOTES_FILE.exists():
        return
    try:
        with open(_NOTES_FILE, encoding="utf-8") as fh:
            _notes = json.load(fh)
    except (json.JSONDecodeError, OSError):
        _notes = {}


def _save() -> None:
    try:
        with open(_NOTES_FILE, "w", encoding="utf-8") as fh:
            json.dump(_notes, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def get_notes(person_id: str) -> list[dict]:
    """Return all notes for a person, newest first."""
    return list(reversed(_notes.get(person_id, [])))


def add_note(person_id: str, content: str) -> dict:
    """Add a note and return it."""
    note = {
        "id":         str(uuid.uuid4()),
        "content":    content.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if person_id not in _notes:
        _notes[person_id] = []
    _notes[person_id].append(note)
    _save()
    return note


def update_note(person_id: str, note_id: str, content: str) -> Optional[dict]:
    """Update a note's content. Returns updated note or None if not found."""
    for note in _notes.get(person_id, []):
        if note["id"] == note_id:
            note["content"]    = content.strip()
            note["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save()
            return note
    return None


def delete_note(person_id: str, note_id: str) -> bool:
    """Delete a note. Returns True if deleted, False if not found."""
    notes = _notes.get(person_id, [])
    original_len = len(notes)
    _notes[person_id] = [n for n in notes if n["id"] != note_id]
    if len(_notes[person_id]) < original_len:
        _save()
        return True
    return False


_load()
