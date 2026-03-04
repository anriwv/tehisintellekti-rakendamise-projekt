import json
import os
from datetime import datetime
from pathlib import Path

from config import HISTORY_DIR

_dir = Path(HISTORY_DIR)


def _ensure_dir():
    _dir.mkdir(exist_ok=True)


def new_session_id() -> str:
    """Generate a unique session ID based on the current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _session_path(session_id: str) -> Path:
    return _dir / f"{session_id}.json"


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Ensure every message has token fields for consistent history format."""
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        normalized.append(
            {
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "token_input": int(msg.get("token_input", 0) or 0),
                "token_output": int(msg.get("token_output", 0) or 0),
            }
        )
    return normalized


def list_sessions() -> list[dict]:
    """
    Return a list of all saved sessions, sorted newest-first.

    Each entry: {"id": str, "title": str, "created_at": str}
    """
    _ensure_dir()
    sessions = []
    for p in sorted(_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "id": data.get("id", p.stem),
                    "title": data.get("title", p.stem),
                    "created_at": data.get("created_at", ""),
                }
            )
        except Exception:
            pass  # skip corrupt files
    return sessions


def load_session(session_id: str) -> list[dict]:
    """Load and return the message list for a given session id."""
    path = _session_path(session_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _normalize_messages(data.get("messages", []))
    except Exception:
        return []


def save_session(session_id: str, messages: list[dict]):
    """
    Save (create or overwrite) a session file.

    The title is derived from the first user message (truncated to 60 chars).
    """
    _ensure_dir()
    messages = _normalize_messages(messages)
    title = session_id  # fallback
    for msg in messages:
        if msg.get("role") == "user":
            title = msg["content"][:60].strip()
            if len(msg["content"]) > 60:
                title += "…"
            break

    payload = {
        "id": session_id,
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }
    _session_path(session_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def delete_session(session_id: str):
    """Delete the JSON file for the given session id, if it exists."""
    path = _session_path(session_id)
    if path.exists():
        os.remove(path)
