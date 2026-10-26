"""
session_store.py -- Save and load session summary JSON files.
"""

import json
import time
from pathlib import Path

SESSIONS_DIR = Path(__file__).parent / "sessions"


def save_session(summary: dict) -> str:
    SESSIONS_DIR.mkdir(exist_ok=True)
    session_id = f"session_{int(time.time() * 1000)}"
    path = SESSIONS_DIR / f"{session_id}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return session_id


def list_sessions() -> list:
    SESSIONS_DIR.mkdir(exist_ok=True)
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            data["id"] = path.stem
            sessions.append(data)
        except Exception:
            pass
    return sessions


def get_session(session_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    data["id"] = session_id
    return data
