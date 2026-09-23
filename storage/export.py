"""Data export and user-controlled deletion operations."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Union
from storage.repository import get_session, get_model, get_captures_by_session, get_session_summary


def export_session_to_json(
    conn: sqlite3.Connection,
    session_id: int,
    out_path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> Dict[str, Any]:
    """Export all session metadata, associated model, and captures to a dict/JSON file.

    Args:
        conn: SQLite connection.
        session_id: The ID of the session to export.
        out_path: Optional file path to write JSON content to.
        indent: Indentation for JSON formatting if saving to file.

    Returns:
        Dict[str, Any]: Exported session data structure.
    """
    session = get_session(conn, session_id)
    if not session:
        raise ValueError(f"Session with ID {session_id} not found.")

    model = None
    if session.get("model_id"):
        model = get_model(conn, session["model_id"])

    captures = get_captures_by_session(conn, session_id)
    summary = get_session_summary(conn, session_id)

    export_data = {
        "session": session,
        "model": model,
        "summary": summary,
        "captures": captures,
    }

    if out_path:
        out_path_obj = Path(out_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path_obj, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=indent, default=str)

    return export_data


def delete_session(conn: sqlite3.Connection, session_id: int) -> bool:
    """Delete a session and all its associated captures.

    Args:
        conn: SQLite connection.
        session_id: The ID of the session to delete.

    Returns:
        bool: True if the session was found and deleted, False otherwise.
    """
    cursor = conn.cursor()
    # Explicitly delete captures first for defensive clean up even without pragma
    cursor.execute("DELETE FROM captures WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    return cursor.rowcount > 0
