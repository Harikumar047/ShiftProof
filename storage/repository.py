"""Repository for typed CRUD operations on models, sessions, and captures."""

import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelRecord:
    id: Optional[int]
    name: str
    version: str
    path: str
    provider_requested: Optional[str] = None
    provider_active: Optional[str] = None
    provider_note: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class SessionRecord:
    id: Optional[int]
    model_id: Optional[int]
    label: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class CaptureRecord:
    id: Optional[int]
    session_id: Optional[int]
    image_path: str
    true_label: str
    condition_type: str
    condition_note: Optional[str] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    correct: Optional[int] = None
    timestamp: Optional[str] = None


def insert_model(
    conn: sqlite3.Connection,
    name: str,
    version: str,
    path: str,
    provider_requested: Optional[str] = None,
    provider_active: Optional[str] = None,
    provider_note: Optional[str] = None,
) -> int:
    """Insert a new model record and return its generated ID."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO models (name, version, path, provider_requested, provider_active, provider_note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, version, path, provider_requested, provider_active, provider_note),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_session(
    conn: sqlite3.Connection,
    model_id: Optional[int],
    label: Optional[str] = None,
) -> int:
    """Insert a new session record and return its generated ID."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sessions (model_id, label)
        VALUES (?, ?)
        """,
        (model_id, label),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_capture(
    conn: sqlite3.Connection,
    session_id: Optional[int],
    image_path: str,
    true_label: str,
    condition_type: str,
    condition_note: Optional[str] = None,
    predicted_label: Optional[str] = None,
    confidence: Optional[float] = None,
    correct: Optional[int] = None,
) -> int:
    """Insert a capture record and return its generated ID.

    If 'correct' is not provided and predicted_label is present,
    correct is automatically derived from (true_label == predicted_label).
    """
    if correct is None and predicted_label is not None:
        correct = 1 if (true_label == predicted_label) else 0

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO captures (
            session_id, image_path, true_label, condition_type,
            condition_note, predicted_label, confidence, correct
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            image_path,
            true_label,
            condition_type,
            condition_note,
            predicted_label,
            confidence,
            correct,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def get_model(conn: sqlite3.Connection, model_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a model record by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_session(conn: sqlite3.Connection, session_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a session record by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_captures_by_session(
    conn: sqlite3.Connection, session_id: int
) -> List[Dict[str, Any]]:
    """Retrieve all capture records for a given session ID."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM captures WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    )
    return [dict(row) for row in cursor.fetchall()]


def get_captures_by_condition(
    conn: sqlite3.Connection, condition_type: str, session_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Retrieve captures matching a condition type, optionally filtered by session."""
    cursor = conn.cursor()
    if session_id is not None:
        cursor.execute(
            """
            SELECT * FROM captures
            WHERE condition_type = ? AND session_id = ?
            ORDER BY id ASC
            """,
            (condition_type, session_id),
        )
    else:
        cursor.execute(
            "SELECT * FROM captures WHERE condition_type = ? ORDER BY id ASC",
            (condition_type,),
        )
    return [dict(row) for row in cursor.fetchall()]


def get_session_summary(
    conn: sqlite3.Connection, session_id: int
) -> Dict[str, int]:
    """Get count of captures grouped by condition_type for a session."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT condition_type, COUNT(*) as count
        FROM captures
        WHERE session_id = ?
        GROUP BY condition_type
        ORDER BY condition_type ASC
        """,
        (session_id,),
    )
    return {row["condition_type"]: row["count"] for row in cursor.fetchall()}


def get_latest_session_with_captures(
    conn: sqlite3.Connection,
) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent session record that has at least one capture recorded."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.*, COUNT(c.id) as capture_count
        FROM sessions s
        JOIN captures c ON s.id = c.session_id
        GROUP BY s.id
        ORDER BY s.id DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone()
    return dict(row) if row else None

