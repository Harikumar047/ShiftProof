"""Database initialization and connection handling for SQLite storage."""

import os
import sqlite3
from typing import Union
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    path TEXT NOT NULL,
    provider_requested TEXT,
    provider_active TEXT,
    provider_note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
    label TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    true_label TEXT NOT NULL,
    condition_type TEXT NOT NULL,
    condition_note TEXT,
    predicted_label TEXT,
    confidence REAL,
    correct INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: Union[str, Path] = "shiftproof.db") -> sqlite3.Connection:
    """Initialize SQLite database connection and create tables if not exist.

    Args:
        db_path: Filepath or ':memory:' for the SQLite database.

    Returns:
        sqlite3.Connection: Configured SQLite database connection with row factory.
    """
    db_path_str = str(db_path)
    if db_path_str != ":memory:":
        parent_dir = os.path.dirname(os.path.abspath(db_path_str))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    conn = sqlite3.connect(db_path_str)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn
