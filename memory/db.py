"""SQLite connection + schema init shared by persistent.py and long_term.py."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    role TEXT NOT NULL,          -- 'user' | 'assistant'
    content TEXT NOT NULL,
    agent TEXT                   -- which agent produced this turn, if assistant
);

CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    department TEXT,
    preferred_email_style TEXT,
    faq_topics_json TEXT DEFAULT '[]',
    frequent_docs_json TEXT DEFAULT '[]'
);
"""


def init_db() -> None:
    config.APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.APP_DB_PATH) as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def get_conn():
    init_db()
    conn = sqlite3.connect(config.APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
