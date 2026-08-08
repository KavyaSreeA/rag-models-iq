"""Module 11 — persistent memory: every conversation turn logged to SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from memory.db import get_conn


def log_turn(session_id: str, role: str, content: str, agent: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, ts, role, content, agent) VALUES (?, ?, ?, ?, ?)",
            (session_id, datetime.now(timezone.utc).isoformat(), role, content, agent),
        )


def get_conversation_sessions(limit: int = 20) -> List[dict]:
    """Return the most recent distinct sessions with a preview of their first message."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT session_id, MIN(ts) AS started_at, COUNT(*) AS turns,
                   (SELECT content FROM conversations c2
                     WHERE c2.session_id = c1.session_id AND c2.role = 'user'
                     ORDER BY c2.ts ASC LIMIT 1) AS preview
            FROM conversations c1
            GROUP BY session_id
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(session_id: str) -> List[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, agent, ts FROM conversations WHERE session_id = ? ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]
