"""Module 11 — long-term memory: user profile (name, department, prefs, FAQs).

Scenario 3/4 from the handbook:
    "Remember that I belong to the Finance Department."  -> update_profile()
    "What department do I belong to?"                    -> get_profile()
"""
from __future__ import annotations

import json
from typing import Optional

from memory.db import get_conn
from schemas import UserProfile


def get_profile(user_id: str) -> UserProfile:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM user_profile WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return UserProfile(user_id=user_id)
    return UserProfile(
        user_id=row["user_id"],
        name=row["name"],
        department=row["department"],
        preferred_email_style=row["preferred_email_style"],
        faq_topics=json.loads(row["faq_topics_json"] or "[]"),
        frequent_docs=json.loads(row["frequent_docs_json"] or "[]"),
    )


def update_profile(
    user_id: str,
    name: Optional[str] = None,
    department: Optional[str] = None,
    preferred_email_style: Optional[str] = None,
    add_faq_topic: Optional[str] = None,
    add_frequent_doc: Optional[str] = None,
) -> UserProfile:
    profile = get_profile(user_id)
    if name:
        profile.name = name
    if department:
        profile.department = department
    if preferred_email_style:
        profile.preferred_email_style = preferred_email_style
    if add_faq_topic and add_faq_topic not in profile.faq_topics:
        profile.faq_topics.append(add_faq_topic)
    if add_frequent_doc and add_frequent_doc not in profile.frequent_docs:
        profile.frequent_docs.append(add_frequent_doc)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_profile (user_id, name, department, preferred_email_style, faq_topics_json, frequent_docs_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name=excluded.name,
                department=excluded.department,
                preferred_email_style=excluded.preferred_email_style,
                faq_topics_json=excluded.faq_topics_json,
                frequent_docs_json=excluded.frequent_docs_json
            """,
            (
                profile.user_id,
                profile.name,
                profile.department,
                profile.preferred_email_style,
                json.dumps(profile.faq_topics),
                json.dumps(profile.frequent_docs),
            ),
        )
    return profile


def remember_fact(user_id: str, key: str, value: str) -> UserProfile:
    """Generic hook used by the Coordinator when it detects a 'remember X' request."""
    key = key.lower().strip()
    if key in ("department", "dept"):
        return update_profile(user_id, department=value)
    if key == "name":
        return update_profile(user_id, name=value)
    if key in ("email_style", "preferred_email_style"):
        return update_profile(user_id, preferred_email_style=value)
    # Fallback: treat unknown facts as an FAQ topic worth remembering.
    return update_profile(user_id, add_faq_topic=f"{key}: {value}")
