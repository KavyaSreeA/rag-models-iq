"""Module 11 — memory tests. Pure SQLite/Python, no API key required."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from memory.long_term import get_profile, remember_fact, update_profile
from memory.persistent import get_conversation_sessions, get_session_messages, log_turn
from memory.short_term import ShortTermMemory


def test_short_term_memory_add_and_trim():
    mem = ShortTermMemory(max_turns=2)
    for i in range(5):
        mem.add_user(f"user {i}")
        mem.add_ai(f"ai {i}")
    messages = mem.as_list()
    assert len(messages) == 4  # 2 turns * 2 messages
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "ai 4"


def test_short_term_memory_clear():
    mem = ShortTermMemory()
    mem.add_user("hello")
    mem.clear()
    assert mem.as_list() == []


def test_long_term_profile_roundtrip():
    profile = update_profile("user1", department="Finance")
    assert profile.department == "Finance"

    fetched = get_profile("user1")
    assert fetched.department == "Finance"
    assert fetched.user_id == "user1"


def test_remember_fact_department():
    remember_fact("user2", "department", "Engineering")
    profile = get_profile("user2")
    assert profile.department == "Engineering"


def test_remember_fact_unknown_key_becomes_faq():
    remember_fact("user3", "favorite_color", "blue")
    profile = get_profile("user3")
    assert any("favorite_color" in t for t in profile.faq_topics)


def test_get_profile_for_unknown_user_returns_empty_profile():
    profile = get_profile("nobody")
    assert profile.department is None
    assert profile.faq_topics == []


def test_persistent_conversation_logging():
    log_turn("session-a", "user", "Hi there")
    log_turn("session-a", "assistant", "Hello!", agent="hr")

    messages = get_session_messages("session-a")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["agent"] == "hr"


def test_get_conversation_sessions_lists_recent_sessions():
    log_turn("session-b", "user", "First message in B")
    sessions = get_conversation_sessions(limit=5)
    ids = [s["session_id"] for s in sessions]
    assert "session-b" in ids
