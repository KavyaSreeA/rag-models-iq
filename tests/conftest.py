"""Shared pytest fixtures.

Redirects the SQLite DB to a temp file per test so tests never touch the
real data/iqmath.db, and exposes a `has_openai_key` marker helper for tests
that need a live LLM call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "test_iqmath.db"
    monkeypatch.setattr(config, "APP_DB_PATH", test_db)
    import memory.db as db_module

    monkeypatch.setattr(db_module.config, "APP_DB_PATH", test_db)
    yield


@pytest.fixture
def has_openai_key() -> bool:
    return bool(config.OPENAI_API_KEY)
