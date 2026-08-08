"""Module 7/8 workflow tests — pure logic pieces that don't need a live LLM."""
from __future__ import annotations

from workflows.sequential import _EMAIL_ADDR_RE, _INTERNAL_HINTS


def test_email_regex_extracts_address():
    text = "Please summarize the handbook and email it to hr@novatech.example please."
    match = _EMAIL_ADDR_RE.search(text)
    assert match is not None
    assert match.group(0) == "hr@novatech.example"


def test_email_regex_no_match_returns_none():
    assert _EMAIL_ADDR_RE.search("Summarize the handbook and email it to HR.") is None


def test_internal_hint_detection():
    assert any(h in "summarize the employee handbook".lower() for h in _INTERNAL_HINTS)
    assert not any(h in "research microsoft's ai products".lower() for h in _INTERNAL_HINTS)


def test_parallel_entity_filtering(monkeypatch):
    from workflows import parallel as parallel_module

    calls = []

    def fake_research_one(entity):
        calls.append(entity)
        return f"findings about {entity}"

    monkeypatch.setattr(parallel_module, "_research_one", fake_research_one)
    monkeypatch.setattr(
        parallel_module,
        "generate_report",
        lambda *a, **k: __import__("schemas").TaskReport(task_summary="ok", department="General"),
    )

    class _FakeLLM:
        def invoke(self, *_a, **_k):
            class _R:
                content = "combined comparison report"

            return _R()

    monkeypatch.setattr(parallel_module, "get_llm", lambda temperature=0.2: _FakeLLM())

    result = parallel_module.run_parallel_research(["Google", "", "Microsoft"], "Research Google and Microsoft")

    assert sorted(calls) == ["Google", "Microsoft"]
    assert result.agent == "parallel"
    assert result.answer == "combined comparison report"
    assert len(result.retrieved_docs) == 2
