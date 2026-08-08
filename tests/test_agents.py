"""Schema validation + coordinator routing logic tests that don't require a
live OpenAI call (classification/LLM calls are monkeypatched or bypassed)."""
from __future__ import annotations

from schemas import AgentName, RouteDecision, TaskReport


def test_task_report_defaults():
    report = TaskReport(task_summary="Did a thing", department="HR")
    assert report.priority == "Medium"
    assert report.actions_taken == []
    assert report.pending_actions == []


def test_route_decision_defaults():
    route = RouteDecision(agent=AgentName.HR, reasoning="policy question")
    assert route.entities == []
    assert route.memory_action is None


def test_coordinator_memory_remember_handler(monkeypatch):
    from agents.coordinator import _handle_memory

    route = RouteDecision(
        agent=AgentName.MEMORY,
        reasoning="user stated their department",
        memory_action="remember",
        memory_key="department",
        memory_value="Finance",
    )
    result = _handle_memory(route, "test_user", "Remember that I belong to the Finance Department.")
    assert "Finance" in result.answer
    assert result.report.department == "Finance"


def test_coordinator_memory_recall_handler_with_no_data():
    from agents.coordinator import _handle_memory

    route = RouteDecision(agent=AgentName.MEMORY, reasoning="user asked for their department", memory_action="recall")
    result = _handle_memory(route, "brand_new_user", "What department do I belong to?")
    assert "don't have anything saved" in result.answer.lower()


def test_coordinator_memory_recall_handler_with_data():
    from agents.coordinator import _handle_memory
    from memory.long_term import update_profile

    update_profile("user_with_dept", department="Finance")
    route = RouteDecision(agent=AgentName.MEMORY, reasoning="recall", memory_action="recall")
    result = _handle_memory(route, "user_with_dept", "What department do I belong to?")
    assert "Finance" in result.answer


def test_generate_report_falls_back_gracefully(monkeypatch):
    from agents import base as agent_base

    class _BoomLLM:
        def with_structured_output(self, schema):
            class _Boom:
                def invoke(self, *_a, **_k):
                    raise RuntimeError("simulated LLM failure")

            return _Boom()

    monkeypatch.setattr(agent_base, "get_llm", lambda temperature=0: _BoomLLM())
    report = agent_base.generate_report("hr", "What is the leave policy?", "You get 18 days.", [])
    assert report.department  # fallback still produces a valid TaskReport
    assert "You get 18 days" in report.task_summary
