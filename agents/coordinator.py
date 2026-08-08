"""Module 6 — Coordinator Agent, Module 9 — Conditional Routing.

Classifies each employee request into a `RouteDecision` (structured output),
then dispatches via a `RunnableBranch` to the right specialized agent or
workflow. This is the single entrypoint the Streamlit app calls per turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

from agents.base import AgentResult, get_llm
from agents.document_agent import run_document_agent
from agents.email_agent import run_email_agent
from agents.hr_agent import run_hr_agent
from agents.python_tool_agent import run_python_agent
from agents.research_agent import run_research_agent
from memory.long_term import get_profile, remember_fact
from schemas import AgentName, RouteDecision, TaskReport

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the routing classifier for NovaTech's Enterprise Operations AI
Assistant. Read the employee's request and decide which single agent or
workflow should handle it:

- hr: leave/attendance/WFH/holiday/benefits/joining/exit policy questions.
- research: external/internet research about ONE subject/company/topic.
- parallel: research that names MULTIPLE independent subjects to research
  and compare (e.g. "Research Google, Microsoft, and OpenAI").
- email: drafting, reading, summarizing, or sending email.
- document: questions about uploaded company documents (handbook, SOPs,
  guidelines, project docs) that are not HR policy specific.
- python: calculations, tables, CSV/pandas processing, report generation.
- memory: the employee is telling you a fact to remember about themselves
  (memory_action='remember', with memory_key one of
  name/department/email_style and memory_value the stated value), OR asking
  you to recall something about themselves, e.g. "what department do I
  belong to?" (memory_action='recall').
- sequential: the request chains multiple distinct actions together, e.g.
  "summarize the handbook and email it to HR" (research/document -> summarize
  -> email -> store).

Pick exactly one. For 'research'/'parallel', list the distinct subjects in
`entities`.""",
        ),
        ("human", "{input}"),
    ]
)


@dataclass
class CoordinatorResult:
    route: RouteDecision
    agent_result: AgentResult


def classify(user_input: str) -> RouteDecision:
    llm = get_llm(temperature=0)
    chain = _CLASSIFIER_PROMPT | llm.with_structured_output(RouteDecision)
    return chain.invoke({"input": user_input})


def _handle_memory(route: RouteDecision, user_id: str, user_input: str) -> AgentResult:
    if route.memory_action == "remember" and route.memory_key and route.memory_value:
        profile = remember_fact(user_id, route.memory_key, route.memory_value)
        answer = f"Got it — I'll remember that your {route.memory_key} is {route.memory_value}."
    else:
        profile = get_profile(user_id)
        parts = []
        if profile.name:
            parts.append(f"name: {profile.name}")
        if profile.department:
            parts.append(f"department: {profile.department}")
        if profile.preferred_email_style:
            parts.append(f"preferred email style: {profile.preferred_email_style}")
        answer = (
            "Here's what I have on file — " + ", ".join(parts)
            if parts
            else "I don't have anything saved about you yet. Tell me, e.g. 'Remember that I belong to the Finance Department.'"
        )
    report = TaskReport(
        task_summary=answer,
        department=profile.department or "General",
        actions_taken=[f"memory:{route.memory_action}"],
    )
    return AgentResult(agent="memory", answer=answer, report=report)


def _build_branch(user_input: str, history: Optional[List[BaseMessage]], user_id: str):
    """Module 9 — RunnableBranch keyed on the classifier's chosen agent label.

    Each branch is a RunnableLambda so agents are only invoked (and pay their
    LLM cost) for the branch that's actually selected.
    """
    return RunnableBranch(
        (lambda r: r.agent == AgentName.HR, RunnableLambda(lambda r: run_hr_agent(user_input, history))),
        (lambda r: r.agent == AgentName.RESEARCH, RunnableLambda(lambda r: run_research_agent(user_input, history))),
        (lambda r: r.agent == AgentName.EMAIL, RunnableLambda(lambda r: run_email_agent(user_input, history))),
        (lambda r: r.agent == AgentName.DOCUMENT, RunnableLambda(lambda r: run_document_agent(user_input, history))),
        (lambda r: r.agent == AgentName.PYTHON, RunnableLambda(lambda r: run_python_agent(user_input, history))),
        (lambda r: r.agent == AgentName.MEMORY, RunnableLambda(lambda r: _handle_memory(r, user_id, user_input))),
        (
            lambda r: r.agent == AgentName.PARALLEL,
            RunnableLambda(lambda r: _run_parallel(r, user_input, history)),
        ),
        (
            lambda r: r.agent == AgentName.SEQUENTIAL,
            RunnableLambda(lambda r: _run_sequential(r, user_input, history)),
        ),
        # default fallback
        RunnableLambda(lambda r: run_document_agent(user_input, history)),
    )


def _run_parallel(route: RouteDecision, user_input: str, history) -> AgentResult:
    from workflows.parallel import run_parallel_research

    return run_parallel_research(route.entities or [user_input], user_input)


def _run_sequential(route: RouteDecision, user_input: str, history) -> AgentResult:
    from workflows.sequential import run_sequential_workflow

    return run_sequential_workflow(user_input, history)


def handle_request(
    user_input: str,
    history: Optional[List[BaseMessage]] = None,
    user_id: str = "default_user",
) -> CoordinatorResult:
    route = classify(user_input)
    branch = _build_branch(user_input, history, user_id)
    agent_result = branch.invoke(route)
    return CoordinatorResult(route=route, agent_result=agent_result)
