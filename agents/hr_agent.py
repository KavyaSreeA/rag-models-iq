"""Module 2 — HR Agent: answers employee questions using RAG over HR policy docs."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import create_retriever_tool

from agents.base import AgentResult, build_react_agent, run_agent
from knowledge_base.retriever import get_retriever

SYSTEM_PROMPT = """You are the HR Agent for NovaTech Solutions Pvt. Ltd.
Answer employee questions about Leave Policy, Attendance Policy, Work From
Home Rules, Holiday List, Employee Benefits, Joining Process, and Exit
Policy. Always use the `search_hr_policies` tool to look up the relevant
policy before answering — do not rely on general knowledge. If the policy
documents don't cover the question, say so plainly and suggest the employee
contact HR directly. Keep answers concise and cite which policy the answer
comes from."""


def _build_agent():
    retriever = get_retriever("hr_policies", k=4)
    tool = create_retriever_tool(
        retriever,
        name="search_hr_policies",
        description="Search NovaTech's HR policy documents (leave, attendance, WFH, holidays, benefits, joining, exit).",
    )
    return build_react_agent([tool], SYSTEM_PROMPT)


def run_hr_agent(user_input: str, history: Optional[List[BaseMessage]] = None) -> AgentResult:
    agent = _build_agent()
    result = run_agent(agent, "hr", user_input, history)
    result.retrieved_docs = [
        {"tool": tc["name"], "query": tc["args"].get("query"), "excerpt": tc.get("result")}
        for tc in result.tool_calls
        if tc["name"] == "search_hr_policies"
    ]
    return result
