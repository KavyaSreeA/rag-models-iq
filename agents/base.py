"""Shared helpers used by every specialized agent.

Each agent is a small LangGraph ReAct agent (tool-calling loop) built with
`create_react_agent`, plus a second lightweight LLM call that turns the
agent's final answer into a structured `TaskReport` (Module 12).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

import config
from schemas import TaskReport


@dataclass
class AgentResult:
    agent: str
    answer: str
    tool_calls: List[dict] = field(default_factory=list)
    retrieved_docs: List[dict] = field(default_factory=list)
    report: Optional[TaskReport] = None


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    config.require_llm_key()
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=config.GROQ_CHAT_MODEL, temperature=temperature, api_key=config.GROQ_API_KEY)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.OPENAI_CHAT_MODEL, temperature=temperature, api_key=config.OPENAI_API_KEY)


def build_react_agent(tools: List[BaseTool], system_prompt: str):
    llm = get_llm()
    return create_react_agent(llm, tools=tools, prompt=system_prompt)


def run_agent(
    agent,
    agent_name: str,
    user_input: str,
    history: Optional[List[BaseMessage]] = None,
    build_report: bool = True,
) -> AgentResult:
    history = history or []
    messages = [*history, HumanMessage(content=user_input)]
    result: dict[str, Any] = agent.invoke({"messages": messages})
    out_messages: List[BaseMessage] = result["messages"]

    final_answer = ""
    tool_calls: List[dict] = []
    for m in out_messages:
        if isinstance(m, AIMessage) and m.content:
            final_answer = m.content
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tool_calls.append({"name": tc.get("name"), "args": tc.get("args")})
        if isinstance(m, ToolMessage):
            for tc in tool_calls:
                if tc.get("name") == m.name and "result" not in tc:
                    tc["result"] = str(m.content)[:500]
                    break

    report = generate_report(agent_name, user_input, final_answer, tool_calls) if build_report else None

    return AgentResult(agent=agent_name, answer=final_answer, tool_calls=tool_calls, report=report)


def generate_report(agent_name: str, user_input: str, answer: str, tool_calls: List[dict]) -> TaskReport:
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(TaskReport)
    prompt = (
        f"Agent: {agent_name}\n"
        f"Employee request: {user_input}\n"
        f"Agent's answer: {answer}\n"
        f"Tools used: {tool_calls}\n\n"
        "Produce a concise structured task report for this interaction."
    )
    try:
        return structured_llm.invoke(prompt)
    except Exception:
        return TaskReport(
            task_summary=answer[:280] or "No summary available.",
            department="General",
            actions_taken=[f"Handled by {agent_name} agent"],
        )
