"""Module 13 — Python Tool Agent: calculations, tables, reports, CSV processing."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import BaseMessage

from agents.base import AgentResult, build_react_agent, run_agent
from tools.python_tool import get_python_tool

SYSTEM_PROMPT = """You are the Python Tool Agent for NovaTech Solutions Pvt. Ltd.
You handle requests that need calculations, table formatting, report
generation, or CSV/pandas data processing (e.g. "calculate employee
attendance percentage", "create a summary table"). Use the python_repl_ast
tool to write and run short Python snippets, then explain the result in
plain language alongside the raw output."""


def _build_agent():
    return build_react_agent([get_python_tool()], SYSTEM_PROMPT)


def run_python_agent(user_input: str, history: Optional[List[BaseMessage]] = None) -> AgentResult:
    agent = _build_agent()
    return run_agent(agent, "python", user_input, history)
