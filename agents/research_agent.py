"""Module 3 — Research Agent: external research via web search + Wikipedia."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import BaseMessage

from agents.base import AgentResult, build_react_agent, run_agent
from tools.search_tools import get_search_tools

SYSTEM_PROMPT = """You are the Research Agent for NovaTech Solutions Pvt. Ltd.
You perform external research using web search and Wikipedia tools, then
summarize findings clearly for a business audience. Always use the search
tools rather than answering from memory for anything time-sensitive or
factual (news, product launches, trends, company info). Cite sources by
name. Keep the summary well-organized with short paragraphs or bullet points."""


def _build_agent():
    return build_react_agent(get_search_tools(), SYSTEM_PROMPT)


def run_research_agent(user_input: str, history: Optional[List[BaseMessage]] = None) -> AgentResult:
    agent = _build_agent()
    return run_agent(agent, "research", user_input, history)
