"""Module 5 — Document Agent: search/answer over uploaded company documents
(handbook, SOPs, project docs, manuals) — separate collection from HR policies."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import create_retriever_tool

from agents.base import AgentResult, build_react_agent, run_agent
from knowledge_base.retriever import get_retriever

SYSTEM_PROMPT = """You are the Document Agent for NovaTech Solutions Pvt. Ltd.
You answer questions using the company's uploaded documents: the Employee
Handbook, Company Guidelines, Project Documentation, and SOP Documents.
Always call `search_company_documents` to find relevant passages before
answering, and quote or paraphrase the source document. If nothing relevant
is found, say so and suggest the employee upload the document or ask HR/IT."""


def _build_agent():
    retriever = get_retriever("company_docs", k=4)
    tool = create_retriever_tool(
        retriever,
        name="search_company_documents",
        description="Search uploaded company documents: employee handbook, guidelines, project docs, SOPs, manuals.",
    )
    return build_react_agent([tool], SYSTEM_PROMPT)


def run_document_agent(user_input: str, history: Optional[List[BaseMessage]] = None) -> AgentResult:
    agent = _build_agent()
    result = run_agent(agent, "document", user_input, history)
    result.retrieved_docs = [
        {"tool": tc["name"], "query": tc["args"].get("query"), "excerpt": tc.get("result")}
        for tc in result.tool_calls
        if tc["name"] == "search_company_documents"
    ]
    return result
