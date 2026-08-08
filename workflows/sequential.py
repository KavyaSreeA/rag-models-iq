"""Module 7 — Sequential Workflow (RunnableSequence via LCEL `|`):

    User Request -> Research -> Summarize -> Generate Email -> Send Email -> Store Conversation

Triggered by the Coordinator when a request chains multiple actions, e.g.
"Summarize the employee handbook and email it to HR." The pipeline decides
whether the "research" step should hit the Document Agent (internal docs) or
the Research Agent (external info) based on the wording of the request.
"""
from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from agents.base import AgentResult, generate_report, get_llm
from agents.document_agent import run_document_agent
from agents.email_agent import run_email_agent
from agents.research_agent import run_research_agent
from memory.persistent import log_turn

_INTERNAL_HINTS = ("handbook", "document", "sop", "guideline", "policy", "uploaded")

_EMAIL_ADDR_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _step_gather_info(state: dict) -> dict:
    request = state["request"]
    use_internal = any(h in request.lower() for h in _INTERNAL_HINTS)
    result: AgentResult = run_document_agent(request) if use_internal else run_research_agent(request)
    state["gathered"] = result.answer
    state["gather_source"] = "document" if use_internal else "research"
    return state


def _step_summarize(state: dict) -> dict:
    llm = get_llm(temperature=0.2)
    prompt = (
        "Summarize the following content in 3-5 concise sentences suitable for "
        f"an internal business email:\n\n{state['gathered']}"
    )
    state["summary"] = llm.invoke(prompt).content
    return state


def _step_generate_email(state: dict) -> dict:
    llm = get_llm(temperature=0.3)
    to = state.get("recipient") or "hr@novatech.example"
    prompt = (
        f"Write a short, professional internal email to {to} with subject and body, "
        f"based on this summary. Reply in the exact format:\nSUBJECT: <subject>\nBODY: <body>\n\n"
        f"Summary:\n{state['summary']}"
    )
    text = llm.invoke(prompt).content
    subject, body = "Summary Report", text
    if "SUBJECT:" in text and "BODY:" in text:
        subject = text.split("SUBJECT:", 1)[1].split("BODY:", 1)[0].strip()
        body = text.split("BODY:", 1)[1].strip()
    state["email_subject"] = subject
    state["email_body"] = body
    return state


def _step_send_email(state: dict) -> dict:
    to = state.get("recipient") or "hr@novatech.example"
    instruction = (
        f"Send an email to {to} with subject '{state['email_subject']}' and this body:\n"
        f"{state['email_body']}"
    )
    result = run_email_agent(instruction)
    state["send_result"] = result.answer
    return state


def _step_store_conversation(state: dict) -> dict:
    session_id = state.get("session_id", "sequential_workflow")
    log_turn(session_id, "user", state["request"])
    log_turn(
        session_id,
        "assistant",
        f"[sequential] gathered via {state['gather_source']}; emailed subject '{state['email_subject']}'",
        agent="sequential",
    )
    return state


_PIPELINE = (
    RunnableLambda(_step_gather_info)
    | RunnableLambda(_step_summarize)
    | RunnableLambda(_step_generate_email)
    | RunnableLambda(_step_send_email)
    | RunnableLambda(_step_store_conversation)
)


def run_sequential_workflow(
    user_input: str,
    history: Optional[List[BaseMessage]] = None,
    recipient: Optional[str] = None,
    session_id: str = "sequential_workflow",
) -> AgentResult:
    recipient_match = _EMAIL_ADDR_RE.search(user_input)
    state = {
        "request": user_input,
        "recipient": recipient or (recipient_match.group(0) if recipient_match else None),
        "session_id": session_id,
    }
    final_state = _PIPELINE.invoke(state)

    answer = (
        f"**Summary:** {final_state['summary']}\n\n"
        f"**Email drafted** (subject: \"{final_state['email_subject']}\") "
        f"to {final_state.get('recipient') or 'hr@novatech.example'}.\n\n"
        f"**Send result:** {final_state['send_result']}"
    )
    report = generate_report(
        "sequential",
        user_input,
        answer,
        [{"name": "gather", "args": {"source": final_state["gather_source"]}}, {"name": "email", "args": {}}],
    )
    return AgentResult(agent="sequential", answer=answer, report=report)
