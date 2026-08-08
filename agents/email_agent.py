"""Module 4 — Email Agent: draft, read, summarize, and send emails."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import BaseMessage

import config
from agents.base import AgentResult, build_react_agent, run_agent
from tools.gmail_tools import get_gmail_tools

SYSTEM_PROMPT = """You are the Email Agent for NovaTech Solutions Pvt. Ltd.
You draft professional emails, read/summarize recent emails, and send emails
using the available tools. Always draft in a professional, courteous tone
appropriate for internal company communication unless told otherwise. Before
sending, make sure the draft includes a clear subject line and a complete
body. If the user only asks to draft (not send), do not call the send tool."""


def _build_agent():
    return build_react_agent(get_gmail_tools(), SYSTEM_PROMPT)


def run_email_agent(user_input: str, history: Optional[List[BaseMessage]] = None) -> AgentResult:
    agent = _build_agent()
    result = run_agent(agent, "email", user_input, history)
    if not config.GMAIL_CONFIGURED:
        result.answer += (
            "\n\n_Note: Gmail credentials aren't configured, so this ran in stub mode "
            "(saved locally instead of sending real email). See README for setup._"
        )
    return result
