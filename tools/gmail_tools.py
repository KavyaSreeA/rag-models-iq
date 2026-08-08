"""Module 4 — Email Agent tools: draft, read, summarize, and send via Gmail.

If `credentials.json` (Google OAuth client secret) is present, wires up
LangChain's real GmailToolkit. Otherwise falls back to a local stub so the
app remains fully runnable and demoable without any Google Cloud setup.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List

from langchain_core.tools import BaseTool, tool

import config


# --- Local stub tools (default / no Google creds) -----------------------

@tool("draft_email")
def _stub_draft_email(to: str, subject: str, body: str) -> str:
    """Draft an email (does not send). Returns the drafted email text."""
    return f"To: {to}\nSubject: {subject}\n\n{body}"


@tool("send_email")
def _stub_send_email(to: str, subject: str, body: str) -> str:
    """'Send' an email by appending it to the local outbox file (data/outbox.jsonl),
    since no real Gmail credentials are configured. Returns a confirmation string."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "to": to,
        "subject": subject,
        "body": body,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(config.OUTBOX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return f"[STUB MODE — no Gmail credentials configured] Saved to local outbox: {config.OUTBOX_PATH}"


@tool("read_recent_emails")
def _stub_read_recent_emails(max_results: int = 5) -> str:
    """Read the most recent emails. In stub mode, reads back the local outbox
    (since there is no real inbox without Gmail credentials)."""
    if not config.OUTBOX_PATH.exists():
        return "[STUB MODE] No emails found — outbox is empty and no Gmail credentials are configured."
    lines = config.OUTBOX_PATH.read_text(encoding="utf-8").splitlines()[-max_results:]
    return "\n---\n".join(lines) if lines else "[STUB MODE] Outbox is empty."


def _stub_tools() -> List[BaseTool]:
    return [_stub_draft_email, _stub_send_email, _stub_read_recent_emails]


# --- Real Gmail integration (used automatically once credentials.json exists) --

def _real_gmail_tools() -> List[BaseTool]:
    from langchain_google_community import GmailToolkit
    from langchain_google_community.gmail.utils import (
        build_resource_service,
        get_gmail_credentials,
    )

    creds = get_gmail_credentials(
        token_file=str(config.GOOGLE_TOKEN_PATH),
        client_secrets_file=str(config.GOOGLE_CREDENTIALS_PATH),
        scopes=config.GMAIL_SCOPES,
    )
    api_resource = build_resource_service(credentials=creds)
    toolkit = GmailToolkit(api_resource=api_resource)
    return toolkit.get_tools()


def get_gmail_tools() -> List[BaseTool]:
    if config.GMAIL_CONFIGURED:
        try:
            return _real_gmail_tools()
        except Exception:
            # Any auth/setup failure -> degrade gracefully to stub mode rather
            # than crashing the whole app.
            return _stub_tools()
    return _stub_tools()
