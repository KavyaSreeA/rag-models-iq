"""Streamlit entrypoint — NovaTech Operations Console.

Employee Chat Interface (Module 1) wired up to the Coordinator Agent, with a
sidebar for knowledge-base management and conversation history, and
expandable per-turn detail panels (agent, tools, retrieved docs, report).

The visual design is a clean, light, professional theme driven mainly by
Streamlit's native theming (.streamlit/config.toml) — which reliably themes
every built-in widget (buttons, file uploader, radios, etc.) — plus a small
CSS layer on top for the header, section labels, and per-agent color chips.
"""
from __future__ import annotations

import uuid

import streamlit as st

import config
from knowledge_base.ingest import build_knowledge_base
from knowledge_base.retriever import collection_doc_count, list_uploaded_sources
from memory.long_term import get_profile
from memory.persistent import get_conversation_sessions, get_session_messages, log_turn
from memory.short_term import ShortTermMemory

st.set_page_config(page_title="NovaTech Operations Console", layout="wide")

# 1x1 transparent pixel — used to suppress Streamlit's built-in chat avatar
# icons entirely, so message identity comes only from the console-styled
# agent chip below rather than any icon/emoji-like glyph.
_BLANK_AVATAR = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
    "QVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

# --- Agent identity: label, short code, accent color ------------------------
# Colors chosen for readable contrast as text-on-light-tint chips (WCAG AA
# against a white/near-white background).
AGENT_META = {
    "hr": ("HR AGENT", "HR", "#2563eb"),
    "research": ("RESEARCH AGENT", "RS", "#7c3aed"),
    "email": ("EMAIL AGENT", "EM", "#059669"),
    "document": ("DOCUMENT AGENT", "DC", "#b45309"),
    "python": ("PYTHON TOOL AGENT", "PY", "#dc2626"),
    "memory": ("MEMORY", "MM", "#0891b2"),
    "sequential": ("SEQUENTIAL WORKFLOW", "SQ", "#db2777"),
    "parallel": ("PARALLEL WORKFLOW", "PL", "#ea580c"),
    "error": ("SYSTEM", "!!", "#b91c1c"),
}


def agent_meta(agent: str):
    return AGENT_META.get(agent, ("ASSISTANT", "AI", "#2563eb"))


# --- Light professional theme (layered on top of .streamlit/config.toml) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --panel-border: #e2e5ea;
        --text-primary: #1a1f28;
        --text-muted: #6b7280;
        --accent: #4338ca;
    }

    /* Header */
    .console-header {
        display: flex; align-items: baseline; gap: 14px;
        border-bottom: 1px solid var(--panel-border);
        padding-bottom: 14px; margin-bottom: 6px;
    }
    .console-header .mark {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 1.65rem; color: var(--text-primary); letter-spacing: 0.3px;
    }
    .console-header .mark span { color: var(--accent); }
    .console-header .sub {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
        color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.3px;
    }

    .console-strip {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        color: var(--text-muted); letter-spacing: 0.4px;
        margin-bottom: 18px; text-transform: uppercase;
    }

    /* Sidebar section labels */
    .panel-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
        color: var(--text-muted); text-transform: uppercase;
        letter-spacing: 1.3px; border-bottom: 1px solid var(--panel-border);
        padding-bottom: 6px; margin: 4px 0 10px 0;
    }

    /* Agent badge chip */
    .agent-chip {
        display: inline-flex; align-items: center; gap: 7px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase;
        padding: 3px 10px; border-radius: 4px; margin-bottom: 8px;
    }
    .agent-chip .dot { width: 7px; height: 7px; border-radius: 50%; }

    /* Chat message cards */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid var(--panel-border);
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }

    .stButton > button {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
        letter-spacing: 0.4px; border-radius: 5px;
    }

    /* Hide the blank-pixel chat avatar entirely — identity comes from the
       agent chip inside the message body instead. */
    div[data-testid="stChatMessage"] > img[alt$="avatar"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state init --------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_employee"
if "memory" not in st.session_state:
    st.session_state.memory = ShortTermMemory()
if "chat" not in st.session_state:
    st.session_state.chat = []  # list[dict]: role, content, meta


def render_agent_chip(agent: str) -> str:
    label, code, color = agent_meta(agent)
    return (
        f'<span class="agent-chip" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55;"><span class="dot" style="background:{color};"></span>'
        f"{label}</span>"
    )


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="panel-label">Knowledge Base</div>', unsafe_allow_html=True)

    key_ready = (config.LLM_PROVIDER == "groq" and config.GROQ_API_KEY) or (
        config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY
    )
    if not key_ready:
        st.warning(
            f"No API key found for LLM_PROVIDER={config.LLM_PROVIDER!r}. "
            f"Set it in your .env file."
        )

    uploaded = st.file_uploader(
        "Upload company documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    target_collection = st.radio(
        "Index into",
        options=["company_docs", "hr_policies"],
        format_func=lambda c: "Company Documents" if c == "company_docs" else "HR Policies",
        horizontal=False,
        label_visibility="collapsed",
    )
    if uploaded and st.button("Index uploaded documents"):
        from knowledge_base.ingest import add_documents

        tmp_dir = config.DATA_DIR / "_uploads"
        tmp_dir.mkdir(exist_ok=True)
        paths = []
        for f in uploaded:
            p = tmp_dir / f.name
            p.write_bytes(f.getbuffer())
            paths.append(p)
        with st.spinner("Indexing..."):
            n = add_documents(paths, collection=target_collection)
        st.success(f"Indexed {n} chunks into '{target_collection}'.")

    if st.button("Build knowledge base from sample_docs"):
        with st.spinner("Building knowledge base..."):
            summary = build_knowledge_base()
        st.success(f"HR: {summary['hr_policies']} chunks | Company: {summary['company_docs']} chunks")

    st.markdown(
        f'<div class="console-strip">HR policies: {collection_doc_count("hr_policies")} chunks '
        f'&nbsp;|&nbsp; Company docs: {collection_doc_count("company_docs")} chunks</div>',
        unsafe_allow_html=True,
    )

    with st.expander("View uploaded documents"):
        st.write("HR policies:", list_uploaded_sources("hr_policies") or "None yet")
        st.write("Company docs:", list_uploaded_sources("company_docs") or "None yet")

    st.markdown('<div class="panel-label">Previous Conversations</div>', unsafe_allow_html=True)
    sessions = get_conversation_sessions(limit=10)
    if not sessions:
        st.caption("No conversations logged yet.")
    for s in sessions:
        label = (s["preview"] or "(empty)")[:38]
        if st.button(label, key=f"sess_{s['session_id']}"):
            st.session_state.session_id = s["session_id"]
            msgs = get_session_messages(s["session_id"])
            st.session_state.chat = [{"role": m["role"], "content": m["content"], "meta": None} for m in msgs]
            st.rerun()

    if st.button("Clear chat", type="secondary"):
        st.session_state.chat = []
        st.session_state.memory.clear()
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown('<div class="panel-label">Employee Profile</div>', unsafe_allow_html=True)
    profile = get_profile(st.session_state.user_id)
    st.caption(f"{profile.name or 'Unknown'} | {profile.department or 'No department on file'}")

# --- Main chat interface ---------------------------------------------------
st.markdown(
    """
    <div class="console-header">
        <div class="mark">NOVATECH <span>// OPS</span></div>
        <div class="sub">Enterprise Operations AI Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="console-strip">HR / Research / Email / Document / Python — '
    "routed automatically by the Coordinator Agent</div>",
    unsafe_allow_html=True,
)

for turn in st.session_state.chat:
    with st.chat_message(turn["role"], avatar=_BLANK_AVATAR):
        meta = turn.get("meta")
        if meta and turn["role"] == "assistant":
            st.markdown(render_agent_chip(meta.get("agent", "")), unsafe_allow_html=True)
        st.markdown(turn["content"])
        if meta:
            with st.expander("Trace"):
                if meta.get("tool_calls"):
                    st.markdown("**Tool execution summary**")
                    st.json(meta["tool_calls"])
                if meta.get("retrieved_docs"):
                    st.markdown("**Retrieved documents**")
                    st.json(meta["retrieved_docs"])
                if meta.get("report"):
                    st.markdown("**Generated report**")
                    st.json(meta["report"])
                    st.download_button(
                        "Download report",
                        data=str(meta["report"]),
                        file_name="report.txt",
                        key=f"dl_{turn['content'][:20]}_{turn.get('ts', '')}",
                    )

user_input = st.chat_input("Type your request...")
if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input, "meta": None})
    st.session_state.memory.add_user(user_input)
    log_turn(st.session_state.session_id, "user", user_input)

    with st.chat_message("user", avatar=_BLANK_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=_BLANK_AVATAR):
        with st.spinner("Processing request..."):
            try:
                from agents.coordinator import handle_request

                coord_result = handle_request(
                    user_input,
                    history=st.session_state.memory.as_list()[:-1],
                    user_id=st.session_state.user_id,
                )
                agent_result = coord_result.agent_result
                answer = agent_result.answer
                meta = {
                    "agent": agent_result.agent,
                    "tool_calls": agent_result.tool_calls,
                    "retrieved_docs": agent_result.retrieved_docs,
                    "report": agent_result.report.model_dump() if agent_result.report else None,
                }
            except Exception as e:  # keep the app alive even if a call fails
                answer = f"Something went wrong handling that request: `{e}`"
                meta = {"agent": "error", "tool_calls": [], "retrieved_docs": [], "report": None}

        st.markdown(render_agent_chip(meta["agent"]), unsafe_allow_html=True)
        st.markdown(answer)
        with st.expander("Trace"):
            if meta["tool_calls"]:
                st.markdown("**Tool execution summary**")
                st.json(meta["tool_calls"])
            if meta["retrieved_docs"]:
                st.markdown("**Retrieved documents**")
                st.json(meta["retrieved_docs"])
            if meta["report"]:
                st.markdown("**Generated report**")
                st.json(meta["report"])

    st.session_state.memory.add_ai(answer)
    log_turn(st.session_state.session_id, "assistant", answer, agent=meta["agent"])
    st.session_state.chat.append({"role": "assistant", "content": answer, "meta": meta})
