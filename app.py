"""Streamlit entrypoint — NovaTech Operations Console.

Employee Chat Interface (Module 1) wired up to the Coordinator Agent, with a
sidebar for knowledge-base management and conversation history, and
expandable per-turn detail panels (agent, tools, retrieved docs, report).

The visual design is a custom "ops console" theme (dark panel, monospace
labels, per-agent color coding) injected via CSS rather than Streamlit's
default look.
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
AGENT_META = {
    "hr": ("HR AGENT", "HR", "#5b8def"),
    "research": ("RESEARCH AGENT", "RS", "#a683f2"),
    "email": ("EMAIL AGENT", "EM", "#3fbf8f"),
    "document": ("DOCUMENT AGENT", "DC", "#e0a940"),
    "python": ("PYTHON TOOL AGENT", "PY", "#e0625a"),
    "memory": ("MEMORY", "MM", "#3fc4c4"),
    "sequential": ("SEQUENTIAL WORKFLOW", "SQ", "#d868a8"),
    "parallel": ("PARALLEL WORKFLOW", "PL", "#e08a3f"),
    "error": ("SYSTEM", "!!", "#c94b4b"),
}


def agent_meta(agent: str):
    return AGENT_META.get(agent, ("ASSISTANT", "AI", "#5b8def"))


# --- Custom console theme ---------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --console-bg: #0b0e13;
        --console-panel: #12161d;
        --console-panel-alt: #161b23;
        --console-border: #232a35;
        --console-text: #d9e1ea;
        --console-muted: #7c8898;
        --console-accent: #4fd1c5;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: var(--console-bg); }

    section[data-testid="stSidebar"] {
        background-color: var(--console-panel);
        border-right: 1px solid var(--console-border);
    }

    /* Console header */
    .console-header {
        display: flex; align-items: baseline; gap: 14px;
        border-bottom: 1px solid var(--console-border);
        padding-bottom: 14px; margin-bottom: 6px;
    }
    .console-header .mark {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 1.65rem; color: var(--console-text); letter-spacing: 0.5px;
    }
    .console-header .mark span { color: var(--console-accent); }
    .console-header .sub {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
        color: var(--console-muted); text-transform: uppercase; letter-spacing: 1.5px;
    }

    .console-strip {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        color: var(--console-muted); letter-spacing: 0.5px;
        margin-bottom: 18px; text-transform: uppercase;
    }

    /* Sidebar section labels */
    .panel-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
        color: var(--console-muted); text-transform: uppercase;
        letter-spacing: 1.5px; border-bottom: 1px solid var(--console-border);
        padding-bottom: 6px; margin: 4px 0 10px 0;
    }

    /* Agent badge chip */
    .agent-chip {
        display: inline-flex; align-items: center; gap: 7px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
        padding: 3px 10px; border-radius: 3px; margin-bottom: 8px;
    }
    .agent-chip .dot { width: 7px; height: 7px; border-radius: 50%; }

    /* Assistant response card */
    .response-card {
        border-left: 3px solid var(--console-border);
        padding: 2px 0 2px 14px; margin: 4px 0 10px 0;
    }

    div[data-testid="stChatMessage"] {
        background-color: var(--console-panel-alt);
        border: 1px solid var(--console-border);
        border-radius: 6px;
    }

    .stButton > button {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
        letter-spacing: 0.5px; border-radius: 3px;
        background-color: var(--console-panel-alt);
        border: 1px solid var(--console-border); color: var(--console-text);
    }
    .stButton > button:hover { border-color: var(--console-accent); color: var(--console-accent); }

    .stChatInputContainer, div[data-testid="stChatInput"] {
        border-color: var(--console-border) !important;
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
