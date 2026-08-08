"""Central configuration: env loading, model settings, filesystem paths.

Every other module imports from here instead of touching os.environ / .env
directly, so switching providers or paths is a one-file change.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

APP_DB_PATH = BASE_DIR / os.getenv("APP_DB_PATH", "data/iqmath.db")
CHROMA_PERSIST_DIR = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
OUTBOX_PATH = DATA_DIR / "outbox.jsonl"
DRIVE_STORAGE_DIR = DATA_DIR / "drive_storage"

# --- LLM provider -------------------------------------------------------
# LLM_PROVIDER selects the chat model backend: "groq" (default) or "openai".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Groq has no embeddings endpoint. Default embeddings provider is the
# Hugging Face hosted Inference API — a lightweight API call, no torch/
# sentence-transformers/transformers install or local model download, which
# matters a lot on resource-limited free hosting (Streamlit Cloud, HF
# Spaces). Options:
#   "huggingface_api"   (default) hosted HF Inference API — needs HUGGINGFACE_API_TOKEN
#   "huggingface_local"  local sentence-transformers model — no key, but
#                         heavy (~1-2GB deps) and slow to cold-boot
#   "openai"              OpenAI embeddings — needs OPENAI_API_KEY
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface_api").lower()
HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# --- Optional integrations ------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GOOGLE_CREDENTIALS_PATH = BASE_DIR / os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
GOOGLE_TOKEN_PATH = BASE_DIR / os.getenv("GOOGLE_TOKEN_PATH", "token.json")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")

# Gmail/Drive are considered "configured" only if the OAuth client secret
# file actually exists on disk. Modules check this flag and fall back to
# local stubs automatically when it's False.
GMAIL_CONFIGURED = GOOGLE_CREDENTIALS_PATH.exists()
DRIVE_CONFIGURED = GOOGLE_CREDENTIALS_PATH.exists()

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
]
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def require_llm_key() -> None:
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file (get one at https://console.groq.com/keys)."
        )
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )


def require_openai_key() -> None:
    """Kept for embeddings/OpenAI-specific paths regardless of LLM_PROVIDER."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
