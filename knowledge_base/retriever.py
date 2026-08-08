"""Chroma vector store + retriever accessors, shared by ingest and agents."""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

import config


@lru_cache(maxsize=None)
def _embeddings() -> Embeddings:
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        config.require_openai_key()
        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)

    if config.EMBEDDING_PROVIDER == "huggingface_local":
        # Local sentence-transformers model — free, no API key, but pulls in
        # torch/transformers (~1-2GB) and is slow to cold-boot on limited
        # free hosting. Prefer huggingface_api unless running fully offline.
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_EMBEDDING_MODEL)

    # Default: Hugging Face's hosted Inference API — a lightweight HTTP call,
    # no heavy local ML deps or model download. Chosen since Groq has no
    # embeddings endpoint of its own. Uses langchain_huggingface (not the
    # deprecated langchain_community.HuggingFaceInferenceAPIEmbeddings,
    # which points at the now-defunct api-inference.huggingface.co host).
    if not config.HUGGINGFACE_API_TOKEN:
        raise RuntimeError(
            "HUGGINGFACE_API_TOKEN is not set. Get a free token at "
            "https://huggingface.co/settings/tokens and add it to your .env "
            "file (or Streamlit secrets), or set EMBEDDING_PROVIDER=huggingface_local "
            "to use a local model instead."
        )
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model=config.HUGGINGFACE_EMBEDDING_MODEL,
        huggingfacehub_api_token=config.HUGGINGFACE_API_TOKEN,
    )


@lru_cache(maxsize=None)
def get_vectorstore(collection: str) -> Chroma:
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=_embeddings(),
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )


def get_retriever(collection: str, k: int = 4) -> VectorStoreRetriever:
    return get_vectorstore(collection).as_retriever(search_kwargs={"k": k})


def collection_doc_count(collection: str) -> int:
    try:
        return get_vectorstore(collection)._collection.count()  # noqa: SLF001
    except Exception:
        return 0


def list_uploaded_sources(collection: str) -> list[str]:
    """Return the distinct source filenames indexed in a collection."""
    try:
        store = get_vectorstore(collection)
        data = store.get(include=["metadatas"])
        sources = {m.get("source") for m in data.get("metadatas", []) if m and m.get("source")}
        return sorted(sources)
    except Exception:
        return []
