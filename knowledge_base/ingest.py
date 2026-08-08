"""Module 10 — Company Knowledge Base.

Loads PDFs/TXT files, splits them, and writes them into a persisted Chroma
vector store. Two logical collections are used:

  - "hr_policies"  -> fed to the HR Agent (leave/attendance/WFH/... policies)
  - "company_docs" -> fed to the Document Agent (handbook, SOPs, project docs)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from knowledge_base.retriever import get_vectorstore

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _load_file(path: Path) -> List[Document]:
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    if path.suffix.lower() in (".txt", ".md"):
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported document type: {path.suffix} ({path.name})")


def add_documents(paths: Iterable[Path], collection: str = "company_docs") -> int:
    """Load, split, and index the given files into the given Chroma collection.

    Returns the number of chunks written.
    """
    docs: List[Document] = []
    for path in paths:
        path = Path(path)
        loaded = _load_file(path)
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)

    if not docs:
        return 0

    chunks = _SPLITTER.split_documents(docs)
    store = get_vectorstore(collection)
    store.add_documents(chunks)
    return len(chunks)


def build_knowledge_base(sample_docs_dir: Path = config.SAMPLE_DOCS_DIR) -> dict:
    """(Re)build both collections from the on-disk sample_docs folder.

    Layout expected:
        data/sample_docs/hr/*.txt|*.pdf          -> "hr_policies"
        data/sample_docs/company/*.txt|*.pdf     -> "company_docs"
    """
    summary = {"hr_policies": 0, "company_docs": 0}

    hr_dir = sample_docs_dir / "hr"
    if hr_dir.exists():
        files = [p for p in hr_dir.iterdir() if p.suffix.lower() in (".txt", ".md", ".pdf")]
        summary["hr_policies"] = add_documents(files, collection="hr_policies")

    company_dir = sample_docs_dir / "company"
    if company_dir.exists():
        files = [p for p in company_dir.iterdir() if p.suffix.lower() in (".txt", ".md", ".pdf")]
        summary["company_docs"] = add_documents(files, collection="company_docs")

    return summary
