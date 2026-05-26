"""Chunk persistence.

Vectors live in pgvector; BM25 is in-memory, so it is rebuilt from the chunk
texts on every startup. For that reason, we persist the chunks as JSONL.
This keeps the single source of truth: vectors and BM25 are fed from the same
chunk set.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def save_chunks(docs: list[Document], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(
                json.dumps(
                    {"page_content": d.page_content, "metadata": d.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_chunks(path: str | Path) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                docs.append(
                    Document(page_content=rec["page_content"], metadata=rec["metadata"])
                )
    return docs
