"""Chunking.

Decision: RecursiveCharacterTextSplitter (with overlap).
Why?
- arXiv abstracts are short and dense; semantic chunking's marginal benefit
  does not justify the extra LLM/embedding cost and complexity. A recursive
  splitter slices along the paragraph→sentence→word hierarchy, is deterministic,
  and is fast.
- Overlap (chunk_overlap) reduces meaning loss at boundaries: even if a sentence
  is split across two chunks, context carries over.
- For full-text ingestion, the same interface can swap to a semantic splitter.

Each chunk carries a stable `chunk_id` — critical for matching in RRF and eval.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, get_settings
from src.ingestion.arxiv_loader import Paper


def build_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    settings = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_papers(
    papers: list[Paper], settings: Settings | None = None
) -> list[Document]:
    """Convert a list of papers into LangChain Document chunks.

    page_content = chunk text; metadata = source traceability (for citations).
    """
    settings = settings or get_settings()
    splitter = build_splitter(settings)
    docs: list[Document] = []

    for paper in papers:
        pieces = splitter.split_text(paper.text)
        for idx, piece in enumerate(pieces):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "chunk_id": f"{paper.paper_id}::{idx}",
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "entry_url": paper.entry_url,
                        "chunk_index": idx,
                    },
                )
            )
    return docs
