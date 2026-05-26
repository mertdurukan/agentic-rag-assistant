"""Retrieval orchestration.

Bundles two modes behind a single interface:
- "baseline": vector search only (reference for comparison).
- "hybrid":   vector + BM25 → RRF → cross-encoder re-rank (the main system).

Using the same interface lets the baseline-vs-hybrid comparison in benchmarks/
be fair and live on a single code path (apples-to-apples).
"""
from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.vector_store import VectorStore

RetrievalMode = Literal["baseline", "hybrid"]


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def retrieve(
        self, query: str, mode: RetrievalMode = "hybrid"
    ) -> list[Document]:
        s = self.settings
        # Explicit mode validation: a typo in a notebook/SDK call must not
        # silently fall back to hybrid (FastAPI Literal and the Gradio Radio
        # protect runtime, but we also need to protect the public Python API).
        if mode not in ("baseline", "hybrid"):
            raise ValueError(
                f"Unknown retrieval mode: {mode!r}. Must be 'baseline' or 'hybrid'."
            )

        vector_hits = self.vector_store.search(query, k=s.retrieval_top_k)

        if mode == "baseline":
            return [d for d, _ in vector_hits][: s.rerank_top_n]

        if self.bm25_index is None:
            raise ValueError("hybrid mode requires a BM25 index.")
        bm25_hits = self.bm25_index.search(query, k=s.retrieval_top_k)
        fused = hybrid_search(query, vector_hits, bm25_hits, k=s.rrf_k)
        return rerank(query, fused, top_n=s.rerank_top_n)
