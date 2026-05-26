"""BM25 keyword (sparse) search.

Why put BM25 alongside vector search?
- Dense embeddings capture semantic similarity but miss EXACT-TERM matches:
  model names, abbreviations, symbol/code terms (e.g. "MoE", "LoRA", "RoPE").
- BM25 is strong on these rare-but-critical exact matches.
- Combining the two (hybrid) raises recall above either one alone.

In-memory BM25 (rank-bm25) is sufficient for the MVP. In production, this
moves to Postgres FTS / OpenSearch — the interface stays the same.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 keyword search over a chunk corpus."""

    def __init__(self, docs: list[Document]) -> None:
        if not docs:
            raise ValueError("BM25Index cannot be built on an empty corpus.")
        self._docs = docs
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in docs])

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return the top-k chunks by BM25 score."""
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._docs, scores, strict=False), key=lambda x: x[1], reverse=True
        )
        return ranked[:k]
