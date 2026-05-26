"""Hybrid retrieval — Reciprocal Rank Fusion (RRF).

Why RRF?
- BM25 and vector scores are on DIFFERENT scales (one probability-like, the
  other distance-based). Adding scores directly is misleading due to scale mismatch.
- RRF uses RANK, not score: score(d) = Σ 1/(k + rank_i(d)). It is scale-independent,
  has a single parameter (k), and is empirically very robust (proven on TREC).
- The k constant softens sharpness at top ranks; k=60 is a common, strong default.

Output: a deduplicated list of Documents in descending RRF order.
"""
from __future__ import annotations

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[tuple[Document, float]]:
    """Fuse several ranked lists via RRF.

    Args:
        ranked_lists: Each list is internally sorted by relevance.
        k: RRF constant.

    Returns:
        (Document, rrf_score) pairs, sorted by score descending.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            cid = doc.metadata["chunk_id"]
            by_id.setdefault(cid, doc)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(by_id[cid], score) for cid, score in fused]


def hybrid_search(
    query: str,
    vector_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    k: int = 60,
) -> list[Document]:
    """Fuse vector and BM25 results via RRF and return a sorted Document list."""
    vec_docs = [d for d, _ in vector_results]
    bm25_docs = [d for d, _ in bm25_results]
    fused = reciprocal_rank_fusion([vec_docs, bm25_docs], k=k)
    return [doc for doc, _ in fused]
