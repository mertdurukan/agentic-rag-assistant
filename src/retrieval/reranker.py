"""Cross-encoder re-ranking.

Why re-ranking?
- Initial retrieval (bi-encoder + BM25) is fast but "broad and noisy": query and
  document are encoded SEPARATELY, missing interaction effects.
- The cross-encoder feeds the query+doc pair to the model JOINTLY → much more
  accurate relevance scoring. It is expensive, so it runs on only the top-N
  candidates, not the full corpus.
- Classic two-stage setup: cheap/wide retrieve → expensive/sharp re-rank.
  Precision improves notably; the context the LLM sees is cleaner (which also
  helps faithfulness).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from src.config import get_settings


@lru_cache
def _get_cross_encoder(model_name: str):
    """Load the cross-encoder model once (loading is expensive)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Re-sort candidates with the cross-encoder; return the best top_n."""
    if not docs:
        return []
    settings = get_settings()
    model = _get_cross_encoder(settings.reranker_model)

    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores, strict=False), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]
