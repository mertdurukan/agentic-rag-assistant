"""Vector search — pgvector (Postgres).

Why pgvector?
- Postgres is already present in most products; no separate vector DB
  infrastructure or operational burden.
- ACID, backups, and SQL filtering come built in. MVP→prod transitions are
  less surprising.
- An alternative like Qdrant may be better at very large scale; unnecessary
  complexity at this size.

Embedding is managed by langchain_postgres.PGVector; we provide a thin wrapper
so the rest of the code stays decoupled from the library API.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.llm.provider import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "arxiv_chunks"


class VectorStore:
    """Dense (semantic) search on top of pgvector."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        from langchain_postgres import PGVector

        self._store = PGVector(
            embeddings=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection=self.settings.database_url,
            use_jsonb=True,
        )

    def add(self, docs: list[Document]) -> None:
        """Embed and write chunks. Idempotent (upsert) via chunk_id."""
        ids = [d.metadata["chunk_id"] for d in docs]
        # Write in batches to limit memory on large corpora.
        batch = 256
        for i in range(0, len(docs), batch):
            self._store.add_documents(docs[i : i + batch], ids=ids[i : i + batch])
            logger.info("Vectors written: %d/%d", min(i + batch, len(docs)), len(docs))

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return the top k similar chunks as (doc, similarity_score).

        Note: PGVector returns a distance (smaller=better); since RRF relies on
        ranks, the absolute score scale does not matter, only the order.
        """
        return self._store.similarity_search_with_score(query, k=k)
