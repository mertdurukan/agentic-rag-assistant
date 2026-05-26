"""Ingestion entrypoint: arXiv → chunk → pgvector + BM25 corpus.

Usage:      python -m scripts.ingest
Idempotent: upserts on chunk_id; re-running does not create duplicates.
"""
from __future__ import annotations

import logging

from src.config import get_settings
from src.ingestion.arxiv_loader import fetch_papers, save_papers
from src.ingestion.chunker import chunk_papers
from src.ingestion.persist import save_chunks
from src.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAPERS_PATH = "data/papers.jsonl"
CHUNKS_PATH = "data/chunks.jsonl"


def main() -> None:
    settings = get_settings()

    logger.info("1/4 Fetching papers from arXiv...")
    papers = fetch_papers(settings)
    save_papers(papers, PAPERS_PATH)

    logger.info("2/4 Chunking...")
    chunks = chunk_papers(papers, settings)
    save_chunks(chunks, CHUNKS_PATH)
    logger.info("Total %d chunks.", len(chunks))

    logger.info("3/4 Writing to pgvector (embedding)...")
    store = VectorStore(settings)
    store.add(chunks)

    logger.info("4/4 Done. BM25 corpus: %s", CHUNKS_PATH)


if __name__ == "__main__":
    main()
