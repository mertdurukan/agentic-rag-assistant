"""Chunker testleri."""
from __future__ import annotations

from src.config import Settings
from src.ingestion.arxiv_loader import Paper
from src.ingestion.chunker import chunk_papers


def _paper(text_len: int = 3000) -> Paper:
    return Paper(
        paper_id="2401.00001",
        title="Test Paper",
        abstract="word " * (text_len // 5),
        authors=["A. Author"],
        categories=["cs.AI"],
        published="2024-01-01T00:00:00",
        pdf_url="http://x/pdf",
        entry_url="http://arxiv.org/abs/2401.00001",
    )


def test_chunk_ids_are_stable_and_unique():
    s = Settings(chunk_size=300, chunk_overlap=50)
    docs = chunk_papers([_paper()], s)
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(ids) == len(set(ids))  # benzersiz
    assert ids[0] == "2401.00001::0"  # stabil format


def test_chunks_carry_source_metadata():
    docs = chunk_papers([_paper()], Settings(chunk_size=300, chunk_overlap=50))
    meta = docs[0].metadata
    assert meta["paper_id"] == "2401.00001"
    assert meta["title"] == "Test Paper"
    assert "entry_url" in meta


def test_long_text_produces_multiple_chunks():
    docs = chunk_papers([_paper(4000)], Settings(chunk_size=300, chunk_overlap=50))
    assert len(docs) > 1
