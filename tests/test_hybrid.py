"""RRF fusion — deterministic, pure-logic tests (no external dependencies)."""
from __future__ import annotations

from langchain_core.documents import Document

from src.retrieval.hybrid import reciprocal_rank_fusion


def _doc(cid: str) -> Document:
    return Document(page_content=cid, metadata={"chunk_id": cid})


def test_rrf_rewards_agreement_across_lists():
    # A is on top in both lists → it must receive the highest RRF score.
    list1 = [_doc("A"), _doc("B"), _doc("C")]
    list2 = [_doc("A"), _doc("C"), _doc("D")]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert fused[0][0].metadata["chunk_id"] == "A"


def test_rrf_deduplicates_documents():
    list1 = [_doc("A"), _doc("B")]
    list2 = [_doc("A"), _doc("B")]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    ids = [d.metadata["chunk_id"] for d, _ in fused]
    assert sorted(ids) == ["A", "B"]


def test_rrf_score_formula():
    fused = reciprocal_rank_fusion([[_doc("X")]], k=60)
    # single list, rank 0 → 1/(60+0)
    assert abs(fused[0][1] - (1 / 60)) < 1e-9


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([], k=60) == []
