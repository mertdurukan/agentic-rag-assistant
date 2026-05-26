"""Config and node helpers — pure-logic tests."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.agent.nodes import _format_context, _parse_json, make_router
from src.config import Settings


def test_category_list_parsing():
    s = Settings(arxiv_categories="cs.AI, cs.LG ,cs.CL")
    assert s.arxiv_category_list == ["cs.AI", "cs.LG", "cs.CL"]


def test_langfuse_disabled_without_keys():
    assert Settings().langfuse_enabled is False
    assert Settings(langfuse_public_key="pk", langfuse_secret_key="sk").langfuse_enabled


def test_parse_json_extracts_embedded_object():
    raw = "Sure! Here is the result:\n```json\n{\"faithful\": true}\n``` done"
    assert _parse_json(raw) == {"faithful": True}


def test_parse_json_returns_empty_on_garbage():
    assert _parse_json("no json here") == {}


def test_format_context_tags_paper_ids():
    docs = [
        Document(page_content="body text", metadata={"paper_id": "2401.1", "title": "T"})
    ]
    out = _format_context(docs)
    assert "[2401.1]" in out and "body text" in out


# --------------------------------------------------------------------------- #
# Router behavior: tests that pin down the max_regen_attempts semantics.
# Value 1 → 1 main generation + 1 regen (2 LLM calls total); 0 → no regen.
# --------------------------------------------------------------------------- #
def test_router_ends_when_faithful():
    route = make_router(Settings(max_regen_attempts=1))
    assert route({"faithful": True, "attempts": 1}) == "end"


def test_router_regenerates_once_when_not_faithful():
    route = make_router(Settings(max_regen_attempts=1))
    # After the first generate, attempts=1 → regen is allowed.
    assert route({"faithful": False, "attempts": 1}) == "regenerate"
    # After the second generate, attempts=2 → no retries left.
    assert route({"faithful": False, "attempts": 2}) == "end"


def test_router_no_regen_when_max_is_zero():
    route = make_router(Settings(max_regen_attempts=0))
    assert route({"faithful": False, "attempts": 1}) == "end"


# --------------------------------------------------------------------------- #
# Retriever mode validation — catch silent typos
# --------------------------------------------------------------------------- #
def test_retriever_rejects_invalid_mode():
    """A mode other than hybrid/baseline must not silently fall back to hybrid."""
    # Test the validation branch directly, without requiring a vector_store.
    # This keeps the test free of embedding/DB dependencies.
    from src.retrieval.pipeline import Retriever

    r = Retriever.__new__(Retriever)  # skip __init__ (no DB connection)
    r.settings = Settings()
    r.vector_store = None
    r.bm25_index = None
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        r.retrieve("q", mode="hyrbid")  # type: ignore[arg-type]
