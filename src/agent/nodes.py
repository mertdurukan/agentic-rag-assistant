"""LangGraph nodes and shared state.

Each node reads `state` and returns a partial update. Dependencies (llm,
retriever) are injected via factory functions → nodes stay testable and pure.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import (
    GENERATION_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    SELF_CHECK_PROMPT,
)
from src.config import Settings
from src.retrieval.pipeline import RetrievalMode, Retriever

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """State carried through the graph."""

    question: str
    mode: RetrievalMode
    sub_queries: list[str]
    documents: list[Document]
    context: str
    answer: str
    faithful: bool
    self_check_reason: str
    attempts: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_json(text: str) -> dict[str, Any]:
    """Safely extract the first JSON object from an LLM output.

    Local models sometimes wrap JSON in markdown or free text; grabbing the first
    { ... } block is a robust fallback.
    """
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _format_context(docs: list[Document]) -> str:
    """Convert chunks into a [paper_id]-tagged context string suitable for citation."""
    blocks = []
    for d in docs:
        pid = d.metadata.get("paper_id", "unknown")
        title = d.metadata.get("title", "")
        blocks.append(f"[{pid}] {title}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _invoke_text(llm: BaseChatModel, prompt: str) -> str:
    # The .content type is `str | list[str | dict]` — our prompts return a
    # single-segment str, but future versions of some providers may return a
    # list. The str() cast satisfies mypy strict and is safe at runtime.
    return str(llm.invoke([HumanMessage(content=prompt)]).content)


# --------------------------------------------------------------------------- #
# Node factories
# --------------------------------------------------------------------------- #
def make_analyze_query(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    def analyze_query(state: AgentState) -> dict:
        question = state["question"]
        raw = _invoke_text(llm, QUERY_ANALYSIS_PROMPT.format(question=question))
        parsed = _parse_json(raw)
        subs = parsed.get("sub_queries") or [question]
        subs = [s for s in subs if isinstance(s, str) and s.strip()] or [question]
        logger.info("Sub-queries: %s", subs)
        return {"sub_queries": subs}

    return analyze_query


def make_retrieve(retriever: Retriever) -> Callable[[AgentState], dict]:
    def retrieve(state: AgentState) -> dict:
        mode: RetrievalMode = state.get("mode", "hybrid")
        seen: dict[str, Document] = {}
        for sub in state.get("sub_queries", [state["question"]]):
            for doc in retriever.retrieve(sub, mode=mode):
                seen.setdefault(doc.metadata["chunk_id"], doc)
        docs = list(seen.values())
        return {"documents": docs, "context": _format_context(docs)}

    return retrieve


def make_generate(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    def generate(state: AgentState) -> dict:
        prompt = GENERATION_PROMPT.format(
            context=state.get("context", ""), question=state["question"]
        )
        answer = _invoke_text(llm, prompt)
        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    return generate


def make_self_check(
    llm: BaseChatModel, settings: Settings
) -> Callable[[AgentState], dict]:
    def self_check(state: AgentState) -> dict:
        if not settings.self_check_enabled:
            return {"faithful": True, "self_check_reason": "self-check disabled"}
        prompt = SELF_CHECK_PROMPT.format(
            context=state.get("context", ""), answer=state.get("answer", "")
        )
        parsed = _parse_json(_invoke_text(llm, prompt))
        faithful = bool(parsed.get("faithful", True))
        return {
            "faithful": faithful,
            "self_check_reason": parsed.get("reason", ""),
        }

    return self_check


def make_router(settings: Settings) -> Callable[[AgentState], str]:
    """Post self-check routing: if not faithful and retries are left, regenerate."""

    def route(state: AgentState) -> str:
        if state.get("faithful", True):
            return "end"
        if state.get("attempts", 0) <= settings.max_regen_attempts:
            return "regenerate"
        return "end"

    return route
