"""LangGraph orchestration.

Why LangGraph (instead of a linear chain)?
- We need multi-step reasoning + CONDITIONAL routing: a loop that goes back to
  generation when self-check fails. This cannot be expressed cleanly with a
  flat pipe ( | ).
- Typed state, deterministic edges, and observability (every node is traced)
  are critical for production-grade flow.

Flow:
    START → analyze_query → retrieve → generate → self_check
                                            ↑                │
                                            └──(not faithful & retries left)
                                                             │
                                                          (else) → END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    AgentState,
    make_analyze_query,
    make_generate,
    make_retrieve,
    make_router,
    make_self_check,
)
from src.config import Settings, get_settings
from src.llm.provider import build_chat_model
from src.retrieval.pipeline import Retriever


def build_graph(retriever: Retriever, settings: Settings | None = None):
    """Return the compiled LangGraph application."""
    settings = settings or get_settings()
    llm = build_chat_model(settings)

    g = StateGraph(AgentState)
    g.add_node("analyze_query", make_analyze_query(llm))
    g.add_node("retrieve", make_retrieve(retriever))
    g.add_node("generate", make_generate(llm))
    g.add_node("self_check", make_self_check(llm, settings))

    g.add_edge(START, "analyze_query")
    g.add_edge("analyze_query", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "self_check")
    g.add_conditional_edges(
        "self_check",
        make_router(settings),
        {"regenerate": "generate", "end": END},
    )
    return g.compile()
