"""Langfuse observability (v3 / OpenTelemetry-based).

Design decision: tracing is OPTIONAL and graceful.
- If Langfuse keys are missing, the system runs without errors; traces simply
  are not collected.
- This keeps local development and CI free of external service dependencies.

Langfuse automatically captures every LangGraph step, latency, tokens, and cost
(once the CallbackHandler is passed as a callback to graph.invoke).
"""
from __future__ import annotations

import logging

from src.config import get_settings

logger = logging.getLogger(__name__)


def get_langfuse_callbacks() -> list:
    """Callback list to pass to LangGraph/LangChain invoke.

    Returns an empty list (no-op) if Langfuse is not configured.
    """
    settings = get_settings()
    if not settings.langfuse_enabled:
        logger.info("Langfuse not configured — tracing disabled.")
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as exc:  # noqa: BLE001 — observability must never break the main flow
        # Log the stacktrace too: callback setup rarely fails, but when it does
        # the exception source is critical for independent debugging.
        logger.warning("Failed to set up Langfuse callback: %s", exc, exc_info=True)
        return []


def configure_langfuse_env() -> None:
    """Fill the environment variables that the Langfuse SDK expects from config."""
    import os

    settings = get_settings()
    if settings.langfuse_enabled:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key or "")
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
