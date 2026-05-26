"""Model-agnostic LLM and embedding provider.

Why this abstraction?
- Development and eval should be local and free (Ollama + Llama 3.1) → zero cost,
  unlimited iteration.
- In the live demo, a hosted model (OpenAI/Anthropic) is more consistent and faster.
- A single config-line change swaps the provider → no vendor lock-in.

LangChain's `BaseChatModel` is the common contract; the rest of the code does
not know which provider was selected.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.config import Settings, get_settings


def build_chat_model(
    settings: Settings | None = None, provider_override: str | None = None
) -> BaseChatModel:
    """Return the appropriate chat model based on config.

    If `provider_override` is set, it is used instead of `settings.llm_provider`
    (the eval harness uses this to set up a judge LLM independent of the
    generation provider).
    """
    settings = settings or get_settings()
    provider = provider_override or settings.llm_provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,  # we want faithfulness in RAG, not creativity
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


@lru_cache
def get_embeddings(model_name: str | None = None) -> Embeddings:
    """Local sentence-transformers embedding model.

    Why local embeddings? Embeddings run once per chunk, and on a large corpus
    hosted-API costs add up fast. bge-small offers a strong size/quality
    tradeoff. lru_cache: the model is loaded once (loading is expensive).
    """
    settings = get_settings()
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name or settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # cosine ~ dot product
    )
