"""Central configuration.

Why pydantic-settings? All settings live in one place, typed and validated.
Loaded from .env; missing or wrong-typed values fail fast at startup rather
than surprising us at runtime. Secrets never end up in the code.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: .env'de boş bırakılan anahtar ("KEY=") "ayarlanmamış"
    # sayılır ve alanın varsayılanı kullanılır. Aksi halde pydantic'e "" olarak
    # ulaşır; EVAL_LLM_PROVIDER gibi Literal alanlarda bu doğrulama hatası verip
    # tüm uygulamayı (ingest dahil) açılışta düşürüyordu.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # --- LLM ---
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    # Eval-only override: fill this in if you want a separate provider as the
    # RAGAS judge. If empty, llm_provider is used. Why separate? Keeping
    # generation on a local/free Ollama while sending the judge to a hosted
    # model that produces solid JSON (Haiku / 4o-mini) reduces RAGAS parse
    # failures and preserves the "system under evaluation ≠ evaluator" principle.
    eval_llm_provider: Literal["ollama", "openai", "anthropic"] | None = None
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # --- Embeddings & reranker (local) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Vector DB ---
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/ragdb"

    # --- Retrieval ---
    retrieval_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150

    # --- arXiv ---
    arxiv_categories: str = "cs.AI,cs.LG,cs.CL"
    arxiv_max_results: int = 600

    # --- Langfuse ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Agent ---
    self_check_enabled: bool = True
    max_regen_attempts: int = 1

    @property
    def arxiv_category_list(self) -> list[str]:
        return [c.strip() for c in self.arxiv_categories.split(",") if c.strip()]

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Single instance (singleton). lru_cache ensures .env is read once."""
    return Settings()
