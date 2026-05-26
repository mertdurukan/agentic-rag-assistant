"""Top-level service: single entry point.

`Assistant.ask(question)` → grounded answer + papers used + faithfulness.
app/ (UI) and eval/ both use this class → single code path, consistent behavior.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.agent.graph import build_graph
from src.config import Settings, get_settings
from src.ingestion.persist import load_chunks
from src.observability.tracing import configure_langfuse_env, get_langfuse_callbacks
from src.retrieval.bm25 import BM25Index
from src.retrieval.pipeline import RetrievalMode, Retriever
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/chunks.jsonl")


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    # The actual context chunks the answer was conditioned on. Eval must use
    # this (retrieving again leads to a "measured contexts ≠ contexts the
    # model saw" mismatch).
    contexts: list[str] = field(default_factory=list)
    faithful: bool = True
    self_check_reason: str = ""
    attempts: int = 0


class Assistant:
    def __init__(
        self,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        configure_langfuse_env()

        vector_store = VectorStore(self.settings)
        bm25_index = None
        chunks_path = Path(chunks_path)
        if chunks_path.exists():
            chunks = load_chunks(chunks_path)
            # The file may also be 0-byte / empty (if ingest aborted halfway).
            # BM25Index cannot be built on an empty corpus; graceful fallback
            # in that case.
            if chunks:
                bm25_index = BM25Index(chunks)
            else:
                logger.warning(
                    "%s is empty — BM25 disabled, only baseline mode will work.",
                    chunks_path,
                )
        else:
            logger.warning(
                "%s not found — BM25 disabled, only baseline mode will work. "
                "Run `python -m scripts.ingest` first.",
                chunks_path,
            )

        self.retriever = Retriever(vector_store, bm25_index, self.settings)
        self.graph = build_graph(self.retriever, self.settings)

    def ask(self, question: str, mode: RetrievalMode = "hybrid") -> Answer:
        callbacks = get_langfuse_callbacks()
        final = self.graph.invoke(
            {"question": question, "mode": mode, "attempts": 0},
            config={"callbacks": callbacks},
        )
        documents = final.get("documents", [])
        seen: dict[str, dict] = {}
        for d in documents:
            pid = d.metadata.get("paper_id")
            if pid and pid not in seen:
                seen[pid] = {
                    "paper_id": pid,
                    "title": d.metadata.get("title", ""),
                    "url": d.metadata.get("entry_url", ""),
                }
        return Answer(
            question=question,
            answer=final.get("answer", ""),
            sources=list(seen.values()),
            contexts=[d.page_content for d in documents],
            faithful=final.get("faithful", True),
            self_check_reason=final.get("self_check_reason", ""),
            attempts=final.get("attempts", 0),
        )
