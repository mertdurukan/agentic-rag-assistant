"""FastAPI service + Gradio UI.

Why FastAPI + Gradio (single process)?
- FastAPI: typed, auto-documented (/docs) JSON API → easy to integrate and test.
- Gradio: fast UI for ML demos, native compatibility with Hugging Face Spaces.
- Mounting Gradio under FastAPI gives a single deploy target (HF Spaces / Railway).

Lazy init: the Assistant is built on the first request → heavy model loads do
not block startup, and /health responds immediately.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.assistant import Assistant
from src.retrieval.pipeline import RetrievalMode

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG Assistant", version="1.0.0")


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """When a backing service (Postgres/Ollama/HF) is unreachable, return a
    short, meaningful 503 to the user instead of a raw stacktrace. Keep the
    full exception on the log side.
    """
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=503,
        content={
            "error": "service_unavailable",
            "detail": (
                f"{type(exc).__name__}: {exc}. "
                "Verify that Postgres (docker compose up -d) and your LLM "
                "provider (Ollama / OPENAI_API_KEY / ANTHROPIC_API_KEY) are up."
            ),
        },
    )


@lru_cache
def get_assistant() -> Assistant:
    return Assistant()


class AskRequest(BaseModel):
    question: str
    mode: RetrievalMode = "hybrid"


class Source(BaseModel):
    paper_id: str
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    faithful: bool
    self_check_reason: str
    attempts: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = get_assistant().ask(req.question, mode=req.mode)
    return AskResponse(
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
        faithful=result.faithful,
        self_check_reason=result.self_check_reason,
        attempts=result.attempts,
    )


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
def _ui_ask(
    question: str,
    mode: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),  # noqa: B008
):
    """Generator that yields intermediate UI states so the user sees progress
    rather than a frozen button during the 5-60s pipeline run."""
    if not question.strip():
        yield "_Please enter a question._", ""
        return

    progress(0.05, desc="Initializing pipeline (cold start may take ~60s)...")
    yield (
        "⏳ _Initializing retrieval pipeline..._\n\n"
        "First request after a cold start can take 30-60 seconds "
        "(model loading + connection warmup). Subsequent requests are 5-15s.",
        "",
    )

    assistant = get_assistant()

    stage_desc = (
        "vector + BM25 + RRF + cross-encoder rerank"
        if mode == "hybrid"
        else "vector only"
    )
    progress(0.30, desc=f"Retrieving relevant papers ({mode} mode)...")
    yield (
        f"⏳ _Retrieving relevant arXiv papers in **{mode}** mode "
        f"({stage_desc})..._",
        "",
    )

    result = assistant.ask(question, mode=mode)  # type: ignore[arg-type]

    progress(0.95, desc="Formatting results...")
    sources_md = "\n".join(
        f"- **[{s['paper_id']}]** {s['title']} — {s['url']}" for s in result.sources
    ) or "_(no sources)_"
    badge = "✅ faithful" if result.faithful else "⚠️ not faithful"

    progress(1.0, desc="Done")
    yield result.answer, f"**{badge}** · {sources_md}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Agentic RAG Assistant") as ui:
        gr.Markdown("# Agentic RAG Assistant for arXiv AI Research")
        with gr.Row():
            q = gr.Textbox(label="Question", lines=2, scale=4)
            mode = gr.Radio(
                ["hybrid", "baseline"], value="hybrid", label="Retrieval mode", scale=1
            )
        btn = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown(label="Sources")
        btn.click(
            _ui_ask,
            inputs=[q, mode],
            outputs=[answer, sources],
            show_progress="full",
        )
    return ui


app = gr.mount_gradio_app(app, build_ui(), path="/")
