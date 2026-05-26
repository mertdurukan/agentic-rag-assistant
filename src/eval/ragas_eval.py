"""RAGAS evaluation harness.

What it does:
- Runs the golden set in both "baseline" (vector-only) and "hybrid"
  (vector + BM25 + rerank) modes.
- Computes RAGAS metrics for each answer: faithfulness, context precision,
  context recall, answer (response) relevancy.
- Writes the real comparison table to benchmarks/baseline_vs_hybrid.md.

The "hybrid + rerank is X better than pure vector" claim is grounded in the
numbers produced here — the code does not fabricate them.

Usage:       python -m src.eval.ragas_eval
Requirements: ingestion must be done + LLM provider must be reachable.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.assistant import Assistant
from src.config import get_settings
from src.eval.eval_dataset import EVAL_SET
from src.llm.provider import build_chat_model, get_embeddings
from src.retrieval.pipeline import RetrievalMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_metrics():
    """Import RAGAS metrics in a version-resilient way."""
    from ragas.metrics import Faithfulness

    metrics = [Faithfulness()]
    try:
        from ragas.metrics import LLMContextPrecisionWithReference

        metrics.append(LLMContextPrecisionWithReference())
    except ImportError:  # legacy name
        from ragas.metrics import ContextPrecision

        metrics.append(ContextPrecision())
    try:
        from ragas.metrics import LLMContextRecall

        metrics.append(LLMContextRecall())
    except ImportError:
        from ragas.metrics import ContextRecall

        metrics.append(ContextRecall())
    try:
        from ragas.metrics import ResponseRelevancy

        metrics.append(ResponseRelevancy())
    except ImportError:
        from ragas.metrics import AnswerRelevancy

        metrics.append(AnswerRelevancy())
    return metrics


@dataclass
class ModeResult:
    """Numeric summary of a mode (baseline/hybrid) — row count and average
    latency are also reported to avoid producing false confidence."""

    means: dict[str, float]
    n_total: int
    avg_latency_s: float


def _run_mode(assistant: Assistant, mode: RetrievalMode) -> ModeResult:
    """Run the golden set in the given mode and return the RAGAS result.

    Critical: we use as `contexts` the chunks the model was ACTUALLY conditioned
    on (from Assistant.ask output). The old version called the retriever a second
    time — that produced different context via the sub_queries-using graph
    retrieval, leading to a "measured contexts ≠ contexts the model saw" mismatch.
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    rows = []
    latencies: list[float] = []
    for item in EVAL_SET:
        t0 = perf_counter()
        ans = assistant.ask(item.question, mode=mode)
        latencies.append(perf_counter() - t0)
        rows.append(
            {
                "user_input": item.question,
                "response": ans.answer,
                "retrieved_contexts": ans.contexts,
                "reference": item.reference,
            }
        )

    dataset = EvaluationDataset.from_list(rows)
    # Decouple the judge LLM from the generation provider: if EVAL_LLM_PROVIDER
    # is set, use it (e.g., generation=Ollama, judge=Anthropic Haiku → more
    # reliable JSON, fewer RAGAS parse failures). If empty, same as LLM_PROVIDER.
    settings = get_settings()
    judge_provider = settings.eval_llm_provider
    if judge_provider:
        logger.info("Eval judge provider override: %s", judge_provider)
    evaluator_llm = LangchainLLMWrapper(
        build_chat_model(provider_override=judge_provider)
    )
    evaluator_emb = LangchainEmbeddingsWrapper(get_embeddings())
    # RAGAS default is 16 workers + 180s timeout. 16 parallel requests on a
    # hosted provider push throttling/queue-time above 180s, and one timed-out
    # row drops the whole eval (with raise_exceptions=True).
    # 4 workers + 600s timeout: still parallel, but within the provider's
    # rate limit.
    run_config = RunConfig(timeout=600, max_workers=4, max_retries=15, max_wait=120)
    # raise_exceptions=False: in a 12-question eval, a single transient network
    # error (e.g. openai.APIConnectionError) was dropping the whole run. To
    # avoid silently inflating scores with NaN, we explicitly report NaN counts
    # below.
    result = evaluate(
        dataset=dataset,
        metrics=_load_metrics(),
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=run_config,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    numeric = df.select_dtypes("number")
    means: dict[str, float] = {}
    for col in numeric.columns:
        total = len(numeric[col])
        valid = int(numeric[col].notna().sum())
        if valid == 0:
            logger.error("Metric %s: 0/%d valid rows — bad signal.", col, total)
            means[col] = float("nan")
            continue
        if valid < total:
            logger.warning(
                "Metric %s: %d/%d valid rows (NaNs excluded from mean).",
                col, valid, total,
            )
        means[col] = round(float(numeric[col].mean()), 4)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return ModeResult(means=means, n_total=len(rows), avg_latency_s=avg_latency)


def write_report(
    baseline: ModeResult,
    hybrid: ModeResult,
    path: str | Path = "benchmarks/baseline_vs_hybrid.md",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted(set(baseline.means) | set(hybrid.means))
    header = "| Setup | n | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 2)
    b_row = (
        f"| Baseline (vector-only) | {baseline.n_total} | "
        + " | ".join(
            f"{baseline.means.get(c, float('nan')):.4f}" for c in cols
        )
        + " |"
    )
    h_row = (
        f"| Hybrid + re-rank (this) | {hybrid.n_total} | "
        + " | ".join(
            f"{hybrid.means.get(c, float('nan')):.4f}" for c in cols
        )
        + " |"
    )
    body = "\n".join(
        [
            "# Baseline vs Hybrid+Re-rank — RAGAS",
            "",
            f"Golden set: {len(EVAL_SET)} questions · LLM: {get_settings().llm_provider}",
            "",
            header,
            sep,
            b_row,
            h_row,
            "",
            "## Latency (end-to-end, mean)",
            "",
            f"- Baseline: **{baseline.avg_latency_s:.2f} s/question**",
            f"- Hybrid + re-rank: **{hybrid.avg_latency_s:.2f} s/question**",
            "",
            "## Analysis",
            "",
            "Write a sentence below: how much hybrid moved each metric versus the",
            "baseline, and the reason for any unexpected result. Be honest — the",
            "value of this repo lies in this analysis being truthful.",
            "",
            "> Auto-generated: `python -m src.eval.ragas_eval`",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "hybrid", "both"],
        default="both",
    )
    args = parser.parse_args()

    assistant = Assistant()
    baseline = hybrid = None
    if args.mode in ("baseline", "both"):
        logger.info("Baseline evaluation...")
        baseline = _run_mode(assistant, "baseline")
        logger.info(
            "Baseline: %s · avg latency=%.2fs · n=%d",
            baseline.means, baseline.avg_latency_s, baseline.n_total,
        )
    if args.mode in ("hybrid", "both"):
        logger.info("Hybrid evaluation...")
        hybrid = _run_mode(assistant, "hybrid")
        logger.info(
            "Hybrid: %s · avg latency=%.2fs · n=%d",
            hybrid.means, hybrid.avg_latency_s, hybrid.n_total,
        )

    if args.mode == "both" and baseline and hybrid:
        out = write_report(baseline, hybrid)
        logger.info("Report written: %s", out)


if __name__ == "__main__":
    main()
