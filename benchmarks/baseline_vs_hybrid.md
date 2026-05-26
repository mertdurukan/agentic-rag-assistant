# Baseline vs Hybrid+Re-rank — RAGAS

Golden set: 12 questions · LLM: openai

| Setup | n | answer_relevancy | context_recall | faithfulness | llm_context_precision_with_reference |
|---|---|---|---|---|---|
| Baseline (vector-only) | 12 | 0.8653 | 0.8333 | 0.8981 | 0.6001 |
| Hybrid + re-rank (this) | 12 | 0.9351 | 0.8583 | 0.9673 | 0.7618 |

## Latency (end-to-end, mean)

- Baseline: **4.92 s/question**
- Hybrid + re-rank: **6.50 s/question**

## Analysis

**Hybrid + re-rank beats baseline on all 4 metrics.**

| Metric | Baseline | Hybrid | Δ (absolute) | Δ (relative) |
|---|---|---|---|---|
| faithfulness | 0.8981 | 0.9673 | +0.069 | +7.7% |
| answer_relevancy | 0.8653 | 0.9351 | +0.070 | +8.1% |
| context_recall | 0.8333 | 0.8583 | +0.025 | +3.0% |
| llm_context_precision_with_reference | 0.6001 | 0.7618 | **+0.162** | **+27.0%** |

- The largest gain is in **context_precision (+27%)** — this is the expected
  outcome: the cross-encoder reranker's job is exactly to reorder the shortlist
  and push irrelevant chunks down. After RRF fusion, selecting the 5 most
  relevant out of 20 candidates directly improves precision.
- **Faithfulness and relevancy at 0.93+** — a strong signal that the RAG chain
  works end to end (grounding + citation + guardrail together do their job).
- **Latency cost:** +1.58 s/question (4.92 → 6.50). BM25 and vector run in
  parallel, so the main cost is the cross-encoder re-rank step; running MiniLM-L6
  on a 5-candidate shortlist is reasonable.

### Setup

- Corpus: most recent 300 arXiv `cs.AI/cs.LG/cs.CL` papers, 934 chunks
  (`ARXIV_MAX_RESULTS=300`).
- Golden set: 12 questions calibrated to the actual papers in the corpus
  (see `src/eval/eval_dataset.py`); each question is directly answered by a
  single paper's abstract.
- Generation + judge LLM: `gpt-4o-mini`.
- Embedding: `BAAI/bge-small-en-v1.5` (local, 384d, cosine).
- Re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local).
- RAGAS run: 4 workers, 600s timeout, 15 retries, `raise_exceptions=False`
  + explicit NaN report.

### NaN report (transparency)

A transient `APIConnectionError` in the `gpt-4o-mini` judge dropped one row in
each of two metrics; those rows were excluded from the mean:
- baseline `llm_context_precision_with_reference`: 10/12 valid rows
- baseline `context_recall`: 10/12 valid rows
- hybrid: all metrics 12/12

With only 12 questions, these missing points increase variance; a 30-50 question
golden set would yield more stable numbers.

### Limitations

- The 12 questions are calibrated to `cs.AI/cs.LG/cs.CL` papers; on other
  domains (e.g., arXiv `math.*`) corpus mismatch may still occur.
- Judge and generation use the same provider (`gpt-4o-mini`). A stronger judge
  (e.g., `gpt-4o` or `claude-3-5-sonnet`) and a cross-provider judge would
  improve score reliability (`EVAL_LLM_PROVIDER` supports this).

> Auto-generated: `python -m src.eval.ragas_eval`
