# DECISIONS — Architectural Decision Records

> Rationale for each major architectural decision in this codebase.

## 1. Hybrid retrieval (BM25 + vector)

Dense embeddings capture semantic similarity but miss **exact-term** matches:
model/technique names, abbreviations (MoE, LoRA, RoPE), symbols. BM25 is strong on
these rare-but-critical matches. Combining the two raises recall above either one
alone. Vector-only fails as "found the concept but missed the exact term"; BM25-only
fails as "matched the word but missed the meaning."

## 2. RRF (Reciprocal Rank Fusion), not score addition

BM25 and vector scores are on different scales (one probability-like, the other
distance-based). Adding them directly creates scale mismatch. RRF uses **rank**, not
score: `score(d) = Σ 1/(k + rank_i(d))`. It is scale-independent, single-parameter
(k≈60), and has proven robustness on TREC. Fewer hyperparameters and lower
over-fitting risk than normalize-and-weight alternatives.

## 3. Cross-encoder re-ranking

Initial retrieval (bi-encoder + BM25) is fast but noisy: query and document are
encoded as separate vectors, missing interaction effects. A cross-encoder takes
query+document **together**, producing much sharper relevance. Because it is
expensive, it runs only on the top-N candidates, not the full corpus. Classic
two-stage retrieval: cheap/wide retrieve → expensive/sharp re-rank. Precision
improves, context to the LLM gets cleaner, faithfulness benefits indirectly.

## 4. LangGraph, not a linear chain

Requirements: multi-step reasoning plus **conditional** routing (e.g., a self-check
loop that retries generation on failure). A flat pipe (`prompt | llm`) cannot
express loops or conditions. LangGraph provides typed state, deterministic and
conditional edges, and step-level tracing.

## 5. Self-check (faithfulness) node

The most dangerous RAG failure mode is "grounded-looking" hallucination — answers
that invent things not in the context. A second LLM call after generation verifies
whether the answer is faithful to the context; if not, it triggers a limited number
of regenerations. A cheap safety layer for generation safety.

**Why regenerate with the same context?** When self-check fails, we do NOT
re-retrieve; we regenerate with the same context. This is intentional: regeneration
helps with "the model misformulated something present in the context or broke the
citation" (decoding drift). It does not fix hallucinations caused by insufficient
context — for that, the guardrail prompt mandates the "I don't have enough grounded
context" answer. Re-retrieve with LLM-expanded sub-queries is a future improvement,
out of scope for this MVP.

## 6. pgvector, not a separate vector DB

Postgres is already present in most stacks; no extra ops burden. ACID, backups, and
SQL filtering are built in. At this scale (a few thousand chunks), Qdrant or Weaviate
would be unnecessary complexity. If scale grows, the interface stays the same and we
can migrate to Qdrant (via the `VectorStore` wrapper).

## 7. Local embedding (bge-small) + cross-encoder

Embeddings run once per chunk; re-ranking runs on every query. Hosted API costs add
up fast. Local sentence-transformers are free and unlimited, freeing iteration
during development. bge-small offers a good size/quality tradeoff.

## 8. Model-agnostic LLM provider

Development and eval run on local Ollama (free); the live demo uses a hosted model
(consistent, fast). A single config change swaps them — no vendor lock-in.
`BaseChatModel` is the common contract.

## 9. RAGAS, not vibes

faithfulness, context precision, context recall, and answer relevancy expose
retriever and generator performance separately and quantitatively. The
baseline-vs-hybrid comparison turns engineering decisions into measurable evidence.

## 10. Recursive chunking, not semantic chunking

arXiv abstracts are short and dense; semantic chunking's marginal benefit does not
justify the extra LLM/embedding cost. Recursive splitting preserves hierarchy, is
deterministic, and is fast. If full-text ingestion is added, the same interface can
swap to semantic.

## Known limitations

- BM25 is in-memory; in production it should move to Postgres FTS or OpenSearch.
- Currently title + abstract only; full-text PDF ingestion is a future addition.
- Self-check uses a single LLM-as-judge; ideally a separate, stronger judge model.
- The golden set is small; expanding it would improve statistical confidence.
