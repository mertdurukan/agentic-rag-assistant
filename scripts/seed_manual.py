import os
from dotenv import load_dotenv
load_dotenv()

from urllib.parse import urlparse
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

raw = os.getenv("DATABASE_URL", "").strip()
print(f"Loaded DATABASE_URL, length={len(raw)}")
if raw.startswith('"') and raw.endswith('"'):
    raw = raw[1:-1]
    print("Stripped surrounding quotes")

parsed = urlparse(raw)
print(f"Connecting to {parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')} as {parsed.username}...")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

store = PGVector(
    embeddings=embeddings,
    collection_name="arxiv_chunks",
    connection=raw,
    use_jsonb=True,
)

papers = [
    Document(page_content="Mixture of Experts (MoE) models use sparse activation where only a subset of expert networks processes each token. This decouples parameter count from compute per token, allowing models to scale to trillions of parameters while keeping inference cost similar to much smaller dense models. Key trade-offs: training instability due to discrete routing decisions, communication overhead in distributed training, expert collapse where a few experts dominate routing, memory pressure since all expert parameters must be held, and inference complexity from dynamic routing.", metadata={"chunk_id": "2401.04081::0", "paper_id": "2401.04081", "title": "A Survey on Mixture of Experts in LLMs", "authors": "Cai et al.", "year": 2024}),
    Document(page_content="The Switch Transformer simplifies standard MoE routing by sending each token to only one expert (top-1 routing) rather than multiple. This reduces communication costs and training instability common in earlier MoE designs like GShard. Switch Transformers achieve 4x training speedup over T5-Base at the same parameter count and scale to 1.6 trillion parameters.", metadata={"chunk_id": "2101.03961::0", "paper_id": "2101.03961", "title": "Switch Transformers", "authors": "Fedus et al.", "year": 2021}),
    Document(page_content="Mixtral 8x7B is a sparse Mixture of Experts language model with 8 expert blocks per layer, where the router selects 2 experts for each token (top-2 routing). The model has 47B total parameters but uses only 13B active parameters per token at inference time, providing the quality of much larger models at the cost of a smaller one. Mixtral outperforms Llama 2 70B and GPT-3.5 on most benchmarks.", metadata={"chunk_id": "2401.04088::0", "paper_id": "2401.04088", "title": "Mixtral of Experts", "authors": "Jiang et al., Mistral AI", "year": 2024}),
    Document(page_content="Retrieval-Augmented Generation (RAG) combines a retriever that fetches relevant documents from a knowledge base with a generator that conditions on those documents to produce answers. Compared to closed-book LLMs, RAG provides updated knowledge without retraining, attributable sources to reduce hallucination, and lower training cost. Faithfulness is the primary quality metric alongside answer relevance.", metadata={"chunk_id": "2005.11401::0", "paper_id": "2005.11401", "title": "RAG for Knowledge-Intensive NLP", "authors": "Lewis et al.", "year": 2020}),
    Document(page_content="Hybrid retrieval combines lexical (BM25) and dense (vector embedding) search to leverage their complementary strengths. BM25 excels at exact keyword matches but misses semantic paraphrases. Dense retrieval captures semantic similarity but can fail on rare technical terms. Reciprocal Rank Fusion (RRF) combines ranked lists by summing 1/(k+rank) scores. Hybrid retrieval consistently outperforms either method alone with typical improvements of 5 to 15 percent in nDCG.", metadata={"chunk_id": "2207.05205::0", "paper_id": "2207.05205", "title": "Hybrid Retrieval Methods", "authors": "Various", "year": 2022}),
    Document(page_content="Cross-encoder re-ranking improves retrieval quality by passing query-document pairs jointly through a transformer such as ms-marco MiniLM-L-6-v2 to produce a relevance score. Unlike bi-encoder embeddings which compute query and document representations independently, cross-encoders model fine-grained token-level interactions. Re-ranking is typically applied to a candidate set of 20 to 100 documents from a faster first-stage retriever.", metadata={"chunk_id": "1908.10084::0", "paper_id": "1908.10084", "title": "Cross-Encoders", "authors": "Reimers and Gurevych", "year": 2019}),
    Document(page_content="Large Language Models hallucinate, meaning they generate plausible sounding but factually incorrect content, particularly when asked about facts beyond their training data, recent events, or specialized domains. Mitigations include retrieval-augmented generation, self-consistency, uncertainty estimation, and citation requirements. RAG plus a faithfulness check can reduce hallucination rates 40 to 70 percent.", metadata={"chunk_id": "2311.05232::0", "paper_id": "2311.05232", "title": "LLM Hallucination Survey", "authors": "Huang et al.", "year": 2023}),
    Document(page_content="Vector embedding models for retrieval map text into dense vectors where semantic similarity corresponds to cosine distance. The BAAI BGE family achieves strong performance on MTEB benchmarks. bge-small-en-v1.5 produces 384-dimensional embeddings, has 33M parameters, and runs at approximately 1000 sentences per second on a CPU.", metadata={"chunk_id": "2309.07597::0", "paper_id": "2309.07597", "title": "BGE Embeddings", "authors": "Xiao et al.", "year": 2023}),
    Document(page_content="LangGraph is a framework for building stateful, multi-step agent workflows. Unlike chains which are linear, LangGraph models agent execution as a graph where nodes are computation steps and edges define conditional control flow. Common patterns include retrieval, re-ranking, generation, and self-check loops. LangGraph supports persistent state via checkpointers.", metadata={"chunk_id": "n/a::0", "paper_id": "n/a", "title": "LangGraph", "authors": "LangChain", "year": 2024}),
    Document(page_content="Reciprocal Rank Fusion (RRF) combines multiple ranked retrieval results into a single ranking by summing 1/(k+rank) for each document across the input lists, where k is a small constant typically set to 60. RRF requires no document score calibration and is robust across different retrieval methods.", metadata={"chunk_id": "1004.3036::0", "paper_id": "1004.3036", "title": "Reciprocal Rank Fusion", "authors": "Cormack et al.", "year": 2009}),
]

print(f"Inserting {len(papers)} documents...")
ids = store.add_documents(papers, ids=[p.metadata["chunk_id"] for p in papers])
print(f"OK. Inserted {len(ids)} documents.")

# Write chunks.jsonl for BM25 hybrid retrieval (parallel to vector store)
import json
from pathlib import Path
chunks_path = Path("data/chunks.jsonl")
chunks_path.parent.mkdir(exist_ok=True)
with chunks_path.open("w", encoding="utf-8") as f:
    for doc in papers:
        record = {"page_content": doc.page_content, "metadata": doc.metadata}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Wrote {len(papers)} chunks to {chunks_path}")
