# Deploy to Hugging Face Spaces

This guide covers step-by-step instructions to deploy `agentic-rag-assistant` to
**Hugging Face Spaces (Docker SDK)**. Target public URL:
`https://huggingface.co/spaces/mertdurukan/agentic-rag-assistant`

## Architectural constraint: managed Postgres required

HF Spaces does not provision Postgres. This project is built on `pgvector`, so an
**external managed Postgres** is required. Recommended providers (all have free tiers):

| Provider | pgvector | Free tier | Notes |
|---|---|---|---|
| **Supabase** | ✅ built-in | 500 MB / 2 projects | Easiest; `CREATE EXTENSION vector;` from the SQL editor |
| **Neon** | ✅ built-in | 0.5 GB / branching | Fast cold-start, branch-friendly |
| **Aiven (PG)** | ✅ via extension | 1 month trial | Not permanently free |

The steps below assume **Supabase**.

---

## 1) Prepare the database on Supabase

1. https://supabase.com → New Project (choose a region on the same continent as your HF Space for latency)
2. **SQL Editor**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. **Project Settings → Database → Connection string**:
   - Take the "URI" format (`postgresql://postgres:[YOUR-PASSWORD]@db.xxxx.supabase.co:5432/postgres`)
   - For SQLAlchemy + psycopg3, replace the prefix with `postgresql+psycopg://`:
     ```
     postgresql+psycopg://postgres:<PASSWORD>@db.xxxx.supabase.co:5432/postgres
     ```
   - This string will be the `DATABASE_URL` secret.

---

## 2) Choose an LLM provider (Ollama does not run on HF Spaces)

You cannot host Ollama on a free CPU Space. Two options:

- **OpenAI**: `LLM_PROVIDER=openai`, `OPENAI_API_KEY=sk-…`, `OPENAI_MODEL=gpt-4o-mini`
- **Anthropic**: `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=sk-ant-…`, `ANTHROPIC_MODEL=claude-3-5-haiku-latest`

To keep generation cheap, **gpt-4o-mini** or **claude-haiku** is recommended
(the README evaluation table was produced with `gpt-4o-mini`).

---

## 3) Create the HF Space

1. https://huggingface.co/new-space
2. **Owner**: `mertdurukan` · **Space name**: `agentic-rag-assistant`
3. **SDK**: Docker · **Template**: Blank · **Hardware**: CPU basic (free)
4. **Visibility**: Public
5. Create Space → an empty git repo opens.

> The README.md frontmatter (`sdk: docker`, `app_port: 8000`) is already correct —
> the HF Space will listen on port 8000.

---

## 4) Add secrets (HF Space → Settings → Variables and secrets)

**As secrets (hidden):**
- `DATABASE_URL` = your Supabase URI (step 1)
- `OPENAI_API_KEY` *or* `ANTHROPIC_API_KEY`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (optional, if enabling observability)

**As variables (visible, passed to build):**
- `LLM_PROVIDER` = `openai` (or `anthropic`)
- `OPENAI_MODEL` = `gpt-4o-mini` (or `ANTHROPIC_MODEL` = `claude-3-5-haiku-latest`)
- `EMBEDDING_MODEL` = `BAAI/bge-small-en-v1.5`
- `EMBEDDING_DIM` = `384`
- `RERANKER_MODEL` = `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `ARXIV_MAX_RESULTS` = `300`
- `SELF_CHECK_ENABLED` = `true`
- `MAX_REGEN_ATTEMPTS` = `1`

---

## 5) Push the code to the HF Space

```bash
# In the repo root (agentic-rag-assistant/), add an HF remote:
git remote add hf https://huggingface.co/spaces/mertdurukan/agentic-rag-assistant

# HF authentication (token: https://huggingface.co/settings/tokens, with "write" scope)
# On git push you'll provide username=mertdurukan, password=<HF_TOKEN>
# Or use a credential helper:
#   git config --global credential.helper store

git push hf main
```

The HF Space build starts automatically. Watch **Build logs** under the "Logs" tab
of the Space page. First build takes ~5-8 minutes (pip install + first model
download happens at runtime).

---

## 6) Ingest the corpus on first run (critical step — answers are empty otherwise)

The vector DB is empty initially. Run this **locally against the remote Supabase**, once:

```bash
# In your .env, temporarily switch DATABASE_URL to the Supabase URI.
# Then:
make ingest        # python -m scripts.ingest
# ~10-25 minutes due to arxiv.Client delay_seconds=3.
# Expected output: "Total ~900+ chunks." + "Vectors written: N/N"

# After it finishes, switch .env back to local.
```

Alternative: SSH into the HF Space once and run `python -m scripts.ingest`.
(Free tier has no SSH; local ingestion is more practical.)

---

## 7) Verify

1. The Space page should render the Gradio UI on load.
2. `https://huggingface.co/spaces/mertdurukan/agentic-rag-assistant?endpoint=/health`
   → `{"status":"ok"}`
3. Ask via UI: "What are the trade-offs of MoE?" → answer + citation + ✅ faithful badge.
4. First question takes ~30-60s (initial request downloads embedding + reranker
   models, ~210 MB). Subsequent questions: 4-8s.

---

## 8) Live Demo link in README

[`README.md`](README.md) already points to this URL:

```
[Live Demo](https://huggingface.co/spaces/mertdurukan/agentic-rag-assistant)
```

If your HF username differs, update this and every other `mertdurukan` reference accordingly.

---

## Troubleshooting

- **Build "out of memory"** → free CPU Space provides 16 GB RAM, requirements fit.
  If unsure whether it finished, try "Restart" → "Factory rebuild".
- **`relation "documents" does not exist`** → migration was not run. Run ingestion
  locally once (step 6); ingest creates the schema automatically.
- **`vector` extension missing error** → run `CREATE EXTENSION vector;` in the Supabase SQL editor.
- **Slow cold start** → first request downloads ~210 MB of models from HF Hub.
  You can pre-download them at build time in the Dockerfile:
  ```dockerfile
  RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
      SentenceTransformer('BAAI/bge-small-en-v1.5'); \
      CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
  ```
- **`OPENAI_API_KEY` not visible** → ensure it's set as a **Secret** in HF Space
  Settings (not a Variable). Restart required.
- **Space in "Stopped" state** → free tier sleeps after 48h of inactivity; a visitor wakes it.
