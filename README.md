# RAG_Advance_System_for_Aplication_Document_Read
The RAG Triad: Understand the components of RAG systems, such as the retriever, generator, &amp; Fusion Module, and how they work together to enhance information retrieval and response generation. Advanced Retrieval Techniques: Explore sparse &amp; dense retrieval methods, including Dense Passage Retrieval , &amp; implement hybrid retrieval approaches for acuracy.



# PDF + Confluence RAG System

A retrieval-augmented generation (RAG) service that indexes **PDFs** from a data directory and, optionally, **Atlassian Confluence** pages. It uses **LLM query expansion**, **multi-query retrieval**, optional **cross-encoder reranking**, and answers with **OpenAI** (`gpt-4o` by default).

## Features

- **PDFs**: all `*.pdf` files under `DATA_DIR` (recursive).
- **Confluence** (optional): pages from listed **space keys** via the REST API; same chunking and vector indexes as PDFs. See **Confluence setup** below.
- **Embeddings**: `text-embedding-3-small`; **Chroma** persistent store (two logical collections when using DPR / parallel modes).
- **Query expansion**: the chat model proposes extra search queries from the user question.
- **Multi-query retrieval**: each query is embedded and searched; hits are merged by chunk id (best similarity kept).
- **First-stage retrieval** (`RETRIEVAL_MODE`): `openai_chroma` (default), `dpr_only`, or `parallel_dpr_openai`. Use **one** mode at a time. Switching between `openai_chroma` and `dpr_only` requires a **forced reindex** (`RAG_FORCE_REINDEX=true` or `POST /reindex` with `force: true`). `parallel_dpr_openai` builds both indexes on ingest.
- **Cross-encoder reranking** (optional): wide Chroma pool per query → top candidates rescored with `cross-encoder/ms-marco-MiniLM-L-6-v2` → top passages to the chat model. Set `RERANK_ENABLED=false` to skip (lighter image, no PyTorch). **Python 3.12** in Docker is recommended; 3.13 may need reranking disabled if PyTorch wheels are missing.
- **FastAPI**: health, query, reindex, expand-only. **`/health`** returns `status`, `indexed_chunks`, `retrieval_mode`.
- **Docker** + **Kubernetes** manifests (PVC for Chroma, Secret for `OPENAI_API_KEY`).

## Quick start

1. Copy **`.env.example`** to **`.env`** and set **`OPENAI_API_KEY`**. Adjust **`DATA_DIR`** / **`CHROMA_DIR`** for local runs if needed.
2. Put PDFs under your data directory (default **`./data`** locally or **`/data`** in the container).
3. Install: **`pip install -r requirements.txt`** (use **Python 3.12** if possible).
4. Run: **`PYTHONPATH=. uvicorn app.main:app --reload --port 8080`**
5. Or build and run Docker: **`docker build -t rag-pdf:latest .`** then run with env file / `-e` flags mapping the same variables.

On startup the app ingests unless the index is already populated; set **`RAG_FORCE_REINDEX=true`** once to rebuild after changing PDFs, Confluence config, or **`RETRIEVAL_MODE`**.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness; `indexed_chunks`, `retrieval_mode`, `status` ok/degraded |
| POST | `/query` | Body: `question`, `use_expansion` (default true). Returns `answer`, `expanded_queries`, `sources` (includes `source_type`, `url` for Confluence when present) |
| POST | `/reindex` | Body: `force` (bool). Returns `files_processed`, `chunks_indexed`, `confluence_pages` |
| POST | `/expand-only` | Debug: expanded queries only |

## Confluence setup

1. Create an **API token**: https://id.atlassian.com/manage-profile/security/api-tokens  
2. In **`.env`** set:
   - **`CONFLUENCE_ENABLED=true`**
   - **`CONFLUENCE_URL`**: Cloud example `https://your-site.atlassian.net/wiki`. For **Server / Data Center**, use the site base that serves **`/rest/api`** (often **no** `/wiki` suffix), e.g. `https://confluence.company.com`
   - **`CONFLUENCE_EMAIL`**: Atlassian account email (Cloud) or username your admin documents for API auth
   - **`CONFLUENCE_API_TOKEN`**: the token (never commit; use Secret in Kubernetes)
   - **`CONFLUENCE_SPACE_KEYS`**: comma-separated **space keys** (e.g. `ENG,DOCS`), not display names
   - Optional: **`CONFLUENCE_MAX_PAGES`**, **`CONFLUENCE_BATCH_LIMIT`** (see `.env.example`)

3. Restart the app or call **`POST /reindex`** with **`force: true`** if you need a full rebuild.

Chunks are tagged with **`source_type: confluence`**, **`page_id`**, **`space_key`**, and **`url`** where available. The LLM context cites Confluence titles and URLs when present.

## Configuration

All runtime settings are environment variables (see **`.env.example`**). Common ones:

- **OpenAI**: `OPENAI_API_KEY`, `CHAT_MODEL`, `EMBEDDING_MODEL`
- **Paths**: `DATA_DIR`, `CHROMA_DIR`
- **Chunks / retrieval**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_PER_QUERY`, `EXPANSION_QUERY_COUNT`
- **Retrieval mode**: `RETRIEVAL_MODE`, `DPR_CTX_MODEL`, `DPR_Q_MODEL`
- **Reranking**: `RERANK_ENABLED`, `CROSS_ENCODER_MODEL`, `RERANK_POOL_PER_QUERY`, `RERANK_MAX_CANDIDATES`, `LLM_CONTEXT_TOP_K`
- **Ingest**: `RAG_FORCE_REINDEX`
- **Confluence**: `CONFLUENCE_*` (see above)

## Query expansion flow

When `use_expansion` is `true` on `/query`:

```mermaid
flowchart LR
  Q[User question] --> E["gpt-4o: expand queries"]
  E --> R[Per query: Chroma wide pool]
  R --> M[Deduplicate by chunk id / bi-encoder score]
  M --> T[Top long-tail candidates]
  T --> C["Cross-encoder (MS MARCO MiniLM) rerank"]
  C --> L[Top passages to gpt-4o]
  L --> A[Final answer]
```

## Matplotlib scatter plots (optional)

Install extra deps and point `CHROMA_DIR` / `DATA_DIR` at your index (same as the API). PNGs go under `plots/` (gitignored).

```bash
pip install -r requirements-viz.txt
export CHROMA_DIR=./chroma_data DATA_DIR=./data
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python scripts/plot_embedding_scatter.py -o plots/chunks_pca.png
PYTHONPATH=. python scripts/plot_embedding_scatter.py --method tsne -o plots/chunks_tsne.png
PYTHONPATH=. python scripts/plot_query_context_scatter.py -o plots/query_context_pca.png
PYTHONPATH=. python scripts/plot_query_context_scatter.py --no-expansion -o plots/query_context_noexpand.png
```

`plot_query_context_scatter.py` defaults to sample finance-style questions; pass more as positional arguments if you like.

## TLS / SSL: CERTIFICATE_VERIFY_FAILED (macOS)

OpenAI **httpx** merges **certifi** with common system CA paths (e.g. Homebrew **`/opt/homebrew/etc/ca-certificates/cert.pem`**). Override with **`SSL_CERT_FILE`** or **`REQUESTS_CA_BUNDLE`** behind a corporate proxy.

Embeddings are computed **outside** Chroma; vectors are passed into `add` / `query`, which avoids some Chroma embedding transport issues.

## Project layout (high level)

- **`app/main.py`** — FastAPI app and lifespan ingest
- **`app/rag.py`** — ingest (PDF + Confluence), retrieval, rerank, answer
- **`app/config.py`** — settings from environment
- **`app/confluence_ingest.py`** — Confluence REST client and HTML-to-text
- **`app/openai_client.py`** — OpenAI client with merged TLS trust store
- **`app/retrieval/`** — OpenAI-only, DPR-only, and parallel DPR+OpenAI retrieval helpers
- **`k8s/`** — Deployment, Service, PVC, Secret example
- **`scripts/`** — Optional visualization scripts

