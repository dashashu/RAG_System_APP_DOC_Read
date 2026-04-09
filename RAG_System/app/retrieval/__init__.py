"""
Pluggable first-stage retrieval for RAG.

Use exactly one mode at a time via settings.retrieval_mode / RETRIEVAL_MODE:

- ``openai_chroma`` — OpenAI embeddings + Chroma (default).
- ``dpr_only`` — facebook/dpr-* encoders + Chroma only (no OpenAI for retrieval).
- ``parallel_dpr_openai`` — both indexes; merge candidate scores (max per chunk id).

Switching between ``openai_chroma`` and ``dpr_only`` requires a full reindex (different vectors).
``parallel_dpr_openai`` builds both indexes on ingest.
"""

from __future__ import annotations

from typing import Literal

RetrievalMode = Literal["openai_chroma", "dpr_only", "parallel_dpr_openai"]

COLLECTION_OPENAI = "rag_pdf_docs"
COLLECTION_DPR = "rag_pdf_docs_dpr"
