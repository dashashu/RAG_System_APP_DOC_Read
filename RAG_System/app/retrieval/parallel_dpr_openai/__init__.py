"""
Parallel dense retrieval: run OpenAI+Chroma and DPR+Chroma, then merge per chunk id.

Use when ``settings.retrieval_mode == "parallel_dpr_openai"``. Ingest must populate
both Chroma collections (see ``RAGService.ingest_pdfs``).

Only one retrieval mode should be active at a time; switch via ``RETRIEVAL_MODE``.
"""

from app.retrieval.parallel_dpr_openai.retriever import merge_by_max_score, retrieve_parallel_merged

__all__ = ["merge_by_max_score", "retrieve_parallel_merged"]
