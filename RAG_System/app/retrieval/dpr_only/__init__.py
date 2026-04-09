"""
DPR-only retrieval: Chroma indexed with DPR context vectors; queries via DPR question encoder.

Use when ``settings.retrieval_mode == "dpr_only"``. Import implementation from
``app.retrieval.dpr_only.retriever`` (``retrieve_merged_by_id``).

Reindex after switching from ``openai_chroma`` — embeddings are not compatible.
"""
