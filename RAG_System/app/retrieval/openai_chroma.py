"""
OpenAI embedding API + Chroma cosine retrieval.

Use this path alone when ``settings.retrieval_mode == "openai_chroma"`` (default).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Chunk id -> (document, metadata, similarity score in [0, 1] from Chroma cosine distance)


def retrieve_merged_by_id(
    collection: Any,
    embed_fn: Callable[[list[str]], list[list[float]]],
    queries: list[str],
    n_results: int,
) -> dict[str, tuple[str, dict[str, Any], float]]:
    by_id: dict[str, tuple[str, dict[str, Any], float]] = {}
    for q in queries:
        q_emb = embed_fn([q])
        res = collection.query(
            query_embeddings=q_emb,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        doc_lists = res.get("documents") or [[]]
        meta_lists = res.get("metadatas") or [[]]
        dist_lists = res.get("distances") or [[]]
        docs = doc_lists[0] if doc_lists else []
        metas = meta_lists[0] if meta_lists else []
        dists = dist_lists[0] if dist_lists else []
        ids_out = res.get("ids") or [[]]
        id_row = ids_out[0] if ids_out else []

        for i, doc in enumerate(docs):
            if not doc:
                continue
            cid = id_row[i] if i < len(id_row) else f"row_{i}"
            meta = metas[i] if i < len(metas) else {}
            dist = float(dists[i]) if i < len(dists) else 1.0
            sim = max(0.0, min(1.0, 1.0 - dist))
            prev = by_id.get(cid)
            if prev is None or sim > prev[2]:
                by_id[cid] = (doc, meta or {}, sim)
    return by_id
