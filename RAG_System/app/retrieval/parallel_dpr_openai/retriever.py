"""Merge OpenAI-Chroma and DPR-Chroma hit lists (max similarity per chunk id)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.retrieval.dpr_common import DPRModels
from app.retrieval import openai_chroma as oai_chroma


def merge_by_max_score(
    a: dict[str, tuple[str, dict[str, Any], float]],
    b: dict[str, tuple[str, dict[str, Any], float]],
) -> dict[str, tuple[str, dict[str, Any], float]]:
    """For each chunk id, keep the document/metadata from the higher-scoring source."""
    merged: dict[str, tuple[str, dict[str, Any], float]] = {}
    for cid in set(a) | set(b):
        best: tuple[str, dict[str, Any], float] | None = None
        for src in (a, b):
            if cid not in src:
                continue
            row = src[cid]
            if best is None or row[2] > best[2]:
                best = row
        if best is not None:
            merged[cid] = best
    return merged


def retrieve_parallel_merged(
    collection_openai: Any,
    collection_dpr: Any,
    embed_openai_fn: Callable[[list[str]], list[list[float]]],
    dpr: DPRModels,
    queries: list[str],
    n_results: int,
) -> dict[str, tuple[str, dict[str, Any], float]]:
    oai_hits = oai_chroma.retrieve_merged_by_id(
        collection_openai, embed_openai_fn, queries, n_results
    )

    def _dpr_embed(qs: list[str]) -> list[list[float]]:
        return dpr.embed_questions(qs)

    dpr_hits = oai_chroma.retrieve_merged_by_id(
        collection_dpr, _dpr_embed, queries, n_results
    )
    return merge_by_max_score(oai_hits, dpr_hits)
