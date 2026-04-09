"""DPR passage index + DPR question queries (no OpenAI retrieval)."""

from __future__ import annotations

from typing import Any

from app.retrieval.dpr_common import DPRModels
from app.retrieval import openai_chroma as oai_chroma


def retrieve_merged_by_id(
    collection: Any,
    dpr: DPRModels,
    queries: list[str],
    n_results: int,
) -> dict[str, tuple[str, dict[str, Any], float]]:
    def _embed(qs: list[str]) -> list[list[float]]:
        return dpr.embed_questions(qs)

    return oai_chroma.retrieve_merged_by_id(collection, _embed, queries, n_results)
