from __future__ import annotations

import glob
import json
import logging
import os

# Chroma telemetry can error on some posthog versions; default off unless user sets env.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pypdf import PdfReader

from app.config import settings
from app.confluence_ingest import fetch_pages_for_spaces, parse_space_keys
from app.openai_client import build_openai_client
from app.retrieval import COLLECTION_DPR, COLLECTION_OPENAI
from app.retrieval import openai_chroma
from app.retrieval.dpr_only import retriever as dpr_only_retriever
from app.retrieval.parallel_dpr_openai.retriever import retrieve_parallel_merged

logger = logging.getLogger(__name__)


def _scores_to_list(raw: Any) -> list[float]:
    if isinstance(raw, (float, int)):
        return [float(raw)]
    try:
        import numpy as np

        arr = np.atleast_1d(np.asarray(raw, dtype=np.float64))
        return [float(x) for x in arr.reshape(-1)]
    except Exception:
        return [float(x) for x in raw]


def _simple_chunk(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception as e:
            logger.warning("extract_text failed for %s: %s", path, e)
            t = ""
        parts.append(t)
    return "\n\n".join(parts)


@dataclass
class IngestResult:
    files_processed: int
    chunks_indexed: int
    confluence_pages: int = 0


class RAGService:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to initialize RAGService")
        self._client = build_openai_client(settings.openai_api_key)
        self._cross_encoder: Any = None
        self._dpr_helper: Any = None
        os.makedirs(settings.chroma_dir, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=settings.chroma_dir)
        # embedding_function=None: we always pass precomputed vectors.
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_OPENAI,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_dpr = self._chroma.get_or_create_collection(
            name=COLLECTION_DPR,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def _dpr_models(self) -> Any:
        if self._dpr_helper is None:
            from app.retrieval.dpr_common import DPRModels

            self._dpr_helper = DPRModels(settings.dpr_ctx_model, settings.dpr_q_model)
        return self._dpr_helper

    def collection_count(self) -> int:
        """Documents in the index used by the active retrieval_mode."""
        if settings.retrieval_mode == "dpr_only":
            return self._collection_dpr.count()
        return self._collection.count()

    def _should_skip_ingest(self, force: bool) -> bool:
        if force:
            return False
        mode = settings.retrieval_mode
        if mode == "openai_chroma":
            return self._collection.count() > 0
        if mode == "dpr_only":
            return self._collection_dpr.count() > 0
        # parallel: both collections must exist with matching counts
        n_o = self._collection.count()
        n_d = self._collection_dpr.count()
        return n_o > 0 and n_d > 0 and n_o == n_d

    def _embed_texts_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed via OpenAI (not via Chroma). Errors are turned into clear RuntimeError text."""
        cleaned = [t.replace("\n", " ") for t in texts]
        try:
            resp = self._client.embeddings.create(
                model=settings.embedding_model,
                input=cleaned,
            )
        except APIStatusError as e:
            hint = ""
            if e.status_code == 401:
                hint = (
                    " Check OPENAI_API_KEY in your environment or .env (correct key, "
                    "active project, not revoked)."
                )
            elif e.status_code == 429:
                hint = " Rate limited—wait and retry or upgrade quota."
            elif e.status_code in (402, 403):
                hint = " Check billing and model access at https://platform.openai.com"
            raise RuntimeError(
                f"OpenAI embeddings HTTP {e.status_code}: {e.message}.{hint}"
            ) from e
        except APITimeoutError as e:
            raise RuntimeError(f"OpenAI embeddings request timed out: {e.message}") from e
        except APIConnectionError as e:
            cause = e.__cause__
            cause_s = repr(cause) if cause is not None else ""
            if cause_s and (
                "CERTIFICATE" in cause_s.upper()
                or "certificate verify failed" in cause_s.lower()
            ):
                raise RuntimeError(
                    f"TLS error calling OpenAI ({cause_s}). "
                    "Try: export SSL_CERT_FILE=/path/to/ca-bundle.pem "
                    "(corporate proxy) or ensure merged CA bundles are installed (see ReadMe)."
                ) from e
            raise RuntimeError(
                f"Cannot reach OpenAI: {e.message}. Underlying: {cause_s or 'unknown'}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"OpenAI embeddings failed ({type(e).__name__}): {e}"
            ) from e
        ordered = sorted(resp.data, key=lambda x: x.index)
        return [x.embedding for x in ordered]

    def ingest_pdfs(self, force: bool = False) -> IngestResult:
        """Index PDFs under ``data_dir`` and, when configured, Confluence pages."""
        if not settings.openai_api_key:
            logger.error("Cannot ingest without OPENAI_API_KEY")
            return IngestResult(0, 0, 0)

        if force:
            for name in (COLLECTION_OPENAI, COLLECTION_DPR):
                try:
                    self._chroma.delete_collection(name)
                except Exception:
                    pass
            self._collection = self._chroma.get_or_create_collection(
                name=COLLECTION_OPENAI,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )
            self._collection_dpr = self._chroma.get_or_create_collection(
                name=COLLECTION_DPR,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )

        if self._should_skip_ingest(force):
            logger.info("Chroma already populated for retrieval_mode=%s; skip ingest", settings.retrieval_mode)
            return IngestResult(0, 0, 0)

        pattern = os.path.join(settings.data_dir, "**", "*.pdf")
        paths = sorted(glob.glob(pattern, recursive=True))
        if not paths:
            pattern = os.path.join(settings.data_dir, "*.pdf")
            paths = sorted(glob.glob(pattern))

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for pdf_path in paths:
            name = os.path.basename(pdf_path)
            body = _extract_pdf_text(pdf_path)
            if not body.strip():
                logger.warning("No text extracted from %s", pdf_path)
                continue
            chunks = _simple_chunk(body, settings.chunk_size, settings.chunk_overlap)
            for i, ch in enumerate(chunks):
                cid = f"pdf_{Path(name).stem}_{i}_{uuid.uuid4().hex[:8]}"
                ids.append(cid)
                documents.append(ch)
                metadatas.append(
                    {
                        "source": name,
                        "source_type": "pdf",
                        "chunk_index": i,
                    }
                )

        confluence_page_count = 0
        if settings.confluence_enabled:
            spaces = parse_space_keys(settings.confluence_space_keys)
            wiki = (settings.confluence_url or "").strip().rstrip("/")
            if not spaces:
                logger.warning("CONFLUENCE_ENABLED but CONFLUENCE_SPACE_KEYS is empty; skipping Confluence")
            elif not wiki:
                logger.warning("CONFLUENCE_ENABLED but CONFLUENCE_URL is empty; skipping Confluence")
            elif not settings.confluence_email or not settings.confluence_api_token:
                logger.warning(
                    "CONFLUENCE_ENABLED but CONFLUENCE_EMAIL or CONFLUENCE_API_TOKEN missing; skipping Confluence"
                )
            else:
                try:
                    pages = fetch_pages_for_spaces(
                        wiki,
                        settings.confluence_email,
                        settings.confluence_api_token,
                        spaces,
                        max_pages=max(1, settings.confluence_max_pages),
                        batch_limit=max(1, min(100, settings.confluence_batch_limit)),
                    )
                except Exception:
                    logger.exception("Confluence fetch failed")
                    pages = []
                confluence_page_count = len(pages)
                for pg in pages:
                    chunks = _simple_chunk(pg.body, settings.chunk_size, settings.chunk_overlap)
                    safe_title = re.sub(r"[^\w\s.-]", "", pg.title)[:80] or "page"
                    for i, ch in enumerate(chunks):
                        cid = f"cf_{pg.page_id}_{i}_{uuid.uuid4().hex[:8]}"
                        ids.append(cid)
                        documents.append(ch)
                        metadatas.append(
                            {
                                "source": f"Confluence: {pg.title}",
                                "source_type": "confluence",
                                "space_key": pg.space_key,
                                "page_id": pg.page_id,
                                "url": pg.url[:500],
                                "chunk_index": i,
                            }
                        )
                n_cf_chunks = sum(1 for m in metadatas if m.get("source_type") == "confluence")
                logger.info("Confluence: fetched %d pages -> %d chunks", confluence_page_count, n_cf_chunks)

        if not documents:
            logger.warning(
                "No chunks to index (no usable PDF text under %s and/or no Confluence content)",
                settings.data_dir,
            )
            return IngestResult(len(paths), 0, confluence_page_count)

        mode = settings.retrieval_mode

        if mode in ("openai_chroma", "parallel_dpr_openai"):
            batch = 128
            for i in range(0, len(documents), batch):
                chunk_docs = documents[i : i + batch]
                embeddings = self._embed_texts_openai(chunk_docs)
                self._collection.add(
                    ids=ids[i : i + batch],
                    embeddings=embeddings,
                    documents=chunk_docs,
                    metadatas=metadatas[i : i + batch],
                )

        if mode in ("dpr_only", "parallel_dpr_openai"):
            dpr = self._dpr_models()
            dpr_batch = 32
            for i in range(0, len(documents), dpr_batch):
                chunk_docs = documents[i : i + dpr_batch]
                dpr_emb = dpr.embed_passages(chunk_docs)
                self._collection_dpr.add(
                    ids=ids[i : i + dpr_batch],
                    embeddings=dpr_emb,
                    documents=chunk_docs,
                    metadatas=metadatas[i : i + dpr_batch],
                )

        logger.info(
            "Indexed %d chunks from %d PDFs, %d Confluence pages (retrieval_mode=%s)",
            len(documents),
            len(paths),
            confluence_page_count,
            mode,
        )
        return IngestResult(len(paths), len(documents), confluence_page_count)

    def expand_queries(self, user_query: str) -> list[str]:
        n = max(1, settings.expansion_query_count)
        system = (
            "You generate diverse search queries to retrieve passages from a document corpus. "
            "Output valid JSON only: {\"queries\": [\"...\", ...]} with exactly "
            f"{n} strings. Queries must be distinct, concise, and cover different angles "
            "of the user's information need. Do not repeat the original verbatim."
        )
        user = f"Original question:\n{user_query}"
        resp = self._client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
            extra = data.get("queries") or data.get("expanded") or []
            if isinstance(extra, str):
                extra = [extra]
            extra = [str(q).strip() for q in extra if str(q).strip()][:n]
        except json.JSONDecodeError:
            logger.warning("Query expansion JSON parse failed; using original only")
            extra = []

        seen: set[str] = set()
        out: list[str] = []
        for q in [user_query] + extra:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                out.append(q)
        return out

    def _get_cross_encoder(self) -> Any:
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s", settings.cross_encoder_model)
            self._cross_encoder = CrossEncoder(settings.cross_encoder_model)
        return self._cross_encoder

    def _rerank_cross_encoder(
        self,
        user_query: str,
        candidates: list[tuple[str, str, dict[str, Any], float]],
    ) -> list[tuple[str, str, dict[str, Any], float, float]]:
        """Rerank (chunk_id, doc, meta, bi_score) with MS MARCO-style cross-encoder.

        Returns tuples (chunk_id, doc, meta, cross_encoder_score, bi_encoder_score) sorted by CE score.
        """
        if not candidates:
            return []
        try:
            model = self._get_cross_encoder()
            pairs = [(user_query, doc) for _, doc, _, _ in candidates]
            raw = model.predict(pairs, batch_size=16, show_progress_bar=False)
            ce_scores = _scores_to_list(raw)
            ranked: list[tuple[str, str, dict[str, Any], float, float]] = []
            for row, ce in zip(candidates, ce_scores, strict=True):
                cid, doc, meta, bi = row
                ranked.append((cid, doc, meta, float(ce), bi))
            ranked.sort(key=lambda x: x[3], reverse=True)
            return ranked
        except Exception:
            logger.exception("Cross-encoder rerank failed; using bi-encoder scores only")
            return [
                (cid, doc, meta, bi, bi)
                for cid, doc, meta, bi in candidates
            ]

    def _retrieve_merged_by_id(
        self,
        queries: list[str],
        n_results: Optional[int] = None,
    ) -> dict[str, tuple[str, dict[str, Any], float]]:
        k = n_results if n_results is not None else settings.top_k_per_query
        mode = settings.retrieval_mode
        if mode == "openai_chroma":
            return openai_chroma.retrieve_merged_by_id(
                self._collection, self._embed_texts_openai, queries, k
            )
        if mode == "dpr_only":
            return dpr_only_retriever.retrieve_merged_by_id(
                self._collection_dpr, self._dpr_models(), queries, k
            )
        return retrieve_parallel_merged(
            self._collection,
            self._collection_dpr,
            self._embed_texts_openai,
            self._dpr_models(),
            queries,
            k,
        )

    def retrieve_merged(self, queries: list[str]) -> list[tuple[str, dict[str, Any], float]]:
        """Return list of (document, metadata, score) best-first; score is similarity 0-1 for cosine."""
        by_id = self._retrieve_merged_by_id(queries)
        ranked = sorted(by_id.items(), key=lambda x: x[1][2], reverse=True)
        return [tpl[1] for tpl in ranked]

    def retrieve_merged_chunks(self, queries: list[str]) -> list[tuple[str, str, dict[str, Any], float]]:
        """Same as retrieve_merged but includes chunk id for each row."""
        by_id = self._retrieve_merged_by_id(queries)
        ranked = sorted(by_id.items(), key=lambda x: x[1][2], reverse=True)
        return [(cid, doc, meta, score) for cid, (doc, meta, score) in ranked]

    def answer(
        self,
        user_query: str,
        use_expansion: bool = True,
    ) -> dict[str, Any]:
        if self.collection_count() == 0:
            return {
                "answer": (
                    "No documents are indexed yet. Add PDFs under the data directory, "
                    "configure Confluence if used, and restart or call /reindex."
                ),
                "expanded_queries": [],
                "sources": [],
            }

        if use_expansion:
            queries = self.expand_queries(user_query)
        else:
            queries = [user_query]

        n_pool = min(
            settings.rerank_pool_per_query,
            max(1, self.collection_count()),
        )
        by_id = self._retrieve_merged_by_id(queries, n_results=n_pool)
        merged_ranked = sorted(by_id.items(), key=lambda x: x[1][2], reverse=True)
        candidates = [
            (cid, doc, meta, bi)
            for cid, (doc, meta, bi) in merged_ranked[: settings.rerank_max_candidates]
        ]

        if settings.rerank_enabled and candidates:
            reranked = self._rerank_cross_encoder(user_query, candidates)
            top_rows = reranked[: settings.llm_context_top_k]
        else:
            top_rows = [
                (cid, doc, meta, bi, bi)
                for cid, doc, meta, bi in candidates[: settings.llm_context_top_k]
            ]

        context_blocks = []
        sources: list[dict[str, Any]] = []
        for cid, doc, meta, ce_score, bi_score in top_rows:
            src = meta.get("source", "unknown")
            src_type = meta.get("source_type", "")
            extra = ""
            if src_type == "confluence" and meta.get("url"):
                extra = f" ({meta.get('url')})"
            context_blocks.append(f"[Source: {src}{extra}]\n{doc}")
            entry: dict[str, Any] = {
                "chunk_id": cid,
                "source": src,
                "source_type": src_type or None,
                "cross_encoder_score": round(ce_score, 4),
                "bi_encoder_score": round(bi_score, 4),
            }
            if meta.get("url"):
                entry["url"] = meta["url"]
            sources.append(entry)

        context = "\n\n---\n\n".join(context_blocks)
        system = (
            "You are a helpful assistant answering using ONLY the provided context from PDF files "
            "and/or Confluence pages. If the context is insufficient, say so clearly. "
            "Cite sources by name or page title (and URL when given) when relevant."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {user_query}"
        resp = self._client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return {
            "answer": answer,
            "expanded_queries": queries,
            "sources": sources,
        }
