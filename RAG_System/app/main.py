import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.rag import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_rag: RAGService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _rag
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not set")
        _rag = None
    else:
        _rag = RAGService()
        force = os.environ.get("RAG_FORCE_REINDEX", "").lower() in ("1", "true", "yes")
        res = _rag.ingest_pdfs(force=force)
        logger.info(
            "Ingest: files=%s chunks=%s confluence_pages=%s",
            res.files_processed,
            res.chunks_indexed,
            res.confluence_pages,
        )
    yield
    _rag = None


app = FastAPI(title="PDF RAG", lifespan=lifespan)


class QueryBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    use_expansion: bool = True


class ReindexBody(BaseModel):
    force: bool = False


@app.get("/health")
def health():
    ok = bool(settings.openai_api_key) and _rag is not None
    n = _rag.collection_count() if _rag else 0
    return {
        "status": "ok" if ok else "degraded",
        "indexed_chunks": n,
        "retrieval_mode": settings.retrieval_mode,
    }


@app.post("/query")
def query(body: QueryBody):
    if not settings.openai_api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    if _rag is None:
        raise HTTPException(503, "RAG not initialized")
    return _rag.answer(body.question, use_expansion=body.use_expansion)


@app.post("/reindex")
def reindex(body: ReindexBody):
    if _rag is None:
        raise HTTPException(503, "RAG not initialized")
    res = _rag.ingest_pdfs(force=body.force)
    return {
        "files_processed": res.files_processed,
        "chunks_indexed": res.chunks_indexed,
        "confluence_pages": res.confluence_pages,
    }


@app.post("/expand-only")
def expand_only(body: QueryBody):
    """Debug: return LLM-expanded queries without retrieval."""
    if not settings.openai_api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    if _rag is None:
        raise HTTPException(503, "RAG not initialized")
    if not body.use_expansion:
        return {"expanded_queries": [body.question]}
    return {"expanded_queries": _rag.expand_queries(body.question)}
