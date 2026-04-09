from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["openai_chroma", "dpr_only", "parallel_dpr_openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_openai_api_key(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("confluence_api_token", "confluence_email", mode="before")
    @classmethod
    def strip_confluence_secrets(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    data_dir: str = "/data"
    chroma_dir: str = "/app/chroma_data"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k_per_query: int = 6
    expansion_query_count: int = 4

    # First-stage retrieval: pick ONE mode (reindex when switching openai_chroma <-> dpr_only)
    retrieval_mode: RetrievalMode = "openai_chroma"
    dpr_ctx_model: str = "facebook/dpr-ctx_encoder-single-nq-base"
    dpr_q_model: str = "facebook/dpr-question_encoder-single-nq-base"

    # Cross-encoder reranking (MS MARCO MiniLM; override with any HF CrossEncoder id)
    rerank_enabled: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Chroma hits per expanded query before merge (long-tail pool)
    rerank_pool_per_query: int = 24
    # Cap unique chunks sent to cross-encoder (latency)
    rerank_max_candidates: int = 80
    # Chunks fed to the chat model after reranking
    llm_context_top_k: int = 12

    # Atlassian Confluence (Cloud REST API); indexed together with PDFs on ingest
    confluence_enabled: bool = False
    # Wiki root, e.g. https://your-site.atlassian.net/wiki
    confluence_url: str = ""
    confluence_email: str = ""
    confluence_api_token: str = ""
    # Comma-separated space keys, e.g. "ENG,DOCS"
    confluence_space_keys: str = ""
    confluence_max_pages: int = 300
    confluence_batch_limit: int = 50


settings = Settings()
