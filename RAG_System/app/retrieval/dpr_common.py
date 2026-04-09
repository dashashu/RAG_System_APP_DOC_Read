"""
Shared DPR (Dense Passage Retrieval) encoders for ``dpr_only`` and ``parallel_dpr_openai``.

Uses Hugging Face ``facebook/dpr-*-single-nq-base`` checkpoints: separate question and
context encoders, L2-normalized for cosine similarity in Chroma.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DPRModels:
    """Lazy-load DPR question + context encoders (CPU by default)."""

    def __init__(
        self,
        ctx_model_id: str = "facebook/dpr-ctx_encoder-single-nq-base",
        q_model_id: str = "facebook/dpr-question_encoder-single-nq-base",
    ) -> None:
        self._ctx_model_id = ctx_model_id
        self._q_model_id = q_model_id
        self._ctx_model: Any = None
        self._ctx_tokenizer: Any = None
        self._q_model: Any = None
        self._q_tokenizer: Any = None
        self._device: Any = None

    def _ensure_loaded(self) -> None:
        if self._ctx_model is not None:
            return
        import torch
        from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading DPR context encoder %s", self._ctx_model_id)
        self._ctx_tokenizer = AutoTokenizer.from_pretrained(self._ctx_model_id)
        self._ctx_model = DPRContextEncoder.from_pretrained(self._ctx_model_id)
        self._ctx_model.to(self._device)
        self._ctx_model.eval()

        logger.info("Loading DPR question encoder %s", self._q_model_id)
        self._q_tokenizer = AutoTokenizer.from_pretrained(self._q_model_id)
        self._q_model = DPRQuestionEncoder.from_pretrained(self._q_model_id)
        self._q_model.to(self._device)
        self._q_model.eval()

    @staticmethod
    def _normalize(emb: Any) -> Any:
        import torch

        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def embed_passages(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """L2-normalized context vectors for indexing."""
        self._ensure_loaded()
        import torch

        out: list[list[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._ctx_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self._device)
                pooled = self._ctx_model(**enc).pooler_output
                pooled = self._normalize(pooled)
                out.extend(pooled.cpu().numpy().tolist())
        return out

    def embed_questions(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """L2-normalized question vectors for query time."""
        self._ensure_loaded()
        import torch

        out: list[list[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._q_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self._device)
                pooled = self._q_model(**enc).pooler_output
                pooled = self._normalize(pooled)
                out.extend(pooled.cpu().numpy().tolist())
        return out
