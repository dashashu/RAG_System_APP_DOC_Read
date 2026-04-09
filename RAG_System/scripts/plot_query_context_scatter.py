#!/usr/bin/env python3
"""
PCA scatter of (1) embedding vectors for expanded search queries and (2) embeddings of
retrieved chunks, for a fixed list of sample questions. Queries use the same embedding
model as the index; chunks load vectors from Chroma by id.

  export CHROMA_DIR=./chroma_data DATA_DIR=./data OPENAI_API_KEY=...
  PYTHONPATH=. python scripts/plot_query_context_scatter.py -o plots/queries_context.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import settings  # noqa: E402
from app.openai_client import build_openai_client  # noqa: E402
from app.rag import RAGService  # noqa: E402

DEFAULT_QUESTIONS = [
    "What was Microsoft's total revenue and how did it change year over year?",
    "How is revenue broken down by business segment?",
    "What does the report say about cloud services and Azure?",
    "What are the main risk factors or legal matters discussed?",
]


def _embed_texts(client, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    batch = 64
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        resp = client.embeddings.create(model=settings.embedding_model, input=chunk)
        by_idx = {item.index: item.embedding for item in resp.data}
        for j in range(len(chunk)):
            out.append(by_idx[j])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Query + retrieved-chunk embedding scatter (PCA)")
    p.add_argument("-o", "--output", type=Path, default=Path("plots/query_context_pca.png"))
    p.add_argument("--no-expansion", action="store_true", help="Use only the literal question (no LLM expansion)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "questions",
        nargs="*",
        help="Extra questions (default: built-in sample questions about annual reports)",
    )
    args = p.parse_args()

    if not settings.openai_api_key:
        print("OPENAI_API_KEY is required.", file=sys.stderr)
        return 1

    questions = list(args.questions) if args.questions else list(DEFAULT_QUESTIONS)
    rag = RAGService()
    if rag.collection_count() == 0:
        ing = rag.ingest_pdfs(force=False)
        if ing.chunks_indexed == 0:
            print("No chunks indexed. Check DATA_DIR and PDFs.", file=sys.stderr)
            return 1

    oai = build_openai_client(settings.openai_api_key)
    rows_emb: list[list[float]] = []
    row_colors: list[int] = []
    row_markers: list[str] = []
    color_by_q: dict[str, int] = {q: i for i, q in enumerate(questions)}

    for q in questions:
        cidx = color_by_q[q]
        if args.no_expansion:
            subqs = [q]
        else:
            subqs = rag.expand_queries(q)

        q_embs = _embed_texts(oai, subqs)
        for j, sq in enumerate(subqs):
            rows_emb.append(q_embs[j])
            row_colors.append(cidx)
            row_markers.append("^")

        ranked = rag.retrieve_merged_chunks(subqs)[:12]
        chunk_ids = [cid for cid, _, _, _ in ranked]
        if not chunk_ids:
            continue
        got = rag._collection.get(ids=chunk_ids, include=["embeddings"])
        ids_got = got["ids"] or []
        embs_got = got.get("embeddings") or []
        emb_map = {i: e for i, e in zip(ids_got, embs_got)}
        for cid, _, _, _ in ranked:
            vec = emb_map.get(cid)
            if vec is None:
                continue
            rows_emb.append(list(vec))
            row_colors.append(cidx)
            row_markers.append("o")

    if len(rows_emb) < 3:
        print("Not enough points to plot (need retrieval + embeddings).", file=sys.stderr)
        return 1

    X = np.asarray(rows_emb, dtype=np.float64)
    xy = PCA(n_components=2, random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=args.dpi)
    cmap = colormaps["tab10"].resampled(max(len(questions), 1))

    for i in range(len(xy)):
        ax.scatter(
            xy[i, 0],
            xy[i, 1],
            c=[cmap(row_colors[i])],
            marker=row_markers[i],
            s=120 if row_markers[i] == "^" else 45,
            alpha=0.75 if row_markers[i] == "^" else 0.55,
            edgecolors="k",
            linewidths=0.2,
        )

    ax.set_title("PCA: expanded queries (^) and retrieved chunks (o) per question color")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    q_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=cmap(i),
            markersize=10,
            label=f"Q{i + 1}: {questions[i][:50]}{'…' if len(questions[i]) > 50 else ''}",
        )
        for i in range(len(questions))
    ]
    style_handles = [
        plt.Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=10, label="Sub-query embedding"),
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=8, label="Retrieved chunk"),
    ]
    ax.legend(handles=q_handles + style_handles, loc="best", fontsize=7)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
