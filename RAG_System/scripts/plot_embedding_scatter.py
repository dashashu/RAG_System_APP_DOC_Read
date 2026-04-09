#!/usr/bin/env python3
"""
2D scatter plot of chunk embeddings from Chroma (PCA or t-SNE), colored by source PDF.

From repo root (after indexing):

  pip install -r requirements-viz.txt
  export CHROMA_DIR=./chroma_data DATA_DIR=./data OPENAI_API_KEY=...
  PYTHONPATH=. python scripts/plot_embedding_scatter.py -o plots/chunks_pca.png
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
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chromadb  # noqa: E402

from app.config import settings  # noqa: E402

COLLECTION = "rag_pdf_docs"


def _fetch_all_embeddings(collection, page_size: int = 256):
    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    offset = 0
    while True:
        batch = collection.get(
            include=["embeddings", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        part_ids = batch.get("ids") or []
        if not part_ids:
            break
        emb = batch.get("embeddings")
        meta = batch.get("metadatas") or []
        if not emb or len(emb) != len(part_ids):
            raise RuntimeError(
                "Missing embeddings in Chroma. Re-index with the same embedding model."
            )
        ids.extend(part_ids)
        embeddings.extend(emb)
        metadatas.extend(meta if len(meta) == len(part_ids) else [{}] * len(part_ids))
        offset += len(part_ids)
        if len(part_ids) < page_size:
            break
    return ids, embeddings, metadatas


def main() -> int:
    p = argparse.ArgumentParser(description="Scatter plot of PDF chunk embeddings")
    p.add_argument("-o", "--output", type=Path, default=Path("plots/chunk_embeddings.png"))
    p.add_argument("--method", choices=("pca", "tsne"), default="pca")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    os.makedirs(settings.chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    try:
        # Read-only: no API key needed if embeddings are already stored
        collection = client.get_collection(name=COLLECTION)
    except Exception as e:
        print(f"Open collection failed: {e}", file=sys.stderr)
        return 1

    if collection.count() == 0:
        print("Empty index. Run the API or: PYTHONPATH=. python -c \"from app.rag import RAGService; RAGService().ingest_pdfs()\"", file=sys.stderr)
        return 1

    _, embeddings, metadatas = _fetch_all_embeddings(collection)
    if len(embeddings) < 2:
        print("Need at least 2 chunks.", file=sys.stderr)
        return 1

    X = np.asarray(embeddings, dtype=np.float64)
    sources = [m.get("source", "unknown") if m else "unknown" for m in metadatas]
    le = LabelEncoder()
    c_idx = le.fit_transform(sources)
    n_cls = len(le.classes_)

    if args.method == "pca":
        red = PCA(n_components=2, random_state=args.seed)
        xy = red.fit_transform(X)
        title = f"Chunk embeddings (PCA) — {len(embeddings)} chunks, {n_cls} PDFs"
        sub = f"Explained variance: {red.explained_variance_ratio_.sum():.1%}"
    else:
        perp = min(args.perplexity, max(5.0, (len(X) - 1) // 3))
        red = TSNE(
            n_components=2,
            random_state=args.seed,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
        )
        xy = red.fit_transform(X)
        title = f"Chunk embeddings (t-SNE) — {len(embeddings)} chunks, {n_cls} PDFs"
        sub = f"perplexity={perp:.1f}"

    cmap = colormaps["tab20"].resampled(max(n_cls, 1))
    fig, ax = plt.subplots(figsize=(10, 8), dpi=args.dpi)
    ax.scatter(xy[:, 0], xy[:, 1], c=c_idx, cmap=cmap, alpha=0.65, s=14, edgecolors="none")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    fig.text(0.5, 0.02, sub, ha="center", fontsize=9, style="italic")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=8,
            label=name,
        )
        for i, name in enumerate(le.classes_)
    ]
    ax.legend(
        handles=handles,
        title="Source PDF",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
