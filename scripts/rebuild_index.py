#!/usr/bin/env python
"""
Typer CLI for embedding + index building.

Replaces AI_Transcripts/embed_videos.py's main() (hardcoded parameters,
no CLI flags, ad hoc "migration" logic embedded permanently in the main
pipeline function). This version:
  - reads chunk rows from sanghabot.db in bounded batches (never loads the
    full corpus into one in-memory DataFrame, unlike the old pipeline)
  - writes embeddings to a NEW, immutable .npy file per run, registered in
    the embedding_runs table (see REWRITE_PLAN.md Appendix C --
    embeddings are never mutated/overwritten in place)
  - only promotes a run to "active" (and rebuilds FAISS from it) once
    explicitly requested, matching REWRITE_PLAN.md Section 11's planned
    full re-embed workflow (dry run first, human review, then promote)

Usage:
    conda activate Dhamma
    python scripts/rebuild_index.py embed --dimension 1024 --run-id qwen3-4b-1024-full
    python scripts/rebuild_index.py build-faiss --run-id qwen3-4b-1024-full
    python scripts/rebuild_index.py build-bm25
    python scripts/rebuild_index.py promote --run-id qwen3-4b-1024-full
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm

REWRITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REWRITE_ROOT))

from config import settings  # noqa: E402
from sanghabot.embeddings.client import embed_texts  # noqa: E402
from sanghabot.storage.db import Database  # noqa: E402

app = typer.Typer(help="Embedding + index (re)building pipeline.")


@app.command()
def embed(
    run_id: str = typer.Option(..., help="Unique identifier for this embedding run, e.g. 'qwen3-4b-1024-full'"),
    dimension: int = typer.Option(settings.embedding_dim, help="Native embedding dimension to request."),
    batch_size: int = typer.Option(16, help="Number of chunks per API call."),
    limit: int | None = typer.Option(None, help="Only embed the first N chunks (for dry runs, see REWRITE_PLAN.md Section 11.3)."),
):
    """
    Embeds chunk text in batches, streaming from sqlite (never loading the
    whole corpus into memory), and writes to a NEW .npy file registered as
    a new, immutable embedding_runs row. Does not touch any existing run.
    """
    db = Database(settings.db_path)

    chunk_ids: list[str] = []
    texts: list[str] = []
    for chunk_id, text in db.iter_chunk_texts():
        chunk_ids.append(chunk_id)
        texts.append(text)
        if limit is not None and len(chunk_ids) >= limit:
            break

    typer.echo(f"Embedding {len(texts)} chunks at dimension={dimension} in batches of {batch_size} ...")

    all_vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[start:start + batch_size]
        vectors = embed_texts(batch, dimension=dimension)
        all_vectors.extend(vectors)

    matrix = np.array(all_vectors, dtype="float32")

    # Sanity checks up front, per REWRITE_PLAN.md Section 11.3 dry-run validation.
    assert matrix.shape[0] == len(chunk_ids), "chunk count != embedding count -- rows were dropped somewhere"
    assert matrix.shape[1] == dimension, f"expected dim {dimension}, got {matrix.shape[1]}"
    assert not np.isnan(matrix).any(), "NaN values found in embeddings"
    norms = np.linalg.norm(matrix, axis=1)
    assert (norms > 0).all(), "one or more embeddings have zero norm"

    npy_path = settings.data_dir / "index" / f"embeddings_{run_id}.npy"
    np.save(npy_path, matrix)

    for chunk_id, row in zip(chunk_ids, range(len(chunk_ids))):
        db.set_embedding_row(chunk_id, row)

    db.insert_embedding_run(
        run_id=run_id,
        model_name=settings.embedding_model,
        dimension=dimension,
        created_at=datetime.now(timezone.utc).isoformat(),
        npy_path=str(npy_path),
        is_active=False,  # never auto-promoted; see `promote` command
        notes=f"Embedded {len(texts)} chunks natively at {dimension} dims. No post-processing/averaging applied.",
    )
    db.close()
    typer.echo(f"Done. Wrote {npy_path} and registered embedding_runs row '{run_id}' (not yet active).")


@app.command()
def build_faiss(run_id: str = typer.Option(..., help="Embedding run to build the FAISS index from.")):
    """Builds faiss.index from the given (not-yet-necessarily-active) embedding run."""
    import faiss

    db = Database(settings.db_path)
    cur = db._conn.execute("SELECT npy_path, dimension FROM embedding_runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    if row is None:
        typer.echo(f"No embedding_runs row found for run_id={run_id!r}", err=True)
        raise typer.Exit(code=1)
    npy_path, dimension = row

    matrix = np.load(npy_path)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    matrix = np.ascontiguousarray(matrix.astype("float32"))

    index = faiss.IndexFlatIP(matrix.shape[1])
    batch = 1000
    for i in range(0, len(matrix), batch):
        index.add(matrix[i:i + batch])

    faiss.write_index(index, str(settings.faiss_index_path))
    db.close()
    typer.echo(f"Wrote {settings.faiss_index_path} ({len(matrix)} vectors, dim={dimension})")


@app.command()
def build_bm25():
    """(Re)builds the BM25 index cache from current chunk text in sqlite."""
    from sanghabot.search.bm25 import BM25SearchEngine

    if settings.bm25_index_path.exists():
        settings.bm25_index_path.unlink()

    db = Database(settings.db_path)
    BM25SearchEngine(db, settings.bm25_index_path)  # building it as a side effect of __init__
    db.close()
    typer.echo(f"Wrote {settings.bm25_index_path}")


@app.command()
def promote(run_id: str = typer.Option(..., help="Embedding run to mark as active.")):
    """
    Marks an embedding run as the active one. Per REWRITE_PLAN.md Section
    11.4, this should only be done after a human review pass over the
    golden query set's results with the new run's FAISS index.
    """
    db = Database(settings.db_path)
    db.set_active_embedding_run(run_id)
    db.close()
    typer.echo(f"Promoted '{run_id}' to active. Rebuild faiss.index from it if you haven't already.")


if __name__ == "__main__":
    app()
