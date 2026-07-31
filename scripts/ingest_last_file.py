#!/usr/bin/env python
"""
One-off: finish ingesting the single file that crashed the original
scripts/ingest_new_blogs.py run (2026-07-26-first-jhana-..., the largest
of the 39 new posts at ~68K chars) -- it failed with a raw network
ReadTimeout during chunk-boundary-refinement embedding, before its
`videos`/`chunks` rows were ever inserted (unlike the other 37, this one
left NO partial state in the DB -- verified separately).

Reuses the same chunk_transcript_semantic() + legacy embedding pipeline as
scripts/ingest_new_blogs.py, but with retry handling around raw
requests.exceptions.RequestException (ReadTimeout/ConnectionError/etc, not
just HTTP 429/5xx) for both the boundary-refinement calls and the final
embedding calls, so a flaky connection can't lose this run either.

Appends this file's new chunk embeddings on top of the
"qwen3-4b-512-legacy-plus-<date>" run already created by
scripts/resume_embed_new_blogs.py (which itself is legacy.npy + the other
37 videos' 96 chunks) -- NOT the original legacy.npy -- so all new content
ends up in one single final combined run.

Usage:
    conda activate Dhamma
    python scripts/ingest_last_file.py
"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import requests
import typer
from tqdm import tqdm

REWRITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REWRITE_ROOT))

from config import settings  # noqa: E402
from sanghabot.embeddings.client import embed_texts  # noqa: E402
from sanghabot.embeddings.legacy_compat import (  # noqa: E402
    LEGACY_MODEL,
    LEGACY_NATIVE_DIM,
    LEGACY_TARGET_DIM,
    legacy_reduce_dimensionality,
)
from sanghabot.ingest.chunker import chunk_transcript_semantic  # noqa: E402
from sanghabot.ingest.parse_blogs import process_markdown_file  # noqa: E402
from sanghabot.models import Chunk, VideoMetadata  # noqa: E402
from sanghabot.storage.db import Database  # noqa: E402

app = typer.Typer()

VIDEO_ID = "2026-07-26-first-jhana-is-all-that-s-necessary-so-that-you-can-see-sangha-uk-364-07-26-26"
PREV_RUN_ID = "qwen3-4b-512-legacy-plus-20260731"
CHUNK_OVERLAP_CHARS = 200
EMBED_BATCH_SIZE = 8


def _embed_with_retry(texts: list[str], max_attempts: int = 6) -> list[np.ndarray]:
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = embed_texts(texts, dimension=LEGACY_NATIVE_DIM, model=LEGACY_MODEL)
            return [np.asarray(v) for v in raw]
        except requests.exceptions.RequestException as e:
            last_exc = e
            wait = min(2 ** attempt, 60)
            typer.echo(f"    [retry {attempt}/{max_attempts}] {type(e).__name__}: {e} -- sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed after {max_attempts} attempts") from last_exc


def _extract_chunks(text: str, chunk_borders: list[int], overlap_chars: int) -> list[str]:
    chunk_texts = []
    idx = 0
    for i in range(len(chunk_borders) + 1):
        start_idx = max(idx - overlap_chars, 0)
        if i < len(chunk_borders):
            end_idx = min(chunk_borders[i] + overlap_chars, len(text))
            chunk_texts.append(text[start_idx:end_idx])
            idx = chunk_borders[i]
        else:
            chunk_texts.append(text[start_idx:])
    return chunk_texts


@app.command()
def run(blogs_dir: Path = typer.Option(settings.raw_blogs_dir)):
    db = Database(settings.db_path)

    existing_video = db.get_video(VIDEO_ID)
    if existing_video is not None:
        typer.echo(f"Video {VIDEO_ID!r} already exists in DB -- aborting to avoid duplicate work.", err=True)
        db.close()
        raise typer.Exit(code=1)

    cur = db._conn.execute("SELECT MAX(embedding_row) FROM chunks")
    max_existing_row = cur.fetchone()[0]
    next_row = (max_existing_row + 1) if max_existing_row is not None else 0
    typer.echo(f"Existing max embedding_row = {max_existing_row}; new rows start at {next_row}")

    fp = blogs_dir / f"{VIDEO_ID}.md"
    post = process_markdown_file(fp)
    transcript = post.transcript
    typer.echo(f"Parsed {VIDEO_ID}: {len(transcript)} chars")

    video_url = f"https://www.youtube.com/watch?v={post.youtube_id}" if post.youtube_id else None
    video = VideoMetadata(
        video_id=VIDEO_ID,
        title=post.title or "",
        blog_url=f"https://dhammarato.com/blog/{VIDEO_ID}",
        url=video_url,
        published_date=post.pub_date if isinstance(post.pub_date, date) else None,
        tags=post.tags,
        video_length_chars=len(transcript),
    )

    def boundary_embed_fn(texts: list[str]) -> list[np.ndarray]:
        return _embed_with_retry(texts)

    typer.echo("Chunking (this calls the embedding API for boundary refinement, may take several minutes) ...")
    borders = chunk_transcript_semantic(transcript, batch_embed_fn=boundary_embed_fn, debug=False)
    typer.echo(f"Chunk borders: {borders}")

    texts_with_overlap = _extract_chunks(transcript, borders, CHUNK_OVERLAP_CHARS)
    texts_no_overlap = _extract_chunks(transcript, borders, 0)
    typer.echo(f"{len(texts_no_overlap)} chunk(s) produced.")

    chunks: list[Chunk] = []
    vectors: list[np.ndarray] = []

    for i in tqdm(range(0, len(texts_with_overlap), EMBED_BATCH_SIZE), desc="Embedding final chunks"):
        batch = texts_with_overlap[i:i + EMBED_BATCH_SIZE]
        raw_vectors = _embed_with_retry(batch)
        batch_matrix = np.array(raw_vectors, dtype="float32")
        reduced = legacy_reduce_dimensionality(batch_matrix, LEGACY_TARGET_DIM)

        for j, vec in enumerate(reduced):
            chunk_idx = i + j
            start_char = borders[chunk_idx - 1] if chunk_idx - 1 >= 0 else 0
            end_char = borders[chunk_idx] if chunk_idx < len(borders) else len(transcript)
            embedding_row = next_row + len(vectors)
            chunks.append(
                Chunk(
                    chunk_id=f"{VIDEO_ID}_{chunk_idx}",
                    video_id=VIDEO_ID,
                    chunk_idx=chunk_idx,
                    text=texts_no_overlap[chunk_idx],
                    summary=post.summary_markdown or None,
                    start_char=start_char,
                    end_char=end_char,
                    embedding_row=embedding_row,
                )
            )
            vectors.append(vec)

    db.insert_video(video)
    db.insert_chunks(chunks)
    typer.echo(f"Inserted video + {len(chunks)} chunk(s) into sanghabot.db.")

    new_matrix = np.array(vectors, dtype="float32")
    assert new_matrix.shape[0] == len(chunks)
    assert new_matrix.shape[1] == LEGACY_TARGET_DIM
    assert not np.isnan(new_matrix).any()
    assert (np.linalg.norm(new_matrix, axis=1) > 0).all()

    cur = db._conn.execute("SELECT npy_path FROM embedding_runs WHERE run_id = ?", (PREV_RUN_ID,))
    row = cur.fetchone()
    if row is None:
        db.close()
        raise RuntimeError(f"Previous run {PREV_RUN_ID!r} not found in embedding_runs")
    prev_npy_path = row[0]
    typer.echo(f"Loading previous combined embeddings from {prev_npy_path} ...")
    prev = np.load(prev_npy_path)
    typer.echo(f"Previous shape: {prev.shape}")

    combined = np.vstack([prev, new_matrix]).astype("float32")
    typer.echo(f"Final combined shape: {combined.shape}")

    run_id = f"qwen3-4b-512-legacy-plus-{datetime.now(timezone.utc).strftime('%Y%m%d')}-final"
    npy_path = settings.data_dir / "index" / f"embeddings_{run_id}.npy"
    np.save(npy_path, combined)
    typer.echo(f"Wrote {npy_path}")

    db.insert_embedding_run(
        run_id=run_id,
        model_name=LEGACY_MODEL,
        dimension=LEGACY_TARGET_DIM,
        created_at=datetime.now(timezone.utc).isoformat(),
        npy_path=str(npy_path),
        is_active=False,
        notes=(
            f"Final combined run: {prev_npy_path} (legacy 23246 + 96 rows from the "
            f"first 37 new videos) plus {len(chunks)} more rows for the 39th/last new "
            "video (2026-07-26-first-jhana...), which crashed the original ingest run "
            "on a raw network ReadTimeout. See scripts/ingest_new_blogs.py, "
            "scripts/resume_embed_new_blogs.py, scripts/ingest_last_file.py."
        ),
    )
    typer.echo(f"Registered embedding_runs row '{run_id}' (not yet active).")
    db.close()
    typer.echo(
        f"\nDone. Next steps:\n"
        f"  python scripts/rebuild_index.py build-faiss --run-id {run_id}\n"
        f"  python scripts/rebuild_index.py build-bm25\n"
        f"  (verify, then) python scripts/rebuild_index.py promote --run-id {run_id}"
    )


if __name__ == "__main__":
    app()
