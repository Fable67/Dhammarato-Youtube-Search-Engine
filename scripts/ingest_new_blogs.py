#!/usr/bin/env python
"""
One-off glue script: ingest new blog markdown files from data/raw/blogs/
into sanghabot.db, chunk them, and embed them into a NEW immutable
embedding_runs artifact appended after the existing legacy 512-dim run.

Context (see chat history for the full investigation): scripts/ingest.py's
`chunk` command is an unimplemented stub, and the existing 512-dim FAISS
index cannot hold vectors of a different dimension, so new content cannot
simply be "added" via the existing rebuild_index.py `embed` command (which
always re-embeds the ENTIRE corpus). This script instead:

  1. Parses new blog posts (via the existing, unmodified
     sanghabot.ingest.parse_blogs) and inserts `videos` rows, this time
     actually persisting transcript text alongside metadata (which
     scripts/ingest.py's `parse` command currently discards).
  2. Chunks each new transcript via the existing, unmodified
     chunk_transcript_semantic() (verified byte-for-byte logically
     identical to the original AI_Transcripts/chunking.py), using the
     LEGACY model/dimension (qwen/qwen3-embedding-8b @ 1024) for boundary
     -refinement embedding calls, since new content must stay in the same
     vector space as the existing corpus to be merged into one FAISS index.
  3. Embeds the new (non-overlapping) chunk texts the same way, then
     applies legacy_reduce_dimensionality() (existing, verbatim-ported
     averaging transform from sanghabot/embeddings/legacy_compat.py) to
     produce 512-dim vectors consistent with the existing stored
     (un-normalized) embeddings.npy convention.
  4. Loads the existing legacy embeddings.npy (READ-ONLY, never modified
     or overwritten), vstacks the new vectors after it, and writes the
     result to a NEW .npy file. embedding_row values for new chunks
     continue from the existing max (23246, 23247, ...).
  5. Registers a NEW embedding_runs row (the existing legacy row is left
     completely untouched, still is_active=1 until explicitly promoted
     via scripts/rebuild_index.py promote).

After this script runs, use the existing, unmodified
scripts/rebuild_index.py `build-faiss` / `build-bm25` / `promote` commands
to finish wiring the new run in.

Usage:
    conda activate Dhamma
    python scripts/ingest_new_blogs.py --dry-run          # first 2 files only, no DB writes
    python scripts/ingest_new_blogs.py                    # full run over data/raw/blogs/
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
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
from sanghabot.ingest.parse_blogs import iter_blog_posts  # noqa: E402
from sanghabot.models import Chunk, VideoMetadata  # noqa: E402
from sanghabot.storage.db import Database  # noqa: E402

app = typer.Typer(help="Ingest new blog posts into an existing sanghabot.db + legacy-compatible embeddings.")

LEGACY_EMBEDDINGS_NPY = Path(
    "/Users/jonas/Programme/Python/Youtube-Video-Recommendation-Engine/AI_Transcripts/embeddings/embeddings.npy"
)
LEGACY_RUN_ID = "qwen3-4b-512-legacy"

CHUNK_OVERLAP_CHARS = 200  # matches AI_Transcripts/embed_videos.py's extract_chunks() overlap
EMBED_BATCH_SIZE = 8       # matches AI_Transcripts/embed_videos.py's default batch_size


def _extract_chunks(text: str, chunk_borders: list[int], overlap_chars: int) -> list[str]:
    """Verbatim port of AI_Transcripts/embed_videos.py's extract_chunks()."""
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


def _boundary_embed_fn(texts: list[str]) -> list[np.ndarray]:
    """
    Used only for the chunker's internal boundary-refinement embedding
    calls (chunk_transcript_semantic's batch_embed_fn). Requests the
    legacy native dimension/model so results stay comparable in spirit to
    how the original corpus was chunked, though refinement similarity is
    scale-invariant to a reduction step, so plain 1024-dim native vectors
    are used directly here (no need to reduce for this internal use).
    """
    vectors = embed_texts(texts, dimension=LEGACY_NATIVE_DIM, model=LEGACY_MODEL)
    return [np.asarray(v) for v in vectors]


@app.command()
def run(
    blogs_dir: Path = typer.Option(settings.raw_blogs_dir, help="Directory of new .md files to ingest."),
    dry_run: bool = typer.Option(False, help="Only process the first 2 files, print results, write nothing to disk/db."),
    limit: int | None = typer.Option(None, help="Only process the first N files (for staged runs)."),
):
    if not blogs_dir.exists():
        typer.echo(f"Blogs directory not found: {blogs_dir}", err=True)
        raise typer.Exit(code=1)

    if not LEGACY_EMBEDDINGS_NPY.exists():
        typer.echo(f"Legacy embeddings.npy not found at {LEGACY_EMBEDDINGS_NPY}", err=True)
        raise typer.Exit(code=1)

    posts = list(iter_blog_posts(blogs_dir))
    if dry_run:
        posts = posts[:2]
    elif limit is not None:
        posts = posts[:limit]

    typer.echo(f"Processing {len(posts)} blog post(s) from {blogs_dir} (dry_run={dry_run}) ...")

    db = Database(settings.db_path)

    # Existing max embedding_row across the whole chunks table -- new rows
    # continue from here, never reusing/overlapping existing indices.
    cur = db._conn.execute("SELECT MAX(embedding_row) FROM chunks")
    max_existing_row = cur.fetchone()[0]
    next_embedding_row = (max_existing_row + 1) if max_existing_row is not None else 0
    typer.echo(f"Existing max embedding_row = {max_existing_row}; new rows start at {next_embedding_row}")

    all_new_vectors: list[np.ndarray] = []
    all_new_chunks: list[Chunk] = []
    videos_inserted = 0
    chunks_created = 0
    skipped_empty_transcript = []

    for post in tqdm(posts, desc="Ingesting posts"):
        video_id = post.file_name.rsplit(".", 1)[0]
        blog_url = f"https://dhammarato.com/blog/{video_id}"
        transcript = post.transcript or ""

        if len(transcript.strip()) < 20:
            skipped_empty_transcript.append(video_id)
            typer.echo(f"  SKIP (empty/too-short transcript): {video_id}")
            continue

        video_url = f"https://www.youtube.com/watch?v={post.youtube_id}" if post.youtube_id else None

        video = VideoMetadata(
            video_id=video_id,
            title=post.title or "",
            blog_url=blog_url,
            url=video_url,
            published_date=post.pub_date if isinstance(post.pub_date, date) else None,
            tags=post.tags,
            video_length_chars=len(transcript),
        )

        # 1. Chunk boundaries via the existing semantic chunker, using the
        #    legacy-dimension embedding function for boundary refinement.
        borders = chunk_transcript_semantic(
            transcript,
            batch_embed_fn=_boundary_embed_fn,
            debug=False,
        )

        # 2. Extract chunk texts: WITH overlap for embedding context (matches
        #    legacy embed_videos.py), WITHOUT overlap for storage/display text.
        texts_with_overlap = _extract_chunks(transcript, borders, CHUNK_OVERLAP_CHARS)
        texts_no_overlap = _extract_chunks(transcript, borders, 0)

        if dry_run:
            typer.echo(f"  {video_id}: {len(transcript)} chars -> {len(texts_no_overlap)} chunk(s), borders={borders}")

        video_chunks: list[Chunk] = []
        video_vectors: list[np.ndarray] = []

        for i in range(0, len(texts_with_overlap), EMBED_BATCH_SIZE):
            batch_with_overlap = texts_with_overlap[i:i + EMBED_BATCH_SIZE]
            if dry_run:
                # Skip real API calls entirely in dry-run mode for the
                # final embedding step (boundary refinement above still
                # makes real calls if refine_with_embeddings=True and the
                # transcript is long enough to have candidate boundaries --
                # acceptable for a 2-file dry run).
                batch_vectors = [np.zeros(LEGACY_NATIVE_DIM, dtype="float32") for _ in batch_with_overlap]
            else:
                raw = embed_texts(batch_with_overlap, dimension=LEGACY_NATIVE_DIM, model=LEGACY_MODEL)
                batch_vectors = [np.asarray(v) for v in raw]

            batch_matrix = np.array(batch_vectors, dtype="float32")
            reduced = legacy_reduce_dimensionality(batch_matrix, LEGACY_TARGET_DIM)

            for j, vec in enumerate(reduced):
                chunk_idx = i + j
                start_char = borders[chunk_idx - 1] if chunk_idx - 1 >= 0 else 0
                end_char = borders[chunk_idx] if chunk_idx < len(borders) else len(transcript)
                embedding_row = next_embedding_row + len(all_new_vectors) + len(video_vectors)

                video_chunks.append(
                    Chunk(
                        chunk_id=f"{video_id}_{chunk_idx}",
                        video_id=video_id,
                        chunk_idx=chunk_idx,
                        text=texts_no_overlap[chunk_idx],
                        summary=post.summary_markdown or None,
                        start_char=start_char,
                        end_char=end_char,
                        embedding_row=embedding_row,
                    )
                )
                video_vectors.append(vec)

        if not dry_run:
            db.insert_video(video)
            db.insert_chunks(video_chunks)
            videos_inserted += 1
            chunks_created += len(video_chunks)
        else:
            typer.echo(f"    would insert video={video_id!r}, {len(video_chunks)} chunk(s)")

        all_new_chunks.extend(video_chunks)
        all_new_vectors.extend(video_vectors)

    if dry_run:
        typer.echo("\nDRY RUN complete -- no DB writes, no .npy written, no embedding_runs row registered.")
        typer.echo(f"Would have created {len(all_new_chunks)} chunk(s) across {len(posts) - len(skipped_empty_transcript)} video(s).")
        if skipped_empty_transcript:
            typer.echo(f"Skipped (empty transcript): {skipped_empty_transcript}")
        db.close()
        return

    if not all_new_vectors:
        typer.echo("No new chunks were created -- nothing to embed or register.")
        db.close()
        return

    # 3. Splice: load existing legacy .npy (read-only), append new vectors,
    #    write to a NEW file. Original is never modified.
    typer.echo(f"\nLoading existing legacy embeddings from {LEGACY_EMBEDDINGS_NPY} ...")
    existing = np.load(LEGACY_EMBEDDINGS_NPY)
    typer.echo(f"Existing shape: {existing.shape}")

    new_matrix = np.array(all_new_vectors, dtype="float32")
    typer.echo(f"New vectors shape: {new_matrix.shape}")

    assert new_matrix.shape[1] == existing.shape[1] == LEGACY_TARGET_DIM, (
        f"Dimension mismatch: existing={existing.shape[1]}, new={new_matrix.shape[1]}, expected={LEGACY_TARGET_DIM}"
    )
    assert not np.isnan(new_matrix).any(), "NaN values found in new embeddings"
    norms = np.linalg.norm(new_matrix, axis=1)
    assert (norms > 0).all(), "one or more new embeddings have zero norm"
    assert new_matrix.shape[0] == len(all_new_chunks), "chunk count != new embedding count"

    combined = np.vstack([existing, new_matrix]).astype("float32")
    typer.echo(f"Combined shape: {combined.shape}")

    run_id = f"qwen3-4b-512-legacy-plus-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    npy_path = settings.data_dir / "index" / f"embeddings_{run_id}.npy"
    np.save(npy_path, combined)
    typer.echo(f"Wrote {npy_path}")

    db.insert_embedding_run(
        run_id=run_id,
        model_name=LEGACY_MODEL,
        dimension=LEGACY_TARGET_DIM,
        created_at=datetime.now(timezone.utc).isoformat(),
        npy_path=str(npy_path),
        is_active=False,  # never auto-promoted; use scripts/rebuild_index.py promote
        notes=(
            f"Appended {len(all_new_chunks)} newly-chunked/embedded rows after the "
            f"existing {existing.shape[0]} legacy vectors (read from {LEGACY_EMBEDDINGS_NPY}, "
            "never modified). New chunks embedded via the same legacy pipeline "
            f"({LEGACY_MODEL} @ {LEGACY_NATIVE_DIM}, averaged down to {LEGACY_TARGET_DIM}) "
            "for vector-space compatibility with the existing corpus. "
            "See scripts/ingest_new_blogs.py."
        ),
    )
    typer.echo(f"Registered embedding_runs row '{run_id}' (not yet active).")

    db.close()
    typer.echo(
        f"\nDone. videos_inserted={videos_inserted}, chunks_created={chunks_created}. "
        f"Next steps:\n"
        f"  python scripts/rebuild_index.py build-faiss --run-id {run_id}\n"
        f"  python scripts/rebuild_index.py build-bm25\n"
        f"  (verify, then) python scripts/rebuild_index.py promote --run-id {run_id}"
    )


if __name__ == "__main__":
    app()
