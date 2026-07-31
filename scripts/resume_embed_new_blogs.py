#!/usr/bin/env python
"""
Resume/finish step for scripts/ingest_new_blogs.py.

Context: the first full run of ingest_new_blogs.py successfully parsed,
chunked, and inserted `videos`/`chunks` rows for 37 of 39 new blog posts
(one was correctly skipped for having a near-empty transcript), but
crashed on the 39th/last post (the largest, ~68K chars) with a raw
`requests.exceptions.ReadTimeout` during an embedding API call for
boundary refinement -- a connection-level timeout that is NOT one of the
transient-error types embed_texts() retries (see
sanghabot/embeddings/client.py: only TransientEmbeddingError, i.e.
HTTP 429/5xx, is retried; a socket-level ReadTimeout propagates
immediately). Since Database.insert_video()/insert_chunks() commit
immediately per-post, the 37 already-processed videos' chunk rows
(chunk boundaries, chunk_id, embedding_row assignments) are safely
persisted in sanghabot.db -- only the actual embedding vectors for those
96 chunks were lost (they only ever existed in memory, spliced into the
.npy in one batch at the very end of the original script, which never
ran).

This script re-embeds exactly those already-chunked rows (embedding_row
>= 23246, i.e. every chunk inserted by the interrupted run) without
re-running the expensive chunker: since chunk_transcript_semantic() is
deterministic and chunk boundaries are already fixed and stored
(start_char/end_char), the exact "with overlap" text used for embedding
(matching AI_Transcripts/embed_videos.py's extract_chunks() with
CHUNK_OVERLAP_CHARS=200) can be reconstructed directly by slicing the
original transcript at [start_char-overlap, end_char+overlap], with no
need to recompute boundaries.

Adds real retry handling around requests.exceptions.RequestException
(covering ReadTimeout/ConnectionError/etc, not just HTTP-level errors)
so a flaky connection doesn't lose an entire run again.

Usage:
    conda activate Dhamma
    python scripts/resume_embed_new_blogs.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
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
from sanghabot.ingest.parse_blogs import process_markdown_file  # noqa: E402
from sanghabot.storage.db import Database  # noqa: E402

app = typer.Typer()

LEGACY_EMBEDDINGS_NPY = Path(
    "/Users/jonas/Programme/Python/Youtube-Video-Recommendation-Engine/AI_Transcripts/embeddings/embeddings.npy"
)
CHUNK_OVERLAP_CHARS = 200
EMBED_BATCH_SIZE = 8
NEW_ROW_START = 23246  # matches the first embedding_row assigned by the interrupted run

# Ordinary function-level Python retry (not tenacity) so we can catch the
# broad requests.exceptions.RequestException umbrella (covers ReadTimeout,
# ConnectionError, etc.) in addition to whatever embed_texts() already
# retries internally for HTTP 429/5xx.
def _embed_with_retry(texts: list[str], max_attempts: int = 6) -> list[np.ndarray]:
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return embed_texts(texts, dimension=LEGACY_NATIVE_DIM, model=LEGACY_MODEL)
        except requests.exceptions.RequestException as e:
            last_exc = e
            wait = min(2 ** attempt, 60)
            typer.echo(f"    [retry {attempt}/{max_attempts}] {type(e).__name__}: {e} -- sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed after {max_attempts} attempts") from last_exc


@app.command()
def run(blogs_dir: Path = typer.Option(settings.raw_blogs_dir)):
    db = Database(settings.db_path)

    cur = db._conn.execute(
        "SELECT chunk_id, video_id, start_char, end_char, embedding_row "
        "FROM chunks WHERE embedding_row >= ? ORDER BY embedding_row",
        (NEW_ROW_START,),
    )
    rows = cur.fetchall()
    typer.echo(f"Found {len(rows)} already-chunked rows needing embeddings (embedding_row >= {NEW_ROW_START}).")
    if not rows:
        typer.echo("Nothing to do.")
        db.close()
        return

    # Sanity: embedding_row must be exactly contiguous 23246..23246+N-1 with
    # no gaps/dupes, or our positional vstack below would misalign vectors.
    rows_by_er = sorted(rows, key=lambda r: r[4])
    expected_ers = list(range(NEW_ROW_START, NEW_ROW_START + len(rows)))
    actual_ers = [r[4] for r in rows_by_er]
    assert actual_ers == expected_ers, (
        f"embedding_row values are not contiguous from {NEW_ROW_START}: "
        f"got {actual_ers[:5]}...{actual_ers[-5:]}"
    )

    # Cache transcript text per video_id (re-parsed from the source .md,
    # never mutated -- chunk boundaries are already fixed/stored).
    transcript_cache: dict[str, str] = {}

    def get_transcript(video_id: str) -> str:
        if video_id not in transcript_cache:
            fp = blogs_dir / f"{video_id}.md"
            post = process_markdown_file(fp)
            transcript_cache[video_id] = post.transcript
        return transcript_cache[video_id]

    texts_with_overlap = []
    for chunk_id, video_id, start_char, end_char, embedding_row in rows_by_er:
        transcript = get_transcript(video_id)
        s = max(start_char - CHUNK_OVERLAP_CHARS, 0)
        e = min(end_char + CHUNK_OVERLAP_CHARS, len(transcript))
        texts_with_overlap.append(transcript[s:e])

    typer.echo(f"Reconstructed {len(texts_with_overlap)} with-overlap texts across {len(transcript_cache)} video(s).")

    all_vectors: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts_with_overlap), EMBED_BATCH_SIZE), desc="Embedding batches"):
        batch = texts_with_overlap[i:i + EMBED_BATCH_SIZE]
        raw = _embed_with_retry(batch)
        batch_matrix = np.array([np.asarray(v) for v in raw], dtype="float32")
        reduced = legacy_reduce_dimensionality(batch_matrix, LEGACY_TARGET_DIM)
        all_vectors.extend(reduced)

    new_matrix = np.array(all_vectors, dtype="float32")
    typer.echo(f"New vectors shape: {new_matrix.shape}")

    assert new_matrix.shape[0] == len(rows_by_er), "vector count != chunk count"
    assert new_matrix.shape[1] == LEGACY_TARGET_DIM, f"expected dim {LEGACY_TARGET_DIM}, got {new_matrix.shape[1]}"
    assert not np.isnan(new_matrix).any(), "NaN values found in new embeddings"
    norms = np.linalg.norm(new_matrix, axis=1)
    assert (norms > 0).all(), "one or more new embeddings have zero norm"

    typer.echo(f"Loading existing legacy embeddings from {LEGACY_EMBEDDINGS_NPY} ...")
    existing = np.load(LEGACY_EMBEDDINGS_NPY)
    typer.echo(f"Existing shape: {existing.shape}")
    assert existing.shape[1] == LEGACY_TARGET_DIM

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
        is_active=False,
        notes=(
            f"Appended {len(rows_by_er)} newly-chunked/embedded rows after the "
            f"existing {existing.shape[0]} legacy vectors (read from {LEGACY_EMBEDDINGS_NPY}, "
            "never modified). New chunks embedded via the same legacy pipeline "
            f"({LEGACY_MODEL} @ {LEGACY_NATIVE_DIM}, averaged down to {LEGACY_TARGET_DIM}) "
            "for vector-space compatibility with the existing corpus. "
            "See scripts/ingest_new_blogs.py + scripts/resume_embed_new_blogs.py "
            "(the initial run crashed on a raw network ReadTimeout on the last/"
            "largest file; chunk rows were already committed, so this resume "
            "step only re-did the embedding step from already-fixed chunk "
            "boundaries, with proper retry handling added)."
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
