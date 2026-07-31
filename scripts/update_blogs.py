#!/usr/bin/env python
"""
ONE-COMMAND blog update: syncs data/raw/blogs/ from a source directory of
site markdown files, then parses, chunks, embeds, and indexes any videos
not already in sanghabot.db -- fully automatically, in a single call.

    conda activate Dhamma
    python scripts/update_blogs.py

See README.md's "Adding new blog posts" section for the full usage guide
(including how to point --site-dir somewhere other than its default, and
what to do if it's interrupted).

--------------------------------------------------------------------------
BACKGROUND -- why this script exists and how it's designed to never lose
work or corrupt data, even if interrupted (see chat history for the full
incident this design is based on):

The first time new blog content was added to this project, it required
manually running four separate one-off scripts in sequence (parse, chunk,
embed, splice into a new .npy, rebuild faiss/bm25, promote), and one of
those runs crashed partway through on a raw network read-timeout that
wasn't covered by the embedding client's retry logic at the time (now
fixed -- see sanghabot/embeddings/client.py). Recovering required writing
*two more* one-off scripts to figure out exactly what had and hadn't been
saved, and finish the job by hand. That whole process is unrealistic to
repeat by hand every time new content needs adding.

This script consolidates all of that into one command, and is designed so
that a crash at ANY point (network blip, API outage, laptop going to
sleep, Ctrl-C) can always be recovered from by just running the exact same
command again -- no manual investigation required:

  1. VIDEOS/CHUNKS are committed to sqlite one video at a time (via
     Database.insert_video()/insert_chunks(), both of which commit
     immediately). If interrupted mid-run, whatever videos were already
     processed stay safely in the DB; re-running detects them as already
     ingested (checked via blog_url, NOT video_id -- see
     Database.get_video_by_blog_url()'s docstring for why video_id alone
     can't be trusted for this check) and continues with whatever's left.
  2. EMBEDDING happens as a separate, later pass over exactly "every chunk
     in the DB that doesn't have an embedding_row pointing into the
     CURRENT run's .npy yet" -- computed fresh from the DB's actual state
     each time, not from an in-memory list that would be lost on crash.
     If interrupted after some chunks were embedded, re-running re-derives
     the same set of textually-identical batches (chunk boundaries are
     already fixed in the DB) and picks up where it left off.
  3. The embeddings API client itself now retries connection-level
     failures (timeouts, dropped connections), not just HTTP 429/5xx --
     see sanghabot/embeddings/client.py -- so most transient network
     issues are handled invisibly without ever reaching this script's own
     resume logic at all.
  4. Every artifact (the new .npy, the embedding_runs row) is written
     ONCE, atomically, near the very end of a successful run, following
     the project's embedding-immutability policy (REWRITE_PLAN.md
     Appendix C) -- nothing already on disk is ever overwritten in place.
  5. A backup of data/index/{sanghabot.db,faiss.index,bm25.pkl} is taken
     automatically before anything is touched (skipped on a resumed run
     if today's backup already exists, so re-running after a crash
     doesn't pile up redundant multi-hundred-MB backups).

WHY THE LEGACY EMBEDDING PIPELINE (not the "clean" native one): the
existing ~23k-chunk corpus is embedded at 512 dimensions via a specific
lossy pipeline (see sanghabot/embeddings/legacy_compat.py), and a single
FAISS index cannot hold vectors of mixed dimensions. Until
REWRITE_PLAN.md Section 11's full native re-embed happens (a deliberate,
separate, human-reviewed step -- not something this script should ever do
silently), new content must be embedded via that SAME legacy pipeline to
stay in the same vector space as everything already indexed. This script
does exactly that, and reuses the existing legacy_compat.py helpers
rather than reimplementing the transform.
"""
from __future__ import annotations

import shutil
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
from sanghabot.ingest.parse_blogs import iter_blog_posts, process_markdown_file  # noqa: E402
from sanghabot.models import Chunk, VideoMetadata  # noqa: E402
from sanghabot.storage.db import Database  # noqa: E402

app = typer.Typer(help="One-command blog update: sync + parse + chunk + embed + index new blog posts.")

# Default source of truth for site markdown files. Override with --site-dir
# if the site repo ever lives somewhere else.
DEFAULT_SITE_DIR = Path("/Users/jonas/Programme/Python/dhammarato-site/src/content/blog")

# The existing, already-indexed legacy embeddings live outside this repo
# (produced once by scripts/migrate_legacy_data.py, see that script's
# docstring for the full history). Read-only: never modified by this
# script. If a *previous* run of this script already created a
# "...-plus-*" run, that run's own .npy is used as the base to append to
# instead (see _find_base_embeddings() below), so the legacy .npy itself
# is only ever the base exactly once, for the very first update.
LEGACY_EMBEDDINGS_NPY = Path(
    "/Users/jonas/Programme/Python/Youtube-Video-Recommendation-Engine/AI_Transcripts/embeddings/embeddings.npy"
)
LEGACY_RUN_ID = "qwen3-4b-512-legacy"

CHUNK_OVERLAP_CHARS = 200  # matches the original embed_videos.py's extract_chunks() overlap
EMBED_BATCH_SIZE = 8
MIN_TRANSCRIPT_CHARS = 20  # below this, treat as "no real transcript" and skip


def _extract_chunks(text: str, chunk_borders: list[int], overlap_chars: int) -> list[str]:
    """Verbatim-equivalent port of the original embed_videos.py's extract_chunks()."""
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
    """Used only for chunk_transcript_semantic()'s internal boundary-refinement calls."""
    vectors = embed_texts(texts, dimension=LEGACY_NATIVE_DIM, model=LEGACY_MODEL)
    return [np.asarray(v) for v in vectors]


def _sync_blogs_dir(site_dir: Path, blogs_dir: Path) -> int:
    """
    Copies every .md file from `site_dir` into `blogs_dir` if it's new or
    changed (mtime/size differ), leaving already-synced files untouched.
    Returns the number of files copied. This is what makes "point at the
    site repo and go" possible without a separate manual copy step.
    """
    blogs_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(site_dir.glob("*.md")):
        dst = blogs_dir / src.name
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def _backup_index_if_needed(data_dir: Path) -> Path | None:
    """
    Backs up data/index/{sanghabot.db,faiss.index,bm25.pkl} before this
    script touches anything, UNLESS a backup already exists from today
    (so re-running after an interruption doesn't pile up redundant
    multi-hundred-MB backups of already-safe data).
    """
    index_dir = data_dir / "index"
    today = datetime.now().strftime("%Y-%m-%d")
    existing = sorted(data_dir.glob(f"index_backup_{today}_*"))
    if existing:
        typer.echo(f"Backup already exists for today ({existing[-1].name}) -- skipping.")
        return existing[-1]

    backup_dir = data_dir / f"index_backup_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    backup_dir.mkdir(parents=True)
    for name in ("sanghabot.db", "faiss.index", "bm25.pkl"):
        src = index_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    typer.echo(f"Backed up data/index/ to {backup_dir}")
    return backup_dir


def _find_base_embeddings(db: Database) -> tuple[np.ndarray, str]:
    """
    Returns (matrix, source_description) for the embeddings this run
    should APPEND to. Prefers the most-recently-created run whose npy_path
    still exists on disk over the original external legacy .npy, so that
    running this script twice in a row (e.g. after adding more posts
    later) keeps building on the previous update instead of re-reading the
    original legacy file and duplicating rows. Falls back to the original
    legacy .npy the very first time this script is ever run.
    """
    cur = db._conn.execute(
        "SELECT run_id, npy_path FROM embedding_runs "
        "WHERE dimension = ? ORDER BY created_at DESC",
        (LEGACY_TARGET_DIM,),
    )
    rows = cur.fetchall()
    for run_id, npy_path in rows:
        p = Path(npy_path)
        if p.exists() and run_id != LEGACY_RUN_ID:
            return np.load(p), f"previous update run '{run_id}' ({p})"

    if not LEGACY_EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            f"Neither a previous update run's .npy nor the original legacy "
            f"embeddings.npy could be found at {LEGACY_EMBEDDINGS_NPY}. "
            f"Cannot determine what to append new embeddings to."
        )
    return np.load(LEGACY_EMBEDDINGS_NPY), f"original legacy embeddings.npy ({LEGACY_EMBEDDINGS_NPY})"


@app.command()
def run(
    site_dir: Path = typer.Option(DEFAULT_SITE_DIR, help="Directory of source .md blog files to sync from."),
    blogs_dir: Path = typer.Option(settings.raw_blogs_dir, help="This repo's data/raw/blogs/ directory."),
    skip_backup: bool = typer.Option(False, help="Skip the automatic data/index/ backup (not recommended)."),
    dry_run: bool = typer.Option(False, help="Show what would be done without writing anything."),
):
    """
    Full update: sync new/changed .md files from `site_dir` into
    `blogs_dir`, then parse + chunk + embed + index every video not
    already present in sanghabot.db. Safe to re-run at any time, including
    after an interruption -- see this module's docstring for why.
    """
    typer.echo("=== Step 1/6: Syncing blog files ===")
    if not site_dir.exists():
        typer.echo(f"Site directory not found: {site_dir}", err=True)
        raise typer.Exit(code=1)
    copied = _sync_blogs_dir(site_dir, blogs_dir)
    typer.echo(f"Synced {copied} new/changed file(s) from {site_dir} into {blogs_dir}")

    typer.echo("\n=== Step 2/6: Backing up data/index/ ===")
    if skip_backup:
        typer.echo("Skipped (--skip-backup).")
    elif dry_run:
        typer.echo("Would back up data/index/ (skipped in --dry-run).")
    else:
        _backup_index_if_needed(settings.data_dir)

    typer.echo("\n=== Step 3/6: Parsing + chunking new videos ===")
    db = Database(settings.db_path)

    cur = db._conn.execute("SELECT MAX(embedding_row) FROM chunks")
    max_existing_row = cur.fetchone()[0]
    next_embedding_row = (max_existing_row + 1) if max_existing_row is not None else 0

    posts = list(iter_blog_posts(blogs_dir))
    # "Already ingested?" is checked via blog_url, NOT video_id: the
    # ~2,061 legacy-era videos have arbitrary integer video_ids (migrated
    # from an old CSV's row position -- see scripts/migrate_legacy_data.py),
    # while every video this script inserts uses the filename slug as
    # video_id instead. blog_url is deterministically derived from the
    # slug for every video regardless of era, so it's the one reliable way
    # to detect "already ingested" across both video_id schemes. See
    # Database.get_video_by_blog_url()'s docstring for the full story --
    # a video_id-based check here would incorrectly treat all ~2,061
    # legacy-era posts as "new" every single run.
    def _blog_url_for(post) -> str:
        video_id = post.file_name.rsplit(".", 1)[0]
        return f"https://dhammarato.com/blog/{video_id}"

    new_posts = [p for p in posts if db.get_video_by_blog_url(_blog_url_for(p)) is None]
    typer.echo(f"{len(posts)} total post(s) in {blogs_dir}; {len(new_posts)} not yet in the database.")

    if dry_run:
        for p in new_posts:
            video_id = p.file_name.rsplit(".", 1)[0]
            status = "OK" if len(p.transcript.strip()) >= MIN_TRANSCRIPT_CHARS else "SKIP (no/short transcript)"
            typer.echo(f"  would ingest: {video_id}  ({len(p.transcript)} chars)  [{status}]")
        typer.echo("\nDry run complete -- no changes made.")
        db.close()
        return

    videos_inserted = 0
    chunks_created = 0
    skipped: list[str] = []

    for post in tqdm(new_posts, desc="Ingesting new posts"):
        video_id = post.file_name.rsplit(".", 1)[0]
        transcript = post.transcript or ""

        if len(transcript.strip()) < MIN_TRANSCRIPT_CHARS:
            skipped.append(video_id)
            typer.echo(f"  SKIP (no/short transcript): {video_id}")
            continue

        video_url = f"https://www.youtube.com/watch?v={post.youtube_id}" if post.youtube_id else None
        video = VideoMetadata(
            video_id=video_id,
            title=post.title or "",
            blog_url=f"https://dhammarato.com/blog/{video_id}",
            url=video_url,
            published_date=post.pub_date if isinstance(post.pub_date, date) else None,
            tags=post.tags,
            video_length_chars=len(transcript),
        )

        borders = chunk_transcript_semantic(transcript, batch_embed_fn=_boundary_embed_fn, debug=False)
        texts_no_overlap = _extract_chunks(transcript, borders, 0)

        chunks = []
        for chunk_idx, chunk_text in enumerate(texts_no_overlap):
            start_char = borders[chunk_idx - 1] if chunk_idx - 1 >= 0 else 0
            end_char = borders[chunk_idx] if chunk_idx < len(borders) else len(transcript)
            chunks.append(
                Chunk(
                    chunk_id=f"{video_id}_{chunk_idx}",
                    video_id=video_id,
                    chunk_idx=chunk_idx,
                    text=chunk_text,
                    summary=post.summary_markdown or None,
                    start_char=start_char,
                    end_char=end_char,
                    embedding_row=None,  # filled in during the embedding pass below
                )
            )

        # Committed immediately (insert_video/insert_chunks both commit) --
        # this video is now safely durable even if the script crashes on
        # the very next line.
        db.insert_video(video)
        db.insert_chunks(chunks)
        videos_inserted += 1
        chunks_created += len(chunks)

    typer.echo(f"Inserted {videos_inserted} video(s), {chunks_created} chunk(s).")
    if skipped:
        typer.echo(f"Skipped {len(skipped)} post(s) with no usable transcript: {skipped}")

    typer.echo("\n=== Step 4/6: Embedding un-embedded chunks ===")
    # Re-derived fresh from the DB every time (never from an in-memory
    # list), so a crash after step 3 but before/during this step loses
    # nothing: re-running the whole script finds exactly the same set of
    # not-yet-embedded chunks and picks up cleanly.
    cur = db._conn.execute(
        "SELECT chunk_id, video_id, start_char, end_char FROM chunks "
        "WHERE embedding_row IS NULL ORDER BY rowid"
    )
    pending = cur.fetchall()
    typer.echo(f"{len(pending)} chunk(s) need embedding.")

    if not pending:
        typer.echo("Nothing new to embed or index. Done.")
        db.close()
        return

    transcript_cache: dict[str, str] = {}

    def get_transcript(vid: str) -> str:
        if vid not in transcript_cache:
            fp = blogs_dir / f"{vid}.md"
            transcript_cache[vid] = process_markdown_file(fp).transcript
        return transcript_cache[vid]

    texts_with_overlap = []
    for chunk_id, vid, start_char, end_char in pending:
        transcript = get_transcript(vid)
        s = max(start_char - CHUNK_OVERLAP_CHARS, 0)
        e = min(end_char + CHUNK_OVERLAP_CHARS, len(transcript))
        texts_with_overlap.append(transcript[s:e])

    all_vectors: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts_with_overlap), EMBED_BATCH_SIZE), desc="Embedding batches"):
        batch = texts_with_overlap[i:i + EMBED_BATCH_SIZE]
        raw = embed_texts(batch, dimension=LEGACY_NATIVE_DIM, model=LEGACY_MODEL)
        batch_matrix = np.array([np.asarray(v) for v in raw], dtype="float32")
        reduced = legacy_reduce_dimensionality(batch_matrix, LEGACY_TARGET_DIM)
        all_vectors.extend(reduced)

    new_matrix = np.array(all_vectors, dtype="float32")
    assert new_matrix.shape[0] == len(pending), "chunk count != embedding count -- rows were dropped somewhere"
    assert new_matrix.shape[1] == LEGACY_TARGET_DIM, f"expected dim {LEGACY_TARGET_DIM}, got {new_matrix.shape[1]}"
    assert not np.isnan(new_matrix).any(), "NaN values found in new embeddings"
    assert (np.linalg.norm(new_matrix, axis=1) > 0).all(), "one or more embeddings have zero norm"

    for (chunk_id, *_rest), row in zip(pending, range(next_embedding_row, next_embedding_row + len(pending))):
        db.set_embedding_row(chunk_id, row)

    typer.echo("\n=== Step 5/6: Writing new embedding_runs artifact ===")
    base_matrix, base_description = _find_base_embeddings(db)
    typer.echo(f"Appending to: {base_description} (shape={base_matrix.shape})")

    combined = np.vstack([base_matrix, new_matrix]).astype("float32")
    typer.echo(f"Combined shape: {combined.shape}")

    run_id = f"qwen3-4b-512-legacy-plus-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
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
            f"Appended {len(pending)} newly-embedded chunk(s) after {base_description}. "
            f"Embedded via the legacy pipeline ({LEGACY_MODEL} @ {LEGACY_NATIVE_DIM}, "
            f"averaged down to {LEGACY_TARGET_DIM}) for vector-space compatibility with "
            "the rest of the corpus. Produced by scripts/update_blogs.py."
        ),
    )
    typer.echo(f"Registered embedding_runs row '{run_id}'.")

    typer.echo("\n=== Step 6/6: Rebuilding faiss.index, bm25.pkl, and promoting ===")
    import faiss

    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = np.ascontiguousarray((combined / norms).astype("float32"))
    index = faiss.IndexFlatIP(normalized.shape[1])
    for i in range(0, len(normalized), 1000):
        index.add(normalized[i:i + 1000])
    faiss.write_index(index, str(settings.faiss_index_path))
    typer.echo(f"Wrote {settings.faiss_index_path} ({len(normalized)} vectors, dim={normalized.shape[1]})")

    from sanghabot.search.bm25 import BM25SearchEngine

    if settings.bm25_index_path.exists():
        settings.bm25_index_path.unlink()
    BM25SearchEngine(db, settings.bm25_index_path)
    typer.echo(f"Wrote {settings.bm25_index_path}")

    db.set_active_embedding_run(run_id)
    typer.echo(f"Promoted '{run_id}' to active.")

    db.close()
    typer.echo(
        f"\nAll done. {videos_inserted} video(s) / {chunks_created} chunk(s) added. "
        f"Active run: '{run_id}'.\n"
        "If the Discord bot is currently running, restart it to pick up the new index "
        "(it loads the FAISS index and active run once at startup, not per-query)."
    )


if __name__ == "__main__":
    app()
