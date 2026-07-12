#!/usr/bin/env python
"""
HISTORICAL RECORD -- already run, kept for documentation/audit purposes.

This script was used ONCE to migrate data out of the original
"Youtube-Video-Recommendation-Engine/AI_Transcripts/" project (a sibling
directory in that older repo) into this project's data/index/sanghabot.db.
That migration is done -- this repo's data/index/ already contains its
output. The script assumes the old AI_Transcripts/ directory exists as a
sibling one level up, which is only true in the original repo layout, not
in this standalone repo. It is kept here as an accurate record of exactly
how the initial data was produced, not as something you should expect to
run again in this repo. If you ever need to re-derive data/index/ from
scratch, use scripts/rebuild_index.py against raw source data instead.

Original docstring follows, unmodified:
---
One-off migration: old AI_Transcripts/embeddings/{metadata.csv,chunks/*.txt,
summaries/*.txt} -> new data/index/sanghabot.db (REWRITE_PLAN.md
Section 10 step 4 / Section 4).

This reads the OLD data files (never writes to them) and populates the NEW
sqlite database with:
  - one `videos` row per distinct video_index
  - one `chunks` row per (video_index, chunk_id), with chunk text and
    summary text inlined (no more 23,246 + 2,061 loose .txt files)
  - embedding_row set to the old `vector_id` (this is the row index into
    the legacy embeddings.npy / faiss.index, which stays valid as long as
    we register that exact .npy as the "legacy-512" embedding_runs row --
    see register step below)
  - one `embedding_runs` row registering the existing 512-dim embeddings.npy
    as-is, explicitly flagged as legacy/lossy in origin per Appendix C, and
    marked as the currently active run (nothing to embed from scratch yet)

Verification performed at the end: row counts are compared against the
source CSV and the loose chunks/summaries directories, per REWRITE_PLAN.md
Section 10 step 4 ("Verify row counts match... before trusting the
migration").

Usage:
    conda activate Dhamma
    python scripts/migrate_legacy_data.py
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REWRITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = REWRITE_ROOT.parent
OLD_DIR = REPO_ROOT / "AI_Transcripts"
OLD_EMBEDDINGS_DIR = OLD_DIR / "embeddings"
OLD_METADATA_CSV = OLD_EMBEDDINGS_DIR / "metadata.csv"
OLD_CHUNKS_DIR = OLD_EMBEDDINGS_DIR / "chunks"
OLD_SUMMARIES_DIR = OLD_EMBEDDINGS_DIR / "summaries"
OLD_FAISS_INDEX = OLD_EMBEDDINGS_DIR / "faiss.index"
OLD_EMBEDDINGS_NPY = OLD_EMBEDDINGS_DIR / "embeddings.npy"

sys.path.insert(0, str(REWRITE_ROOT))
from config import settings  # noqa: E402
from sanghabot.models import Chunk, VideoMetadata  # noqa: E402
from sanghabot.storage.db import Database  # noqa: E402

LEGACY_RUN_ID = "qwen3-4b-512-legacy"
LEGACY_NOTES = (
    "Legacy embeddings migrated as-is from AI_Transcripts/embeddings/embeddings.npy. "
    "Originally generated at 1024 dims, then reduced to 512 dims via a lossy "
    "'average adjacent pairs' transform (see AI_Transcripts/embed_videos.py: "
    "reduce_dimensionality()). The 1024-dim originals were overwritten and no "
    "longer exist. This run is NOT a clean native 512-dim request -- do not "
    "reuse this as a template for future embedding runs. See "
    "REWRITE_PLAN.md Appendix C and Section 11 (planned full "
    "re-embed to 1024 dims, sequenced after the parity gate passes)."
)


def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    print(f"[migrate] Reading {OLD_METADATA_CSV} ...")
    if not OLD_METADATA_CSV.exists():
        print(f"[migrate] ERROR: source metadata not found at {OLD_METADATA_CSV}")
        sys.exit(1)

    df = pd.read_csv(OLD_METADATA_CSV)
    print(f"[migrate] {len(df)} rows in source metadata.csv")

    # Fresh target DB: if a prior migration run left one, start clean so
    # re-running this script is idempotent rather than accumulating dupes.
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.db_path.exists():
        backup = settings.db_path.with_suffix(".db.bak")
        print(f"[migrate] Existing {settings.db_path} found -- backing up to {backup}")
        shutil.move(str(settings.db_path), str(backup))

    db = Database(settings.db_path)

    # ------------------------------------------------------------------
    # 1. Videos: one distinct row per video_index
    # ------------------------------------------------------------------
    video_cols = ["video_index", "title", "url", "blog_url", "tags", "video_length_chars"]
    video_rows = df[video_cols].drop_duplicates(subset="video_index")
    videos = []
    for _, row in video_rows.iterrows():
        tags_raw = row["tags"]
        tags = [t.strip() for t in str(tags_raw).split(",")] if pd.notna(tags_raw) else []
        videos.append(
            VideoMetadata(
                video_id=str(int(row["video_index"])),
                title=str(row["title"]) if pd.notna(row["title"]) else "",
                blog_url=str(row["blog_url"]) if pd.notna(row["blog_url"]) else "",
                url=str(row["url"]) if pd.notna(row["url"]) else None,
                published_date=None,  # not present in legacy metadata.csv
                tags=tags,
                video_length_chars=int(row["video_length_chars"]) if pd.notna(row["video_length_chars"]) else None,
            )
        )
    print(f"[migrate] Inserting {len(videos)} distinct videos ...")
    db.insert_videos(videos)

    # ------------------------------------------------------------------
    # 2. Chunks: text + summary inlined from the loose .txt files
    # ------------------------------------------------------------------
    print(f"[migrate] Inlining chunk text from {OLD_CHUNKS_DIR} and summaries from "
          f"{OLD_SUMMARIES_DIR} into sqlite rows (this replaces {len(df)} + "
          f"~2000 loose files with a single db file) ...")

    summary_cache: dict[str, str] = {}
    chunks: list[Chunk] = []
    missing_chunk_files = 0
    missing_summary_files = 0

    batch_size = 2000
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        for _, row in batch.iterrows():
            video_id = str(int(row["video_index"]))
            chunk_idx = int(row["chunk_id"])
            chunk_id = f"{video_id}_{chunk_idx}"

            text_path = OLD_CHUNKS_DIR / str(row["text_file"])
            if text_path.exists():
                text = read_text_file(text_path)
            else:
                missing_chunk_files += 1
                text = ""

            summary_file = str(row["summary_file"])
            if summary_file in summary_cache:
                summary = summary_cache[summary_file]
            else:
                summary_path = OLD_SUMMARIES_DIR / summary_file
                if summary_path.exists():
                    summary = read_text_file(summary_path)
                else:
                    missing_summary_files += 1
                    summary = ""
                summary_cache[summary_file] = summary

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    video_id=video_id,
                    chunk_idx=chunk_idx,
                    text=text,
                    summary=summary or None,
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    embedding_row=int(row["vector_id"]),
                )
            )
        db.insert_chunks(chunks)
        print(f"[migrate]   inserted {min(start + batch_size, len(df))}/{len(df)} chunks")
        chunks = []

    if missing_chunk_files:
        print(f"[migrate] WARNING: {missing_chunk_files} chunk text files were missing on disk")
    if missing_summary_files:
        print(f"[migrate] WARNING: {missing_summary_files} summary files were missing on disk")

    # ------------------------------------------------------------------
    # 3. Register the legacy embedding run (Appendix C policy: register,
    #    never silently pretend it's a clean native request)
    # ------------------------------------------------------------------
    print("[migrate] Registering legacy embedding_runs row ...")
    db.insert_embedding_run(
        run_id=LEGACY_RUN_ID,
        model_name="qwen/qwen3-embedding-8b",
        dimension=512,
        created_at=datetime.now(timezone.utc).isoformat(),
        npy_path=str(OLD_EMBEDDINGS_NPY),
        is_active=True,
        notes=LEGACY_NOTES,
    )

    # ------------------------------------------------------------------
    # 4. Verification: row counts must match source data before trusting
    #    this migration (REWRITE_PLAN.md Section 10 step 4).
    # ------------------------------------------------------------------
    print("[migrate] Verifying row counts ...")
    expected_chunks = len(df)
    expected_videos = df["video_index"].nunique()
    actual_chunks = db.count_chunks()
    actual_videos = db.count_videos()

    print(f"[migrate]   videos: expected={expected_videos} actual={actual_videos}")
    print(f"[migrate]   chunks: expected={expected_chunks} actual={actual_chunks}")

    ok = True
    if actual_videos != expected_videos:
        print("[migrate]   MISMATCH on video count!")
        ok = False
    if actual_chunks != expected_chunks:
        print("[migrate]   MISMATCH on chunk count!")
        ok = False

    if ok:
        print("[migrate] Row counts match. Migration verified.")
    else:
        print("[migrate] Row count MISMATCH -- do not trust this migration until resolved.")
        sys.exit(2)

    db.close()
    print(f"[migrate] Done. New database at {settings.db_path}")


if __name__ == "__main__":
    main()
