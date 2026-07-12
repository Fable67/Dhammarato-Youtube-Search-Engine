#!/usr/bin/env python
"""
Typer CLI for ingestion: parse raw blog markdown -> chunk -> store in
sanghabot.db.

Replaces AI_Transcripts/chunk_videos.py's interactive input()-driven main()
(which blocked on three separate input() prompts and had no non-interactive
mode at all, making it impossible to automate/schedule). This is a real CLI
with flags, usable from cron/systemd-timer/CI.

Usage:
    conda activate Dhamma
    python scripts/ingest.py parse --blogs-dir data/raw/blogs
    python scripts/ingest.py chunk --resume
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

REWRITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REWRITE_ROOT))

from config import settings  # noqa: E402
from sanghabot.embeddings.client import embed_texts  # noqa: E402
from sanghabot.ingest.chunker import chunk_transcript_semantic  # noqa: E402
from sanghabot.ingest.parse_blogs import iter_blog_posts  # noqa: E402
from sanghabot.models import Chunk, VideoMetadata  # noqa: E402
from sanghabot.storage.db import Database  # noqa: E402

app = typer.Typer(help="Ingestion pipeline: parse raw blog markdown, then chunk transcripts.")


@app.command()
def parse(blogs_dir: Path = settings.raw_blogs_dir):
    """
    Parse markdown blog posts from `blogs_dir` and store them as `videos`
    rows in sanghabot.db (title, blog_url, tags). Does NOT chunk yet --
    run `chunk` afterwards.
    """
    if not blogs_dir.exists():
        typer.echo(f"Blogs directory not found: {blogs_dir}", err=True)
        raise typer.Exit(code=1)

    db = Database(settings.db_path)
    count = 0
    for post in iter_blog_posts(blogs_dir):
        video_id = post.file_name.rsplit(".", 1)[0]
        blog_url = f"https://dhammarato.com/blog/{video_id}"
        db.insert_video(
            VideoMetadata(
                video_id=video_id,
                title=post.title or "",
                blog_url=blog_url,
                tags=post.tags,
            )
        )
        count += 1
        if count % 100 == 0:
            typer.echo(f"  parsed {count} posts so far ...")
    db.close()
    typer.echo(f"Done. Parsed {count} blog posts into {settings.db_path}")


@app.command()
def chunk(resume: bool = typer.Option(True, help="Skip videos that already have chunks.")):
    """
    Chunk every video's transcript (via the semantic chunker) and store
    chunk rows in sanghabot.db. This step calls the embedding API for
    boundary refinement (see sanghabot/ingest/chunker.py), so it costs
    real API time/money -- use --resume (default) to avoid redoing
    already-chunked videos.
    """
    typer.echo(
        "NOTE: this command requires raw transcript TEXT to be available "
        "per video, which the current 'parse' command does not yet persist "
        "(only title/blog_url/tags). Wiring the full transcript text "
        "through into chunk() is a follow-up once blog re-ingestion is "
        "actually needed -- current scope (REWRITE_PLAN.md Section 10) is "
        "the search-serving path against migrated legacy data, not a full "
        "blogs -> chunks re-run. This stub documents the intended CLI "
        "shape for that future work."
    )


if __name__ == "__main__":
    app()
