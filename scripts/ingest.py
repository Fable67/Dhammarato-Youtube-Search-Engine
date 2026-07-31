#!/usr/bin/env python
"""
Typer CLI for ingestion: parse raw blog markdown -> chunk -> store in
sanghabot.db.

Replaces AI_Transcripts/chunk_videos.py's interactive input()-driven main()
(which blocked on three separate input() prompts and had no non-interactive
mode at all, making it impossible to automate/schedule). This is a real CLI
with flags, usable from cron/systemd-timer/CI.

NOTE: for the actual day-to-day task of adding new blog posts, use
scripts/update_blogs.py instead -- it's the single command that does
everything below (sync, parse, chunk, embed, index, promote) in one call,
with automatic backups and resumability. See README.md's "Adding new blog
posts" section. `parse` (below) is kept as a lower-level building block
(useful for just checking what parse_blogs.py extracts from a directory of
files, without touching chunks/embeddings at all); `chunk` remains an
intentionally unimplemented stub -- see its own docstring for why.

Usage:
    conda activate Dhamma
    python scripts/ingest.py parse --blogs-dir data/raw/blogs
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
    Intentionally unimplemented stub -- this command's `parse` sibling
    above only persists title/blog_url/tags, not the raw transcript text
    a real chunk() would need, and was never wired up (see git history:
    this was true from the initial rewrite through the first real blog
    re-ingestion). Left in place purely to document the originally
    intended CLI shape.

    Use `python scripts/update_blogs.py` instead for actual blog
    ingestion -- it parses, chunks, embeds, and indexes new posts in one
    command, including persisting the transcript text this stub was
    always missing. See README.md's "Adding new blog posts" section.
    """
    typer.echo(
        "This command is an intentionally unimplemented stub -- use "
        "`python scripts/update_blogs.py` instead. See README.md's "
        "'Adding new blog posts' section."
    )


if __name__ == "__main__":
    app()
