"""
Parses raw markdown blog posts (YAML frontmatter + transcript + summary
sections) into VideoMetadata records ready for storage.

Ported from AI_Transcripts/parse_blogs.py. The old version wrote directly
to a giant CSV held fully in memory (dhammarato_transcripts.csv, 114MB).
This version yields records incrementally and the caller (scripts/ingest.py)
is responsible for streaming them into sqlite -- no full-corpus DataFrame
is ever held in memory (see REWRITE_PLAN.md Section 7 for why this
matters: it removes the underlying RAM-pressure problem the old codebase
worked around by spraying chunk/summary text into thousands of loose files).

NOTE: this module returns raw parsed markdown records (title, transcript
text, summary text, tags, youtube_id) -- turning these into DB rows still
requires running the chunker (sanghabot/ingest/chunker.py) to establish
chunk boundaries, and is wired together in scripts/ingest.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml


@dataclass
class ParsedBlogPost:
    file_name: str
    title: str | None
    pub_date: str | None
    author: str | None
    categories: list[str]
    tags: list[str]
    description: str | None
    youtube_id: str | None
    summary_markdown: str
    transcript: str


def extract_yaml_frontmatter(text: str) -> tuple[dict, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if match:
        metadata = yaml.safe_load(match.group(1)) or {}
        content = text[match.end():]
        return metadata, content
    return {}, text


def extract_section(content: str, header: str) -> str:
    pattern = rf"(?:^|\n){re.escape(header)}\s*\n(.*?)(?=\n### |\n## |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""


def remove_connect_section(content: str) -> str:
    pattern = r"\n### Connect with Dhammarato and Sangha Friends.*?(?=\n### |\n## |\Z)"
    return re.sub(pattern, "", content, flags=re.DOTALL)


def extract_youtube_id(video_section: str) -> str | None:
    match = re.search(r"embed/([^?\"/]+)", video_section)
    return match.group(1) if match else None


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def process_markdown_file(filepath: Path) -> ParsedBlogPost:
    raw_text = filepath.read_text(encoding="utf-8")
    metadata, content = extract_yaml_frontmatter(raw_text)
    content = remove_connect_section(content)

    transcript = extract_section(content, "### Transcript")
    video_section = extract_section(content, "### Video")
    youtube_id = extract_youtube_id(video_section)

    summary = ""
    split_after_transcript = re.split(r"\n### Transcript\s*\n", content, maxsplit=1)
    if len(split_after_transcript) > 1:
        summary = re.sub(r"^(.*?)(?=\n### |\n## |\Z)", "", split_after_transcript[1], flags=re.DOTALL).strip()

    transcript = re.sub(r"\*\*(.*?)\*\*", r"\1", transcript)

    return ParsedBlogPost(
        file_name=filepath.name,
        title=metadata.get("title"),
        pub_date=metadata.get("pubDate"),
        author=metadata.get("author"),
        categories=_as_list(metadata.get("categories")),
        tags=_as_list(metadata.get("tags")),
        description=metadata.get("description"),
        youtube_id=youtube_id,
        summary_markdown=summary,
        transcript=transcript,
    )


def iter_blog_posts(folder_path: Path | str) -> Iterator[ParsedBlogPost]:
    """Streams parsed posts one at a time -- never holds the full corpus
    in memory (unlike the old parse_blogs.py, which built one giant
    in-memory DataFrame for the entire blog corpus)."""
    folder = Path(folder_path)
    for file in sorted(folder.glob("*.md")):
        try:
            yield process_markdown_file(file)
        except Exception as e:
            print(f"[parse_blogs] Error processing {file}: {e}")
