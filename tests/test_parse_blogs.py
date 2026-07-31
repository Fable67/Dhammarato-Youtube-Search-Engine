"""
Tests for sanghabot.ingest.parse_blogs.

Covers the H2-vs-H3 "### Transcript" heading regression: a handful of
posts in the real site content use "## Transcript" (H2) instead of the
usual "### Transcript" (H3) -- e.g. posts with a shallower overall heading
structure that also use a single "#" for the top-level title instead of
"##". Before this fix, process_markdown_file() only ever matched H3,
silently parsing these posts to an empty transcript instead of failing
loudly -- discovered directly while ingesting new blog posts (see
scripts/update_blogs.py's module docstring).
"""
from pathlib import Path

import pytest

from sanghabot.ingest.parse_blogs import process_markdown_file


def _write(tmp_path: Path, name: str, content: str) -> Path:
    fp = tmp_path / name
    fp.write_text(content, encoding="utf-8")
    return fp


_FRONTMATTER = """---
title: "Test Post"
pubDate: 2024-01-01
tags: [foo, bar]
---
"""


def test_h3_transcript_header_is_parsed(tmp_path):
    content = _FRONTMATTER + """
## Test Post

### Video

<iframe src="https://www.youtube.com/embed/abc123?rel=0"></iframe>

### Transcript

**Speaker A:** Hello world, this is the transcript.

### Summary

A short summary.
"""
    post = process_markdown_file(_write(tmp_path, "test.md", content))
    assert post.transcript == "Speaker A: Hello world, this is the transcript."
    assert post.youtube_id == "abc123"
    # Pre-existing behavior (unrelated to the H2/H3 fix under test here):
    # the next section's own header line is retained in summary_markdown.
    assert post.summary_markdown == "### Summary\n\nA short summary."


def test_h2_transcript_header_is_parsed():
    """
    Regression test: some real posts use "## Transcript" instead of
    "### Transcript". Must still extract the transcript text, not
    silently produce an empty string.
    """
    content = _FRONTMATTER + """
# Test Post

## Transcript

I have a question about practice.

## Want More? Watch the Whole Video Here

Some footer content.
"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fp = _write(Path(d), "test.md", content)
        post = process_markdown_file(fp)
    assert post.transcript == "I have a question about practice."


def test_post_with_no_transcript_section_yields_empty_transcript(tmp_path):
    """
    A small number of posts (essays, differently-organized dialogues) have
    no Transcript section at all. These must yield an empty transcript
    (safely skippable by callers) rather than crashing or guessing at
    some other section as a fallback.
    """
    content = _FRONTMATTER + """
## Some Essay Title

This is just prose with no Transcript header at all.

## Another Section

More prose.
"""
    post = process_markdown_file(_write(tmp_path, "test.md", content))
    assert post.transcript == ""


def test_h3_preferred_when_both_would_technically_be_absent_is_moot(tmp_path):
    """Sanity check: when only H3 is present (the common case), it's used as before."""
    content = _FRONTMATTER + """
### Transcript

Just H3 here.
"""
    post = process_markdown_file(_write(tmp_path, "test.md", content))
    assert post.transcript == "Just H3 here."


def test_real_h2_transcript_file_parses_correctly():
    """
    End-to-end check against a real, previously-broken file from the
    actual site content, if available on this machine. Skipped gracefully
    if the sibling site repo isn't present (e.g. CI, a fresh clone).
    """
    real_file = Path(
        "/Users/jonas/Programme/Python/dhammarato-site/src/content/blog/"
        "2024-04-14-you-dont-have-to-fix-the-world-sangha-uk-210-04-14-24-SMALigFMRxM.md"
    )
    if not real_file.exists():
        pytest.skip("Sibling dhammarato-site repo not present on this machine.")
    post = process_markdown_file(real_file)
    assert len(post.transcript) > 1000
    assert "psychology" in post.transcript.lower()
