"""
Tests for sanghabot.rag.pipeline: context building, timestamp estimation, and citation handling.

All tests use hand-constructed SearchResult objects and do not touch the network,
database, or LLM API (no calls to generate_answer).
"""
from __future__ import annotations

from typing import Optional

from sanghabot.models import SearchResult
from sanghabot.rag.pipeline import (
    Citation,
    build_context_block,
    build_user_prompt,
    estimate_timestamp,
)


def make_result(
    chunk_id: str,
    title: str = "Test Video",
    text: str = "Test transcript text",
    start_char: Optional[int] = None,
    percentage_score: float = 85.0,
) -> SearchResult:
    """Helper to construct a SearchResult for testing."""
    return SearchResult(
        chunk_id=chunk_id,
        video_id=chunk_id.split("_")[0],
        title=title,
        blog_url="https://example.com/blog/test",
        text=text,
        summary=None,
        score=0.0,
        percentage_score=percentage_score,
        source="semantic",
        url=f"https://youtube.com/watch?v={chunk_id.split('_')[0]}",
        start_char=start_char,
    )


class TestEstimateTimestamp:
    """Tests for estimate_timestamp() function."""

    def test_none_returns_zero_timestamp(self):
        """estimate_timestamp(None) should return '00:00:00'."""
        assert estimate_timestamp(None) == "00:00:00"

    def test_zero_returns_zero_timestamp(self):
        """estimate_timestamp(0) should return '00:00:00'."""
        assert estimate_timestamp(0) == "00:00:00"

    def test_small_start_char_gives_seconds(self):
        """estimate_timestamp(17) should give ~1 second (17.0 / 17 = 1)."""
        # 17 chars -> 1 second
        assert estimate_timestamp(17) == "00:00:01"

    def test_one_minute_of_chars(self):
        """estimate_timestamp for 60 seconds worth of chars."""
        # 60 seconds * 17 chars/second = 1020 chars
        assert estimate_timestamp(1020) == "00:01:00"

    def test_one_hour_of_chars(self):
        """estimate_timestamp for 1 hour (3600 seconds)."""
        # 3600 seconds * 17 chars/second = 61200 chars
        assert estimate_timestamp(61200) == "01:00:00"

    def test_mixed_hms(self):
        """estimate_timestamp for a mixed time (e.g., 1:23:45)."""
        # 1h + 23m + 45s = 5025 seconds
        # 5025 * 17 = 85425 chars
        assert estimate_timestamp(85425) == "01:23:45"

    def test_formatting_with_leading_zeros(self):
        """Verify leading zeros are present in output."""
        # 5 seconds -> should be "00:00:05"
        result = estimate_timestamp(85)
        assert result == "00:00:05"
        # All parts should have leading zeros if needed
        assert len(result) == 8  # HH:MM:SS


class TestBuildContextBlock:
    """Tests for build_context_block() function."""

    def test_empty_results_returns_empty_strings(self):
        """build_context_block([]) should return empty string and empty citation list."""
        context, citations = build_context_block([])
        assert context == ""
        assert citations == []

    def test_single_result_formats_correctly(self):
        """build_context_block with one result formats with [1] marker."""
        result = make_result("vid1_0", title="Test Title", text="Sample text")
        context, citations = build_context_block([result])

        # Context should include the [1] marker
        assert "[1]" in context
        assert "Test Title" in context
        assert "Sample text" in context
        assert "Relevance:" in context
        assert "85.0%" in context

    def test_multiple_results_are_numbered(self):
        """build_context_block with multiple results uses 1, 2, 3 markers."""
        results = [
            make_result("vid1_0", title="First", text="First text"),
            make_result("vid1_1", title="Second", text="Second text"),
            make_result("vid1_2", title="Third", text="Third text"),
        ]
        context, citations = build_context_block(results)

        # All markers should be present and in order
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        # Verify order by checking positions
        pos1 = context.find("[1]")
        pos2 = context.find("[2]")
        pos3 = context.find("[3]")
        assert pos1 < pos2 < pos3

    def test_citations_list_matches_results(self):
        """build_context_block citations should have same length and 1-indexed numbers."""
        results = [
            make_result("vid1_0", title="First"),
            make_result("vid1_1", title="Second"),
        ]
        context, citations = build_context_block(results)

        assert len(citations) == len(results)
        assert citations[0].number == 1
        assert citations[1].number == 2

    def test_citation_fields_match_result(self):
        """Citation fields should correctly map from SearchResult."""
        result = make_result(
            "vid123_5",
            title="Awesome Talk",
            start_char=1000,
            percentage_score=92.5,
        )
        context, citations = build_context_block([result])

        citation = citations[0]
        assert citation.title == "Awesome Talk"
        assert citation.percentage_score == 92.5
        assert citation.chunk_id == "vid123_5"
        assert citation.blog_url == "https://example.com/blog/test"
        assert citation.url == "https://youtube.com/watch?v=vid123"
        # timestamp should be estimated from start_char=1000
        assert citation.timestamp == estimate_timestamp(1000)

    def test_context_excludes_summary_field(self):
        """The context block should use .text, not .summary."""
        result = SearchResult(
            chunk_id="vid_0",
            video_id="vid",
            title="Title",
            blog_url="https://example.com/blog",
            text="This is the actual transcript text",
            summary="This is a pre-generated summary",
            score=1.0,
            percentage_score=100.0,
            source="semantic",
        )
        context, citations = build_context_block([result])

        # Context should include the actual text, not the summary
        assert "This is the actual transcript text" in context
        assert "pre-generated summary" not in context

    def test_citation_with_none_url(self):
        """Citation should handle result.url = None gracefully."""
        result = SearchResult(
            chunk_id="vid_0",
            video_id="vid",
            title="Title",
            blog_url="https://example.com/blog",
            text="Text",
            summary=None,
            score=1.0,
            percentage_score=100.0,
            source="semantic",
            url=None,  # No YouTube URL
        )
        context, citations = build_context_block([result])

        citation = citations[0]
        assert citation.url is None  # Should be preserved


class TestBuildUserPrompt:
    """Tests for build_user_prompt() function."""

    def test_includes_query_text(self):
        """build_user_prompt should include the query verbatim."""
        query = "What is right effort?"
        context = "[1] Sample\n---"
        prompt = build_user_prompt(query, context)

        assert query in prompt
        assert "What is right effort?" in prompt

    def test_includes_context_block(self):
        """build_user_prompt should include the context block verbatim."""
        query = "Test query"
        context = "[1] Title: Test\nRelevance: 90%\nText content\n---"
        prompt = build_user_prompt(query, context)

        assert context in prompt
        assert "[1]" in prompt

    def test_includes_grounding_instructions(self):
        """build_user_prompt should remind about citing rules."""
        query = "Test"
        context = "[1] Text"
        prompt = build_user_prompt(query, context)

        # Should mention excerpts and citation rules
        assert "numbered excerpts" in prompt.lower() or "excerpt" in prompt.lower()
        assert "citation" in prompt.lower() or "cite" in prompt.lower()

    def test_empty_context_still_valid(self):
        """build_user_prompt should work even with empty context."""
        query = "Test query"
        context = ""
        prompt = build_user_prompt(query, context)

        assert query in prompt
        # Should not error; structure should still be intact


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_context_and_prompt_together(self):
        """Context block and user prompt should work together."""
        results = [
            make_result("vid1_0", title="Talk 1", text="First excerpt"),
            make_result("vid1_1", title="Talk 2", text="Second excerpt"),
        ]
        context, citations = build_context_block(results)
        prompt = build_user_prompt("Example query", context)

        # Prompt should contain markers and context
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Talk 1" in prompt
        assert "Talk 2" in prompt
        assert "Example query" in prompt
        assert len(citations) == 2

    def test_numbered_results_and_citations_aligned(self):
        """Context markers [N] and Citation.number should always match."""
        results = [
            make_result(f"vid_{i}", title=f"Result {i}")
            for i in range(5)
        ]
        context, citations = build_context_block(results)

        # Should have 5 citations with numbers 1-5
        assert len(citations) == 5
        for i, citation in enumerate(citations, start=1):
            assert citation.number == i
            # Also verify the context has [i] marker
            assert f"[{i}]" in context
