"""
Shared data models used across ingest/embeddings/search/storage/bot.

These replace the old codebase's pattern of passing around loosely-typed
plain dicts (e.g. `res["video_index"]`, `res["chunk_index"]`, ...), which
was directly responsible for at least one production bug: `search.py`
indexed `final_results[i]` in a branch with no `for` loop defining `i`
(a NameError waiting to happen). Typed dataclasses don't prevent every bug,
but they make this specific class of mistake (referencing a field or index
that was never actually set) far easier to catch via static analysis/tests
instead of a crash in production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    blog_url: str
    url: Optional[str] = None  # YouTube URL, distinct from blog_url
    published_date: Optional[date] = None
    tags: list[str] = field(default_factory=list)
    video_length_chars: Optional[int] = None


@dataclass
class Chunk:
    chunk_id: str  # "{video_id}_{chunk_idx}"
    video_id: str
    chunk_idx: int
    text: str
    summary: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    # Row index into the *active* embedding_runs .npy file. None until embedded.
    embedding_row: Optional[int] = None


@dataclass
class TermSuggestion:
    """
    A single query word not found verbatim in the BM25 corpus vocabulary,
    optionally paired with a fuzzy-matched "did you mean" suggestion.

    Used to give users a transparent, non-blocking heads-up (see
    sanghabot/bot/discord_bot.py's format_typo_notice()) when a keyword
    looks like a typo (e.g. "duka" vs. "dukkha") without ever refusing or
    altering the actual search -- the original query is still searched
    as-is regardless of what's found here.
    """
    term: str
    suggestion: Optional[str] = None


@dataclass
class QueryIntent:
    use_semantic: bool
    use_bm25: bool
    weights: dict[str, float]
    exact_phrases: list[str] = field(default_factory=list)
    meaningful_words: list[str] = field(default_factory=list)
    is_question: bool = False


@dataclass
class SearchResult:
    chunk_id: str
    video_id: str
    title: str
    blog_url: str
    text: str
    summary: Optional[str]
    score: float
    percentage_score: float
    source: Literal["semantic", "bm25", "fusion"]

    # Optional fields carried over from the old dict-based results, kept for
    # parity comparison against the legacy engine's output (see
    # tests/test_parity_old_vs_new.py). Not all callers need these.
    url: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    video_length_chars: Optional[int] = None
