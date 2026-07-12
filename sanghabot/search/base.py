"""
Common protocol for search engines (semantic, BM25).

Both concrete engines load their indices exactly once, in __init__, and
never touch disk again inside search(). This is the direct fix for the
old codebase's worst performance bug: semantic_search.py called
faiss.read_index() inside search(), and keyword_search.py called
pd.read_csv() + pickle.load() inside search() -- meaning every single
Discord message reloaded a 45MB FAISS index and a 55MB BM25 pickle from
disk. Combined with discord_bot_channel.py constructing a brand new
CombinedSearchEngine per message, this produced the 8-14 second query
times documented in AI_Transcripts/search_engine.log.
"""
from __future__ import annotations

from typing import Protocol

from sanghabot.models import SearchResult


class SearchEngine(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]:
        ...
