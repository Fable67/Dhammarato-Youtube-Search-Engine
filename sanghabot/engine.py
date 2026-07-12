"""
CombinedSearchEngine: composed once at process startup (see bot/discord_bot.py
on_ready), never reconstructed per request.

Replaces AI_Transcripts/search.py :: CombinedSearchEngine, fixing:
  - The per-message engine reconstruction in discord_bot_channel.py (this
    class is built once and held for the process lifetime).
  - The duplicated analyze_query_intent/reciprocal_rank_fusion logic
    (this module imports the single canonical copies from
    sanghabot/search/intent.py and sanghabot/search/fusion.py).
  - The search.py:247-249 NameError crash bug: percentage_score is now
    ALWAYS populated via an explicit loop in every branch below, including
    the semantic-only and bm25-only branches, so there is no code path
    that indexes an undefined loop variable.
"""
from __future__ import annotations

from sanghabot.models import SearchResult, TermSuggestion
from sanghabot.search.base import SearchEngine
from sanghabot.search.fusion import reciprocal_rank_fusion
from sanghabot.search.intent import analyze_query_intent


class CombinedSearchEngine:
    def __init__(
        self,
        semantic_engine: SearchEngine,
        bm25_engine: SearchEngine,
        stopwords: set[str] | None = None,
        top_k_per_engine: int = 100,
        rrf_k_penalty: int = 60,
    ):
        self.semantic_engine = semantic_engine
        self.bm25_engine = bm25_engine
        self.stopwords = stopwords or set()
        self.top_k_per_engine = top_k_per_engine
        self.rrf_k_penalty = rrf_k_penalty

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        intent = analyze_query_intent(query, stopwords=self.stopwords)

        semantic_results: list[SearchResult] = []
        bm25_results: list[SearchResult] = []

        if intent.use_semantic:
            semantic_results = self.semantic_engine.search(query, k=self.top_k_per_engine)
        if intent.use_bm25:
            bm25_results = self.bm25_engine.search(query, k=self.top_k_per_engine)

        if intent.use_semantic and intent.use_bm25:
            final_results = reciprocal_rank_fusion(
                semantic_results=semantic_results,
                bm25_results=bm25_results,
                semantic_weight=intent.weights["semantic"],
                bm25_weight=intent.weights["bm25"],
                top_k=top_k,
                k_penalty=self.rrf_k_penalty,
                exact_phrases=intent.exact_phrases,
            )
        elif intent.use_semantic:
            final_results = semantic_results[:top_k]
            _populate_percentage_score_by_top_score(final_results)
            for r in final_results:
                r.source = "semantic"
        else:
            final_results = bm25_results[:top_k]
            _populate_percentage_score_by_top_score(final_results)
            for r in final_results:
                r.source = "bm25"

        return final_results

    def check_query_terms(self, query: str) -> list[TermSuggestion]:
        """
        Delegates typo/OOV keyword detection to the BM25 engine's corpus
        vocabulary (see sanghabot/search/bm25.py::check_query_terms).
        Never raises and never changes search() behavior -- this is purely
        informational, consumed by the Discord bot to post a transparent,
        temporary "heads up" notice (see discord_bot.py's perform_search).

        Uses getattr with a no-op fallback rather than a hard dependency on
        BM25SearchEngine specifically, so any SearchEngine implementation
        satisfying the base Protocol (e.g. in tests, or a future engine
        swap) degrades gracefully to "nothing to flag" instead of crashing.
        """
        check_fn = getattr(self.bm25_engine, "check_query_terms", None)
        if check_fn is None:
            return []
        return check_fn(query)


def _populate_percentage_score_by_top_score(results: list[SearchResult]) -> None:
    """
    Populates percentage_score as a fraction of the top result's score.

    This is the explicit-loop replacement for the old search.py's two
    single-engine branches. The old bm25-only branch already did this
    correctly with a for-loop; the old semantic-only branch did NOT
    (it referenced an undefined `i`, causing a NameError -- see
    REWRITE_PLAN.md Section 0). Both branches now share this one
    implementation, so neither can silently diverge or omit the loop.
    """
    if not results:
        return
    top_score = results[0].score
    for r in results:
        r.percentage_score = (r.score / top_score * 100) if top_score else 0.0
