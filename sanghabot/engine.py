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
  - A quoted-exact-phrase branch-coverage bug found during a later
    investigation (see sanghabot/search/phrase.py's module docstring for
    the full story): a query that is JUST a quoted phrase with nothing
    else (e.g. '"hot dog"') sets use_semantic=False, use_bm25=True (see
    sanghabot/search/intent.py), which used to route straight to the
    bm25-only branch below WITHOUT ever considering exact_phrases at all
    -- so quoting a phrase had literally no effect on ranking for the
    single most common quoted-query shape. This is fixed below by always
    building exact-phrase tiers whenever intent.exact_phrases is
    non-empty and routing through reciprocal_rank_fusion() whenever there
    is at least one real signal to fuse (semantic+bm25, or a non-empty
    exact-phrase tier), rather than only when both engines happen to be
    active.
"""
from __future__ import annotations

from sanghabot.models import SearchResult, TermSuggestion
from sanghabot.search.base import SearchEngine
from sanghabot.search.fusion import reciprocal_rank_fusion
from sanghabot.search.intent import analyze_query_intent
from sanghabot.search.phrase import (
    EXACT_PHRASE_ALL_WEIGHT,
    EXACT_PHRASE_ANY_WEIGHT,
    ExactPhraseTiers,
    build_exact_phrase_tiers,
)


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

        exact_phrase_tiers = self._build_exact_phrase_tiers(intent.exact_phrases)
        has_exact_phrase_signal = exact_phrase_tiers is not None and not exact_phrase_tiers.is_empty

        if (intent.use_semantic and intent.use_bm25) or has_exact_phrase_signal:
            final_results = reciprocal_rank_fusion(
                semantic_results=semantic_results,
                bm25_results=bm25_results,
                semantic_weight=intent.weights["semantic"],
                bm25_weight=intent.weights["bm25"],
                top_k=top_k,
                k_penalty=self.rrf_k_penalty,
                exact_phrase_all_results=exact_phrase_tiers.all_phrases_results if exact_phrase_tiers else None,
                exact_phrase_any_results=exact_phrase_tiers.any_phrase_results if exact_phrase_tiers else None,
                exact_phrase_all_weight=EXACT_PHRASE_ALL_WEIGHT,
                exact_phrase_any_weight=EXACT_PHRASE_ANY_WEIGHT,
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

    def _build_exact_phrase_tiers(self, exact_phrases: list[str]) -> ExactPhraseTiers | None:
        """
        Returns the ALL/ANY exact-phrase tiers for `exact_phrases` (see
        sanghabot/search/phrase.py), or None if there are no quoted
        phrases to look up, or if `self.bm25_engine` doesn't implement
        `find_exact_phrase_results` (mirrors check_query_terms()'s
        getattr-based graceful degradation below, e.g. for a test double
        or a future SearchEngine implementation that doesn't support
        this).

        None is deliberately distinct from an ExactPhraseTiers with two
        empty lists: the former means "phrase lookup wasn't attempted at
        all," the latter means "it was attempted and genuinely found
        nothing" (e.g. a misspelled Pali term) -- both end up not
        contributing anything to fusion, but the has_exact_phrase_signal
        check in search() above only needs to know whether there's an
        empty-vs-populated result either way.
        """
        if not exact_phrases:
            return None
        find_fn = getattr(self.bm25_engine, "find_exact_phrase_results", None)
        if find_fn is None:
            return None
        return build_exact_phrase_tiers(exact_phrases, self.bm25_engine)

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
