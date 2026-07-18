"""
Tests for sanghabot.engine.CombinedSearchEngine.

Covers two things:
  1. check_query_terms(), which delegates typo/OOV keyword detection to
     whichever underlying search engine exposes it (in practice,
     BM25SearchEngine -- see sanghabot/search/bm25.py).
  2. search()'s exact-phrase routing, including a direct regression test
     for the branch-coverage bug found during investigation (see
     sanghabot/search/phrase.py's module docstring): a query that is
     JUST a quoted phrase (use_semantic=False, use_bm25=True) used to
     route straight to the bm25-only branch WITHOUT ever considering
     intent.exact_phrases at all, so quoting a phrase had zero effect on
     ranking for the single most common quoted-query shape.

Uses tiny fake SearchEngine stand-ins (not real BM25/FAISS) so these
tests are fast and isolate the routing/delegation logic itself from the
real BM25Plus/FAISS/sqlite machinery (that's covered separately in
tests/test_bm25.py and tests/test_phrase.py).
"""
from sanghabot.engine import CombinedSearchEngine
from sanghabot.models import SearchResult, TermSuggestion


def make_result(chunk_id: str, score: float = 1.0, text: str = "") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        video_id=chunk_id.split("_")[0],
        title=f"Title {chunk_id}",
        blog_url="https://example.com/blog/x",
        text=text or f"text for {chunk_id}",
        summary=None,
        score=score,
        percentage_score=0.0,
        source="bm25",
    )


class FakeEngineWithCheck:
    def __init__(self, suggestions):
        self._suggestions = suggestions
        self.last_query = None

    def search(self, query, k):
        return []

    def check_query_terms(self, query):
        self.last_query = query
        return self._suggestions


class FakeEngineWithoutCheck:
    """Simulates a SearchEngine implementation that doesn't implement
    check_query_terms (e.g. SemanticSearchEngine, or a future engine)."""

    def search(self, query, k):
        return []


class FakeBM25EngineWithPhraseSupport:
    """
    A fake bm25_engine that returns pre-baked search() results AND
    supports find_exact_phrase_results(), so CombinedSearchEngine.search()
    can be tested end-to-end (routing + fusion + exact-phrase boosting)
    without any real BM25Plus/sqlite machinery.
    """

    def __init__(self, search_results=None, phrase_results_by_phrase=None):
        self._search_results = search_results or []
        self._phrase_results_by_phrase = phrase_results_by_phrase or {}
        self.search_calls: list[str] = []
        self.phrase_calls: list[str] = []

    def search(self, query, k):
        self.search_calls.append(query)
        return list(self._search_results)

    def find_exact_phrase_results(self, phrase):
        self.phrase_calls.append(phrase)
        return list(self._phrase_results_by_phrase.get(phrase, []))


class FakeSemanticEngine:
    def __init__(self, search_results=None):
        self._search_results = search_results or []
        self.search_calls: list[str] = []

    def search(self, query, k):
        self.search_calls.append(query)
        return list(self._search_results)


def test_check_query_terms_delegates_to_bm25_engine():
    expected = [TermSuggestion(term="duka", suggestion="dukkha")]
    bm25 = FakeEngineWithCheck(expected)
    semantic = FakeEngineWithoutCheck()
    engine = CombinedSearchEngine(semantic_engine=semantic, bm25_engine=bm25)

    result = engine.check_query_terms("duka meditation")

    assert result == expected
    assert bm25.last_query == "duka meditation"


def test_check_query_terms_returns_empty_list_when_bm25_engine_lacks_method():
    bm25 = FakeEngineWithoutCheck()
    semantic = FakeEngineWithoutCheck()
    engine = CombinedSearchEngine(semantic_engine=semantic, bm25_engine=bm25)

    result = engine.check_query_terms("anything")

    assert result == []


def test_check_query_terms_passes_through_empty_suggestions():
    bm25 = FakeEngineWithCheck([])
    semantic = FakeEngineWithoutCheck()
    engine = CombinedSearchEngine(semantic_engine=semantic, bm25_engine=bm25)

    assert engine.check_query_terms("dukkha") == []


# ---------------------------------------------------------------------------
# search() exact-phrase routing, including the branch-coverage bug regression
# ---------------------------------------------------------------------------

def test_pure_quoted_phrase_query_still_applies_exact_phrase_boost():
    """
    Direct regression test for the branch-coverage bug: a query that is
    JUST a quoted phrase (e.g. '"hot dog"') sets use_semantic=False,
    use_bm25=True (see sanghabot/search/intent.py), which used to route
    straight to the bm25-only branch in engine.py WITHOUT ever looking at
    intent.exact_phrases -- so quoting had no effect on ranking. This
    test uses plain bag-of-words bm25_results ordered so that a
    NON-matching chunk ("bag_of_words_winner") would win under the old
    behavior, and asserts the TRUE exact-phrase match ("true_phrase_match")
    is now ranked first instead.
    """
    bag_of_words_winner = make_result("bag_of_words_winner", score=99.0, text="hot hot hot hot hot")
    true_phrase_match = make_result("true_phrase_match", score=5.0, text="a hot dog with mustard")

    bm25 = FakeBM25EngineWithPhraseSupport(
        search_results=[bag_of_words_winner, true_phrase_match],
        phrase_results_by_phrase={"hot dog": [true_phrase_match]},
    )
    semantic = FakeSemanticEngine()
    engine = CombinedSearchEngine(semantic_engine=semantic, bm25_engine=bm25)

    results = engine.search('"hot dog"', top_k=5)

    assert semantic.search_calls == [], "use_semantic=False for a pure quoted phrase -- semantic must not be called"
    assert bm25.phrase_calls == ["hot dog"], "exact-phrase lookup must be invoked for a pure bm25-only quoted query"
    assert len(results) == 2, "non-matching candidate must still be present, not excluded"
    assert results[0].chunk_id == "true_phrase_match"
    assert results[1].chunk_id == "bag_of_words_winner"


def test_quoted_phrase_with_no_corpus_match_falls_back_gracefully():
    """
    Regression test for the old exclusion-filter bug: when the quoted
    phrase matches NOTHING in the corpus (e.g. a misspelling, or a
    tokenization mismatch like "paticca samuppada" vs. the corpus's
    "paticcasamuppada"), the query must still return the underlying
    bm25/semantic results -- never collapse to zero results.
    """
    fallback_result = make_result("fallback_result", score=10.0)
    bm25 = FakeBM25EngineWithPhraseSupport(
        search_results=[fallback_result],
        phrase_results_by_phrase={},  # no match for any phrase
    )
    semantic = FakeSemanticEngine()
    engine = CombinedSearchEngine(semantic_engine=semantic, bm25_engine=bm25)

    results = engine.search('"pattica sammupadha"', top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "fallback_result"


def test_quoted_phrase_with_extra_context_uses_both_engines_and_boost():
    """
    A quoted phrase with extra context outside the quotes (e.g.
    '"right effort" and how it fits into daily life') sets
    use_semantic=True AND use_bm25=True (see intent.py) -- this already
    routed through reciprocal_rank_fusion() even before the bug fix, but
    this test confirms the exact-phrase tiers are additionally applied on
    top of the normal two-engine fusion, not replacing it.
    """
    semantic_hit = make_result("semantic_hit", score=0.9)
    bm25_hit = make_result("bm25_hit", score=10.0)
    exact_match = make_result("exact_match", score=3.0, text="right effort in daily life")

    bm25 = FakeBM25EngineWithPhraseSupport(
        search_results=[bm25_hit],
        phrase_results_by_phrase={"right effort": [exact_match]},
    )
    semantic = FakeSemanticEngine(search_results=[semantic_hit])
    engine = CombinedSearchEngine(semantic_engine=semantic, bm25_engine=bm25)

    results = engine.search('"right effort" and how it fits into daily life', top_k=5)

    assert semantic.search_calls, "use_semantic=True here -- semantic must still be consulted"
    chunk_ids = {r.chunk_id for r in results}
    assert chunk_ids == {"semantic_hit", "bm25_hit", "exact_match"}
    assert results[0].chunk_id == "exact_match", "exact-phrase match should rank first given its extra boost"


def test_unquoted_query_never_triggers_phrase_lookup():
    bm25 = FakeBM25EngineWithPhraseSupport(search_results=[make_result("v1_0")])
    semantic = FakeSemanticEngine(search_results=[make_result("v1_0")])
    engine = CombinedSearchEngine(semantic_engine=semantic, bm25_engine=bm25)

    engine.search("blue green red", top_k=5)

    assert bm25.phrase_calls == [], "no quoted phrase in the query -- phrase lookup must never be invoked"


def test_exact_phrase_lookup_gracefully_skipped_when_bm25_engine_lacks_support():
    """
    If the bm25_engine doesn't implement find_exact_phrase_results (e.g.
    a minimal test double, or some future SearchEngine implementation),
    a quoted query must still work -- just without the boost -- rather
    than raising.
    """
    bm25 = FakeEngineWithoutCheck()  # no find_exact_phrase_results either
    semantic = FakeEngineWithoutCheck()
    engine = CombinedSearchEngine(semantic_engine=semantic, bm25_engine=bm25)

    results = engine.search('"hot dog"', top_k=5)

    assert results == []
