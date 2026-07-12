"""
Tests for sanghabot.engine.CombinedSearchEngine.check_query_terms(), which
delegates typo/OOV keyword detection to whichever underlying search engine
exposes it (in practice, BM25SearchEngine -- see sanghabot/search/bm25.py).

Uses tiny fake SearchEngine stand-ins (not real BM25/FAISS) so these tests
are fast and isolate the delegation logic itself.
"""
from sanghabot.engine import CombinedSearchEngine
from sanghabot.models import TermSuggestion


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
