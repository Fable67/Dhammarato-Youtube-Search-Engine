"""
Tests for sanghabot.search.phrase.build_exact_phrase_tiers().

Uses a small fake stand-in for BM25SearchEngine (exposing only
find_exact_phrase_results(), the one method build_exact_phrase_tiers()
actually calls) so these tests are fast, deterministic, and exercise the
ALL/ANY tiering logic in isolation from BM25Plus/sqlite entirely. See
tests/test_bm25.py for coverage of find_exact_phrase_results() itself
against a real (fabricated) corpus.
"""
from sanghabot.models import SearchResult
from sanghabot.search.phrase import ExactPhraseTiers, build_exact_phrase_tiers


def make_result(chunk_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        video_id=chunk_id.split("_")[0],
        title=f"Title {chunk_id}",
        blog_url="https://example.com/blog/x",
        text=f"text for {chunk_id}",
        summary=None,
        score=score,
        percentage_score=0.0,
        source="bm25",
    )


class FakeBM25Engine:
    """
    Maps a phrase string to a pre-baked, already-ranked list of
    SearchResults -- mirrors BM25SearchEngine.find_exact_phrase_results()'s
    contract (ranked best-first, empty list for "no match") without
    needing a real BM25Plus index or sqlite database.
    """

    def __init__(self, matches_by_phrase: dict[str, list[SearchResult]]):
        self._matches_by_phrase = matches_by_phrase
        self.calls: list[str] = []

    def find_exact_phrase_results(self, phrase: str) -> list[SearchResult]:
        self.calls.append(phrase)
        return self._matches_by_phrase.get(phrase, [])


def test_no_phrases_returns_empty_tiers():
    engine = FakeBM25Engine({})
    tiers = build_exact_phrase_tiers([], engine)
    assert tiers.is_empty
    assert tiers.all_phrases_results == []
    assert tiers.any_phrase_results == []
    assert engine.calls == [], "should not call the engine at all when there are no phrases"


def test_single_phrase_all_and_any_tiers_are_identical():
    matches = [make_result("v1_0"), make_result("v1_1")]
    engine = FakeBM25Engine({"hot dog": matches})

    tiers = build_exact_phrase_tiers(["hot dog"], engine)

    assert not tiers.is_empty
    assert tiers.all_phrases_chunk_ids == ["v1_0", "v1_1"]
    assert tiers.any_phrase_chunk_ids == ["v1_0", "v1_1"]
    assert engine.calls == ["hot dog"]


def test_single_phrase_with_no_matches_is_empty():
    engine = FakeBM25Engine({"hot dog": []})
    tiers = build_exact_phrase_tiers(["hot dog"], engine)
    assert tiers.is_empty


def test_multi_phrase_all_tier_is_intersection():
    engine = FakeBM25Engine({
        "anatta": [make_result("both"), make_result("anatta_only")],
        "anicca": [make_result("both"), make_result("anicca_only")],
    })

    tiers = build_exact_phrase_tiers(["anatta", "anicca"], engine)

    assert tiers.all_phrases_chunk_ids == ["both"]


def test_multi_phrase_any_tier_is_union():
    engine = FakeBM25Engine({
        "anatta": [make_result("both"), make_result("anatta_only")],
        "anicca": [make_result("both"), make_result("anicca_only")],
    })

    tiers = build_exact_phrase_tiers(["anatta", "anicca"], engine)

    assert set(tiers.any_phrase_chunk_ids) == {"both", "anatta_only", "anicca_only"}


def test_multi_phrase_chunk_matching_all_also_appears_in_any_tier():
    engine = FakeBM25Engine({
        "anatta": [make_result("both")],
        "anicca": [make_result("both")],
    })

    tiers = build_exact_phrase_tiers(["anatta", "anicca"], engine)

    assert "both" in tiers.all_phrases_chunk_ids
    assert "both" in tiers.any_phrase_chunk_ids


def test_multi_phrase_no_overlap_gives_empty_all_tier_but_populated_any_tier():
    engine = FakeBM25Engine({
        "anatta": [make_result("anatta_only")],
        "anicca": [make_result("anicca_only")],
    })

    tiers = build_exact_phrase_tiers(["anatta", "anicca"], engine)

    assert tiers.all_phrases_results == []
    assert set(tiers.any_phrase_chunk_ids) == {"anatta_only", "anicca_only"}
    # ANY-only tier means the query still gets a meaningful boost even
    # though no single chunk contains both quoted phrases.
    assert not tiers.is_empty


def test_multi_phrase_neither_phrase_matches_anything_is_empty():
    engine = FakeBM25Engine({"anatta": [], "anicca": []})
    tiers = build_exact_phrase_tiers(["anatta", "anicca"], engine)
    assert tiers.is_empty


def test_multi_phrase_ranking_prefers_best_rank_across_phrases():
    """
    A chunk that was rank 0 (best) for one phrase should rank ahead, in
    the ANY tier, of a chunk that was only rank 5 for its best phrase --
    even though both only match one phrase each.
    """
    far_result = make_result("far_match")
    engine = FakeBM25Engine({
        "anatta": [make_result(f"filler_{i}") for i in range(5)] + [far_result],
        "anicca": [make_result("near_match")],
    })

    tiers = build_exact_phrase_tiers(["anatta", "anicca"], engine)

    assert tiers.any_phrase_chunk_ids.index("near_match") < tiers.any_phrase_chunk_ids.index("far_match")


def test_three_phrases_all_tier_requires_every_phrase():
    engine = FakeBM25Engine({
        "a": [make_result("all_three"), make_result("a_and_b")],
        "b": [make_result("all_three"), make_result("a_and_b")],
        "c": [make_result("all_three")],
    })

    tiers = build_exact_phrase_tiers(["a", "b", "c"], engine)

    assert tiers.all_phrases_chunk_ids == ["all_three"]
    assert set(tiers.any_phrase_chunk_ids) == {"all_three", "a_and_b"}


def test_exact_phrase_tiers_dataclass_defaults_are_empty():
    tiers = ExactPhraseTiers()
    assert tiers.is_empty
    assert tiers.all_phrases_chunk_ids == []
    assert tiers.any_phrase_chunk_ids == []
