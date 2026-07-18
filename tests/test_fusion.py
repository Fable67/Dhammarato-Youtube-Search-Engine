"""
Tests for sanghabot.search.fusion.reciprocal_rank_fusion().

Includes a direct regression test for the old search.py:249 crash:

    elif intent["use_semantic"]:
        final_results = semantic_results[:top_k]
        final_results[i]["percentage_score"] = final_results[i]["score"] * 100
        # NameError: name 'i' is not defined -- no loop in this branch.

That branch lived in engine.py/search dispatch, not in fusion.py itself,
but the fix that makes it structurally impossible is that
`percentage_score` assignment now always happens through this module's
loop-based reconstruction (see sanghabot/engine.py), never through a
bare indexed assignment outside a loop.

Also includes regression coverage for the exact-phrase EXCLUSION bug
found later (see sanghabot/search/phrase.py's module docstring for the
full story): this module used to drop any candidate not literally
containing every quoted phrase, which could -- and, verified directly
against the real corpus, DID -- reduce an entire result set to zero
whenever the corpus never spells the quoted phrase exactly as typed
(e.g. "paticca samuppada" vs. the corpus's actual "paticcasamuppada").
Exact-phrase matching is now supplied as two additional *ranked lists*
(`exact_phrase_all_results` / `exact_phrase_any_results`) that are fused
in exactly the same way as the semantic/BM25 lists -- boosting, never
excluding -- so partial/non-matching candidates are always retained.
"""
from sanghabot.models import SearchResult
from sanghabot.search.fusion import reciprocal_rank_fusion


def make_result(chunk_id: str, text: str = "some text") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        video_id=chunk_id.split("_")[0],
        title=f"Title {chunk_id}",
        blog_url="https://example.com/blog/x",
        text=text,
        summary=None,
        score=0.0,
        percentage_score=0.0,
        source="semantic",
    )


def test_fusion_orders_by_combined_rank():
    semantic_results = [make_result("v1_0"), make_result("v1_1"), make_result("v1_2")]
    bm25_results = [make_result("v1_2"), make_result("v1_0"), make_result("v1_1")]

    fused = reciprocal_rank_fusion(
        semantic_results, bm25_results,
        semantic_weight=0.5, bm25_weight=0.5, top_k=3,
    )

    assert len(fused) == 3
    # v1_0 is rank 0 in semantic and rank 1 in bm25 -> best combined score
    # v1_2 is rank 2 in semantic and rank 0 in bm25
    # v1_1 is rank 1 in semantic and rank 2 in bm25
    # With equal weights, v1_0 and v1_2 tie in structure but v1_0 has a
    # slightly better combined rank sum (0+1=1) vs v1_2 (2+0=2), so v1_0
    # should be first.
    assert fused[0].chunk_id == "v1_0"


def test_fusion_populates_percentage_score_for_every_result():
    semantic_results = [make_result("v1_0")]
    bm25_results = [make_result("v1_1")]

    fused = reciprocal_rank_fusion(semantic_results, bm25_results, top_k=5)

    assert len(fused) == 2
    for r in fused:
        # This is the exact field that was left unset in the old crash bug.
        assert r.percentage_score > 0
        assert r.source == "fusion"


def test_fusion_handles_empty_semantic_results_without_crashing():
    fused = reciprocal_rank_fusion([], [make_result("v1_0")], top_k=3)
    assert len(fused) == 1
    assert fused[0].percentage_score > 0


def test_fusion_handles_empty_bm25_results_without_crashing():
    fused = reciprocal_rank_fusion([make_result("v1_0")], [], top_k=3)
    assert len(fused) == 1
    assert fused[0].percentage_score > 0


def test_fusion_handles_both_empty_without_crashing():
    fused = reciprocal_rank_fusion([], [], top_k=3)
    assert fused == []


def test_fusion_respects_top_k_trim():
    semantic_results = [make_result(f"v1_{i}") for i in range(10)]
    fused = reciprocal_rank_fusion(semantic_results, [], top_k=3)
    assert len(fused) == 3


def test_exact_phrase_match_boosts_without_excluding_others():
    """
    Direct regression test for the removed exclusionary filter: v1_1
    (which does NOT match the exact phrase) must still be present in the
    fused output, just ranked below v1_0 (which does match).
    """
    semantic_results = [
        make_result("v1_1", text="talks about something else entirely"),
        make_result("v1_0", text="talks about right effort in depth"),
    ]
    fused = reciprocal_rank_fusion(
        semantic_results, [], top_k=5,
        exact_phrase_all_results=[make_result("v1_0", text="talks about right effort in depth")],
        exact_phrase_any_results=[make_result("v1_0", text="talks about right effort in depth")],
    )
    assert len(fused) == 2, "non-matching candidate must NOT be dropped"
    assert {r.chunk_id for r in fused} == {"v1_0", "v1_1"}
    # v1_0 gets an extra boost from both exact-phrase tiers, so it must
    # outrank v1_1 despite starting at a worse semantic rank (index 1).
    assert fused[0].chunk_id == "v1_0"
    assert fused[1].chunk_id == "v1_1"


def test_no_exact_phrase_matches_falls_back_to_plain_fusion():
    """
    Regression test for the old exclusion-filter bug where a query with
    zero literal phrase matches anywhere in the corpus (e.g. because the
    corpus spells a Pali/Sanskrit term differently, like "paticcasamuppada"
    instead of "paticca samuppada") reduced the ENTIRE result set to zero.
    Passing empty exact-phrase lists must behave identically to not
    passing any at all.
    """
    semantic_results = [make_result("v1_0"), make_result("v1_1")]
    bm25_results = [make_result("v1_1"), make_result("v1_0")]

    fused_without_phrases = reciprocal_rank_fusion(semantic_results, bm25_results, top_k=5)
    fused_with_empty_phrases = reciprocal_rank_fusion(
        semantic_results, bm25_results, top_k=5,
        exact_phrase_all_results=[], exact_phrase_any_results=[],
    )

    assert len(fused_without_phrases) == 2
    assert [r.chunk_id for r in fused_without_phrases] == [r.chunk_id for r in fused_with_empty_phrases]
    for a, b in zip(fused_without_phrases, fused_with_empty_phrases):
        assert a.percentage_score == b.percentage_score


def test_exact_phrase_result_outside_both_engines_still_appears():
    """
    Regression test for the recall-gap finding: a true exact-phrase match
    that falls entirely outside both engines' own candidate windows (never
    appears in semantic_results or bm25_results at all) must still surface
    in the final fused output, using the SearchResult object carried by
    the exact-phrase list itself -- not be silently dropped for lack of a
    known result object.
    """
    semantic_results = [make_result("v1_0"), make_result("v1_1")]
    bm25_results = [make_result("v1_1"), make_result("v1_0")]
    # v2_0 never appears in either engine's own results.
    exact_only_result = make_result("v2_0", text="the exact phrase appears here")

    fused = reciprocal_rank_fusion(
        semantic_results, bm25_results, top_k=5,
        exact_phrase_all_results=[exact_only_result],
        exact_phrase_any_results=[exact_only_result],
    )

    chunk_ids = {r.chunk_id for r in fused}
    assert "v2_0" in chunk_ids, "exact-phrase-only match must not be dropped"
    # It should also rank first, since it benefits from both the ALL and
    # ANY tier's top-rank contribution while the others only have
    # semantic/bm25 contributions.
    assert fused[0].chunk_id == "v2_0"


def test_all_tier_ranks_above_any_tier_only_match():
    """
    A chunk present in BOTH the ALL tier and the ANY tier (i.e. it matches
    every quoted phrase) should outrank a chunk present in only the ANY
    tier (i.e. it matches just one of several quoted phrases), all else
    being equal -- this is the ALL > ANY > none priority ordering
    requested explicitly.
    """
    semantic_results = [make_result("both_phrases"), make_result("one_phrase_only")]
    all_tier = [make_result("both_phrases")]
    any_tier = [make_result("both_phrases"), make_result("one_phrase_only")]

    fused = reciprocal_rank_fusion(
        semantic_results, [], top_k=5,
        exact_phrase_all_results=all_tier,
        exact_phrase_any_results=any_tier,
    )

    assert [r.chunk_id for r in fused][0] == "both_phrases"


def test_fusion_dedups_same_chunk_appearing_in_both_engines():
    semantic_results = [make_result("v1_0")]
    bm25_results = [make_result("v1_0")]
    fused = reciprocal_rank_fusion(semantic_results, bm25_results, top_k=5)
    assert len(fused) == 1
