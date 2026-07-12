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


def test_fusion_exact_phrase_filter_excludes_non_matching_results():
    semantic_results = [
        make_result("v1_0", text="talks about right effort in depth"),
        make_result("v1_1", text="talks about something else entirely"),
    ]
    fused = reciprocal_rank_fusion(
        semantic_results, [], top_k=5, exact_phrases=["right effort"],
    )
    assert len(fused) == 1
    assert fused[0].chunk_id == "v1_0"


def test_fusion_dedups_same_chunk_appearing_in_both_engines():
    semantic_results = [make_result("v1_0")]
    bm25_results = [make_result("v1_0")]
    fused = reciprocal_rank_fusion(semantic_results, bm25_results, top_k=5)
    assert len(fused) == 1
