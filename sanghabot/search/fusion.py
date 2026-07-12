"""
Canonical Reciprocal Rank Fusion (RRF) implementation.

This is the ONE implementation, replacing two near-duplicate copies from
the old codebase:
  - AI_Transcripts/keyword_search.py: reciprocal_rank_fusion()
  - AI_Transcripts/search.py:         CombinedSearchEngine._reciprocal_rank_fusion()

It also structurally eliminates the crash bug present in the old
search.py:247-249:

    elif intent["use_semantic"]:
        final_results = semantic_results[:top_k]
        final_results[i]["percentage_score"] = final_results[i]["score"] * 100
        # `i` is never defined here -- there is no loop in this branch.
        # NameError waiting to happen the moment this branch is exercised.

See sanghabot/engine.py for how percentage_score is now always populated
via an explicit loop in every branch, and tests/test_fusion.py +
tests/test_parity_old_vs_new.py for regression coverage of this exact bug.
"""
from __future__ import annotations

import re

from sanghabot.models import SearchResult


def reciprocal_rank_fusion(
    semantic_results: list[SearchResult],
    bm25_results: list[SearchResult],
    semantic_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 5,
    k_penalty: int = 60,
    exact_phrases: list[str] | None = None,
) -> list[SearchResult]:
    """
    Combines results from semantic search and BM25 using Reciprocal Rank
    Fusion. RRF ignores raw scores (which live on totally different scales
    between the two engines) and merges purely based on rank position.

    k_penalty is a smoothing constant (industry standard is usually 60).
    """
    # Apply strict exact-phrase filtering before fusion, same as old code.
    if exact_phrases:
        def contains_phrases(res: SearchResult) -> bool:
            text = (res.text or "").lower()
            return all(re.search(re.escape(p), text) for p in exact_phrases)

        semantic_results = [r for r in semantic_results if contains_phrases(r)]
        bm25_results = [r for r in bm25_results if contains_phrases(r)]

    # NOTE on a subtle old-code quirk (documented, not reproduced): the old
    # search.py/keyword_search.py guarded BM25 dedup via a `debug_info` dict
    # that is only ever populated when `debug=True`. In production
    # (debug=False), that guard was always a no-op, so a duplicate chunk_id
    # within bm25_results would have its score added more than once. This
    # never manifests in practice because BM25's top_n_indices are produced
    # via np.argsort, which cannot yield duplicate indices -- so old
    # production behavior and this corrected, always-deduped implementation
    # are functionally identical. See tests/golden/KNOWN_DIVERGENCES.md.
    fused_scores: dict[str, float] = {}
    result_objects: dict[str, SearchResult] = {}
    seen_semantic: set[str] = set()
    seen_bm25: set[str] = set()

    # 1. Score semantic results based on rank.
    for rank, res in enumerate(semantic_results):
        unique_id = res.chunk_id
        if unique_id not in result_objects:
            result_objects[unique_id] = res
        if unique_id not in seen_semantic:
            seen_semantic.add(unique_id)
            score = semantic_weight * (1.0 / (k_penalty + rank))
            fused_scores[unique_id] = fused_scores.get(unique_id, 0.0) + score

    # 2. Score BM25 results based on rank.
    for rank, res in enumerate(bm25_results):
        unique_id = res.chunk_id
        if unique_id not in result_objects:
            result_objects[unique_id] = res
        if unique_id not in seen_bm25:
            seen_bm25.add(unique_id)
            score = bm25_weight * (1.0 / (k_penalty + rank))
            fused_scores[unique_id] = fused_scores.get(unique_id, 0.0) + score

    # 3. Sort by fused score, descending.
    sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # 4. Reconstruct the final list. Always via an explicit loop, so
    #    percentage_score is unconditionally populated for every branch that
    #    calls this function -- unlike the old search.py's use_semantic-only
    #    branch which indexed an undefined loop variable.
    final_results: list[SearchResult] = []
    for unique_id, fused_score in sorted_chunks[:top_k]:
        res = result_objects[unique_id]
        res.score = fused_score
        res.percentage_score = fused_score * k_penalty * 100
        res.source = "fusion"
        final_results.append(res)

    return final_results
