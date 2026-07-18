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

EXACT-PHRASE HANDLING (see sanghabot/search/phrase.py for the full
rationale): earlier versions of this function enforced quoted phrases
(e.g. `"hot dog"`) as an EXCLUSIONARY filter -- any candidate not
literally containing the phrase was dropped from the result set outright.
This was found to be a real, user-visible bug: (a) it could zero out the
ENTIRE result set when the corpus never spells the phrase exactly as
quoted (verified directly: "paticca samuppada" with a space appears in
zero corpus chunks, since transcripts always spell it as one word,
"paticcasamuppada" -- so `What is "paticca samuppada"?` returned zero
results even though the underlying semantic/BM25 engines had good
candidates); and (b) it was never even applied for pure bm25-only quoted
queries in the first place, due to a branch-coverage bug in engine.py, so
plain bag-of-words BM25 term-frequency scoring won unopposed for the most
common case (a query that is *just* a quoted phrase).

Exact-phrase matching is now instead supplied by the caller as up to two
additional ranked SearchResult lists (`exact_phrase_all_results` /
`exact_phrase_any_results`, produced by
sanghabot/search/phrase.py::build_exact_phrase_tiers()) and fused using
the exact same weighted-reciprocal-rank mechanism already used to blend
semantic and BM25 below -- never an exclusion. A chunk that doesn't
appear in either list simply doesn't receive that extra contribution to
its fused score; it is never removed from consideration. This means
partial matches are still returned (just typically ranked lower than true
exact matches when enough exist), and queries with zero exact matches
anywhere in the corpus gracefully fall back to ordinary semantic/BM25
ranking with no special-casing needed by the caller.

These are full SearchResult lists, not bare chunk_id lists, deliberately:
a true exact-phrase match frequently falls entirely outside both engines'
own top-100-per-engine candidate window (verified directly against the
real corpus -- literal phrase adjacency is a different signal than
bag-of-words term frequency or semantic similarity, so a chunk built
around a phrase can still score low on either engine's own metric). If
only chunk_ids were accepted, this function would have no SearchResult
object to attach a fused score to for exact matches outside that window,
silently losing exactly the recall-gap cases this feature exists to fix.
"""
from __future__ import annotations

from sanghabot.models import SearchResult


def reciprocal_rank_fusion(
    semantic_results: list[SearchResult],
    bm25_results: list[SearchResult],
    semantic_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 5,
    k_penalty: int = 60,
    exact_phrase_all_results: list[SearchResult] | None = None,
    exact_phrase_any_results: list[SearchResult] | None = None,
    exact_phrase_all_weight: float = 1.0,
    exact_phrase_any_weight: float = 0.5,
) -> list[SearchResult]:
    """
    Combines results from semantic search and BM25 (plus, optionally, up
    to two exact-phrase-match rank lists -- see module docstring) using
    Reciprocal Rank Fusion. RRF ignores raw scores (which live on totally
    different scales between engines) and merges purely based on rank
    position.

    k_penalty is a smoothing constant (industry standard is usually 60).

    `exact_phrase_all_results` / `exact_phrase_any_results`: ranked
    SearchResult lists (see module docstring for why these are full
    result objects, not bare chunk_ids). A chunk_id appearing in one of
    these lists but not in `semantic_results`/`bm25_results` is still
    registered as a known result (using the SearchResult object supplied
    here), so it can appear in the final fused output purely on the
    strength of its exact-phrase match.
    """
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
    total_weight = 0.0

    def _add_rank_list(results: list[SearchResult], weight: float) -> None:
        nonlocal total_weight
        if not results:
            return
        total_weight += weight
        seen: set[str] = set()
        for rank, res in enumerate(results):
            unique_id = res.chunk_id
            if unique_id in seen:
                continue
            seen.add(unique_id)
            # Register this chunk's result object if it's not already
            # known from an earlier list (e.g. an exact-phrase match that
            # fell outside both engines' own candidate windows) -- see
            # module docstring for why this must be allowed to introduce
            # NEW chunks, not just re-rank already-known ones.
            if unique_id not in result_objects:
                result_objects[unique_id] = res
            score = weight * (1.0 / (k_penalty + rank))
            fused_scores[unique_id] = fused_scores.get(unique_id, 0.0) + score

    # 1. Score every rank list -- semantic and BM25 always; the two
    #    exact-phrase tiers only when the caller supplied them (a query
    #    with no quoted phrases passes neither, and this loop is then
    #    functionally identical to the pre-exact-phrase-feature version).
    _add_rank_list(semantic_results, semantic_weight)
    _add_rank_list(bm25_results, bm25_weight)
    _add_rank_list(exact_phrase_all_results or [], exact_phrase_all_weight)
    _add_rank_list(exact_phrase_any_results or [], exact_phrase_any_weight)

    # 2. Sort by fused score, descending.
    sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # 3. Reconstruct the final list. Always via an explicit loop, so
    #    percentage_score is unconditionally populated for every branch that
    #    calls this function -- unlike the old search.py's use_semantic-only
    #    branch which indexed an undefined loop variable.
    #
    #    percentage_score is normalized against the theoretical maximum
    #    possible fused_score for THIS call: a chunk ranked #1 (rank=0) in
    #    every single rank list that was actually supplied would score
    #    total_weight * (1/k_penalty); dividing by that (rather than
    #    assuming weights always sum to 1, which the original two-list-only
    #    version could get away with) keeps percentage_score bounded to
    #    <=100 regardless of how many extra weighted rank lists (e.g. the
    #    exact-phrase tiers) are active for a given query. Guarded against
    #    total_weight == 0 (only possible if every list passed was empty,
    #    which already returns an empty final_results below).
    max_possible_score = (total_weight / k_penalty) if total_weight > 0 else 0.0
    final_results: list[SearchResult] = []
    for unique_id, fused_score in sorted_chunks[:top_k]:
        res = result_objects[unique_id]
        res.score = fused_score
        res.percentage_score = (fused_score / max_possible_score * 100) if max_possible_score else 0.0
        res.source = "fusion"
        final_results.append(res)

    return final_results
