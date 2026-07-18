"""
Tiered exact-phrase-match ranking signal for quoted queries.

Replaces the old, buggy design where a quoted phrase (e.g. `"hot dog"`)
was enforced as an EXCLUSIONARY filter inside reciprocal_rank_fusion():
any candidate not literally containing the phrase was dropped entirely,
and -- because of a branch-coverage bug in engine.py -- that filter was
never even applied for pure bm25-only queries (a quoted phrase with
nothing else in the query), so plain bag-of-words BM25 term-frequency
scoring won unopposed. Concretely, for the query '"hot dog"' this
produced a #1 result that says "hot" fifteen times and never once says
"dog", ranked above a chunk that actually contains "hot diggity dog".

This module instead treats exact-phrase matching as one or two
additional *ranked lists* that get fed into reciprocal_rank_fusion()
alongside the semantic and BM25 lists (see fusion.py), using the same
rank-fusion mechanism already used to blend those two engines. This has
two important properties, both required by direct user instruction:

  1. Exact matches are prioritized, not exclusively required: a chunk
     that only partially matches (or doesn't match at all) is never
     dropped from the result set -- it simply doesn't benefit from the
     extra exact-phrase-list contribution to its fused RRF score, so it
     naturally settles below true exact matches (when there are enough
     of them) rather than disappearing.
  2. Graceful, silent fallback when no exact match exists anywhere in
     the corpus: verified directly against the real corpus that the
     correctly-spelled Pali/Sanskrit term "paticca samuppada" (with a
     space) appears in ZERO chunks -- the corpus always spells it as one
     word, "paticcasamuppada" -- so the OLD exclusionary filter reduced
     `What is "paticca samuppada"?` to ZERO results even though the
     underlying semantic/BM25 engines found good candidates. Under this
     design, an empty exact-phrase-candidate list simply contributes
     nothing to fusion, and the query falls back to normal hybrid
     ranking untouched.

Tiered priority for MULTIPLE quoted phrases in one query (e.g.
'"anatta" "anicca"'), per explicit user instruction, in priority order:
  1. chunks containing ALL quoted phrases (intersection)
  2. chunks containing ANY quoted phrase (union)
  3. chunks containing no quoted phrase (ordinary semantic/BM25 ranking)

For a single-phrase query (the common case), tiers 1 and 2 are the same
set, so this collapses to the simple two-list case with no special
casing required by callers.

Full SearchResult objects (not bare chunk_ids) are carried through this
module, not just IDs: verified directly against the real corpus that a
large fraction of true exact-phrase matches for common two-word phrases
rank far outside the semantic/BM25 engines' own top-100-per-engine
candidate window (bag-of-words term frequency and semantic similarity
are not the same signal as literal phrase adjacency, so a chunk built
entirely around a phrase can still score low on either engine's own
metric). If only chunk_ids were carried, reciprocal_rank_fusion() would
have no SearchResult object to attach a fused score to for any exact
match outside that window, silently losing exactly the recall-gap cases
this feature exists to fix. See BM25SearchEngine.find_exact_phrase_results()
for where these SearchResult objects come from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sanghabot.models import SearchResult
from sanghabot.search.bm25 import BM25SearchEngine

# Weights used when feeding the ALL/ANY tiers into reciprocal_rank_fusion's
# weighted-sum-of-reciprocal-ranks formula (see fusion.py). Chosen to be
# comparable to or higher than the existing quoted-phrase bm25_weight
# (0.85, see sanghabot/search/intent.py) so exact matches reliably surface
# first when enough of them exist, while remaining plain RRF inputs (not a
# hard override), so they blend with -- rather than replace -- the
# semantic/BM25 signal instead of clobbering it.
#
# Empirically compared against a lower weight pair (0.6 / 0.3) on the real
# corpus for several representative queries ("hot dog", "anatta" "anicca",
# and the question-wrapped "paticca samuppada" case): both weight pairs
# produced IDENTICAL rank ordering in every case tested -- RRF is primarily
# rank-driven, so the weight mainly matters near close ties at the cutoff
# boundary. The stronger pair is used as the default since it best matches
# the requirement that exact matches should clearly dominate when present.
EXACT_PHRASE_ALL_WEIGHT = 1.0
EXACT_PHRASE_ANY_WEIGHT = 0.5


@dataclass
class ExactPhraseTiers:
    """
    The two ranked candidate lists produced by build_exact_phrase_tiers(),
    ready to be fed into reciprocal_rank_fusion() as additional weighted
    rank lists.

    `all_phrases_results`: SearchResults for chunks containing every
    quoted phrase, ranked best-first.
    `any_phrase_results`: SearchResults for chunks containing at least one
    quoted phrase, ranked best-first.

    For a single-phrase query these two lists contain the same chunks in
    the same order (distinct list objects, same content).
    """

    all_phrases_results: list[SearchResult] = field(default_factory=list)
    any_phrase_results: list[SearchResult] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.all_phrases_results and not self.any_phrase_results

    @property
    def all_phrases_chunk_ids(self) -> list[str]:
        return [r.chunk_id for r in self.all_phrases_results]

    @property
    def any_phrase_chunk_ids(self) -> list[str]:
        return [r.chunk_id for r in self.any_phrase_results]


def build_exact_phrase_tiers(
    exact_phrases: list[str],
    bm25_engine: BM25SearchEngine,
) -> ExactPhraseTiers:
    """
    Given the list of quoted phrases from analyze_query_intent()
    (`intent.exact_phrases`), returns the ALL-tier and ANY-tier result
    lists described in the module docstring.

    Each phrase's own matches are found and ranked independently via
    BM25SearchEngine.find_exact_phrase_results() (see that method's
    docstring for the two-stage candidate-narrowing + verification
    approach). Behavior for common cases:

      - No phrases (`exact_phrases == []`): returns an empty
        ExactPhraseTiers (both lists empty) -- callers should treat this
        as "no exact-phrase signal," not an error.
      - One phrase: both tiers contain the same chunks in the same order
        (trivially -- every chunk matching a single-element phrase set
        matches "all" and "any" of it).
      - Multiple phrases: `all_phrases_results` covers only chunks
        matching every phrase (ranked by the best rank any single phrase
        gave that chunk); `any_phrase_results` covers every chunk
        matching at least one phrase (same ranking rule). A chunk
        matching all phrases still also appears in the ANY list --
        reciprocal_rank_fusion() sums contributions across lists, so such
        a chunk naturally benefits from both tiers rather than needing
        deduplication here.
    """
    if not exact_phrases:
        return ExactPhraseTiers()

    per_phrase_matches: list[list[SearchResult]] = [
        bm25_engine.find_exact_phrase_results(phrase) for phrase in exact_phrases
    ]

    if len(per_phrase_matches) == 1:
        matches = per_phrase_matches[0]
        return ExactPhraseTiers(all_phrases_results=list(matches), any_phrase_results=list(matches))

    result_by_chunk_id: dict[str, SearchResult] = {}
    per_phrase_id_lists: list[list[str]] = []
    for matches in per_phrase_matches:
        ids: list[str] = []
        for res in matches:
            result_by_chunk_id.setdefault(res.chunk_id, res)
            ids.append(res.chunk_id)
        per_phrase_id_lists.append(ids)

    per_phrase_sets = [set(ids) for ids in per_phrase_id_lists]
    all_set = set.intersection(*per_phrase_sets)
    any_set = set.union(*per_phrase_sets)

    # Rank within each tier by the best (lowest/earliest) rank a chunk_id
    # achieved in any single phrase's own ranked list -- i.e. a chunk that
    # was the #1 exact match for one of the phrases ranks ahead of a chunk
    # that was only a middling match for every phrase. This is a simple,
    # predictable tie-breaker; exact relative ordering among multi-phrase
    # matches is a secondary concern next to the ALL > ANY > none priority
    # itself, which is enforced by feeding these as two separate
    # differently-weighted RRF lists (see EXACT_PHRASE_ALL_WEIGHT /
    # EXACT_PHRASE_ANY_WEIGHT above and fusion.py).
    def _rank_within(chunk_id_set: set[str]) -> list[SearchResult]:
        best_rank: dict[str, int] = {}
        for ids in per_phrase_id_lists:
            for rank, chunk_id in enumerate(ids):
                if chunk_id not in chunk_id_set:
                    continue
                if chunk_id not in best_rank or rank < best_rank[chunk_id]:
                    best_rank[chunk_id] = rank
        ordered_ids = [cid for cid, _ in sorted(best_rank.items(), key=lambda pair: pair[1])]
        return [result_by_chunk_id[cid] for cid in ordered_ids]

    return ExactPhraseTiers(
        all_phrases_results=_rank_within(all_set),
        any_phrase_results=_rank_within(any_set),
    )
