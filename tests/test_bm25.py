"""
Tests for sanghabot.search.bm25.

Covers check_query_terms() -- the typo/OOV keyword detection used to
power a transparent, non-blocking Discord notice (see
tests/test_discord_bot_formatting.py and
sanghabot/bot/discord_bot.py's perform_search()) -- and
find_exact_phrase_results(), the exact-phrase-match ranking signal used
to power quoted-query boosting (see sanghabot/search/phrase.py and
sanghabot/search/fusion.py).

Uses a small, fabricated in-memory-ish corpus (tmp_path sqlite db) rather
than the real production data/index/bm25.pkl, so these tests are fast,
deterministic, and don't depend on the real corpus's exact contents.
"""
from sanghabot.models import Chunk, VideoMetadata
from sanghabot.search.bm25 import BM25SearchEngine
from sanghabot.storage.db import Database


def make_engine(tmp_path) -> BM25SearchEngine:
    db = Database(tmp_path / "test.db")
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks(
        [
            Chunk(
                chunk_id="v1_0", video_id="v1", chunk_idx=0,
                text="The Buddha taught about dukkha, the jhana states, and anatta.",
            ),
            Chunk(
                chunk_id="v1_1", video_id="v1", chunk_idx=1,
                text="Breath meditation and mindfulness practice lead toward nibbana.",
            ),
        ]
    )
    return BM25SearchEngine(db, tmp_path / "bm25.pkl")


def make_phrase_engine(tmp_path) -> BM25SearchEngine:
    """
    A larger, more varied fabricated corpus specifically for exercising
    find_exact_phrase_results()'s two-stage candidate-narrowing +
    verification logic, mirroring the exact real-corpus scenarios found
    during investigation (see sanghabot/search/bm25.py's docstring):
      - a true, contiguous multi-word phrase ("hot dog")
      - the SAME two words present, but NOT adjacent/not a real phrase
        (BM25 bag-of-words would score this highly on term frequency
        alone, which is exactly the bug being fixed)
      - punctuation/clause-boundary false-positive case ("right effort"
        vs. "that's right, effort." -- these must NOT match)
      - a compound-word transliteration case ("paticca samuppada" query
        vs. corpus text "paticcasamuppada" as one word)
      - plural tolerance ("breath" query matching "breaths" in text)
    """
    db = Database(tmp_path / "phrase_test.db")
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks(
        [
            Chunk(
                chunk_id="v1_0", video_id="v1", chunk_idx=0,
                text="I love my hot dog, mustard and all, it's the best hot dog in town.",
            ),
            Chunk(
                chunk_id="v1_1", video_id="v1", chunk_idx=1,
                text="It was so hot today. My dog wanted to go for a walk anyway.",
            ),
            Chunk(
                chunk_id="v1_2", video_id="v1", chunk_idx=2,
                text="We did have to put out right effort. It was not complicated.",
            ),
            Chunk(
                chunk_id="v1_3", video_id="v1", chunk_idx=3,
                text="Well, that's right, effort is not the same as struggle.",
            ),
            Chunk(
                chunk_id="v1_4", video_id="v1", chunk_idx=4,
                text="The Buddha taught the doctrine of paticcasamuppada in depth.",
            ),
            Chunk(
                chunk_id="v1_5", video_id="v1", chunk_idx=5,
                text="Notice your breaths as you sit quietly and observe the mind.",
            ),
            Chunk(
                chunk_id="v1_6", video_id="v1", chunk_idx=6,
                text="Nothing relevant is mentioned in this chunk at all.",
            ),
        ]
    )
    return BM25SearchEngine(db, tmp_path / "phrase_test.pkl")


def test_known_word_produces_no_suggestion(tmp_path):
    engine = make_engine(tmp_path)
    suggestions = engine.check_query_terms("dukkha")
    assert suggestions == []


def test_typo_of_known_word_gets_suggestion(tmp_path):
    engine = make_engine(tmp_path)
    suggestions = engine.check_query_terms("duka")
    assert len(suggestions) == 1
    assert suggestions[0].term == "duka"
    assert suggestions[0].suggestion == "dukkha"


def test_gibberish_word_gets_no_suggestion(tmp_path):
    engine = make_engine(tmp_path)
    suggestions = engine.check_query_terms("xqzwplk")
    assert len(suggestions) == 1
    assert suggestions[0].term == "xqzwplk"
    assert suggestions[0].suggestion is None


def test_short_words_are_not_flagged(tmp_path):
    engine = make_engine(tmp_path)
    # "is" and "the" are short/common and below the default min_word_len,
    # so even though they aren't literally in this tiny fabricated corpus,
    # they must not be flagged as unknown/typo'd.
    suggestions = engine.check_query_terms("is the way")
    assert suggestions == []


def test_multiple_typos_are_each_reported_once(tmp_path):
    engine = make_engine(tmp_path)
    # Note: "jhanas" is deliberately NOT used here -- the existing BM25
    # tokenizer's suffix-stripper (_simple_s_stripper) already stems it to
    # "jhana" before it ever reaches check_query_terms, so it would never
    # be flagged (correctly -- it's already resolved upstream).
    suggestions = engine.check_query_terms("duka duka anata")
    terms = {s.term for s in suggestions}
    assert terms == {"duka", "anata"}
    by_term = {s.term: s.suggestion for s in suggestions}
    assert by_term["duka"] == "dukkha"
    assert by_term["anata"] == "anatta"


def test_mixed_known_and_unknown_words_only_flags_unknown(tmp_path):
    engine = make_engine(tmp_path)
    suggestions = engine.check_query_terms("dukkha and duka")
    assert len(suggestions) == 1
    assert suggestions[0].term == "duka"


def test_case_insensitive_matching(tmp_path):
    engine = make_engine(tmp_path)
    suggestions = engine.check_query_terms("DUKKHA")
    assert suggestions == []


def test_empty_query_returns_no_suggestions(tmp_path):
    engine = make_engine(tmp_path)
    assert engine.check_query_terms("") == []


def test_quoted_phrase_query_is_still_tokenized_and_checked(tmp_path):
    engine = make_engine(tmp_path)
    suggestions = engine.check_query_terms('"duka" meditation')
    terms = {s.term for s in suggestions}
    assert "duka" in terms


# ---------------------------------------------------------------------------
# find_exact_phrase_results()
# ---------------------------------------------------------------------------

def test_exact_phrase_finds_true_contiguous_match(tmp_path):
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("hot dog")
    chunk_ids = {r.chunk_id for r in results}
    assert "v1_0" in chunk_ids, "v1_0 literally contains 'hot dog' twice"


def test_exact_phrase_excludes_bag_of_words_only_match(tmp_path):
    """
    v1_1 ("It was so hot today. My dog wanted...") contains both words
    'hot' and 'dog' -- which would score highly under plain BM25
    bag-of-words term-frequency scoring -- but NOT as a contiguous
    phrase. This is the exact bug this method exists to fix: it must NOT
    be returned as an exact-phrase match.
    """
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("hot dog")
    chunk_ids = {r.chunk_id for r in results}
    assert "v1_1" not in chunk_ids


def test_exact_phrase_avoids_punctuation_boundary_false_positive(tmp_path):
    """
    v1_3 ("Well, that's right, effort is not the same...") contains the
    stemmed tokens 'right' and 'effort' as ADJACENT tokens once
    punctuation is stripped by the tokenizer, but they are not the
    literal phrase "right effort" in the original text (they're split
    across a comma / clause boundary). Only v1_2 (true contiguous phrase)
    should match.
    """
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("right effort")
    chunk_ids = {r.chunk_id for r in results}
    assert "v1_2" in chunk_ids
    assert "v1_3" not in chunk_ids


def test_exact_phrase_matches_compound_word_transliteration(tmp_path):
    """
    The corpus spells the term as one word, "paticcasamuppada" (v1_4),
    while the query is typed with a space, "paticca samuppada" -- this
    must still be found via the merged-single-token candidate path (see
    find_exact_phrase_results()'s docstring, stage 1(b)).
    """
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("paticca samuppada")
    chunk_ids = {r.chunk_id for r in results}
    assert "v1_4" in chunk_ids


def test_exact_phrase_tolerates_trailing_plural(tmp_path):
    """
    Query "breath" should also match "breaths" in v1_5, mirroring
    sanghabot/highlight.py's existing trailing "s"/"es" plural
    tolerance rule.
    """
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("breath")
    chunk_ids = {r.chunk_id for r in results}
    assert "v1_5" in chunk_ids


def test_exact_phrase_returns_empty_list_for_no_match(tmp_path):
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("nonexistent phrase xyz")
    assert results == []


def test_exact_phrase_returns_empty_list_for_empty_phrase(tmp_path):
    engine = make_phrase_engine(tmp_path)
    assert engine.find_exact_phrase_results("") == []
    assert engine.find_exact_phrase_results("   ") == []


def test_exact_phrase_single_word_matches_simple_membership(tmp_path):
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("effort")
    chunk_ids = {r.chunk_id for r in results}
    # Both v1_2 (true "right effort") and v1_3 ("...effort is not...")
    # contain the standalone word "effort" -- a single-word phrase has no
    # adjacency requirement, so both are legitimate matches.
    assert "v1_2" in chunk_ids
    assert "v1_3" in chunk_ids


def test_exact_phrase_results_are_ranked_by_bm25_score_descending(tmp_path):
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("hot dog")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_exact_phrase_results_are_full_search_results_with_text(tmp_path):
    engine = make_phrase_engine(tmp_path)
    results = engine.find_exact_phrase_results("hot dog")
    assert len(results) >= 1
    for r in results:
        assert r.text
        assert r.chunk_id
        assert r.video_id == "v1"
