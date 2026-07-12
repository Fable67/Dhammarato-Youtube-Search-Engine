"""
Tests for sanghabot.search.bm25, focused on check_query_terms() -- the
typo/OOV keyword detection used to power a transparent, non-blocking
Discord notice (see tests/test_discord_bot_formatting.py and
sanghabot/bot/discord_bot.py's perform_search()).

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
