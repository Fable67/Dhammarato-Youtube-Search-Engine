"""
Tests for sanghabot.search.intent.analyze_query_intent().

Includes an explicit regression test for the unreachable elif branch found
in both old-code copies (keyword_search.py / search.py):

    elif is_question or len(raw_words) >= 5:      # catches ALL is_question
        ...
    elif is_question or len(raw_words) >= 20:      # unreachable
        ...

We assert that a long (>=20 word), non-question query is routed the same
way as any other >=5-word query (semantic-heavy, both engines run) --
proving there's no separate/dead "only semantic" code path silently
lurking that could diverge from tested behavior.

Also covers two later fixes to quoted-phrase handling (see
sanghabot/search/intent.py's module docstring for the full rationale):

  1. Graduated semantic weight: a quoted-phrase query's semantic_weight now
     scales from 0.15 (a bare quote, no extra context) up to 0.5 (>=8 extra
     context words), instead of a single fixed 0.85/0.15 split regardless
     of how much natural-language context surrounds the quote(s).
  2. Contraction/possessive-safe single-quote detection: a single `'`
     inside a word (don't, it's, y'all, Buddha's) must NOT be treated as a
     phrase delimiter, while genuine single-quoted phrases ('jhana',
     'right effort') still must be.
"""
from sanghabot.search.intent import (
    GRADUATED_SEMANTIC_WEIGHT_MAX,
    GRADUATED_SEMANTIC_WEIGHT_MIN,
    GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS,
    analyze_query_intent,
)


def test_bare_quoted_phrase_routes_bm25_heavy_at_minimum_semantic_weight():
    intent = analyze_query_intent('"right effort"')
    assert intent.use_bm25 is True
    assert intent.weights["bm25"] == 0.85
    assert intent.weights["semantic"] == GRADUATED_SEMANTIC_WEIGHT_MIN
    assert intent.exact_phrases == ["right effort"]
    # A bare quote (no extra context) does not need semantic search run at
    # all -- this is unchanged from before the graduated-weight fix.
    assert intent.use_semantic is False


def test_quoted_phrase_with_extra_context_also_uses_semantic():
    intent = analyze_query_intent('"right effort" and how it relates to jhana practice')
    assert intent.use_bm25 is True
    assert intent.use_semantic is True


def test_question_routes_semantic_heavy():
    intent = analyze_query_intent("What is the capital of dukkha?")
    assert intent.is_question is True
    assert intent.use_semantic is True
    assert intent.use_bm25 is True
    assert intent.weights["semantic"] == 0.8
    assert intent.weights["bm25"] == 0.2


def test_long_query_routes_same_as_five_plus_words_no_dead_branch():
    long_query = " ".join(["word"] * 25)  # 25 raw words, not a question
    intent = analyze_query_intent(long_query)
    assert intent.is_question is False
    assert len(long_query.split()) >= 20
    # Must land in the >=5-words bucket (semantic-heavy), since the old
    # >=20-words-only bucket was unreachable dead code and is intentionally
    # not reproduced here.
    assert intent.use_semantic is True
    assert intent.use_bm25 is True
    assert intent.weights["semantic"] == 0.8
    assert intent.weights["bm25"] == 0.2


def test_single_meaningful_word_routes_bm25_leaning():
    intent = analyze_query_intent("dukkha")
    assert intent.use_bm25 is True
    assert intent.use_semantic is True
    assert intent.weights["bm25"] == 0.7
    assert intent.weights["semantic"] == 0.3


def test_two_meaningful_words_routes_bm25_leaning():
    intent = analyze_query_intent("right effort")
    assert intent.use_bm25 is True
    assert intent.use_semantic is True
    assert intent.weights["bm25"] == 0.6
    assert intent.weights["semantic"] == 0.4


def test_default_hybrid_for_three_to_four_word_non_question():
    intent = analyze_query_intent("blue green red")
    assert intent.use_bm25 is True
    assert intent.use_semantic is True
    assert intent.weights == {"semantic": 0.5, "bm25": 0.5}


def test_stopwords_reduce_meaningful_word_count():
    intent = analyze_query_intent("the dukkha", stopwords={"the"})
    assert intent.meaningful_words == ["dukkha"]
    assert intent.weights["bm25"] == 0.7


def test_empty_query_does_not_crash():
    intent = analyze_query_intent("")
    assert intent.exact_phrases == []
    assert intent.meaningful_words == []


# ---------------------------------------------------------------------------
# Graduated semantic weight for quoted-phrase queries
# ---------------------------------------------------------------------------

def test_graduated_weight_ramps_up_with_extra_context_word_count():
    """
    semantic_weight must strictly increase as more non-quoted context words
    surround the same quoted phrase, from the minimum (bare quote) up to
    the maximum (>= GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS extra words).
    """
    queries_by_extra_word_count = [
        ('"right effort"', 0),
        ('"right effort" meditation', 1),
        ('"right effort" and meditation practice', 3),
        ('"right effort" and how it fits into the eightfold path in daily life', 11),
    ]
    prev_weight = None
    for query, expected_extra in queries_by_extra_word_count:
        intent = analyze_query_intent(query)
        extra = len(intent.meaningful_words) - sum(len(p.split()) for p in intent.exact_phrases)
        assert extra == expected_extra, f"{query!r}: expected {expected_extra} extra words, got {extra}"
        weight = intent.weights["semantic"]
        if prev_weight is not None:
            assert weight > prev_weight, f"{query!r}: semantic weight did not increase ({prev_weight} -> {weight})"
        prev_weight = weight


def test_graduated_weight_caps_at_maximum_for_long_context():
    intent = analyze_query_intent(
        '"right effort" and how it fits into the eightfold path in daily life'
    )
    assert intent.weights["semantic"] == GRADUATED_SEMANTIC_WEIGHT_MAX
    assert intent.weights["bm25"] == 1.0 - GRADUATED_SEMANTIC_WEIGHT_MAX


def test_graduated_weight_does_not_exceed_maximum_beyond_ramp_words():
    """
    Once extra context reaches GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS, adding
    even more context words must not push semantic_weight past the max.
    """
    long_query = '"jhana" ' + " ".join(["context"] * (GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS + 20))
    intent = analyze_query_intent(long_query)
    assert intent.weights["semantic"] == GRADUATED_SEMANTIC_WEIGHT_MAX


def test_graduated_weight_bm25_and_semantic_always_sum_to_one():
    queries = [
        '"hot dog"',
        '"right effort" meditation',
        '"right effort" and how it fits into the eightfold path in daily life',
        'What is "paticca samuppada" and how does it relate to suffering in daily life?',
    ]
    for q in queries:
        intent = analyze_query_intent(q)
        assert abs(intent.weights["semantic"] + intent.weights["bm25"] - 1.0) < 1e-9, q


def test_multiple_quoted_phrases_sum_their_word_counts_for_extra_context():
    intent = analyze_query_intent("'Right noble view' vs. 'right view' vs. 'wrong view'")
    assert intent.exact_phrases == ["right noble view", "right view", "wrong view"]
    # meaningful_words: right, noble, view, vs., right, view, vs., wrong, view = 9
    # exact_phrases word counts: 3 + 2 + 2 = 7 -> extra = 2 ("vs." x2)
    extra = len(intent.meaningful_words) - sum(len(p.split()) for p in intent.exact_phrases)
    assert extra == 2
    assert GRADUATED_SEMANTIC_WEIGHT_MIN < intent.weights["semantic"] < GRADUATED_SEMANTIC_WEIGHT_MAX


# ---------------------------------------------------------------------------
# Single-quote / contraction / possessive handling bug fix
# ---------------------------------------------------------------------------

def test_single_quoted_phrase_is_detected():
    intent = analyze_query_intent("How to practice with 'Anapanasati'?")
    assert intent.exact_phrases == ["anapanasati"]


def test_contraction_dont_is_not_mistaken_for_a_phrase_delimiter():
    intent = analyze_query_intent("Why don't I feel satisfaction during 'jhana' practice?")
    assert intent.exact_phrases == ["jhana"]
    assert "don't" in intent.meaningful_words


def test_contraction_its_is_not_mistaken_for_a_phrase_delimiter():
    intent = analyze_query_intent("What's the difference between 'jhana' and 'samadhi'?")
    assert intent.exact_phrases == ["jhana", "samadhi"]
    assert "what's" in intent.meaningful_words


def test_possessive_is_not_mistaken_for_a_phrase_delimiter():
    intent = analyze_query_intent("What is the Buddha's teaching on 'anatta'?")
    assert intent.exact_phrases == ["anatta"]
    assert "buddha's" in intent.meaningful_words


def test_leading_contraction_like_yall_is_not_mistaken_for_a_phrase_delimiter():
    intent = analyze_query_intent("y'all should know 'dukkha'")
    assert intent.exact_phrases == ["dukkha"]
    assert "y'all" in intent.meaningful_words


def test_meaningful_words_do_not_retain_stray_single_quote_characters():
    """
    Direct regression test for the bug: meaningful_words used to contain
    "'anapanasati'?" (quotes and punctuation still attached) instead of
    the clean word "anapanasati".
    """
    intent = analyze_query_intent("How to practice the eightfold noble path with 'Anapanasati'?")
    assert "anapanasati" not in [w for w in intent.meaningful_words if "'" in w]
    for word in intent.meaningful_words:
        assert "'" not in word or word in {"don't", "it's", "y'all", "what's", "buddha's"} or "'" not in word.strip("'")


def test_multiple_single_quoted_phrases_all_detected():
    intent = analyze_query_intent("'Right noble view' vs. 'right view' vs. 'wrong view'")
    assert intent.exact_phrases == ["right noble view", "right view", "wrong view"]


def test_mixed_single_and_double_quotes_both_detected():
    intent = analyze_query_intent('"right effort" and \'wrong effort\'')
    assert set(intent.exact_phrases) == {"right effort", "wrong effort"}


def test_single_quote_regression_meaningful_words_are_clean():
    """
    Ensures no stray apostrophe/quote characters leak into meaningful_words
    for a variety of single-quoted queries -- this directly affects
    sanghabot/highlight.py's build_highlight_terms() output quality.
    """
    intent = analyze_query_intent("How to practice the eightfold noble path with 'Anapanasati'?")
    assert intent.meaningful_words == [
        "how", "to", "practice", "the", "eightfold", "noble", "path", "with", "anapanasati", "?",
    ]
