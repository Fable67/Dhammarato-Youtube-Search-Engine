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
"""
from sanghabot.search.intent import analyze_query_intent


def test_quoted_phrase_routes_bm25_heavy():
    intent = analyze_query_intent('"right effort"')
    assert intent.use_bm25 is True
    assert intent.weights["bm25"] == 0.85
    assert intent.weights["semantic"] == 0.15
    assert intent.exact_phrases == ["right effort"]


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
