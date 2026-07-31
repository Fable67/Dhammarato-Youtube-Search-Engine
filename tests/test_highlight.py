"""
Tests for sanghabot.highlight -- transcript search-term bolding.

Covers:
  - should_highlight()'s routing rule (must stay the exact logical inverse
    of analyze_query_intent()'s own semantic-heavy condition).
  - build_highlight_terms()'s dedup of exact_phrases + meaningful_words.
  - highlight_text()'s case-insensitivity, word-boundary safety, plural
    tolerance, stray-asterisk escaping, and empty-terms passthrough.
"""
from sanghabot.highlight import build_highlight_terms, highlight_text, should_highlight
from sanghabot.models import QueryIntent
from sanghabot.search.intent import analyze_query_intent


def test_should_highlight_true_for_single_keyword():
    intent = analyze_query_intent("dukkha")
    assert should_highlight(intent, "dukkha") is True


def test_should_highlight_true_for_two_word_query():
    intent = analyze_query_intent("right effort")
    assert should_highlight(intent, "right effort") is True


def test_should_highlight_true_for_exact_phrase():
    intent = analyze_query_intent('"right effort"')
    assert should_highlight(intent, '"right effort"') is True


def test_should_highlight_true_for_three_to_four_word_non_question():
    intent = analyze_query_intent("western secular buddhism")
    assert should_highlight(intent, "western secular buddhism") is True


def test_should_highlight_false_for_question():
    query = "What is the capital of dukkha?"
    intent = analyze_query_intent(query)
    assert intent.is_question is True
    assert should_highlight(intent, query) is False


def test_should_highlight_false_for_long_non_question_query():
    query = " ".join(["word"] * 6)  # 6 raw words, not a question
    intent = analyze_query_intent(query)
    assert intent.is_question is False
    assert should_highlight(intent, query) is False


def test_should_highlight_matches_intent_routing_exactly():
    """
    should_highlight() must mirror analyze_query_intent()'s own routing
    priority order (intent.py:168-181): exact_phrases wins outright first;
    only otherwise does is_question/len(raw_words) >= 5 apply.
    """
    queries = [
        "dukkha",
        "right effort",
        '"right effort"',
        "blue green red",
        "western secular buddhism",
        "What is the capital of dukkha?",
        "how can we tell if something is wholesome",
        "define free will and its limits today",
    ]
    for q in queries:
        intent = analyze_query_intent(q)
        raw_words = q.replace('"', " ").split()
        if intent.exact_phrases:
            expected = True
        else:
            expected = not (intent.is_question or len(raw_words) >= 5)
        assert should_highlight(intent, q) == expected, q


def test_should_highlight_true_for_quoted_phrase_inside_a_question():
    """
    Regression test for the reported bug: a quoted single-word keyword
    embedded in a natural-language question must still be highlighted,
    even though the query is a question and has >= 5 raw words.
    """
    query = "What is the difference between 'sukha' and 'piti'"
    intent = analyze_query_intent(query)
    assert intent.exact_phrases == ["sukha", "piti"]
    assert intent.is_question is True
    assert should_highlight(intent, query) is True


def test_should_highlight_true_for_quoted_phrase_in_long_non_question_query():
    query = "please explain 'sukha' and 'piti' further clearly now"
    intent = analyze_query_intent(query)
    assert intent.exact_phrases == ["sukha", "piti"]
    assert should_highlight(intent, query) is True


def test_should_highlight_true_for_double_quoted_phrase_inside_a_question():
    query = '"right effort" and how it fits into the eightfold path in daily life'
    intent = analyze_query_intent(query)
    assert intent.exact_phrases == ["right effort"]
    assert intent.is_question is True
    assert should_highlight(intent, query) is True


def test_build_highlight_terms_combines_exact_phrases_and_meaningful_words():
    intent = QueryIntent(
        use_semantic=True, use_bm25=True, weights={"semantic": 0.4, "bm25": 0.6},
        exact_phrases=["right effort"], meaningful_words=["right", "effort"],
    )
    terms = build_highlight_terms(intent)
    assert terms == ["right effort", "right", "effort"]


def test_build_highlight_terms_dedupes():
    intent = QueryIntent(
        use_semantic=True, use_bm25=True, weights={"semantic": 0.5, "bm25": 0.5},
        exact_phrases=["dukkha"], meaningful_words=["dukkha"],
    )
    terms = build_highlight_terms(intent)
    assert terms == ["dukkha"]


def test_build_highlight_terms_empty_when_intent_has_no_terms():
    intent = QueryIntent(
        use_semantic=True, use_bm25=False, weights={"semantic": 1.0, "bm25": 0.0},
    )
    assert build_highlight_terms(intent) == []


def test_highlight_text_bolds_case_insensitive_match():
    result = highlight_text("The Buddha taught about Dukkha and suffering.", ["dukkha"])
    assert result == "The Buddha taught about **Dukkha** and suffering."


def test_highlight_text_word_boundary_safe():
    # "cat" must not match inside "category".
    result = highlight_text("This category is unrelated to cats or cat.", ["cat"])
    assert "**category**" not in result
    assert "category" in result
    assert "**cat**" in result
    assert "**cats**" in result  # plural tolerance


def test_highlight_text_plural_tolerance():
    result = highlight_text("Breath and breaths and breathing.", ["breath"])
    assert "**Breath**" in result
    assert "**breaths**" in result
    # "breathing" should NOT be highlighted -- not a simple s/es plural.
    assert "**breathing**" not in result
    assert "breathing" in result


def test_highlight_text_multi_word_phrase():
    result = highlight_text("He spoke about right effort in daily life.", ["right effort"])
    assert "**right effort**" in result


def test_highlight_text_multiple_terms():
    result = highlight_text("Dukkha and jhana were both discussed.", ["dukkha", "jhana"])
    assert "**Dukkha**" in result
    assert "**jhana**" in result


def test_highlight_text_empty_terms_returns_text_unchanged_but_escaped():
    text = "Plain text with no markers."
    assert highlight_text(text, []) == text


def test_highlight_text_escapes_preexisting_asterisks():
    text = "The show M*A*S*H was mentioned."
    result = highlight_text(text, [])
    assert "*" not in result.replace("\\*", "")  # every remaining '*' must be escaped
    assert result == "The show M\\*A\\*S\\*H was mentioned."


def test_highlight_text_escapes_asterisks_even_with_terms_present():
    text = "M*A*S*H and dukkha both mentioned."
    result = highlight_text(text, ["dukkha"])
    assert "M\\*A\\*S\\*H" in result
    assert "**dukkha**" in result


def test_highlight_text_no_match_leaves_text_effectively_unchanged():
    result = highlight_text("Nothing relevant here.", ["dukkha"])
    assert result == "Nothing relevant here."


def test_highlight_text_longest_term_preferred_over_substring_word():
    # "right" alone should not fragment the "right effort" phrase match.
    result = highlight_text("Practicing right effort daily.", ["right effort", "right"])
    assert "**right effort**" in result
    assert result.count("**") == 2  # exactly one bolded span, not two
