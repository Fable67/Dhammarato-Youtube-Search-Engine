"""
Transcript highlighting: bolds the user's searched keyword(s) in the
transcript text posted to Discord, so someone scanning a long passage can
immediately spot why a given chunk matched their search.

Design (see chat history / investigation for full rationale):
  - Applied whenever the query has explicit keywords to highlight: either
    quoted exact phrase(s) (regardless of question-ness or length -- a
    quoted phrase is always an explicit, literal keyword the user asked
    for), or a short, non-question query under 5 words. This mirrors
    analyze_query_intent()'s own routing priority order
    (sanghabot/search/intent.py:168-181), where `exact_phrases` is checked
    FIRST and wins outright, before the is_question/length check is even
    considered. Long/question queries with NO quoted phrases are answered
    by semantic search, which matches on meaning rather than literal words,
    so bolding a handful of incidental word overlaps there would be noisy
    and potentially misleading rather than helpful. See should_highlight().
  - Terms highlighted are exactly analyze_query_intent()'s own
    exact_phrases + meaningful_words -- i.e. the same terms the routing
    logic already decided were "the keywords," not a separately invented
    list. The caller (sanghabot/bot/discord_bot.py) passes
    HIGHLIGHT_STOPWORDS (below) into that analyze_query_intent() call so
    common English function words (the, is, and, between, ...) don't get
    swept into meaningful_words and bolded alongside the user's actual
    keywords -- e.g. without this, `'sukha' and 'piti'` would highlight
    the literal word "and" in transcript text, and a question like `What
    is the difference between 'sukha' and 'piti'` would highlight "what",
    "is", "the", "difference", "between", and "and" in addition to the
    two real keywords. This is purely a *display* filter: it is NEVER
    passed to engine.search()'s own internal analyze_query_intent() call
    (see sanghabot/engine.py), so it has zero effect on search ranking or
    routing -- BM25SearchEngine's own `stopwords` param is intentionally
    left unapplied for parity with the old engine (see bm25.py's module
    docstring and tests/golden/KNOWN_DIVERGENCES.md #3); this is a wholly
    separate, highlight-only list.
  - Matching is case-insensitive, word-boundary-safe, and tolerant of a
    simple trailing "s"/"es" plural (e.g. searching "breath" also
    highlights "breaths"). This is a deliberately simple, predictable rule
    -- not a reversal of BM25's lossy suffix-stripper (see bm25.py's
    _simple_s_stripper docstring for why that reversal is ambiguous by
    design and not worth attempting here).
  - Any literal "*" already present in raw chunk text (a rare but real
    case -- e.g. a transcript mentioning "M*A*S*H", or a stray "**" that
    survived migration) is escaped to "\\*" before inserting our own
    "**term**" markers, so we never accidentally clobber/collide with
    pre-existing characters or produce unbalanced markdown.

This module is intentionally pure (no I/O, no Discord/DB dependencies) so
it's fully unit-testable in isolation -- see tests/test_highlight.py.
"""
from __future__ import annotations

import re

from sanghabot.models import QueryIntent

# Common English function/filler words that carry no keyword meaning on
# their own and should never be bolded in transcript text, even when they
# survive into analyze_query_intent()'s meaningful_words (that function has
# no stopword list applied by default -- see build_highlight_terms() and
# the module docstring above for why the caller passes this set in).
# Deliberately conservative/short: only words with essentially zero
# standalone search value in this domain (articles, common prepositions,
# conjunctions, auxiliary/copula verbs, question words, basic pronouns).
# Content words that happen to be short (e.g. "self", "mind", "path") are
# intentionally NOT included.
HIGHLIGHT_STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but", "nor", "so", "yet",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "as",
    "how", "what", "why", "when", "where", "who", "whom", "which",
    "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
}

# Matches analyze_query_intent()'s own semantic-heavy routing condition
# (sanghabot/search/intent.py:177: `elif is_question or len(raw_words) >= 5`)
# -- but, like that function, this is only reached/consulted when there are
# no exact_phrases; see should_highlight() below.
_LONG_QUERY_WORD_COUNT = 5


def should_highlight(intent: QueryIntent, query: str) -> bool:
    """
    True whenever there are explicit keywords to highlight: quoted exact
    phrase(s) (always -- regardless of question-ness or query length, since
    a quoted term is an explicit, literal keyword the user asked for), or a
    short, non-question query under 5 raw words. False for questions/long
    queries with NO quoted phrases, where literal word-overlap highlighting
    would be noisy rather than helpful (see module docstring).

    Mirrors analyze_query_intent()'s own priority order (intent.py:168-181):
    `exact_phrases` is checked first and wins outright; the is_question/
    length check is only consulted as a fallback when there are no exact
    phrases.
    """
    if intent.exact_phrases:
        return True
    raw_words = query.replace('"', " ").replace("'", " ").split()
    return not (intent.is_question or len(raw_words) >= _LONG_QUERY_WORD_COUNT)


def build_highlight_terms(intent: QueryIntent) -> list[str]:
    """
    Returns the deduplicated list of terms to highlight: the query's exact
    (quoted) phrases plus its meaningful (non-stopword) words, exactly as
    analyze_query_intent() already computed them -- no separate term
    extraction logic. Order-preserving, case as originally lowercased by
    analyze_query_intent().
    """
    seen: set[str] = set()
    terms: list[str] = []
    for term in (*intent.exact_phrases, *intent.meaningful_words):
        term = term.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _build_pattern(terms: list[str]) -> re.Pattern:
    # Longest-first so a multi-word exact phrase (e.g. "right effort") is
    # tried before its individual constituent words, avoiding a phrase
    # being partially/incorrectly highlighted word-by-word instead of as
    # one contiguous span.
    ordered = sorted(terms, key=len, reverse=True)
    alternation = "|".join(re.escape(t) for t in ordered)
    # Optional trailing "s"/"es" plural tolerance (see module docstring).
    return re.compile(rf"\b(?:{alternation})(?:es|s)?\b", re.IGNORECASE)


def highlight_text(text: str, terms: list[str]) -> str:
    """
    Returns `text` with every case-insensitive, word-boundary-safe match of
    any term in `terms` (plus a tolerated trailing "s"/"es") wrapped in
    Discord bold markers ("**match**").

    Always escapes any pre-existing literal "*" in `text` first (turning it
    into the literal, non-magic "\\*"), regardless of whether `terms` is
    empty -- so callers can unconditionally route chunk text through this
    function and get safe-to-render output either way.
    """
    safe_text = text.replace("*", "\\*")
    if not terms:
        return safe_text

    pattern = _build_pattern(terms)
    return pattern.sub(lambda m: f"**{m.group(0)}**", safe_text)
