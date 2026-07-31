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
    list.
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
