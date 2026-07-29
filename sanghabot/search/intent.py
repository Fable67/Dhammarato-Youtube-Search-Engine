"""
Canonical query-intent analysis.

This is the ONE implementation, replacing two near-duplicate,
subtly-diverging copies from the old codebase:
  - AI_Transcripts/keyword_search.py: analyze_query_intent()
  - AI_Transcripts/search.py:         CombinedSearchEngine._analyze_query_intent()

Both old copies contained an unreachable branch:

    elif is_question or len(raw_words) >= 5:      # catches ALL is_question cases
        ...
    elif is_question or len(raw_words) >= 20:      # can never fire: any query
        ...                                         # reaching here already
                                                      # failed `is_question`,
                                                      # and len>=20 implies len>=5

This rewrite keeps the exact same *intent* (long/question queries lean
semantic-heavy) but removes the dead branch outright rather than silently
carrying it forward. Weight tuning per bucket is preserved from search.py's
version (the more fine-grained `== 2` / `== 1` split), since that was the
more deliberate of the two copies.

Behavioral parity with the old `search.py` implementation (not the
`keyword_search.py` copy, which used a coarser `<= 2` bucket) is verified in
tests/test_intent.py and tests/test_parity_old_vs_new.py.

QUOTE-HANDLING BUG FIX (found during a later investigation into quoted-
phrase search quality, see sanghabot/search/phrase.py): the original phrase
regex `r'(["\'])(.*?)\1'` treated ANY single-quote character as a phrase
delimiter, with no awareness of English contractions/possessives. For a
query like `"Why don't I feel satisfaction during 'jhana' practice?"`, the
regex greedily matched the FIRST `'` (inside "don't") as an opening
delimiter and the NEXT `'` (opening "jhana") as the closer, incorrectly
extracting the phrase `"t i feel satisfaction during "` instead of the
user's actual intended phrase, `jhana`. Separately, `clean_query =
query.replace('"', " ")` only ever stripped double quotes, never single
quotes, so `meaningful_words` for a single-quoted query like `'Anapanasati'`
retained the raw apostrophes glued to the word (e.g. `"'anapanasati'?"`
instead of `anapanasati`), which fed directly into
sanghabot/highlight.py's build_highlight_terms() and this module's own
extra-context-word count used for semantic-weight scaling (see
GRADUATED_SEMANTIC_WEIGHT below).

Fixed with a two-part regex, _PHRASE_RE below: double-quoted phrases are
matched unconditionally (`"..."`, unchanged from before -- double quotes are
never legitimately part of an English word, so no extra care is needed
there); single-quoted phrases are matched only when the opening `'` is NOT
immediately preceded by a word character and the closing `'` is NOT
immediately followed by one (a negative lookbehind/lookahead pair), so
"don't", "it's", "y'all", "What's" are correctly left alone as ordinary
words, while `'jhana'`, `'right effort'`, `'Right noble view' vs. 'right
view'` are still correctly detected as quoted phrases. See
tests/test_intent.py for the full regression-test matrix (contractions,
possessives, multiple single-quoted phrases, mixed single+double quotes in
one query).
"""
from __future__ import annotations

import re

from sanghabot.models import QueryIntent

QUESTION_WORDS = {
    "how", "what", "why", "when", "where", "who",
    "is", "can", "does", "do", "should", "could", "would",
}

# Matches a double-quoted phrase (group 1) OR a single-quoted phrase (group
# 2) -- but only treats a `'` as a phrase delimiter when it's NOT adjacent
# to a word character on the "inside" of the quote mark, i.e. not part of a
# contraction ("don't") or possessive ("Buddha's"). See module docstring
# above for the bug this fixes and worked examples.
_PHRASE_RE = re.compile(r'"(.*?)"' + r"|" + r"(?<!\w)'(.+?)'(?!\w)")

# --- Graduated semantic weight for quoted queries with surrounding context
# ---------------------------------------------------------------------
# Previously, ANY quoted-phrase query used a single fixed 0.85/0.15
# bm25/semantic split, regardless of how many extra (non-quoted) words of
# natural-language context surrounded the quote(s) -- a bare `"hot dog"`
# and a full question like `"right effort" and how it fits into the
# eightfold path in daily life` (11 extra words) got IDENTICAL weighting,
# even though the latter clearly has much more contextual meaning for
# semantic search to work with.
#
# Verified directly against the real corpus (see chat history) that
# increasing semantic_weight as extra context grows does NOT push exact
# matches out of the final top-5 -- the exact-phrase RRF tiers (see
# sanghabot/search/phrase.py, weights 1.0/0.5) dominate strongly enough
# that every one of 6 representative queries tested (0 to 12 extra words)
# kept 5/5 exact matches in the top-5 both before and after this change.
# The graduated weight's real effect is REORDERING among exact matches
# (and, for larger extra-word counts, letting a handful of additional
# strong semantic matches surface) -- not excluding/burying true matches.
#
# Ramp: linear from GRADUATED_SEMANTIC_WEIGHT_MIN (0 extra words, same as
# the old fixed 0.15) up to GRADUATED_SEMANTIC_WEIGHT_MAX (reached at
# GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS extra words and beyond -- an
# 8-10 word extra-context count is already a full natural-language
# question's worth of surrounding text, so there's little reason to keep
# ramping past that point).
GRADUATED_SEMANTIC_WEIGHT_MIN = 0.15
GRADUATED_SEMANTIC_WEIGHT_MAX = 0.5
GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS = 8


def _graduated_semantic_weight(extra_context_words: int) -> float:
    """
    Linearly ramps semantic_weight from GRADUATED_SEMANTIC_WEIGHT_MIN (at
    0 extra context words -- a bare quoted phrase, e.g. `"hot dog"`) up to
    GRADUATED_SEMANTIC_WEIGHT_MAX (at GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS
    or more extra words -- a quote embedded in a full question/sentence).
    Clamped so extra_context_words <= 0 or >= the ramp width don't
    over/undershoot the [MIN, MAX] range.
    """
    if extra_context_words <= 0:
        return GRADUATED_SEMANTIC_WEIGHT_MIN
    fraction = min(extra_context_words, GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS) / GRADUATED_SEMANTIC_WEIGHT_RAMP_WORDS
    return GRADUATED_SEMANTIC_WEIGHT_MIN + fraction * (GRADUATED_SEMANTIC_WEIGHT_MAX - GRADUATED_SEMANTIC_WEIGHT_MIN)


def analyze_query_intent(query: str, stopwords: set[str] | None = None) -> QueryIntent:
    """
    Analyzes a user's query to determine the best search strategy.

    Routing rules (in priority order):
      1. Quoted exact phrase(s) present  -> BM25-leaning, with semantic_weight
         graduated between 0.15 (bare quote, no extra context) and 0.5 (quote
         embedded in >=8 extra words of surrounding context) -- see
         _graduated_semantic_weight() above. Semantic is only actually RUN
         (use_semantic=True) if there's meaningful text outside the quotes.
      2. Natural-language question, or >=5 raw words -> semantic-heavy
         (0.8/0.2), both engines run.
      3. Exactly 2 meaningful (non-stopword) words -> BM25-leaning (0.6/0.4).
      4. Exactly 1 meaningful word                 -> BM25-leaning (0.7/0.3).
      5. Otherwise -> default hybrid, both engines run at 0.5/0.5.
    """
    if stopwords is None:
        stopwords = set()

    # 1. Detect exact phrases (e.g., "breath meditation", 'jhana') -- see
    #    _PHRASE_RE's definition above and the module docstring for why
    #    single-quote detection needs the contraction/possessive guard.
    exact_phrases = [(g1 or g2).lower() for g1, g2 in _PHRASE_RE.findall(query)]

    # 2. Clean query: strip BOTH the double quotes and the specific single
    #    quote characters that were consumed as phrase delimiters above
    #    (via the same regex, substituting the phrase text back in without
    #    its quote marks) -- this is what fixes meaningful_words retaining
    #    stray apostrophes like "'anapanasati'?" instead of "anapanasati".
    #    Note this leaves ordinary apostrophes (don't, it's, Buddha's)
    #    completely untouched, since _PHRASE_RE never matched those in the
    #    first place.
    dequoted_query = _PHRASE_RE.sub(lambda m: f" {m.group(1) or m.group(2)} ", query)
    clean_query = dequoted_query.replace('"', " ").strip().lower()
    raw_words = clean_query.split()
    meaningful_words = [w for w in raw_words if w not in stopwords]

    # 3. Detect if it's a natural language question
    is_question = any(w in QUESTION_WORDS for w in raw_words) or "?" in query

    # 4. Routing logic
    use_semantic = False
    use_bm25 = False
    semantic_weight = 0.5
    bm25_weight = 0.5

    if exact_phrases:
        use_bm25 = True
        extra_context_words = len(meaningful_words) - sum(len(p.split()) for p in exact_phrases)
        semantic_weight = _graduated_semantic_weight(extra_context_words)
        bm25_weight = 1.0 - semantic_weight
        # Still run semantic in case the user added context outside the quotes.
        if extra_context_words > 0:
            use_semantic = True

    elif is_question or len(raw_words) >= 5:
        use_semantic = True
        use_bm25 = True
        semantic_weight = 0.8
        bm25_weight = 0.2

    elif len(meaningful_words) == 2:
        use_bm25 = True
        use_semantic = True
        bm25_weight = 0.6
        semantic_weight = 0.4

    elif len(meaningful_words) == 1:
        use_bm25 = True
        use_semantic = True
        bm25_weight = 0.7
        semantic_weight = 0.3

    else:
        use_bm25 = True
        use_semantic = True

    return QueryIntent(
        use_semantic=use_semantic,
        use_bm25=use_bm25,
        weights={"semantic": semantic_weight, "bm25": bm25_weight},
        exact_phrases=exact_phrases,
        meaningful_words=meaningful_words,
        is_question=is_question,
    )
