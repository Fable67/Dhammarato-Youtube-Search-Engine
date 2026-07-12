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
"""
from __future__ import annotations

import re

from sanghabot.models import QueryIntent

QUESTION_WORDS = {
    "how", "what", "why", "when", "where", "who",
    "is", "can", "does", "do", "should", "could", "would",
}


def analyze_query_intent(query: str, stopwords: set[str] | None = None) -> QueryIntent:
    """
    Analyzes a user's query to determine the best search strategy.

    Routing rules (in priority order):
      1. Quoted exact phrase(s) present  -> BM25-heavy (0.85/0.15), semantic
         only added back in if there's meaningful text outside the quotes.
      2. Natural-language question, or >=5 raw words -> semantic-heavy
         (0.8/0.2), both engines run.
      3. Exactly 2 meaningful (non-stopword) words -> BM25-leaning (0.6/0.4).
      4. Exactly 1 meaningful word                 -> BM25-leaning (0.7/0.3).
      5. Otherwise -> default hybrid, both engines run at 0.5/0.5.
    """
    if stopwords is None:
        stopwords = set()

    # 1. Detect exact phrases (e.g., "breath meditation")
    exact_phrases = [m[1].lower() for m in re.findall(r'(["\'])(.*?)\1', query)]

    # 2. Clean query
    clean_query = query.replace('"', " ").strip().lower()
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
        bm25_weight = 0.85
        semantic_weight = 0.15
        # Still run semantic in case the user added context outside the quotes.
        if len(meaningful_words) > sum(len(p.split()) for p in exact_phrases):
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
