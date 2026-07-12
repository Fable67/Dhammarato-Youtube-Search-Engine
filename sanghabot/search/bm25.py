"""
BM25Plus-backed lexical search engine.

Same load-once discipline as semantic.py: the BM25 index is built/loaded
exactly ONCE, in __init__. search() never touches disk. This fixes
AI_Transcripts/keyword_search.py, which called `pd.read_csv(self.metadata_path)`
and `self._load_or_build_index(metadata)` *inside* `search()` on every call.

NOTE on tokenization/stemming (intentional, temporary parity decision):
the old codebase's `simple_s_stripper()` was a 30-line undocumented
heuristic suffix-stripper with a hardcoded exception list, reimplementing
(worse) what a real stemming library already does. It would normally be
the first thing to delete in a rewrite. It is intentionally KEPT HERE,
verbatim, for one reason only: REWRITE_PLAN.md's parity-testing
gate (Section 9a) requires this engine's BM25 rankings to match the old
engine's rankings against the same migrated data, and BM25 rankings are
sensitive to exact tokenization. Swapping in a better stemmer (e.g. a real
Snowball/Porter implementation) is a good, separate, tracked follow-up --
but it must happen AFTER parity is proven with the old behavior preserved,
never silently bundled into this rewrite. See tests/golden/KNOWN_DIVERGENCES.md.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Plus
from rapidfuzz import fuzz, process

from sanghabot.models import SearchResult, TermSuggestion
from sanghabot.storage.db import Database

_TOKEN_RE = re.compile(r"\b\w+\b")


def _simple_s_stripper(word: str) -> str:
    """
    Verbatim port of AI_Transcripts/keyword_search.py's simple_s_stripper.
    Kept as-is for BM25 ranking parity -- see module docstring above.
    """
    word = word.lower()
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        if not (word.endswith("eies") or word.endswith("aies")):
            return word[:-3] + "y"
    elif word.endswith("es"):
        if not (word.endswith("aes") or word.endswith("ees") or word.endswith("oes")):
            return word[:-1]
    elif word.endswith("s"):
        if not (word.endswith("us") or word.endswith("ss")):
            return word[:-1]
    return word


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    """
    Matches AI_Transcripts/keyword_search.py's _tokenize() exactly: the old
    code applies simple_s_stripper to every token but does NOT actually
    filter stopwords at tokenize time (that line is commented out in the
    old code -- `# return [t for t in tokens if len(t) > 2 and t not in
    self.stopwords]`). We reproduce that (surprising, but real) behavior
    for parity: `stopwords` is accepted for API symmetry with
    analyze_query_intent(), but is intentionally NOT applied here, matching
    old production behavior. See tests/golden/KNOWN_DIVERGENCES.md.
    """
    if not isinstance(text, str):
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [_simple_s_stripper(t) for t in tokens]


class BM25SearchEngine:
    def __init__(
        self,
        db: Database,
        index_cache_path: Path | str,
        stopwords: set[str] | None = None,
    ):
        self._db = db
        self.index_cache_path = Path(index_cache_path)
        self.stopwords = stopwords or set()

        # chunk_ids in the exact order they were fed into BM25Plus, so a
        # score-array index can be mapped back to a chunk_id.
        self._chunk_ids: list[str]
        self._bm25: BM25Plus
        self._bm25, self._chunk_ids = self._load_or_build_index()

        # Cached once, at load time (not per-query): the full set of
        # stemmed tokens BM25 actually knows about, and the same set as a
        # list for rapidfuzz.process (which needs a sequence, not a set).
        # Used by check_query_terms() below for typo/OOV detection only --
        # never touches actual search()/get_scores() ranking behavior.
        self._vocab: set[str] = set(self._bm25.idf.keys())
        self._vocab_list: list[str] = list(self._vocab)

    def _load_or_build_index(self) -> tuple[BM25Plus, list[str]]:
        if self.index_cache_path.exists():
            with open(self.index_cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["bm25"], cached["chunk_ids"]

        chunk_ids: list[str] = []
        tokenized_corpus: list[list[str]] = []
        for chunk_id, text in self._db.iter_chunk_texts():
            chunk_ids.append(chunk_id)
            tokenized_corpus.append(tokenize(text, self.stopwords))

        bm25 = BM25Plus(tokenized_corpus)

        self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_cache_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

        return bm25, chunk_ids

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        tokenized_query = tokenize(query, self.stopwords)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:k]

        chunk_ids_in_order = []
        scores_in_order = []
        for idx in top_n_indices:
            score = scores[idx]
            if score <= 0:
                continue
            chunk_ids_in_order.append(self._chunk_ids[idx])
            scores_in_order.append(float(score))

        if not chunk_ids_in_order:
            return []

        chunks = self._db.get_chunks_by_ids(chunk_ids_in_order)
        chunks_by_id = {c.chunk_id: c for c in chunks}

        video_ids = list({c.video_id for c in chunks})
        videos_by_id = self._db.get_videos_by_ids(video_ids)

        results: list[SearchResult] = []
        for chunk_id, score in zip(chunk_ids_in_order, scores_in_order):
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            video = videos_by_id.get(chunk.video_id)
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    video_id=chunk.video_id,
                    title=video.title if video else "",
                    blog_url=video.blog_url if video else "",
                    text=chunk.text,
                    summary=chunk.summary,
                    score=score,
                    percentage_score=0.0,
                    source="bm25",
                    url=video.url if video else None,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    video_length_chars=video.video_length_chars if video else None,
                )
            )
        return results

    def check_query_terms(
        self,
        query: str,
        min_word_len: int = 4,
        score_cutoff: float = 80.0,
    ) -> list[TermSuggestion]:
        """
        Flags query words not present (after stemming) in the BM25 corpus
        vocabulary, and, where possible, attaches a fuzzy "did you mean"
        suggestion pulled from that same vocabulary (e.g. "duka" -> "dukkha",
        "jhanas" -> "jhana").

        This is purely informational: it never changes what search()
        actually searches for, and never blocks a search from running --
        see sanghabot/bot/discord_bot.py's perform_search(), which surfaces
        the result of this method as a temporary, auto-deleting Discord
        notice alongside the (unmodified) real search results.

        Design notes:
          - Uses the exact same tokenize() used to build the BM25 index and
            to tokenize queries at search time, so "found" here means
            precisely "found by the real matching logic," not some looser
            secondary definition.
          - min_word_len=4 filters out short function words (the, of, is,
            ...) that would otherwise constantly false-positive; it also
            means a small number of very short nonsense inputs (e.g. "zapx")
            may occasionally surface a low-value suggestion -- acceptable
            since this is cosmetic and never affects actual results.
          - Deduplicates repeated words in the query (each unique unknown
            term is reported once).
          - score_cutoff=80 (rapidfuzz.fuzz.ratio, 0-100 scale) was picked
            by direct measurement against this project's real corpus
            vocabulary: genuine typos on domain terms (e.g. "duka"->"dukkha"
            scores exactly 80.0, "sammma"->"samma" and "anata"->"anatta"
            score ~91, "meditaton"->"meditation" ~95) all clear this bar,
            while true gibberish (e.g. "xqzwplk", "blorpzenit") mostly finds
            no match at all at this cutoff. A handful of short (4-5 char)
            nonsense inputs may occasionally surface a low-value match
            (e.g. "zapx"->"zap" at ~86) -- acceptable since this only
            affects notice text, never actual search results.
        """
        tokens = tokenize(query, self.stopwords)

        seen: set[str] = set()
        suggestions: list[TermSuggestion] = []
        for token in tokens:
            if len(token) < min_word_len or token in seen:
                continue
            seen.add(token)
            if token in self._vocab:
                continue

            match = process.extractOne(
                token, self._vocab_list, scorer=fuzz.ratio, score_cutoff=score_cutoff
            )
            suggestion = match[0] if match else None
            suggestions.append(TermSuggestion(term=token, suggestion=suggestion))

        return suggestions
