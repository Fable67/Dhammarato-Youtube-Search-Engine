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

    def find_exact_phrase_results(self, phrase: str) -> list[SearchResult]:
        """
        Returns full SearchResult objects for every chunk that contains
        `phrase` as a true exact phrase, ranked by BM25 term-frequency
        score (descending). Used to power exact-phrase-match boosting for
        quoted queries (see sanghabot/search/phrase.py and
        sanghabot/search/fusion.py) -- this replaces the old, buggy
        exclusionary substring filter that lived in fusion.py, which (a)
        could zero out an entire result set when the literal phrase
        didn't appear verbatim anywhere (e.g. the corpus spells "paticca
        samuppada" as one word, "paticcasamuppada", so the two-word
        quoted form matched nothing), and (b) was never even invoked for
        pure bm25-only queries due to a branch-coverage bug in engine.py.

        Deliberately returns full SearchResult objects, not just
        chunk_ids: verified directly against the real corpus that most
        exact-phrase matches for a common two-word phrase (e.g. "hot
        dog", "right effort") rank far outside the semantic/BM25 engines'
        own top-100-per-engine candidate window (up to 80% of true phrase
        matches, for some phrases, never appear in either engine's own
        results at all, since bag-of-words term frequency and semantic
        similarity are not the same signal as literal phrase adjacency).
        If this method only returned chunk_ids, reciprocal_rank_fusion()
        would have no SearchResult to attach a fused score to for any
        chunk_id outside that window, silently losing exactly the
        recall-gap cases this feature exists to fix.

        Design, in two stages so this stays fast on a 23k-chunk corpus
        (a naive full-corpus regex scan measured ~4.5s/query -- too slow
        for interactive search):

          Stage 1 (candidate narrowing, ~20-70ms): reuse the BM25Plus
          index's already-in-memory `doc_freqs` (per-chunk stemmed-token
          frequency dicts, built once at index-load time, no new index
          file needed) to cheaply find every chunk that *could* contain
          the phrase. A chunk is a candidate if EITHER:
            (a) all of the phrase's stemmed tokens are present somewhere
                in that chunk (bag-of-words AND) -- covers ordinary
                multi-word phrases like "hot dog", "right effort"; or
            (b) the phrase's stemmed tokens, concatenated with no
                separator, appear as a single stemmed token in that
                chunk -- covers compound-word transliterations like
                "paticca samuppada" / "paticcasamuppada". Verified
                directly: 1202 chunks contain "paticcasamuppada" as one
                word, only 2 contain "paticca" and "samuppada" both
                present as separate words, and these two candidate sets
                are almost entirely disjoint.
          This stage only ever narrows candidates -- it never itself
          decides a "true" phrase match (that would produce false
          positives on documents where the words merely co-occur
          unrelatedly), it just avoids doing stage 2's more expensive
          check against the whole corpus.

          Stage 2 (verification, candidates only): fetch just the
          candidate chunks' raw (unstemmed, un-tokenized) text from the
          DB and regex-match against it directly -- NOT against
          re-tokenized text. This distinction matters: an earlier
          approach that re-tokenized candidate text and checked for
          token-list adjacency produced false positives across sentence/
          clause boundaries (tokenizing strips punctuation, so "that's
          right, effort." and "right? Effort," both collapse to the
          adjacent tokens ['right', 'effort'] even though they are not
          the phrase "right effort" in the original text). A
          word-boundary-safe regex on the ORIGINAL lowercased text
          (`\\bright\\s+effort(?:es|s)?\\b`) does not have this problem,
          because it still sees the intervening punctuation. A second
          regex checks for the merged single-token form (also
          word-boundary-safe) to catch the compound-word case from stage
          1(b). Both patterns tolerate a single trailing "s"/"es", for
          the same reason sanghabot/highlight.py does (e.g. "breath" also
          matching "breaths").

        Returns an empty list (never raises) if `phrase` is empty/
        whitespace, or if no chunk matches -- callers (see
        sanghabot/search/phrase.py) treat that as "no exact-phrase
        signal for this phrase," letting normal semantic/BM25 ranking
        take over untouched, rather than needing special-case handling.
        """
        stemmed_tokens = tokenize(phrase, self.stopwords)
        if not stemmed_tokens:
            return []

        merged_token = "".join(stemmed_tokens)
        doc_freqs = self._bm25.doc_freqs

        candidate_indices: list[int] = []
        for i, freqs in enumerate(doc_freqs):
            if all(t in freqs for t in stemmed_tokens):
                candidate_indices.append(i)
                continue
            if len(stemmed_tokens) > 1 and merged_token in freqs:
                candidate_indices.append(i)

        if not candidate_indices:
            return []

        candidate_ids = [self._chunk_ids[i] for i in candidate_indices]
        chunks = self._db.get_chunks_by_ids(candidate_ids)
        if not chunks:
            return []

        raw_words = [w for w in re.split(r"\s+", phrase.strip().lower()) if w]
        literal_pattern = re.compile(
            r"\b" + r"\s+".join(re.escape(w) for w in raw_words) + r"(?:es|s)?\b"
        )
        merged_pattern = (
            re.compile(r"\b" + re.escape(merged_token) + r"(?:es|s)?\b")
            if len(stemmed_tokens) > 1
            else None
        )

        scores = self._bm25.get_scores(stemmed_tokens)
        score_by_index = {self._chunk_ids[i]: scores[i] for i in candidate_indices}

        matched_chunks: list[tuple] = []
        for chunk in chunks:
            text_low = chunk.text.lower()
            matched = bool(literal_pattern.search(text_low))
            if not matched and merged_pattern is not None:
                matched = bool(merged_pattern.search(text_low))
            if matched:
                score = score_by_index.get(chunk.chunk_id, 0.0)
                matched_chunks.append((chunk, score))

        if not matched_chunks:
            return []

        matched_chunks.sort(key=lambda pair: pair[1], reverse=True)

        video_ids = list({chunk.video_id for chunk, _ in matched_chunks})
        videos_by_id = self._db.get_videos_by_ids(video_ids)

        results: list[SearchResult] = []
        for chunk, score in matched_chunks:
            video = videos_by_id.get(chunk.video_id)
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    video_id=chunk.video_id,
                    title=video.title if video else "",
                    blog_url=video.blog_url if video else "",
                    text=chunk.text,
                    summary=chunk.summary,
                    score=float(score),
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
