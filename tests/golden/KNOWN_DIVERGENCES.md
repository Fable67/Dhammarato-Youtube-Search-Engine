# Known, documented divergences between old and new engine output

Per rewrite/REWRITE_PLAN.md Section 9a.C.2: any divergence between the old
engine's captured golden output and the new engine's output must be listed
here with a one-line justification before the parity test is allowed to
treat it as an expected pass rather than a failure. Nothing gets silently
waved through.

## 0. IMPORTANT: a real bug was found and fixed in the old code during parity testing

During the first parity test run, 25 of 27 golden queries failed. Root
cause: `AI_Transcripts/semantic_search.py`'s `search()` ran
`SELECT * FROM metadata WHERE rowid - 1 IN (?, ?, ...)` using FAISS's
rank-ordered result indices as the `IN`-list, then assigned FAISS's scores
back onto the SQL result rows purely positionally
(`candidates["semantic_score"] = D[0]`). SQLite does **not** preserve
`IN`-list order in its results -- it returns rows in `rowid` order. This
meant the semantic_score attached to each chunk was silently wrong
whenever the returned rowids weren't already ascending relative to FAISS's
rank order (which is nearly always), corrupting which chunk actually
"won" any given semantic search.

Reproduced directly: for the query "Freedom", FAISS's true top match was
chunk `1410_2` (score 0.5827), but the buggy positional assignment handed
that top score to chunk `226_0` instead (an unrelated, lower-ranked
match), while the real top match landed near the bottom of the reported
ranking.

Per explicit user instruction, this exact bug -- and only this bug -- was
fixed directly in `AI_Transcripts/semantic_search.py` (see git diff for
the change): `candidates` is now re-indexed by `vector_id` (which is
always identical to `rowid - 1`, verified directly against the live
database) to match FAISS's actual rank order (`I[0]`) before scores are
assigned. Nothing else in the old codebase was touched. The golden query
capture was re-run against this fixed version, and this parity suite
compares the new engine against those corrected captures.

This finding directly validates the user's stated motivation for planning
a full re-embed from scratch (rewrite/REWRITE_PLAN.md Section 11): the
concern that "there is some underlying bug hidden somewhere" in the old
pipeline was correct. This particular bug lived in the query-time
scoring/lookup code, not in the embeddings themselves, so it does not by
itself require re-embedding to fix -- but it's exactly the class of
silent, previously-undiscovered issue that motivates treating a from-
scratch rebuild as real due diligence rather than excessive caution.

## 5. The embeddings API is not perfectly deterministic call-to-call

After fixing the bug in #0 above, 17/27 golden queries passed exactly, but
11 still showed small mismatches -- all either (a) percentage_score
differing by well under 2% relative, or (b) near-tied candidates swapping
order (same set of chunk_ids, different order), never a genuinely
different/wrong candidate appearing.

Root-caused directly: querying the SAME text twice, in separate API calls,
through `sanghabot/embeddings/legacy_compat.py`'s `legacy_embed_query()`
(which calls the OpenRouter-hosted Qwen3 embedding model), produces
vectors with cosine similarity ~0.99999994 -- not exactly 1.0. Repeated
FAISS searches against the same index using these two very-slightly-
different query vectors return the same top candidates but with scores
that drift by roughly 1e-4 to 1e-3, occasionally enough to flip the order
of two near-tied candidates. This is inherent non-determinism in the
hosted embedding service (most likely floating-point non-associativity in
batched GPU inference), present in both the old and new code equally --
it is not a bug in either.

Consequence for the parity test: `tests/test_parity_old_vs_new.py`'s
original `1e-6` relative-error tolerance on `percentage_score` assumed
exact determinism that does not exist at the embedding-API layer. This has
been loosened to a tolerance that accounts for this measured noise floor
(see the test file for the exact value), and near-tied order swaps among
otherwise-identical candidate sets are treated as passing, not failing.
This is a measured, explained adjustment -- not a silent loosening to make
red tests green.

## 1. Query embedding pipeline for legacy-data parity testing

The new production embedding client (`sanghabot/embeddings/client.py`)
requests dimensions natively and applies no query expansion. To compare
against the OLD FAISS index (built from vectors that went through query
expansion + qwen3-embedding-8b @ 1024 dims + averaging-pool to 512),
`tests/test_parity_old_vs_new.py` uses
`sanghabot/embeddings/legacy_compat.py`, a deliberately quarantined shim
that reproduces the old pipeline exactly. This is not a divergence in
behavior between old and new *logic* -- it is a necessary adapter so both
engines are tested against the same vector space. See
`legacy_compat.py`'s module docstring for full detail. This shim is
scheduled for deletion once REWRITE_PLAN.md Section 11 (full re-embed to
1024 dims) lands and the FAISS index is rebuilt from clean vectors.

## 2. search.py:247-249 NameError crash bug is dead code, not a live bug

Verified directly (see `scripts/capture_golden_queries.py` module comment
and manual check against `analyze_query_intent()`): every *reachable*
routing branch in the old code sets `use_bm25 = True`. The only branch
that sets `use_bm25 = False` (`elif is_question or len(raw_words) >= 20`)
is unreachable, because any `is_question` case is already caught by the
preceding `elif`, and any `len >= 20` query already satisfies `len >= 5`
and is also already caught. This means the crash bug in
`search.py:247-249` (`final_results[i]` with no defining loop) cannot
actually be triggered by any real query today.

Consequence for parity testing: no golden capture exercises this crash, so
there is nothing to "match." The new code (`sanghabot/engine.py`) still
fixes this class of bug structurally (see
`_populate_percentage_score_by_top_score`, used identically by both the
semantic-only and bm25-only branches), and `tests/test_fusion.py` /
`tests/test_intent.py` cover it as a regression test for correctness, not
for old-vs-new parity.

## 3. BM25 tokenizer/stemmer is intentionally preserved verbatim (for now)

`sanghabot/search/bm25.py`'s `_simple_s_stripper()` is a verbatim port of
the old codebase's heuristic suffix-stripper, including its quirks (e.g.
stopwords are accepted as a parameter but never actually applied, matching
the old code's commented-out filter line). This is deliberate: BM25
rankings are sensitive to exact tokenization, and parity requires matching
rankings against the same corpus. Replacing this with a proper stemming
library (e.g. Snowball/Porter) is a good follow-up, but must happen AFTER
the parity gate passes, as its own tracked change -- not silently bundled
into this rewrite. See `bm25.py`'s module docstring.

## 4. RRF dedup-guard difference (functionally identical, not a real divergence)

The old code's BM25-result dedup guard inside `reciprocal_rank_fusion` was
only active when `debug=True` (a `debug_info` dict populated only in debug
mode gated the "already scored" check). In production (`debug=False`) that
guard was structurally a no-op. This never manifested as an actual bug
because BM25's `top_n_indices` are produced via `np.argsort`, which cannot
yield duplicate indices within a single call -- so old production output
and the new, always-deduped `sanghabot/search/fusion.py` implementation
are functionally identical for every real query. See the inline comment in
`fusion.py` for detail.
