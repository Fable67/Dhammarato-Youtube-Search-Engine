"""
LEGACY COMPATIBILITY SHIM -- for parity testing against migrated old data
ONLY. Do not use this for any new embedding work.

The old codebase's FAISS index (AI_Transcripts/embeddings/faiss.index) was
built from vectors that went through a very specific, now-deprecated
pipeline (see AI_Transcripts/embed_videos.py and semantic_search.py):

  1. Query expansion: for queries of <=2 words, prepend "Discussion about "
     (semantic_search.py: _expand_query).
  2. Embed via OpenRouter using model "qwen/qwen3-embedding-8b" requesting
     1024 dimensions (embed_videos.py: batch_embed_fn_openrouter).
  3. Reduce 1024 -> 512 dims via averaging adjacent pairs
     (embed_videos.py: reduce_dimensionality) -- the lossy, irreversible
     transform documented in REWRITE_PLAN.md Appendix C.
  4. L2-normalize.

To compare the new engine's semantic search behavior against the old
engine's behavior on the SAME migrated FAISS index (necessary for the
Section 9a parity gate), query vectors must go through this exact same
legacy pipeline -- otherwise a query embedded via the new default client
(sanghabot/embeddings/client.py, native dimension request, no query
expansion, no averaging) would land in a different vector space than the
one the legacy FAISS index was built in, and any mismatch would be
meaningless noise rather than a real signal about code correctness.

Once the planned full re-embed to 1024 dims (REWRITE_PLAN.md Section 11)
happens and the FAISS index is rebuilt from clean native embeddings, this
entire module becomes dead and should be deleted.
"""
from __future__ import annotations

import numpy as np

from sanghabot.embeddings.client import embed_texts

LEGACY_MODEL = "qwen/qwen3-embedding-8b"
LEGACY_NATIVE_DIM = 1024
LEGACY_TARGET_DIM = 512

# Must match scripts/migrate_legacy_data.py's LEGACY_RUN_ID. Used by
# sanghabot/bot/discord_bot.py to decide whether the currently-active
# embedding_runs row requires this legacy query-embedding pipeline, or
# whether a clean native embed_query() call is appropriate (i.e. once
# REWRITE_PLAN.md Section 11's full re-embed has happened and a new run
# has been promoted to active).
LEGACY_RUN_MARKER = "qwen3-4b-512-legacy"


def legacy_expand_query(query: str) -> str:
    """Verbatim port of semantic_search.py's _expand_query."""
    words = query.strip().split()
    if len(words) <= 2:
        return f"Discussion about {query}"
    return query


def legacy_reduce_dimensionality(vectors: np.ndarray, target_dim: int = LEGACY_TARGET_DIM) -> np.ndarray:
    """Verbatim port of embed_videos.py's reduce_dimensionality."""
    current_dim = vectors.shape[1]
    if current_dim == target_dim:
        return vectors
    if current_dim % target_dim != 0:
        raise ValueError(f"Cannot predictably reduce {current_dim} to {target_dim} via pooling.")
    factor = current_dim // target_dim
    return vectors.reshape(vectors.shape[0], target_dim, factor).mean(axis=2)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def legacy_embed_query(query: str) -> np.ndarray:
    """
    Full legacy pipeline for a single query string: expand -> embed at
    1024 natively -> average-pool to 512 -> normalize. Matches
    SemanticSearchEngine.search()'s query-embedding steps in the old
    AI_Transcripts/semantic_search.py exactly.
    """
    clean_query = query.replace('"', " ").strip()
    expanded = legacy_expand_query(clean_query)
    raw_vec = embed_texts([expanded], dimension=LEGACY_NATIVE_DIM, model=LEGACY_MODEL)[0]
    reduced = legacy_reduce_dimensionality(np.array([raw_vec]), LEGACY_TARGET_DIM)[0]
    return _normalize(reduced).astype("float32")
