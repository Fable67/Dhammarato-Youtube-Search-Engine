"""
FAISS-backed semantic search engine.

The single most important discipline in this file: the FAISS index is
loaded exactly ONCE, in __init__. search() never touches disk. This is the
direct fix for AI_Transcripts/semantic_search.py, which called
`faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)` *inside*
`search()` -- meaning every single query re-read the entire index from
disk, and combined with the old Discord bot reconstructing the whole
engine per message, produced multi-second query latency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from sanghabot.models import SearchResult
from sanghabot.storage.db import Database


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class SemanticSearchEngine:
    def __init__(
        self,
        index_path: Path | str,
        db: Database,
        embed_fn: Callable[[str], np.ndarray],
    ):
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        # Loaded once. Never reloaded inside search().
        self._index = faiss.read_index(str(index_path))
        self._db = db
        self._embed_fn = embed_fn

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        clean_query = query.replace('"', " ").strip()
        if not clean_query:
            return []

        query_vec = self._embed_fn(clean_query)
        query_vec = _normalize(np.asarray(query_vec)).astype("float32")

        scores, row_indices = self._index.search(np.array([query_vec]), k)
        scores = scores[0]
        row_indices = row_indices[0]

        # Drop any -1 padding FAISS returns when fewer than k matches exist.
        valid = [(s, r) for s, r in zip(scores, row_indices) if r != -1]
        if not valid:
            return []

        rows_in_order = [int(r) for _, r in valid]
        chunks_by_row = self._db.get_chunks_by_embedding_rows(rows_in_order)

        video_ids = list({c.video_id for c in chunks_by_row.values()})
        videos_by_id = self._db.get_videos_by_ids(video_ids)

        results: list[SearchResult] = []
        for score, row in valid:
            chunk = chunks_by_row.get(int(row))
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
                    score=float(score),
                    percentage_score=0.0,  # populated by caller (engine.py) as needed
                    source="semantic",
                    url=video.url if video else None,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    video_length_chars=video.video_length_chars if video else None,
                )
            )
        return results
