"""
SQLite storage layer.

Replaces, in one file:
  - AI_Transcripts/embeddings/metadata.csv
  - AI_Transcripts/embeddings/metadata.db
  - AI_Transcripts/embeddings/chunks/*.txt        (23,246 loose files)
  - AI_Transcripts/embeddings/summaries/*.txt     (2,061 loose files)

One connection PER THREAD, opened lazily and held for that thread's
lifetime (WAL mode allows multiple connections to the same file to read
concurrently, and to write without blocking readers). No per-search-call
CSV reload, no per-result file opens -- both of which were serious
performance smells in the old code (semantic_search.py / keyword_search.py
opened two files per result, per query, for up to 1000 results per
engine).

Why per-thread instead of one shared connection: the Discord bot runs
search() inside `asyncio.to_thread(...)` (see sanghabot/bot/discord_bot.py),
which executes on a worker thread pool -- NOT the thread that constructs
the Database object during startup. A single sqlite3.Connection is tied to
the thread that created it by default (and is not safe to share across
threads even with that restriction lifted, since concurrent use of one
connection object from multiple threads isn't supported). A thread-local
connection pool gives every thread its own connection to the same
database file, which is both safe and, in WAL mode, still fast.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from sanghabot.models import Chunk, VideoMetadata

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id            TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    blog_url            TEXT NOT NULL,
    url                 TEXT,
    published_date      TEXT,
    tags                TEXT,  -- JSON array
    video_length_chars  INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(video_id),
    chunk_idx       INTEGER NOT NULL,
    text            TEXT NOT NULL,
    summary         TEXT,
    start_char      INTEGER,
    end_char        INTEGER,
    embedding_row   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_row ON chunks(embedding_row);

-- See REWRITE_PLAN.md Appendix C: embeddings are never mutated in
-- place. Each embedding artifact is a new, immutable row here.
CREATE TABLE IF NOT EXISTS embedding_runs (
    run_id        TEXT PRIMARY KEY,
    model_name    TEXT NOT NULL,
    dimension     INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    npy_path      TEXT NOT NULL,
    is_active     INTEGER DEFAULT 0,
    notes         TEXT
);

-- Resumable ingestion/embedding checkpointing, replacing the old
-- hand-rolled progress.json.
CREATE TABLE IF NOT EXISTS progress (
    stage           TEXT PRIMARY KEY,  -- e.g. "chunk", "embed:<run_id>"
    completed_ids   TEXT NOT NULL,     -- JSON array of video_id/chunk_id
    updated_at      TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

        # Run schema creation once, up front, from the constructing thread,
        # so every later thread just opens a plain connection to an
        # already-initialized database file.
        conn = sqlite3.connect(str(self.path))
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Returns this thread's own connection, opening one on first use.
        See the module docstring for why this is thread-local rather than
        a single shared connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path))
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Closes this thread's connection, if it has one open. Note: in a
        multi-threaded setting (e.g. the Discord bot's worker thread pool),
        other threads' connections are each closed automatically when
        their thread exits and the connection object is garbage collected;
        this method only closes the calling thread's own connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Videos
    # ------------------------------------------------------------------
    def insert_video(self, v: VideoMetadata) -> None:
        self._conn.execute(
            """
            INSERT INTO videos (video_id, title, blog_url, url, published_date, tags, video_length_chars)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title=excluded.title,
                blog_url=excluded.blog_url,
                url=excluded.url,
                published_date=excluded.published_date,
                tags=excluded.tags,
                video_length_chars=excluded.video_length_chars
            """,
            (
                v.video_id, v.title, v.blog_url, v.url,
                v.published_date.isoformat() if v.published_date else None,
                json.dumps(v.tags), v.video_length_chars,
            ),
        )
        self._conn.commit()

    def insert_videos(self, videos: list[VideoMetadata]) -> None:
        rows = [
            (
                v.video_id, v.title, v.blog_url, v.url,
                v.published_date.isoformat() if v.published_date else None,
                json.dumps(v.tags), v.video_length_chars,
            )
            for v in videos
        ]
        self._conn.executemany(
            """
            INSERT INTO videos (video_id, title, blog_url, url, published_date, tags, video_length_chars)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title=excluded.title,
                blog_url=excluded.blog_url,
                url=excluded.url,
                published_date=excluded.published_date,
                tags=excluded.tags,
                video_length_chars=excluded.video_length_chars
            """,
            rows,
        )
        self._conn.commit()

    def get_video(self, video_id: str) -> VideoMetadata | None:
        cur = self._conn.execute(
            "SELECT video_id, title, blog_url, url, published_date, tags, video_length_chars "
            "FROM videos WHERE video_id = ?",
            (video_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_video(row)

    def get_video_by_blog_url(self, blog_url: str) -> VideoMetadata | None:
        """
        Looks up a video by its blog_url rather than video_id.

        Why this exists: video_id is NOT a stable/consistent identifier
        across the corpus's history. The original ~2,061 videos were
        migrated from a legacy CSV where video_id was an arbitrary integer
        (row position at migration time -- see scripts/migrate_legacy_data.py),
        while videos ingested since (see scripts/update_blogs.py) use the
        source markdown filename's slug as video_id instead. blog_url,
        however, is deterministically derived from the filename slug for
        EVERY video regardless of which era it was ingested in
        (f"https://dhammarato.com/blog/{slug}"), so it's the one reliable
        way to check "has this specific blog post already been ingested,
        under whatever video_id it happened to get" -- which a naive
        get_video(slug) lookup cannot answer for the ~2,061 legacy-era rows.
        """
        cur = self._conn.execute(
            "SELECT video_id, title, blog_url, url, published_date, tags, video_length_chars "
            "FROM videos WHERE blog_url = ?",
            (blog_url,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_video(row)

    def get_videos_by_ids(self, video_ids: list[str]) -> dict[str, VideoMetadata]:
        if not video_ids:
            return {}
        placeholders = ",".join("?" for _ in video_ids)
        cur = self._conn.execute(
            f"SELECT video_id, title, blog_url, url, published_date, tags, video_length_chars "
            f"FROM videos WHERE video_id IN ({placeholders})",
            video_ids,
        )
        return {row[0]: _row_to_video(row) for row in cur.fetchall()}

    def get_max_end_char(self, video_id: str) -> int | None:
        cur = self._conn.execute(
            "SELECT MAX(end_char) FROM chunks WHERE video_id = ?", (video_id,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------
    def insert_chunks(self, chunks: list[Chunk]) -> None:
        rows = [
            (
                c.chunk_id, c.video_id, c.chunk_idx, c.text, c.summary,
                c.start_char, c.end_char, c.embedding_row,
            )
            for c in chunks
        ]
        self._conn.executemany(
            """
            INSERT INTO chunks
                (chunk_id, video_id, chunk_idx, text, summary, start_char, end_char, embedding_row)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                text=excluded.text,
                summary=excluded.summary,
                start_char=excluded.start_char,
                end_char=excluded.end_char,
                embedding_row=excluded.embedding_row
            """,
            rows,
        )
        self._conn.commit()

    def set_embedding_row(self, chunk_id: str, embedding_row: int) -> None:
        self._conn.execute(
            "UPDATE chunks SET embedding_row = ? WHERE chunk_id = ?",
            (embedding_row, chunk_id),
        )
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        cur = self._conn.execute(
            "SELECT chunk_id, video_id, chunk_idx, text, summary, start_char, end_char, embedding_row "
            "FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Batch fetch, preserving the input order (important: FAISS/BM25
        return ranked ids, and callers depend on that order surviving)."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        cur = self._conn.execute(
            f"SELECT chunk_id, video_id, chunk_idx, text, summary, start_char, end_char, embedding_row "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        by_id = {row[0]: _row_to_chunk(row) for row in cur.fetchall()}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_chunks_by_embedding_rows(self, rows: list[int]) -> dict[int, Chunk]:
        """Used by the semantic engine: FAISS returns row indices, not chunk_ids."""
        if not rows:
            return {}
        placeholders = ",".join("?" for _ in rows)
        cur = self._conn.execute(
            f"SELECT chunk_id, video_id, chunk_idx, text, summary, start_char, end_char, embedding_row "
            f"FROM chunks WHERE embedding_row IN ({placeholders})",
            rows,
        )
        result = {}
        for row in cur.fetchall():
            chunk = _row_to_chunk(row)
            result[chunk.embedding_row] = chunk
        return result

    def iter_chunk_texts(self):
        """Streamed (chunk_id, text) pairs for BM25 corpus building -- never
        loads the full corpus into a pandas DataFrame."""
        cur = self._conn.execute("SELECT chunk_id, text FROM chunks ORDER BY rowid")
        for row in cur:
            yield row[0], row[1]

    def count_chunks(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks")
        return cur.fetchone()[0]

    def count_videos(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM videos")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Embedding runs (Appendix C immutability policy)
    # ------------------------------------------------------------------
    def insert_embedding_run(
        self, run_id: str, model_name: str, dimension: int,
        created_at: str, npy_path: str, is_active: bool = False,
        notes: str = "",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO embedding_runs (run_id, model_name, dimension, created_at, npy_path, is_active, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO NOTHING
            """,
            (run_id, model_name, dimension, created_at, npy_path, int(is_active), notes),
        )
        self._conn.commit()

    def set_active_embedding_run(self, run_id: str) -> None:
        self._conn.execute("UPDATE embedding_runs SET is_active = 0")
        self._conn.execute(
            "UPDATE embedding_runs SET is_active = 1 WHERE run_id = ?", (run_id,)
        )
        self._conn.commit()

    def get_active_embedding_run(self) -> dict | None:
        cur = self._conn.execute(
            "SELECT run_id, model_name, dimension, created_at, npy_path, notes "
            "FROM embedding_runs WHERE is_active = 1 LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0], "model_name": row[1], "dimension": row[2],
            "created_at": row[3], "npy_path": row[4], "notes": row[5],
        }


def _row_to_video(row) -> VideoMetadata:
    from datetime import date as _date
    published = _date.fromisoformat(row[4]) if row[4] else None
    tags = json.loads(row[5]) if row[5] else []
    return VideoMetadata(
        video_id=row[0], title=row[1], blog_url=row[2], url=row[3],
        published_date=published, tags=tags, video_length_chars=row[6],
    )


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        chunk_id=row[0], video_id=row[1], chunk_idx=row[2], text=row[3],
        summary=row[4], start_char=row[5] or 0, end_char=row[6] or 0,
        embedding_row=row[7],
    )
