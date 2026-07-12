import concurrent.futures
import threading
from datetime import date

import pytest

from sanghabot.models import Chunk, VideoMetadata
from sanghabot.storage.db import Database


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


def test_insert_and_get_video(db):
    v = VideoMetadata(
        video_id="v1", title="Talk 1", blog_url="https://x.com/1",
        published_date=date(2026, 1, 1), tags=["meditation", "jhana"],
    )
    db.insert_video(v)
    fetched = db.get_video("v1")
    assert fetched is not None
    assert fetched.video_id == "v1"
    assert fetched.title == "Talk 1"
    assert fetched.tags == ["meditation", "jhana"]
    assert fetched.published_date == date(2026, 1, 1)


def test_insert_video_upsert_updates_fields(db):
    db.insert_video(VideoMetadata(video_id="v1", title="Old", blog_url="https://x.com/1"))
    db.insert_video(VideoMetadata(video_id="v1", title="New", blog_url="https://x.com/1"))
    fetched = db.get_video("v1")
    assert fetched.title == "New"


def test_insert_and_batch_fetch_chunks_preserves_order(db):
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    chunks = [
        Chunk(chunk_id=f"v1_{i}", video_id="v1", chunk_idx=i, text=f"text {i}")
        for i in range(5)
    ]
    db.insert_chunks(chunks)

    # Request out of natural/insertion order -- result order must match request order.
    requested = ["v1_3", "v1_0", "v1_4"]
    fetched = db.get_chunks_by_ids(requested)
    assert [c.chunk_id for c in fetched] == requested


def test_get_chunks_by_ids_skips_missing_ids(db):
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks([Chunk(chunk_id="v1_0", video_id="v1", chunk_idx=0, text="hi")])
    fetched = db.get_chunks_by_ids(["v1_0", "does_not_exist"])
    assert len(fetched) == 1
    assert fetched[0].chunk_id == "v1_0"


def test_set_and_get_embedding_row(db):
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks([Chunk(chunk_id="v1_0", video_id="v1", chunk_idx=0, text="hi")])
    db.set_embedding_row("v1_0", 42)
    chunk = db.get_chunk("v1_0")
    assert chunk.embedding_row == 42

    by_row = db.get_chunks_by_embedding_rows([42])
    assert 42 in by_row
    assert by_row[42].chunk_id == "v1_0"


def test_embedding_runs_are_never_overwritten_only_flagged_active(db):
    db.insert_embedding_run(
        run_id="legacy-512", model_name="qwen/qwen3-embedding-4b",
        dimension=512, created_at="2026-01-01T00:00:00Z",
        npy_path="embeddings_legacy_512.npy", is_active=True,
        notes="Legacy averaging-hack output, migrated as-is.",
    )
    db.insert_embedding_run(
        run_id="qwen3-4b-1024-2026", model_name="qwen/qwen3-embedding-4b",
        dimension=1024, created_at="2026-08-01T00:00:00Z",
        npy_path="embeddings_1024.npy", is_active=False,
    )
    active = db.get_active_embedding_run()
    assert active["run_id"] == "legacy-512"

    db.set_active_embedding_run("qwen3-4b-1024-2026")
    active = db.get_active_embedding_run()
    assert active["run_id"] == "qwen3-4b-1024-2026"
    assert active["dimension"] == 1024

    # The old run row must still exist -- "never overwritten, only flagged" policy.
    cur = db._conn.execute("SELECT COUNT(*) FROM embedding_runs")
    assert cur.fetchone()[0] == 2


def test_iter_chunk_texts_streams_all_chunks(db):
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks([
        Chunk(chunk_id=f"v1_{i}", video_id="v1", chunk_idx=i, text=f"text {i}")
        for i in range(3)
    ])
    texts = list(db.iter_chunk_texts())
    assert len(texts) == 3
    assert set(t[1] for t in texts) == {"text 0", "text 1", "text 2"}


def test_count_chunks_and_videos(db):
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks([Chunk(chunk_id="v1_0", video_id="v1", chunk_idx=0, text="hi")])
    assert db.count_videos() == 1
    assert db.count_chunks() == 1


def test_get_max_end_char(db):
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks([
        Chunk(chunk_id="v1_0", video_id="v1", chunk_idx=0, text="a", start_char=0, end_char=100),
        Chunk(chunk_id="v1_1", video_id="v1", chunk_idx=1, text="b", start_char=100, end_char=250),
    ])
    assert db.get_max_end_char("v1") == 250


def test_database_usable_from_a_different_thread_than_it_was_created_in(db):
    """
    Regression test for a real production bug: the Discord bot constructs
    Database in the main thread (see sanghabot/bot/discord_bot.py's
    on_ready), but engine.search() runs inside asyncio.to_thread(...),
    which executes on a worker thread pool -- a DIFFERENT thread than the
    one that created the Database object. A plain sqlite3.Connection
    raises "SQLite objects created in a thread can only be used in that
    same thread" in this situation. Database must transparently give each
    thread its own connection instead.
    """
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks([Chunk(chunk_id="v1_0", video_id="v1", chunk_idx=0, text="hi")])

    main_thread_id = threading.get_ident()
    result_holder = {}

    def query_from_worker_thread():
        result_holder["thread_id"] = threading.get_ident()
        return db.get_chunks_by_ids(["v1_0"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(query_from_worker_thread)
        chunks = future.result()

    assert result_holder["thread_id"] != main_thread_id, "test setup error: worker didn't run on a different thread"
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "v1_0"


def test_database_usable_concurrently_from_multiple_threads(db):
    """Multiple overlapping searches (e.g. two users querying the Discord
    bot at nearly the same time) must not corrupt or crash each other's
    reads."""
    db.insert_video(VideoMetadata(video_id="v1", title="Talk 1", blog_url="https://x.com/1"))
    db.insert_chunks([
        Chunk(chunk_id=f"v1_{i}", video_id="v1", chunk_idx=i, text=f"text {i}")
        for i in range(20)
    ])

    def query(i):
        return db.get_chunks_by_ids([f"v1_{i}"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(query, range(20)))

    for i, chunks in enumerate(results):
        assert len(chunks) == 1
        assert chunks[0].chunk_id == f"v1_{i}"
