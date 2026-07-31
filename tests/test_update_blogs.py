"""
Tests for scripts/update_blogs.py -- the one-command blog update script.

Covers the pure, easily-isolated helper functions:
  - _extract_chunks(): with/without-overlap text slicing.
  - _sync_blogs_dir(): copies new/changed files, skips unchanged ones.
  - _backup_index_if_needed(): backs up once per day, skips on re-run.

Does NOT exercise the full `run()` command end-to-end (that requires real
API calls and a real legacy embeddings.npy on disk -- see the manual
sandboxed testing described in chat history instead). These tests focus on
the parts of the script that are pure/local-filesystem-only and therefore
fast and deterministic to unit test.

Note: scripts/ is not a package (no __init__.py) and update_blogs.py does
its own sys.path manipulation at import time (consistent with the other
scripts/*.py files), so it's imported here via importlib rather than a
normal package-relative import.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "update_blogs.py"


@pytest.fixture(scope="module")
def update_blogs_module():
    spec = importlib.util.spec_from_file_location("update_blogs_test_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _extract_chunks()
# ---------------------------------------------------------------------------

def test_extract_chunks_no_overlap_matches_borders_exactly(update_blogs_module):
    text = "0123456789ABCDEFGHIJ"  # 20 chars
    borders = [5, 12]
    chunks = update_blogs_module._extract_chunks(text, borders, overlap_chars=0)
    assert chunks == ["01234", "56789AB", "CDEFGHIJ"]
    assert "".join(chunks) == text


def test_extract_chunks_with_overlap_extends_each_side(update_blogs_module):
    text = "0123456789ABCDEFGHIJ"  # 20 chars
    borders = [5, 12]
    chunks = update_blogs_module._extract_chunks(text, borders, overlap_chars=2)
    # chunk 0: [max(0-2,0) : min(5+2,20)] = [0:7]
    assert chunks[0] == text[0:7]
    # chunk 1: [max(5-2,0) : min(12+2,20)] = [3:14]
    assert chunks[1] == text[3:14]
    # chunk 2 (final, no upper border): [max(12-2,0) : end] = [10:]
    assert chunks[2] == text[10:]


def test_extract_chunks_no_borders_returns_whole_text_as_one_chunk(update_blogs_module):
    text = "just one chunk here"
    chunks = update_blogs_module._extract_chunks(text, [], overlap_chars=0)
    assert chunks == [text]


def test_extract_chunks_overlap_never_exceeds_text_bounds(update_blogs_module):
    text = "short"
    chunks = update_blogs_module._extract_chunks(text, [2], overlap_chars=1000)
    # overlap far exceeding text length must clamp to [0, len(text)], not error/overrun.
    assert chunks[0] == text[0:5]
    assert chunks[1] == text[0:]


# ---------------------------------------------------------------------------
# _sync_blogs_dir()
# ---------------------------------------------------------------------------

def test_sync_blogs_dir_copies_new_files(update_blogs_module, tmp_path):
    site_dir = tmp_path / "site"
    blogs_dir = tmp_path / "blogs"
    site_dir.mkdir()
    (site_dir / "a.md").write_text("content a")
    (site_dir / "b.md").write_text("content b")

    copied = update_blogs_module._sync_blogs_dir(site_dir, blogs_dir)

    assert copied == 2
    assert (blogs_dir / "a.md").read_text() == "content a"
    assert (blogs_dir / "b.md").read_text() == "content b"


def test_sync_blogs_dir_skips_unchanged_files_on_rerun(update_blogs_module, tmp_path):
    site_dir = tmp_path / "site"
    blogs_dir = tmp_path / "blogs"
    site_dir.mkdir()
    (site_dir / "a.md").write_text("content a")

    first = update_blogs_module._sync_blogs_dir(site_dir, blogs_dir)
    second = update_blogs_module._sync_blogs_dir(site_dir, blogs_dir)

    assert first == 1
    assert second == 0


def test_sync_blogs_dir_recopies_changed_files(update_blogs_module, tmp_path):
    import time

    site_dir = tmp_path / "site"
    blogs_dir = tmp_path / "blogs"
    site_dir.mkdir()
    src = site_dir / "a.md"
    src.write_text("v1")
    update_blogs_module._sync_blogs_dir(site_dir, blogs_dir)
    assert (blogs_dir / "a.md").read_text() == "v1"

    # Ensure a detectably later mtime, then change content+size.
    time.sleep(0.05)
    src.write_text("v2 is longer than v1")
    copied = update_blogs_module._sync_blogs_dir(site_dir, blogs_dir)

    assert copied == 1
    assert (blogs_dir / "a.md").read_text() == "v2 is longer than v1"


def test_sync_blogs_dir_leaves_extra_files_in_blogs_dir_untouched(update_blogs_module, tmp_path):
    """
    A file present in blogs_dir but not (any longer) in site_dir should be
    left alone -- this script only ever adds/updates, never deletes.
    """
    site_dir = tmp_path / "site"
    blogs_dir = tmp_path / "blogs"
    site_dir.mkdir()
    blogs_dir.mkdir()
    (blogs_dir / "old_only.md").write_text("still here")

    update_blogs_module._sync_blogs_dir(site_dir, blogs_dir)

    assert (blogs_dir / "old_only.md").read_text() == "still here"


# ---------------------------------------------------------------------------
# _backup_index_if_needed()
# ---------------------------------------------------------------------------

def test_backup_creates_new_dir_and_copies_index_files(update_blogs_module, tmp_path):
    data_dir = tmp_path / "data"
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "sanghabot.db").write_text("db contents")
    (index_dir / "faiss.index").write_text("faiss contents")
    (index_dir / "bm25.pkl").write_text("bm25 contents")

    backup_dir = update_blogs_module._backup_index_if_needed(data_dir)

    assert backup_dir is not None
    assert (backup_dir / "sanghabot.db").read_text() == "db contents"
    assert (backup_dir / "faiss.index").read_text() == "faiss contents"
    assert (backup_dir / "bm25.pkl").read_text() == "bm25 contents"


def test_backup_skips_second_call_same_day(update_blogs_module, tmp_path):
    data_dir = tmp_path / "data"
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "sanghabot.db").write_text("db contents")

    first = update_blogs_module._backup_index_if_needed(data_dir)
    second = update_blogs_module._backup_index_if_needed(data_dir)

    assert first == second  # same backup dir returned, not a new one
    backups = list(data_dir.glob("index_backup_*"))
    assert len(backups) == 1


def test_backup_handles_missing_index_files_gracefully(update_blogs_module, tmp_path):
    """No crash if data/index/ doesn't have all three files yet (e.g. a
    brand-new setup before any indexing has ever run)."""
    data_dir = tmp_path / "data"
    (data_dir / "index").mkdir(parents=True)

    backup_dir = update_blogs_module._backup_index_if_needed(data_dir)

    assert backup_dir is not None
    assert backup_dir.exists()
