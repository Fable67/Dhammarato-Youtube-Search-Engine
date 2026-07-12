#!/usr/bin/env python
"""
HISTORICAL RECORD -- already run, kept for documentation/audit purposes.

This script was used ONCE, in the original "Youtube-Video-Recommendation-
Engine" repo, to capture golden query output from that repo's
AI_Transcripts/search.py before this rewrite existed as its own project.
The resulting golden captures already live in tests/golden/ in this repo.
This script assumes AI_Transcripts/ exists as a sibling directory, which is
only true in that original repo, not here. It's kept as an accurate record
of exactly how tests/golden/ was produced, not as something expected to
run again in this standalone repo.

Original docstring follows, unmodified:
---
Capture ground-truth search output from the CURRENTLY RUNNING, UNTOUCHED
old implementation (AI_Transcripts/search.py :: CombinedSearchEngine).

This script implements REWRITE_PLAN.md Section 9a.C.1. It must be
run BEFORE any new search code is trusted, against the old code exactly as
it exists in the repo today. It never imports anything from the new
`sanghabot` package -- only from the legacy `AI_Transcripts/` directory.

What it produces:
  - tests/golden/frozen_data_manifest.json
        sha256 checksums of the data files the old engine was tested
        against, so a later parity run can detect if the underlying data
        has drifted since capture (in which case a mismatch means "data
        changed", not "new code is wrong").
  - tests/golden/queries/<slug>.json
        one file per query, with the full ordered result list exactly as
        returned by the old CombinedSearchEngine.search(), including a
        captured exception (if any) for the query that's expected to
        exercise the known search.py:249 NameError crash.

Run this from anywhere; it locates AI_Transcripts/ relative to this repo
via REPO_ROOT below. It does NOT modify anything in AI_Transcripts/.

Usage:
    conda activate Dhamma
    python scripts/capture_golden_queries.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REWRITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = REWRITE_ROOT.parent
OLD_CODE_DIR = REPO_ROOT / "AI_Transcripts"
GOLDEN_DIR = REWRITE_ROOT / "tests" / "golden"
QUERIES_DIR = GOLDEN_DIR / "queries"

# Data files whose checksums get frozen into the manifest. These are the
# files the old CombinedSearchEngine actually reads from at construction /
# search time.
DATA_FILES_TO_CHECKSUM = [
    "embeddings/metadata.csv",
    "embeddings/metadata.db",
    "embeddings/faiss.index",
    "embeddings/bm25+_index.pkl",
    "embeddings/embeddings.npy",
]

# The query set: real historical queries pulled from search_engine.log
# (verbatim user queries, not synthetic), plus deliberately chosen queries
# to exercise every analyze_query_intent() branch, plus edge cases.
#
# NOTE on the known search.py:247-249 NameError crash bug (documented in
# REWRITE_PLAN.md Section 0 and Section 9a): that branch only executes when
# `intent["use_semantic"] is True and intent["use_bm25"] is False`. Verified
# directly against the live analyze_query_intent() logic: every *reachable*
# branch sets `use_bm25 = True` unconditionally; the only branch that sets
# `use_bm25 = False` is the dead/unreachable
# `elif is_question or len(raw_words) >= 20` branch (unreachable because any
# is_question case is already caught by the preceding elif, and any
# len>=20 case already implies len>=5, also already caught). This means the
# crash bug cannot actually be triggered by ANY real query today -- it is
# latent dead code, not a live production bug. We do NOT fabricate a fake
# "crash" capture here; that would misrepresent the old system's actual
# ground-truth behavior. Instead, this is captured as a documented,
# structural finding (see tests/golden/KNOWN_DIVERGENCES.md), and the new
# fusion.py is unit-tested (tests/test_fusion.py) to prove it cannot
# reproduce this bug class even if a future routing change made the branch
# reachable.
QUERY_SET: list[str] = [
    # --- Real historical queries from search_engine.log ---
    "Freedom",
    "Define free will",
    "Disliking the world",
    "Fire",
    "How can we tell if something (like a thought) is wholesome or unwholesome? "
    "On a side note: How are kusula and akusula defined in the suttas?",
    "How to practice?",
    "Mudita",
    "Nothingness",
    "Personality disorder",
    "Rebirth",
    "Sukha",
    "This is a test",
    "Tractor",
    "Victimhood",
    "Western secular buddhism",
    "What am I doing here?",
    "What are the four noble truths?",

    # --- Deliberate coverage of every analyze_query_intent() branch ---
    '"right effort"',                                   # exact phrase -> bm25-heavy
    '"right effort" and how it fits into the eightfold path in daily life',  # phrase + extra context -> both
    "What is the nature of consciousness according to early Buddhism?",  # question -> semantic-heavy
    "dukkha",                                            # 1 meaningful word -> bm25-leaning
    "right effort",                                      # 2 meaningful words -> bm25-leaning
    "blue green red",                                     # 3+ words, non-question -> default hybrid
    " ".join(["meditation"] * 25),                        # >=20 words, non-question: would have hit
                                                           # old code's dead branch IF it were reachable;
                                                           # confirmed it still resolves via the
                                                           # >=5-words bucket instead (use_bm25=True),
                                                           # so no crash occurs -- this query exists to
                                                           # prove that in the captured ground truth.

    # --- Edge cases ---
    "",
    "a",
    "asdkfjhaslkdjfhlaskdjfh nonsense gibberish query",
]


def _slugify(query: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip().lower()).strip("_")
    if not slug:
        slug = "empty_query"
    # Include a short hash suffix to disambiguate queries that collapse to
    # the same slug after stripping punctuation (e.g. 'right effort' vs
    # '"right effort"' both slugify to 'right_effort' -- without this, the
    # second capture would silently overwrite the first).
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:70]}_{digest}"


def _sha256_of_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_frozen_manifest() -> dict:
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    for rel_path in DATA_FILES_TO_CHECKSUM:
        full_path = OLD_CODE_DIR / rel_path
        checksum = _sha256_of_file(full_path)
        size = full_path.stat().st_size if full_path.exists() else None
        manifest["files"][rel_path] = {"sha256": checksum, "size_bytes": size}
    return manifest


@dataclass
class CapturedQuery:
    query: str
    captured_at: str
    results: list[dict] = field(default_factory=list)
    raised_exception: str | None = None


def capture_query(engine, query: str) -> CapturedQuery:
    captured = CapturedQuery(query=query, captured_at=datetime.now(timezone.utc).isoformat())
    try:
        results = engine.search(query, top_k=3)
        # Keep the full dict output as-is (old code returns plain dicts).
        # Convert any non-JSON-serializable values (numpy types) defensively.
        clean_results = []
        for r in results:
            clean = {}
            for k, v in r.items():
                try:
                    json.dumps(v)
                    clean[k] = v
                except TypeError:
                    clean[k] = str(v)
            clean_results.append(clean)
        captured.results = clean_results
    except Exception as e:
        captured.raised_exception = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    return captured


def main() -> None:
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[capture] Freezing data manifest from {OLD_CODE_DIR} ...")
    manifest = build_frozen_manifest()
    manifest_path = GOLDEN_DIR / "frozen_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[capture] Wrote {manifest_path}")
    for rel_path, info in manifest["files"].items():
        status = "OK" if info["sha256"] else "MISSING"
        print(f"    [{status}] {rel_path} ({info['size_bytes']} bytes)")

    # Import the OLD code, unmodified, exactly as it runs in production.
    # It relies on relative imports and relative file paths, so both
    # sys.path and cwd must point at AI_Transcripts/ during this import.
    sys.path.insert(0, str(OLD_CODE_DIR))
    import os
    original_cwd = os.getcwd()
    os.chdir(OLD_CODE_DIR)
    try:
        from search import CombinedSearchEngine  # noqa: E402  (old, untouched code)

        print("[capture] Constructing old CombinedSearchEngine (debug=False) ...")
        engine = CombinedSearchEngine(debug=False)

        old_code_git_sha = _get_git_sha_for_path(OLD_CODE_DIR)

        for query in QUERY_SET:
            slug = _slugify(query)
            print(f"[capture] Query: {query!r} -> {slug}.json")
            captured = capture_query(engine, query)

            out = asdict(captured)
            out["old_code_git_sha"] = old_code_git_sha
            out_path = QUERIES_DIR / f"{slug}.json"
            out_path.write_text(json.dumps(out, indent=2, default=str))

            if captured.raised_exception:
                print(f"    -> raised exception (expected for the crash-probe query): "
                      f"{captured.raised_exception.splitlines()[0]}")
            else:
                print(f"    -> {len(captured.results)} results")
    finally:
        os.chdir(original_cwd)
        sys.path.remove(str(OLD_CODE_DIR))

    print(f"[capture] Done. Golden queries written to {QUERIES_DIR}")


def _get_git_sha_for_path(path: Path) -> str | None:
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT, text=True,
        ).strip()
        return sha or None
    except Exception:
        return None


if __name__ == "__main__":
    main()
