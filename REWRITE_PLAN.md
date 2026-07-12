# Sanghabot Rewrite Plan

Status: design complete, implementation not started.
Scope: this rewrite covers only what currently lives in `AI_Transcripts/`.
The root-level legacy code (`WebApp/`, `helpers/`, `search_videos.py`,
`discord_bot_command.py`, `transcribe_videos.py`, `transcripts/`,
`videos*.csv`, `sanghabot.service`) is **deprecated and out of scope**.
Nothing in the existing repo is deleted or modified by this plan — this
directory (`rewrite/`) is the root of a fresh implementation that will
eventually replace `AI_Transcripts/` once verified at parity.

**The current `AI_Transcripts/` code is working and running right now.**
That running system is the ground truth the rewrite must match. This plan
therefore treats parity testing against the old code's actual output —
especially `search.py`'s search results — as a first-class requirement, not
an afterthought. See §9a for the full methodology and §10 step 11 for the
acceptance gate this implies.

**Update, implementation phase:** parity testing against the old code
surfaced a real, previously undiscovered bug in
`AI_Transcripts/semantic_search.py` (see `tests/golden/KNOWN_DIVERGENCES.md`
§0 for full detail) — the old code silently attached the wrong semantic
relevance score to the wrong chunk on essentially every semantic search,
due to a SQL result ordering assumption that didn't hold. Per explicit
instruction, this exact bug (and only this bug) was fixed directly in the
old file so a correct golden baseline could be captured; nothing else in
`AI_Transcripts/` was touched. This is direct, concrete validation of the
concern that motivated §11 (a full re-embed to catch undiscovered issues)
— the process of building this rewrite is already finding real bugs.

---

## 0. Why this rewrite exists

The current `AI_Transcripts/` implementation works, but has accumulated
serious issues found during a full code review:

- A live Discord bot token hardcoded and committed to git.
- Every incoming Discord message reconstructs the entire search engine from
  scratch: rereads a 12MB metadata CSV, unpickles a 55MB BM25 index, and
  reloads a 45MB FAISS index from disk. Confirmed via `search_engine.log`:
  8-14 seconds per query. A commit message even admits this was a deliberate
  (mis-guided) fix for RAM pressure: *"Search engine is not kept in memory
  for discord but but loaded per request."*
- `analyze_query_intent()` and `reciprocal_rank_fusion()` duplicated
  near-verbatim across `keyword_search.py` and `search.py`, with subtly
  different thresholds, and a live crash bug (`NameError: i`) in one branch
  of `search.py` that has no `for` loop despite indexing `final_results[i]`.
- 23,246 loose `chunks/*.txt` files and 2,061 loose `summaries/*.txt` files
  on disk instead of using the sqlite database that already sits right next
  to them.
- Three different, divergent implementations of `batch_embed_fn_openrouter`
  across three files with different signatures and embedding dimensions.
- Bogus/fabricated dependency pins (`pandas==3.0.3`, `Requests==2.34.2` —
  neither version has ever existed).
- No tests (only an ad hoc `test.py` scratch script), no CLI (interactive
  `input()` prompts drive the batch pipelines), broad `except Exception`
  blocks that retry unrecoverable bugs for 126+ seconds before giving up.
- A one-way-door mistake: original embeddings were generated at 1024
  dimensions, then shrunk to 512 dims via a lossy "average adjacent pairs"
  transform, and the 1024-dim originals were overwritten in the process.
  The embedding model (Qwen3-Embedding, MRL-capable) natively supports
  requesting any dimension from 32 up, and also supports simple truncation
  as a lossless-enough shrink method — the custom averaging step was both
  unnecessary and irreversible. This rewrite makes that class of mistake
  structurally impossible (see Appendix C).

None of this is a RAM problem. Loaded-but-idle indices sitting in memory are
fine and cheap on the new machine. The actual fix is: **load each index
exactly once, at process start, into a long-lived singleton, and never touch
disk again in the hot request path.**

---

## 1. Target directory layout

This `rewrite/` directory becomes the new project root. Final shape:

```
rewrite/
├── REWRITE_PLAN.md              # this file
├── requirements.txt
├── requirements-dev.txt         # pytest, ruff, mypy — kept separate from runtime deps
├── .env.example
├── .gitignore                   # data/, *.log, .DS_Store, __pycache__
├── config.py
├── data/
│   ├── raw/blogs/*.md           # tracked — the one irreplaceable input
│   └── index/                   # gitignored, fully rebuildable
│       ├── faiss.index
│       ├── bm25.pkl
│       └── sanghabot.db         # sqlite: metadata + chunk text + summary text + embedding_runs
├── sanghabot/
│   ├── __init__.py
│   ├── models.py                 # dataclasses shared across modules (Chunk, VideoMetadata, SearchResult, QueryIntent)
│   ├── ingest/
│   │   ├── parse_blogs.py        # markdown+frontmatter -> videos table
│   │   └── chunker.py            # ported chunking.py, cleaned up (this file was already good)
│   ├── embeddings/
│   │   ├── client.py              # ONE OpenRouter embed function, retry via tenacity, dimension-aware
│   │   └── build_index.py         # builds faiss + bm25 + populates embedding_runs
│   ├── search/
│   │   ├── base.py                # SearchEngine protocol
│   │   ├── semantic.py            # FAISS engine, index loaded once in __init__
│   │   ├── bm25.py                 # BM25 engine, index loaded once in __init__
│   │   ├── intent.py               # ONE analyze_query_intent(), unit tested
│   │   └── fusion.py                # ONE reciprocal_rank_fusion(), unit tested
│   ├── engine.py                    # CombinedSearchEngine, composed once at startup
│   ├── storage/
│   │   └── db.py                     # sqlite access layer, single connection, WAL mode
│   └── bot/
│       └── discord_bot.py             # thin: on_ready builds engine once; on_message just queries it
├── scripts/
│   ├── ingest.py                  # typer CLI, replaces input()-driven chunk_videos.py
│   └── rebuild_index.py           # typer CLI, replaces embed_videos.py main()
├── tests/
│   ├── conftest.py
│   ├── test_intent.py
│   ├── test_fusion.py
│   ├── test_chunker.py
│   ├── test_db.py
│   ├── test_engine_golden_queries.py
│   ├── golden/
│   │   ├── frozen_data_manifest.json    # checksums of the exact old-code data snapshot used
│   │   └── queries/*.json               # one file per query: input + old-code's full output
│   └── test_parity_old_vs_new.py        # runs the same queries through the new engine, diffs vs golden/
└── Dockerfile
```

---

## 2. Config (`config.py`)

Single `pydantic-settings` `Settings` class, loaded from `.env` (gitignored),
with `.env.example` committed as a template. Replaces every scattered
`open(".openrouterapikey")` / hardcoded relative path in the old codebase.

```python
class Settings(BaseSettings):
    discord_token: str
    guild_id: int
    channel_id: int
    openrouter_api_key: str
    embedding_model: str = "qwen/qwen3-embedding-4b"
    embedding_dim: int = 1024   # new default going forward; see Appendix C

    data_dir: Path = Path("data")
    db_path: Path = data_dir / "index" / "sanghabot.db"
    faiss_index_path: Path = data_dir / "index" / "faiss.index"
    bm25_index_path: Path = data_dir / "index" / "bm25.pkl"

    search_top_k_per_engine: int = 100   # was hardcoded 1000 in old code — needs profiling, see Appendix B
    search_final_k: int = 3
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
```

Every module imports `from config import settings`. No module re-reads
secrets from disk on every call.

---

## 3. Data model (`sanghabot/models.py`)

```python
@dataclass
class Chunk:
    chunk_id: str          # "{video_id}_{chunk_idx}"
    video_id: str
    chunk_idx: int
    text: str
    summary: str | None
    start_char: int
    end_char: int
    embedding_row: int | None   # row index into the active embedding_runs .npy

@dataclass
class VideoMetadata:
    video_id: str
    title: str
    blog_url: str
    published_date: date | None
    tags: list[str]

@dataclass
class QueryIntent:
    use_semantic: bool
    use_bm25: bool
    weights: dict[str, float]
    exact_phrases: list[str]

@dataclass
class SearchResult:
    chunk_id: str
    video_id: str
    title: str
    blog_url: str
    text: str
    summary: str | None
    score: float
    percentage_score: float
    source: Literal["semantic", "bm25", "fusion"]
```

`SearchResult` replaces the ad hoc dicts the old code passed around
(`final_results[i]["score"]` etc.). Typed dataclass fields make the
`NameError: i` bug class structurally harder to reintroduce, and mypy will
flag any missing-field access at review time instead of at 2am in
production.

---

## 4. Storage (`sanghabot/storage/db.py`)

SQLite schema. One file replaces `metadata.csv`, `metadata.db`, and the
23k+2k loose `.txt` files.

```sql
CREATE TABLE videos (
    video_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    blog_url        TEXT NOT NULL,
    published_date  TEXT,
    tags            TEXT            -- JSON array
);

CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(video_id),
    chunk_idx       INTEGER NOT NULL,
    text            TEXT NOT NULL,
    summary         TEXT,
    start_char      INTEGER,
    end_char        INTEGER,
    embedding_row   INTEGER          -- row index into the *active* embedding_runs .npy
);
CREATE INDEX idx_chunks_video_id ON chunks(video_id);

-- see Appendix C for why this table exists and the immutability rule around it
CREATE TABLE embedding_runs (
    run_id        TEXT PRIMARY KEY,   -- e.g. "qwen3-4b-1024-2026-07-09"
    model_name    TEXT NOT NULL,
    dimension     INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    npy_path      TEXT NOT NULL,      -- data/index/embeddings_{run_id}.npy
    is_active     INTEGER DEFAULT 0   -- which run FAISS is currently built from
);
```

`db.py` API surface:

```python
class Database:
    def __init__(self, path: Path): ...            # one connection, opened once, WAL mode
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]: ...  # single WHERE IN batch query
    def get_video(self, video_id: str) -> VideoMetadata: ...
    def insert_video(self, v: VideoMetadata) -> None: ...
    def insert_chunks(self, chunks: list[Chunk]) -> None: ...             # bulk insert
    def iter_chunk_texts(self) -> Iterator[tuple[str, str]]: ...           # streamed, for BM25 corpus build
```

---

## 5. Search layer

**`search/base.py`**
```python
class SearchEngine(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...
```

**`search/semantic.py`**
```python
class SemanticSearchEngine:
    def __init__(self, index_path: Path, db: Database, embed_fn: Callable[[str], np.ndarray]):
        self._index = faiss.read_index(str(index_path))   # loaded ONCE, here, never again
        self._db = db
        self._embed_fn = embed_fn

    def search(self, query: str, k: int) -> list[SearchResult]: ...
        # pure in-memory FAISS search + one batched db.get_chunks_by_ids() call
```

**`search/bm25.py`** — same rule: index built/loaded once in `__init__`,
`search()` never touches disk.

**`search/intent.py`** — the one canonical implementation of query-intent
routing (quoted phrase -> bm25-heavy, question -> semantic-heavy, single
keyword -> bm25-heavy). Replaces the two divergent copies in
`keyword_search.py` and `search.py`.

**`search/fusion.py`** — the one canonical reciprocal rank fusion
implementation, always loop-based over `final_results`, eliminating the
missing-loop crash bug class entirely.

**`engine.py`**
```python
class CombinedSearchEngine:
    def __init__(self, semantic: SemanticSearchEngine, bm25: BM25SearchEngine, db: Database):
        ...  # composed once, at app startup — never reconstructed per request

    def search(self, query: str, top_k: int = settings.search_final_k) -> list[SearchResult]:
        intent = analyze_query_intent(query)
        # dispatch through the single fusion implementation regardless of branch taken
```

---

## 6. Bot (`sanghabot/bot/discord_bot.py`)

```python
engine: CombinedSearchEngine | None = None   # module-level, set exactly once

@bot.event
async def on_ready():
    global engine
    db = Database(settings.db_path)
    semantic = SemanticSearchEngine(settings.faiss_index_path, db, embed_query)
    bm25 = BM25SearchEngine(settings.bm25_index_path, db)
    engine = CombinedSearchEngine(semantic, bm25, db)
    logger.info("Engine loaded, bot ready.")

@bot.event
async def on_message(message):
    ...
    results = engine.search(query)   # zero disk I/O in the hot path — milliseconds, not seconds
```

Discord token comes from `settings.discord_token`, never hardcoded.

---

## 7. Pipeline flow (confirmed with project owner)

```
data/raw/blogs/*.md   (markdown, YAML frontmatter + transcript — the one irreplaceable input)
     │  ingest/parse_blogs.py
     ▼
videos table in sanghabot.db   (one row per video: id, title, transcript, metadata)
     │  scripts/ingest.py  (calls ingest/chunker.py, the ported chunking.py logic)
     ▼
chunks table in sanghabot.db   (one row per chunk: text, boundaries, video_id)
     │  scripts/rebuild_index.py  (calls embeddings/client.py + embeddings/build_index.py)
     ▼
embedding_runs row + embeddings_{run_id}.npy + faiss.index + bm25.pkl
     (chunk text stays in the chunks table — never re-exported to loose files)
```

This is the same three-stage shape as the current pipeline
(`parse_blogs.py` -> `chunk_videos.py` -> `embed_videos.py`) — that shape is
correct and is being kept. What changes is the **medium** data moves through
between stages:

| Stage | Old | New |
|---|---|---|
| parse_blogs output | one big CSV held fully in memory | rows written directly into `videos` table; no giant CSV ever loaded whole |
| chunk_videos | reads CSV fully via pandas, writes another CSV | streams rows from `videos` table via cursor, writes chunks into `chunks` table incrementally |
| embed_videos | reads chunked CSV, embeds, writes `.npy` + loose `.txt` files + `metadata.csv`/`.db` | reads chunk rows from sqlite in bounded batches, embeds, writes vectors to `embeddings_{run_id}.npy`, writes text back into the same `chunks` table it read from — zero loose text files |

Context on why the old code split chunk/summary text into loose files: the
old `embed_videos.py` held a full pandas DataFrame in memory during
embedding generation, and evicted large text columns to disk per-row to
keep that DataFrame's RAM footprint down. That was a real, reasonable
motivation at the time — not just clutter. The new design removes the
underlying problem (nothing ever loads the *entire* corpus into one
in-memory DataFrame; sqlite + bounded batch cursors replace that pattern
outright), so there's no RAM pressure left to work around with
file-spraying in the first place.

---

## 8. Ingestion & indexing scripts

`scripts/ingest.py` (typer CLI, replaces `chunk_videos.py`'s `input()` loop):
```
python scripts/ingest.py chunk --resume
```

`scripts/rebuild_index.py` (replaces `embed_videos.py main()`):
```
python scripts/rebuild_index.py --rebuild-faiss --rebuild-bm25
```

Both use `tenacity` for retry, scoped to specific transient errors only
(`requests.HTTPError` with 429/5xx status). Anything else (`KeyError`,
`TypeError`, programming bugs) fails fast and loud instead of retrying 6
times with exponential backoff (the old code's bare `except Exception`
pattern wasted up to ~126 seconds of sleep per row before giving up on
unrecoverable errors). Checkpointing uses a `progress` table in sqlite
instead of a hand-rolled `progress.json`.

---

## 9. Tests (concrete list)

- `test_intent.py` — quoted phrase routes bm25-heavy; question routes
  semantic-heavy; single keyword routes bm25-heavy; long query (>20 words)
  does not hit a dead/unreachable branch (regression test for the exact
  unreachable-elif bug found in the old code).
- `test_fusion.py` — known rank lists in -> expected fused order out; empty
  result list from one engine doesn't crash; `percentage_score` always
  populated (regression test for the `search.py` `NameError: i` bug).
- `test_chunker.py` — known transcript fixture -> expected chunk boundaries
  (snapshot test).
- `test_db.py` — insert + batch fetch round-trip; `get_chunks_by_ids`
  preserves input order.
- `test_engine_golden_queries.py` — 5-10 real queries pulled from the old
  `search_engine.log`, asserting the same top result `video_id` still comes
  back after any refactor — regression safety net for future reindexing or
  logic changes.

---

## 9a. Parity testing: proving the new code matches the old code

The old implementation is currently working and running in production.
That is the ground truth. The rewrite is not "done" because it looks
cleaner — it is done when it is demonstrated, mechanically, to produce the
same (or better, with any difference explained) results as the code
currently running. This is especially true for `search.py`'s output, since
that is the exact user-facing behavior of the bot today.

**This is not optional and not a one-time sanity check.** It is a
first-class artifact of the rewrite: a frozen snapshot of real queries and
the old system's real output, checked into `tests/golden/`, that the new
engine must reproduce (within an explicitly defined tolerance) before the
old code is allowed to be retired.

### C.1 Capture ground truth from the currently-running old code

Before writing a single line of the new `search`/`engine` modules, run a
capture pass against the **existing, untouched** `AI_Transcripts/search.py`
(`CombinedSearchEngine`) exactly as it runs today:

1. Freeze the data snapshot: record checksums (`sha256`) of
   `dhammarato_transcripts_chunked.csv`, `embeddings/metadata.csv`,
   `embeddings/faiss.index`, `embeddings/bm25+_index.pkl`, and
   `embeddings/embeddings.npy` at the moment of capture, into
   `tests/golden/frozen_data_manifest.json`. The new engine must be tested
   against a database/index built from this exact same frozen snapshot —
   otherwise a mismatch could just mean "the underlying data changed," not
   "the new code is wrong."
2. Assemble a query set covering the full behavior space, not just happy
   paths:
   - 15-20 real historical queries pulled verbatim from `search_engine.log`
     (actual user queries, not synthetic ones).
   - At least one query per `analyze_query_intent()` branch: quoted-phrase
     query, question-form query, single-keyword query, long (>20 word)
     query, and a query intentionally chosen to hit the *fixed* crash bug
     path (`use_semantic` branch in the old `search.py:247-249`) — this one
     is expected to have thrown `NameError` in old code, so its "ground
     truth" is explicitly the exception, not a result list. Capture that a
     production bug exists rather than silently working around it.
   - A handful of edge cases: empty string, single character, non-English
     text if the bot supports it, a query matching zero results.
3. For every query in the set, run it through the **actual currently
   running** `CombinedSearchEngine.search()` (old code, untouched, in
   `AI_Transcripts/search.py`) and record the full raw output — not just
   top-1 video_id, but the entire ordered result list with every field
   (`score`, `percentage_score`, `text`, `summary`, `video_id`, source
   engine, everything `search()` currently returns) — into
   `tests/golden/queries/<slugified_query>.json`:

```json
{
  "query": "how do I deal with anxiety during meditation",
  "captured_at": "2026-07-09T23:50:00Z",
  "old_code_git_sha": "<commit hash of AI_Transcripts/search.py at capture time>",
  "expected_intent": {"use_semantic": true, "use_bm25": true, "weights": {"semantic": 0.7, "bm25": 0.3}},
  "results": [
    {"chunk_id": "...", "video_id": "...", "score": 0.8123, "percentage_score": 81.23, "text": "...", "summary": "...", "source": "fusion"},
    ...
  ],
  "raised_exception": null
}
```

   A small capture script (`scripts/capture_golden_queries.py`, run once
   against the old code, never against the new code) produces these files.
   It imports `AI_Transcripts.search.CombinedSearchEngine` directly and
   calls `.search()` — it does not go through Discord, to keep capture fast
   and deterministic.

### C.2 Define the comparison rule ("same output" precisely)

Exact floating-point equality is too strict for anything involving FAISS/
BM25 recomputation paths and too easy to accidentally satisfy by literally
reusing the old code. The parity test (`test_parity_old_vs_new.py`)
compares new-engine output to the captured golden file per query with
these explicit rules:

- **Result identity and order**: the ordered list of `chunk_id`s returned
  must match exactly for the top `search_final_k` (currently 3) results,
  unless a difference is explicitly annotated and justified in the golden
  file (e.g., a documented old-code bug being intentionally fixed).
- **Scores**: `percentage_score` must match within a tolerance of `1e-6`
  relative error — not bit-exact, since RRF math may be re-expressed, but
  functionally identical.
- **Intent routing**: `analyze_query_intent()` output (`use_semantic`,
  `use_bm25`, weights) must match exactly per query, since this determines
  *which* branch of behavior is exercised at all.
- **Known bug divergence is allowed, but must be explicit and documented,
  never silent.** The one query designed to hit the old `NameError: i`
  crash is asserted to have raised in the golden capture, and asserted to
  **succeed** (return a valid result list) in the new engine — with a
  comment in the test pointing at this plan's §0 writeup of the bug. Any
  other divergence between old and new output must be added to a
  `tests/golden/KNOWN_DIVERGENCES.md` file with a one-line justification
  before the parity test is allowed to mark it as an expected-pass rather
  than a failure.

### C.3 Running the parity suite

```
pytest tests/test_parity_old_vs_new.py -v
```

This test builds the new engine (`CombinedSearchEngine` from
`sanghabot/engine.py`) against a `sanghabot.db`/`faiss.index`/`bm25.pkl`
that were migrated/rebuilt from the exact frozen snapshot recorded in
`frozen_data_manifest.json` (checksums re-verified at test start — if the
source data files don't match the manifest, the test errors out loudly
instead of silently comparing against stale or drifted data). It then runs
every query in `tests/golden/queries/*.json` through the new engine and
applies the C.2 comparison rules.

### C.4 When parity is required, and what happens on failure

- Parity capture (`scripts/capture_golden_queries.py` run against old code)
  happens **once, early** — ideally right after rollout step 1
  (token rotation), before any new search code is written, so the ground
  truth reflects the currently-running system, not a moving target.
- The parity suite is run continuously during rewrite steps 5-7 (intent/
  fusion, semantic/bm25 engines, bot wiring) — every change to search logic
  re-runs it, the same way a unit test suite would.
- A failing parity test blocks progress: either the new code has a genuine
  bug (fix it), or the divergence is intentional (document it in
  `KNOWN_DIVERGENCES.md` and update the test's expectation explicitly —
  never just delete/loosen an assertion to make it pass).
- **The old `AI_Transcripts/` code is not retired (rollout step 10) until
  the full parity suite passes with zero undocumented divergences.** This
  is the actual acceptance criterion for the rewrite being "done," not
  "the new code looks nicer" or "it seems to work in manual testing."

---

## 10. Rollout steps

1. Rotate the Discord bot token (independent of code); new token only ever
   lives in `.env`, never in git.
2. **Capture golden parity data now, against the still-untouched old
   code** (§9a.C.1): run `scripts/capture_golden_queries.py` against the
   current, working `AI_Transcripts/search.py`, freeze the data manifest,
   and commit `tests/golden/` before any new search code exists. This is
   the ground truth every later step gets checked against.
3. Scaffold the `rewrite/` package skeleton (empty modules + signatures
   per this doc).
4. Write `storage/db.py` + a one-off migration script that imports the
   existing `metadata.csv`/`metadata.db` + loose `embeddings/chunks|
   summaries/*.txt` files from `AI_Transcripts/` into `data/index/
   sanghabot.db`. Verify row counts match (23,246 chunks, 2,061 summaries)
   before trusting the migration.
5. Port `chunking.py` -> `ingest/chunker.py`, largely unchanged (it was
   already the best-written file in the old codebase).
6. Implement `search/intent.py` + `search/fusion.py` with unit tests
   passing first, then run the parity suite (§9a) against them.
7. Implement `semantic.py` / `bm25.py` with strict load-once discipline;
   verify the latency drop manually (old baseline: 8-14s/query from
   `search_engine.log`) and re-run the parity suite.
8. Wire `engine.py` + `bot/discord_bot.py`; run the full parity suite
   end-to-end through `CombinedSearchEngine.search()` before any smoke
   test against a dev Discord server/guild, and before touching the
   production bot.
9. Write `requirements.txt` with real, verified version pins (fixing the
   old fictional `pandas==3.0.3` / `Requests==2.34.2` pins).
10. Write a new `Dockerfile` for the `rewrite/` entrypoint.
11. **Gate: the parity suite (§9a) must pass with zero undocumented
    divergences from the golden captures.** Only once this holds, and the
    new implementation has also run stably in a dev/staging environment,
    is retiring the old `AI_Transcripts/` implementation even considered.
    Deletion of old code is an explicit separate decision, not part of
    this plan, and has not been approved yet.
12. **Only after step 11 passes**: perform the full from-scratch re-embed
    to 1024 dimensions described in §11 below. This is intentionally
    sequenced after parity is proven, not bundled into it — see §11 for
    why, and for the separate validation approach this step requires
    (parity vs. old code and correctness of fresh embeddings are two
    different questions, checked two different ways).

---

## 11. Full re-embed to 1024 dimensions (post-parity)

Once the rewrite has passed the parity gate (step 11 above) against the
*migrated legacy data* (embeddings as they exist today, 512-dim, averaging-
hack origin), a separate, deliberate follow-up is planned: **re-embed every
chunk from scratch at 1024 dimensions**, using the existing chunk
boundaries/text as-is (chunking itself is not being redone — see scope
note below).

### 11.1 Why this is being done, and why it's sequenced after parity

Two independent motivations:

1. **The embedding-dimension incident (Appendix C)**: the current 512-dim
   vectors are the output of a lossy, irreversible averaging transform, not
   a clean native embedding. Re-embedding at 1024 natively replaces them
   with a correct artifact instead of continuing to build on the damaged
   one.
2. **Suspected hidden bug**: it's possible something was silently wrong
   somewhere earlier in the historical pipeline (embedding generation,
   batching, retry logic, or the migration between old CSV/pickle formats
   over time) that has never been surfaced. A full from-scratch regeneration
   of every embedding is one of the most effective ways to surface such a
   bug, because it removes every opportunity for a historical mistake to be
   silently carried forward by a migration or copy step.

This is sequenced **after** the parity gate (§9a, step 11) rather than
folded into the initial build, on purpose: parity testing answers "is the
new code correct, given the same data the old code used?" Re-embedding
answers a different question — "is the data itself correct?" Bundling both
into one step would make it impossible to tell, if something looked wrong,
whether the *code* or the *data* was the cause. Keeping them sequential
means a parity failure can only mean a code bug (data is held constant),
and any issue found during/after re-embedding can only mean a data or
embedding-pipeline issue (code is already proven correct by that point).

### 11.2 Scope

- **In scope**: regenerating embedding vectors for all ~23,246 existing
  chunks, at 1024 dimensions, via `sanghabot/embeddings/client.py`, using
  the current chunk text and boundaries as-is.
- **Out of scope for this step**: re-running the chunker
  (`ingest/chunker.py`) against the raw markdown. Chunk boundaries/text are
  reused unchanged. If the dry run or full re-embed surfaces evidence that
  chunk *boundaries* (not just embeddings) are suspect, that's a separate
  follow-up decision, not something this step silently expands to cover.

### 11.3 Step 1 — small-sample dry run (required before the full run)

Given real API cost/time for ~23k chunks, a dry run happens first:

1. Select a sample of 50-100 chunks — not random: deliberately include a
   mix of (a) typical chunks, (b) the shortest chunks in the corpus, (c)
   the longest chunks in the corpus, (d) chunks from videos with unusual
   characters/non-English content if any exist, and (e) chunks that were
   flagged in the "previous text/summary migration logic" compatibility
   shim in the old `embed_videos.py` — those are the most likely spot for a
   historical bug to be hiding.
2. Run them through the new `embeddings/client.py` requesting 1024 dims
   natively (no averaging, no post-processing — per Appendix C policy).
3. Validate the dry-run output before proceeding to the full run:
   - Every returned vector has exactly 1024 dimensions, no `NaN`/`Inf`
     values, and non-zero norm.
   - Chunk count in == embedding count out (no silently dropped rows).
   - Sanity-check semantic behavior: pick 2-3 known-similar chunk pairs
     (e.g. two chunks from the same talk on the same topic) and 2-3
     known-dissimilar pairs, compute cosine similarity, and confirm
     similar pairs score meaningfully higher than dissimilar pairs. This
     is a smoke test for "the embedding pipeline is doing something
     sensible," not a formal quality benchmark.
   - Compare the dry-run sample's nearest-neighbor search results (using
     the new 1024-dim vectors) against the same chunks' results under the
     existing 512-dim vectors. Large, inexplicable divergences for
     unremarkable chunks are a signal to investigate before committing to
     the full run — this is the concrete mechanism for catching the
     "hidden bug" concern, applied cheaply on a small sample first.
4. Only proceed to the full run once the dry run passes these checks.

### 11.4 Step 2 — full re-embed

1. Run `scripts/rebuild_index.py --embed-only --dimension 1024` (or
   equivalent) against all chunks, writing to a **new** `.npy` file and a
   **new** `embedding_runs` row (per Appendix C policy — never overwritten
   in place). The existing 512-dim run stays on disk, `is_active = 0`,
   untouched.
2. Rebuild `faiss.index` from the new 1024-dim run.
3. Re-run the full parity suite's *structural* checks (embedding count
   matches chunk count, no NaN/zero vectors across the whole corpus, no
   crashes) plus a spot check of the golden query set from §9a: results
   are not expected to be byte-identical to the old 512-dim golden capture
   (the whole point is the vectors are different/better now), but the
   **top-1 result per golden query should remain topically/semantically
   plausible** — a human review pass over the golden query set's new
   top-3 results, flagging anything that looks obviously wrong (a sign the
   suspected hidden bug was real and just found).
4. Only after this human review pass is clean does the new 1024-dim run
   get marked `is_active = 1` and become the run FAISS/BM25/the bot
   actually serve from.
5. Document actual API cost and wall-clock time spent in this step's
   commit message / a short note in `data/index/` — useful reference for
   estimating cost if a re-embed is ever needed again.

---

## Appendix A: Deprecated / out of scope

The following existing paths are deprecated and are **not** touched, ported,
or deleted by this plan: `WebApp/`, `helpers/`, `search_videos.py`,
`discord_bot_command.py`, `transcribe_videos.py`, `transcripts/`,
`videos.csv`, `videos_transcribed.csv`, `videos_transcribed_chunked.csv`,
`sanghabot.service` (old systemd unit, pointed at the old path).

## Appendix B: Open parameters to validate empirically

- `search_top_k_per_engine`: old code hardcoded `1000` per engine before
  RRF trimmed to 3. New default is `100`, but this needs a profiling pass
  against the golden-query test set to confirm ranking quality doesn't
  regress before shipping — not a guess to leave unchecked.
- FAISS index type: `IndexFlatIP` remains fine at the current ~23k chunk
  scale. Switching to `IndexHNSWFlat`/`IndexIVFFlat` is a lever for later,
  only if corpus size grows substantially — not needed now.

## Appendix C: The embedding dimension incident, and the policy it produced

**What happened:** embeddings were originally generated at 1024 dimensions.
When RAM pressure came up on a different machine, dimensionality was
reduced to 512 by averaging adjacent value pairs — a custom, lossy,
non-reversible transform — and the result overwrote the original 1024-dim
embeddings. The 1024-dim originals no longer exist; recovering them requires
a full re-embed of all ~23k chunks (real API cost, real time).

**Why it was avoidable:** Qwen3-Embedding models are MRL-capable ("supports
user-defined output dimensions ranging from 32 to 2560" per the model
card). This means the model can be asked directly for a smaller native
dimension, or a stored high-dimension vector can simply be truncated
(`emb[:, :512]`), and either approach preserves semantic quality far better
than averaging adjacent pairs — because MRL training specifically front-loads
importance into earlier dimensions, while pairwise averaging corresponds to
no meaningful direction in the embedding space at all. The custom
dimensionality-reduction function was well-written and well-documented, it
just solved a problem the vendor had already solved better, and did so
irreversibly.

**Policy going forward:**
1. Embeddings are never mutated or overwritten in place. Every embedding
   artifact is identified by `(model_name, dimension, generated_at)` via the
   `embedding_runs` table and treated as immutable once written.
2. Dimensionality changes are new artifacts, not in-place transforms.
   Reducing dimension, if ever needed again, means slicing a stored native
   vector (`emb[:, :n]`) into a *new* `.npy` file registered as a new
   `embedding_runs` row — the original is never deleted.
3. New embedding runs default to requesting 1024 dimensions natively from
   the API (see `config.py`), not derived via any custom post-processing.
4. The existing 512-dim embeddings (the averaging-hack output) are migrated
   as-is for the initial rewrite and parity-testing phase (§9a) — registered
   as an `embedding_runs` row explicitly flagged in a code comment as
   legacy/lossy in origin, so future maintainers don't mistake it for a
   clean native-512 request. This is intentionally temporary: §11 covers a
   planned full re-embed to 1024 dimensions, sequenced after the parity
   gate passes, both to replace this lossy artifact and to check for any
   other undiscovered issues in the historical data.
