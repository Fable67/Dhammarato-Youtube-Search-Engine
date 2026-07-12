# Dhammarato YouTube Search Engine (Sanghabot)

A Discord bot that lets people search a large library of Dhamma talk
transcripts using natural language -- ask a question or type a few
keywords, and the bot finds the most relevant moments across thousands of
talks and posts them back with links to the video and full transcript.

This repo is a clean rewrite of an earlier prototype. It's a standalone
project now -- it doesn't depend on any other repo to run.

---

## What it actually does, in plain words

1. **You type something in Discord** -- a question ("what did he say about
   letting go of anger?"), a phrase in quotes, or just a keyword.
2. **The bot decides how to search.** Short keyword? It leans on classic
   keyword search. A full question? It leans on "semantic" search (finds
   passages that mean the same thing, even with different words). Most of
   the time it uses both and blends the results.
3. **It searches two ways at once:**
   - **Keyword search (BM25):** the same kind of ranking search engines
     have used for decades -- finds passages containing your actual words.
   - **Semantic search:** turns your query into a vector of numbers (an
     "embedding") using an AI model, and finds transcript chunks whose
     vectors are closest to it -- i.e. passages that are *about* the same
     thing, not just using the same words.
4. **It combines both result lists** into one ranked list (a technique
   called Reciprocal Rank Fusion), so you get the benefits of both
   approaches.
5. **It replies in Discord** with the top few matches: a link to the video
   (jumping to roughly the right spot), a link to the full blog
   transcript, and the actual passage of text.

All of this happens in well under a second per search, because the search
indexes are loaded into memory once when the bot starts, and never
reloaded from disk while it's running.

---

## Where the data comes from

The searchable content is transcripts of Dhamma talks (originally YouTube
videos), each broken into chunks (a few paragraphs at a time) with an AI
summary. All of that is already prepared and lives in `data/index/`:

- `sanghabot.db` -- a single SQLite database file containing every video's
  title/URL and every transcript chunk's text + summary.
- `faiss.index` -- the semantic search index (pre-computed embeddings for
  every chunk, in a format that supports fast "find similar" lookups).
- `bm25.pkl` -- the keyword search index (pre-computed word statistics for
  every chunk).

This data was migrated from an earlier prototype and is *the* actual
content the bot searches. If it's missing, the bot has nothing to search.

---

## Project layout

```
config.py                  Central settings (reads .env), one source of truth
sanghabot/
  models.py                 Shared data types (a Chunk, a SearchResult, etc.)
  storage/db.py              All database (SQLite) access
  search/
    intent.py                 Decides HOW to search a given query
    fusion.py                  Combines keyword + semantic results into one list
    semantic.py                The semantic (AI embedding) search engine
    bm25.py                     The keyword search engine
  engine.py                  Ties intent + both search engines + fusion together
  embeddings/
    client.py                  Talks to the embedding API (OpenRouter) for new embeddings
    legacy_compat.py            Reproduces the OLD embedding pipeline exactly
                                 (needed because the current index was built
                                 with it -- see "A note on embeddings" below)
  ingest/
    parse_blogs.py              Reads raw blog markdown files into structured data
    chunker.py                   Splits a transcript into semantically coherent chunks
  bot/discord_bot.py          The actual Discord bot (entry point)
scripts/
  migrate_legacy_data.py     One-time script that built data/index/ (already run)
  capture_golden_queries.py  One-time script used during development (already run)
  rebuild_index.py           Rebuilds the search indexes (e.g. after re-embedding)
  ingest.py                  CLI for parsing/chunking new transcripts
data/
  index/                     The actual search data the bot uses (see above)
  raw/blogs/                 Original markdown source files (if present)
tests/                       Automated tests, including a comparison against
                             the old prototype's real output (see REWRITE_PLAN.md)
```

---

## Running it locally (without Docker)

1. Create a Python environment with the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in real values:
   ```
   DISCORD_TOKEN=your-bot-token
   GUILD_ID=your-discord-server-id
   CHANNEL_ID=the-channel-id-the-bot-should-watch
   OPENROUTER_API_KEY=your-openrouter-api-key
   ```
3. Make sure `data/index/sanghabot.db`, `faiss.index`, and `bm25.pkl`
   exist (they should already be present in this repo).
4. Run the bot:
   ```
   python -m sanghabot.bot.discord_bot
   ```

---

## Running it with Docker

The Docker setup here is intentionally a bit different from a "normal"
Dockerfile you might have seen, so it's worth explaining *why*.

**The image does NOT contain the application code.** It only contains
Python and the installed dependencies. The actual code (`sanghabot/`,
`config.py`, `scripts/`), the data (`data/`), and the secrets (`.env`) are
all mounted into the container from your local disk at startup, instead of
being baked into the image with `COPY`.

**Why this is useful:** dependencies (the stuff in `requirements.txt`)
rarely change and take a while to install, so it makes sense to build them
into an image once. Application code changes constantly while you're
developing, though -- if the code were baked into the image, every small
edit would mean rebuilding the whole image just to test it. With the code
mounted in from disk instead, you edit a `.py` file on your machine and the
running container sees the change immediately, because it's reading the
exact same file.

**Important limitation to understand:** mounting the code doesn't make the
bot magically reload itself. Python doesn't watch files for changes. If
you edit code while the bot is running, you still need to restart the
container (`docker compose restart sanghabot`) for the change to take
effect -- you just don't need to *rebuild* the image to do that. A full
rebuild is only needed when `requirements.txt` changes (a new/updated
dependency).

### Using docker-compose (recommended)

```
docker compose up --build
```

This builds the image once (installing dependencies) and starts the bot
with your local `sanghabot/`, `config.py`, `scripts/`, `data/`, and `.env`
all mounted in, exactly as described above. After that, for a normal code
change:

```
docker compose restart sanghabot
```

You only need `docker compose up --build` again if you changed
`requirements.txt`.

### Using plain `docker run` (if you don't want docker-compose)

```
docker build -t sanghabot .

docker run -d \
  --name sanghabot \
  -v "$(pwd)/sanghabot:/app/sanghabot:ro" \
  -v "$(pwd)/config.py:/app/config.py:ro" \
  -v "$(pwd)/scripts:/app/scripts:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.env:/app/.env:ro" \
  sanghabot
```

### If you'd rather have a fully self-contained image instead

If you ever deploy this somewhere you can't easily bind-mount local files
(e.g. a managed container platform), the mount-based approach isn't the
right fit. In that case, add these lines to the Dockerfile's runtime stage
(right before the `USER appuser` line) to bake the code in the traditional
way, and drop the corresponding volumes from `docker-compose.yml`:

```dockerfile
COPY config.py ./
COPY sanghabot/ ./sanghabot/
COPY scripts/ ./scripts/
```

You'd still want to mount `data/` (large data files, not source code) and
`.env` (secrets should never be baked into an image) as volumes even in
that setup.

---

## A note on embeddings (why there's a "legacy_compat.py")

The bot turns your search query into an "embedding" (a list of numbers
representing its meaning) using an AI model, then compares it against the
pre-computed embeddings of every transcript chunk. Those pre-computed
embeddings were generated by an earlier version of this project using a
specific pipeline (a particular model, a particular way of shrinking the
embedding size). For the bot's search to make sense, a new query has to be
turned into a number-vector using that *exact same* pipeline -- otherwise
it's like comparing measurements in different units.

`sanghabot/embeddings/legacy_compat.py` exists to reproduce that original
pipeline exactly, so today's searches stay compatible with the
already-built index. `sanghabot/embeddings/client.py` is the "clean," new
way of generating embeddings, meant for whenever the data gets
regenerated from scratch with better settings (planned -- see
`REWRITE_PLAN.md`, Section 11). The bot automatically figures out at
startup which of the two to use, based on what's recorded in the database.

---

## Testing

Fast tests (no network calls, run in under a second):
```
pytest -m "not slow"
```

There's also a slower "parity" test suite that compares this rewrite's
search results against real output captured from the original prototype,
to make sure the rewrite behaves the same (or better, with differences
explicitly documented). See `tests/golden/KNOWN_DIVERGENCES.md` for a
plain-language account of what was found and fixed while doing that
comparison -- including a real, previously-undiscovered bug in the
original code that this rewrite fixes.

---

## Background / design history

`REWRITE_PLAN.md` in this repo is the original design document written
before this rewrite was built. It goes into a lot more technical detail
about *why* things are structured this way, what was wrong with the
previous version, and what's planned next (e.g. eventually re-generating
all embeddings from scratch at a higher quality setting). Worth a read if
you want the full story, but everything you need to actually run and
maintain the bot day-to-day is in this README.
