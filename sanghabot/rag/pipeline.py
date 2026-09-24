"""
RAG-based answer generation: retrieval, context building, and generation pipeline.

Combines search results into a grounded prompt context and calls the LLM
to generate answers strictly from the retrieved excerpts.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings
from sanghabot.engine import CombinedSearchEngine
from sanghabot.models import SearchResult
from sanghabot.rag.generator import generate_answer
from sanghabot.rag.prompts import SYSTEM_PROMPT


@dataclass
class Citation:
    """One numbered source citation shown to the user alongside the generated answer."""

    number: int  # 1-indexed, matches the [N] marker used in the prompt/answer
    title: str
    url: str | None  # YouTube URL if available
    blog_url: str
    timestamp: str  # HH:MM:SS estimate, derived from start_char
    percentage_score: float
    chunk_id: str


@dataclass
class RagAnswer:
    query: str
    answer: str  # raw text returned by the LLM (with [N] markers)
    citations: list[Citation]
    model: str
    num_sources: int


def estimate_timestamp(start_char: int | None) -> str:
    """
    Estimates a video timestamp in HH:MM:SS format from a character position.

    Uses the same empirical heuristic as sanghabot/bot/discord_bot.py's
    perform_search() (lines 297-301): dividing start_char by 17.0, which
    was found empirically to map transcript character positions to seconds
    across this project's corpus. This constant is kept in sync deliberately;
    if the heuristic is updated in discord_bot.py, this should be updated too.

    Args:
        start_char: Character position in the transcript (0-indexed), or None.

    Returns:
        Formatted timestamp string, e.g. "01:23:45" (HH:MM:SS).
    """
    estimated_seconds = int((start_char or 0) / 17.0)
    hours = estimated_seconds // 3600
    minutes = (estimated_seconds % 3600) // 60
    secs = estimated_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_context_block(results: list[SearchResult]) -> tuple[str, list[Citation]]:
    """
    Builds a numbered context block and corresponding citations from search results.

    Formats the results as a numbered list (1-indexed) suitable for inclusion in
    a user prompt, with each result showing its title, relevance, and raw transcript
    excerpt (NOT the summary, which should not be cited as if it were the original
    text). Returns both the formatted context string and a parallel list of Citation
    objects with matching 1-indexed numbers.

    Args:
        results: Ranked list of SearchResult objects from the search engine.

    Returns:
        A tuple of (context_block: str, citations: list[Citation]).
        context_block is ready to be embedded in a user prompt.
        citations are parallel to results[i] with 1-indexed number field.
    """
    if not results:
        return "", []

    context_lines = []
    citations = []

    for i, result in enumerate(results, start=1):
        # Add numbered block to context
        context_lines.append(f"[{i}] Title: {result.title}")
        context_lines.append(f"Relevance: {result.percentage_score:.1f}%")
        context_lines.append("Transcript excerpt:")
        context_lines.append(result.text)
        context_lines.append("---")

        # Build parallel citation
        citation = Citation(
            number=i,
            title=result.title,
            url=result.url,
            blog_url=result.blog_url,
            timestamp=estimate_timestamp(result.start_char),
            percentage_score=result.percentage_score,
            chunk_id=result.chunk_id,
        )
        citations.append(citation)

    context_block = "\n".join(context_lines)
    return context_block, citations


def build_user_prompt(query: str, context_block: str) -> str:
    """
    Combines a query and numbered context block into a complete user-role message.

    Args:
        query: The user's question.
        context_block: Formatted numbered excerpts (from build_context_block).

    Returns:
        A complete user prompt ready to pass to the LLM.
    """
    return (
        f"Question: {query}\n\n"
        f"Numbered transcript excerpts:\n\n"
        f"{context_block}\n\n"
        f"Answer the question above using ONLY the numbered excerpts, "
        f"following the citation and grounding rules you were given."
    )


class RagPipeline:
    """
    End-to-end RAG pipeline: search, context building, and answer generation.

    Wraps a CombinedSearchEngine and calls it to retrieve relevant excerpts,
    builds a grounded prompt, and calls the LLM to generate a citation-aware answer.
    """

    def __init__(self, search_engine: CombinedSearchEngine):
        """
        Initialize the RAG pipeline with a search engine.

        Args:
            search_engine: A CombinedSearchEngine instance (dependency-injected,
                not constructed here).
        """
        self.search_engine = search_engine

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> RagAnswer:
        """
        Generate an answer to a query using RAG (retrieval + generation).

        Calls search_engine.search() to retrieve top_k results, builds a numbered
        context block and prompt, calls generate_answer() with the LLM, and returns
        a RagAnswer with citations. If no results are found, returns a short "no
        sources found" message without calling the LLM (to save an API call).

        Args:
            query: The user's question.
            top_k: Number of results to retrieve. Defaults to settings.rag_top_k_results.
            model: Model identifier (e.g. "google/gemini-2.5-flash").
                Defaults to settings.rag_model.
            temperature: Sampling temperature [0.0, 2.0].
                Defaults to settings.rag_temperature.
            top_p: Nucleus sampling [0.0, 1.0]. Defaults to settings.rag_top_p.

        Returns:
            RagAnswer with answer text and citations.

        Raises:
            PermanentGenerationError: If the LLM call fails permanently (auth, bad request, etc.).
            TransientGenerationError: If the LLM call fails transiently (rate limit, 5xx, timeout).
                These are retried internally by generate_answer(); if all retries are exhausted,
                this exception is re-raised.
        """
        k = top_k if top_k is not None else settings.rag_top_k_results
        results = self.search_engine.search(query, top_k=k)

        # Short-circuit if no results: return a brief "no sources" message
        # without calling the LLM.
        if not results:
            return RagAnswer(
                query=query,
                answer="I could not find any matching sources in the library for your query. Please try rephrasing or using different terms.",
                citations=[],
                model=model or settings.rag_model,
                num_sources=0,
            )

        # Build context and citations
        context_block, citations = build_context_block(results)
        user_prompt = build_user_prompt(query, context_block)

        # Call the LLM
        answer_text = generate_answer(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )

        return RagAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            model=model or settings.rag_model,
            num_sources=len(results),
        )


def build_search_engine() -> CombinedSearchEngine:
    """
    Builds a CombinedSearchEngine, replicating the exact logic from discord_bot.py.

    This function mirrors sanghabot/bot/discord_bot.py::build_engine() (lines 57-99).
    It is duplicated here (rather than imported from discord_bot) to keep the RAG
    module independent of the Discord bot. If the engine construction logic changes
    in discord_bot.py, this function should be updated to match.

    The key logic: if the active embedding run's dimension equals LEGACY_TARGET_DIM
    (512), use the legacy query-embedding pipeline (query expansion + averaging) for
    compatibility with that run's vectors. Otherwise, use the standard embedding client.

    Returns:
        A fully-constructed CombinedSearchEngine ready to call .search().
    """
    # Import here to avoid circular imports and to defer loading heavy
    # dependencies (faiss, bm25, etc.) until this function is actually called.
    from sanghabot.embeddings.client import embed_query
    from sanghabot.embeddings.legacy_compat import LEGACY_TARGET_DIM, legacy_embed_query
    from sanghabot.search.bm25 import BM25SearchEngine
    from sanghabot.search.semantic import SemanticSearchEngine
    from sanghabot.storage.db import Database

    database = Database(settings.db_path)

    # The FAISS index must be queried with vectors from the SAME embedding
    # pipeline it was built from, or scores are meaningless (see
    # sanghabot/embeddings/legacy_compat.py). Until REWRITE_PLAN.md Section
    # 11's full re-embed happens and a new run is promoted to active, the
    # active run's vectors are (partly or wholly) products of the legacy
    # pipeline (query expansion + 1024-dim native embed + average-pool to
    # 512) -- so we must use that exact query-embedding pipeline, not a
    # plain native embed_query() call, whenever the active run's stored
    # DIMENSION is the legacy 512 target dimension.
    #
    # Gated on dimension (LEGACY_TARGET_DIM), not an exact run_id string
    # match: run_ids are expected to keep evolving as new content is
    # appended on top of the legacy vector space (e.g.
    # "qwen3-4b-512-legacy-plus-<date>", produced by
    # scripts/ingest_new_blogs.py splicing newly-embedded chunks -- via
    # the SAME legacy averaging pipeline, for vector-space compatibility --
    # onto the original legacy embeddings.npy) well before REWRITE_PLAN.md
    # Section 11's full native re-embed happens. Any run whose dimension is
    # NOT the legacy 512 is, by construction, a clean native run and uses
    # the standard client.
    active_run = database.get_active_embedding_run()
    if active_run is not None and active_run["dimension"] == LEGACY_TARGET_DIM:
        embed_fn = legacy_embed_query
    else:
        embed_fn = lambda q: embed_query(q, dimension=settings.embedding_dim)  # noqa: E731

    semantic = SemanticSearchEngine(settings.faiss_index_path, database, embed_fn=embed_fn)
    bm25 = BM25SearchEngine(database, settings.bm25_index_path)
    return CombinedSearchEngine(
        semantic_engine=semantic,
        bm25_engine=bm25,
        top_k_per_engine=settings.search_top_k_per_engine,
        rrf_k_penalty=settings.rrf_k_penalty,
    )
