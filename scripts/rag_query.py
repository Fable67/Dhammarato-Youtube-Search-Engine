#!/usr/bin/env python
"""
Typer CLI for querying the RAG pipeline.

Builds a search engine, constructs a RAG pipeline, and generates a grounded
answer to a user query with inline citations.

Usage:
    python scripts/rag_query.py --query "What is right effort?" --top-k 5
    python scripts/rag_query.py -q "What is right effort?" --model google/gemini-2.5-flash
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

REWRITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REWRITE_ROOT))

from config import settings  # noqa: E402
from sanghabot.rag.generator import (  # noqa: E402
    PermanentGenerationError,
    TransientGenerationError,
)
from sanghabot.rag.pipeline import RagPipeline, build_search_engine  # noqa: E402

app = typer.Typer(help="Query the RAG pipeline for grounded Dhamma answers.")


@app.command()
def query(
    query_text: str = typer.Option(
        ...,
        "--query",
        "-q",
        help="The question to ask the RAG system.",
    ),
    top_k: int = typer.Option(
        settings.rag_top_k_results,
        "--top-k",
        help="Number of transcript excerpts to retrieve.",
    ),
    model: str = typer.Option(
        settings.rag_model,
        "--model",
        help="LLM model identifier (e.g. 'google/gemini-2.5-flash').",
    ),
    temperature: float = typer.Option(
        settings.rag_temperature,
        "--temperature",
        help="Sampling temperature [0.0, 2.0].",
    ),
    top_p: float = typer.Option(
        settings.rag_top_p,
        "--top-p",
        help="Nucleus sampling threshold [0.0, 1.0].",
    ),
):
    """
    Query the RAG pipeline and return a grounded answer with citations.

    Retrieves the top-k most relevant transcript excerpts, builds a numbered
    context prompt, and calls the LLM to generate an answer strictly grounded
    in those excerpts. Returns the answer text and a numbered sources list.
    """
    try:
        typer.echo(f"Building search engine...")
        engine = build_search_engine()

        typer.echo(f"Initializing RAG pipeline...")
        pipeline = RagPipeline(engine)

        typer.echo(f"Querying: {query_text!r}")
        typer.echo("")

        result = pipeline.answer(
            query=query_text,
            top_k=top_k,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )

        # Print header
        typer.echo("=" * 70)
        typer.echo(f"QUERY: {result.query}")
        typer.echo(f"MODEL: {result.model}")
        typer.echo(f"SOURCES: {result.num_sources}")
        typer.echo("=" * 70)
        typer.echo("")

        # Print answer
        typer.echo(result.answer)
        typer.echo("")

        # Print sources
        if result.citations:
            typer.echo("=" * 70)
            typer.echo("SOURCES")
            typer.echo("=" * 70)
            for citation in result.citations:
                typer.echo(f"[{citation.number}] {citation.title}")
                typer.echo(f"    Relevance: {citation.percentage_score:.1f}%")
                typer.echo(f"    Timestamp: {citation.timestamp}")
                if citation.url:
                    typer.echo(f"    YouTube: {citation.url}")
                typer.echo(f"    Blog: {citation.blog_url}")
                typer.echo(f"    Chunk ID: {citation.chunk_id}")
                typer.echo("")

    except PermanentGenerationError as e:
        typer.echo(f"Error: Permanent generation failure: {e}", err=True)
        raise typer.Exit(code=1)
    except TransientGenerationError as e:
        typer.echo(
            f"Error: Transient generation failure (retried and exhausted): {e}",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
