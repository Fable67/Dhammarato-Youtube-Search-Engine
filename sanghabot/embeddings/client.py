"""
The ONE OpenRouter embedding client.

Replaces three divergent copies of `batch_embed_fn_openrouter` found in the
old codebase (AI_Transcripts/embed_videos.py, AI_Transcripts/chunk_videos.py,
AI_Transcripts/discord_bot_channel.py), each with a different signature
and/or a different embedding dimension baked in.

Policy (see REWRITE_PLAN.md Appendix C -- the embedding-dimension
incident): this client ALWAYS requests the target dimension natively from
the API. It never derives a smaller dimension via custom post-processing
(e.g. the old averaging-adjacent-pairs hack in embed_videos.py's
reduce_dimensionality()). Qwen3-Embedding models are MRL-capable and
support requesting dimensions natively from 32 up to their max, which is
both simpler and preserves far more semantic signal than any custom
pooling transform -- and unlike that old transform, it's trivially
reversible: a smaller dimension is always just `vec[:n]` of a larger
native request, never a destructive combination of values.
"""
from __future__ import annotations

import time

import numpy as np
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


class TransientEmbeddingError(Exception):
    """Raised for retryable failures (rate limits, 5xx server errors)."""


class PermanentEmbeddingError(Exception):
    """Raised for non-retryable failures (bad request, auth, malformed response)."""


def _post_embeddings(texts: list[str], model: str, dimension: int, api_key: str) -> list[np.ndarray]:
    try:
        response = requests.post(
            url=OPENROUTER_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "Sanghabot Rewrite",
            },
            json={"model": model, "input": texts, "dimensions": dimension},
            timeout=60,
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        # Connection-level failures (read timeouts, DNS blips, dropped
        # connections) never even produce an HTTP response, so they can't
        # be caught by the status-code checks below. Discovered directly
        # in production: a real `ReadTimeout` on a large batch mid-run
        # propagated straight past this function and crashed a long-running
        # ingestion script, losing all in-memory progress since the last
        # per-item checkpoint. Route these through the exact same
        # TransientEmbeddingError retry path as HTTP 429/5xx -- a timeout
        # or dropped connection is definitionally a transient condition
        # worth retrying, never a reason to give up immediately.
        raise TransientEmbeddingError(f"Connection-level error calling embeddings API: {e}") from e

    if response.status_code == 429 or response.status_code >= 500:
        raise TransientEmbeddingError(
            f"Transient error from embeddings API: {response.status_code} {response.text[:200]}"
        )
    if response.status_code >= 400:
        raise PermanentEmbeddingError(
            f"Embeddings API rejected request: {response.status_code} {response.text[:500]}"
        )

    try:
        payload = response.json()
    except ValueError as e:
        raise PermanentEmbeddingError(
            f"Embeddings response was not valid JSON: {response.text[:500]}"
        ) from e

    # OpenRouter sometimes wraps upstream provider errors (including
    # transient ones like 429 "Model busy, retry later") inside an HTTP 200
    # response with an `error` key in the body, rather than reflecting them
    # in the actual HTTP status code. Discovered directly while testing the
    # standalone rewrite: a plain status-code check let one of these slip
    # through as an unretried PermanentEmbeddingError. Detect this shape
    # explicitly and route it through the same transient-retry path as a
    # real HTTP 429/5xx.
    if "error" in payload:
        error_info = payload["error"]
        error_message = str(error_info.get("message", error_info))
        error_code = error_info.get("code")
        looks_transient = (
            error_code == 429
            or "429" in error_message
            or "busy" in error_message.lower()
            or "overloaded" in error_message.lower()
        )
        if looks_transient:
            raise TransientEmbeddingError(f"Transient error embedded in 200 response: {error_message[:300]}")
        raise PermanentEmbeddingError(f"Embeddings API returned an error payload: {error_message[:500]}")

    try:
        data = payload["data"]
    except KeyError as e:
        raise PermanentEmbeddingError(
            f"Embeddings response missing expected 'data' field: {response.text[:500]}"
        ) from e

    return [np.array(item["embedding"], dtype="float32") for item in data]


@retry(
    retry=retry_if_exception_type(TransientEmbeddingError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def embed_texts(
    texts: list[str],
    dimension: int | None = None,
    model: str | None = None,
) -> list[np.ndarray]:
    """
    Requests `dimension` natively from the embeddings API for every text.

    Only retries on TransientEmbeddingError -- HTTP 429/5xx, an error
    payload smuggled inside an HTTP 200 response, or a connection-level
    failure (timeout, dropped connection, DNS blip) that never produced an
    HTTP response at all. Anything else -- malformed responses, auth
    failures, bad requests -- raises immediately as PermanentEmbeddingError
    instead of silently retrying 6 times with exponential backoff, which is
    what the old code's bare `except Exception` did (wasting up to ~126s
    per row on unrecoverable errors before giving up).
    """
    dim = dimension if dimension is not None else settings.embedding_dim
    mdl = model if model is not None else settings.embedding_model
    if not settings.openrouter_api_key:
        raise PermanentEmbeddingError(
            "OPENROUTER_API_KEY is not set. Check your .env file (see .env.example)."
        )
    return _post_embeddings(texts, model=mdl, dimension=dim, api_key=settings.openrouter_api_key)


def embed_query(query: str, dimension: int | None = None, model: str | None = None) -> np.ndarray:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([query], dimension=dimension, model=model)[0]
