"""
The OpenRouter chat-completions client for RAG-based answer generation.

Mirrors the battle-tested error-handling and retry patterns from
sanghabot/embeddings/client.py, adapted for chat completions. Uses the same
tenacity-based retry strategy to handle transient API failures (rate limits,
5xx errors, connection timeouts) separately from permanent failures (bad
requests, auth, malformed responses).

The key insight from embeddings/client.py that applies equally here:
OpenRouter sometimes wraps upstream provider errors (including transient ones
like "Model busy, retry later") inside an HTTP 200 response with an `error`
key in the body. This function detects that pattern explicitly and retries it,
rather than silently treating it as a permanent failure.
"""
from __future__ import annotations

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class TransientGenerationError(Exception):
    """Raised for retryable failures (rate limits, 5xx server errors, connection timeouts)."""


class PermanentGenerationError(Exception):
    """Raised for non-retryable failures (bad request, auth, malformed response)."""


def _post_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str,
) -> str:
    """
    POST a chat-completion request to OpenRouter and return the generated answer.

    Args:
        system_prompt: System role content (e.g., grounding/instruction rules).
        user_prompt: User role content (the query + retrieved excerpts).
        model: Model identifier (e.g., "google/gemini-2.5-flash").
        temperature: Sampling temperature [0.0, 2.0].
        top_p: Nucleus sampling threshold [0.0, 1.0].
        max_tokens: Maximum tokens to generate.
        api_key: OpenRouter API key for authorization.

    Returns:
        The generated answer string (content from the assistant message).

    Raises:
        TransientGenerationError: On rate-limit, 5xx, connection failure, or
            transient error smuggled inside an HTTP 200 response.
        PermanentGenerationError: On 4xx errors (except 429), malformed
            responses, missing fields, or other unrecoverable failures.
    """
    try:
        response = requests.post(
            url=OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "Sanghabot RAG",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        # Connection-level failures (read timeouts, DNS blips, dropped
        # connections) never even produce an HTTP response, so they can't
        # be caught by the status-code checks below. These are definitionally
        # transient conditions worth retrying, never a reason to give up
        # immediately.
        raise TransientGenerationError(
            f"Connection-level error calling chat-completion API: {e}"
        ) from e

    if response.status_code == 429 or response.status_code >= 500:
        raise TransientGenerationError(
            f"Transient error from chat-completion API: {response.status_code} {response.text[:200]}"
        )
    if response.status_code >= 400:
        raise PermanentGenerationError(
            f"Chat-completion API rejected request: {response.status_code} {response.text[:500]}"
        )

    try:
        payload = response.json()
    except ValueError as e:
        raise PermanentGenerationError(
            f"Chat-completion response was not valid JSON: {response.text[:500]}"
        ) from e

    # OpenRouter sometimes wraps upstream provider errors (including
    # transient ones like 429 "Model busy, retry later") inside an HTTP 200
    # response with an `error` key in the body, rather than reflecting them
    # in the actual HTTP status code. Detect this shape explicitly and route
    # it through the transient-retry path.
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
            raise TransientGenerationError(
                f"Transient error embedded in 200 response: {error_message[:300]}"
            )
        raise PermanentGenerationError(
            f"Chat-completion API returned an error payload: {error_message[:500]}"
        )

    try:
        choices = payload["choices"]
        if not choices:
            raise KeyError("choices array is empty")
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise PermanentGenerationError(
            f"Chat-completion response missing expected 'choices[0].message.content' structure: {response.text[:500]}"
        ) from e

    return content


@retry(
    retry=retry_if_exception_type(TransientGenerationError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def generate_answer(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Generate an answer using a chat-completion model, with exponential backoff retry.

    Only retries on TransientGenerationError -- HTTP 429/5xx, an error payload
    smuggled inside an HTTP 200 response, or a connection-level failure (timeout,
    dropped connection, DNS blip). Anything else raises immediately as
    PermanentGenerationError instead of silently retrying.

    Args:
        system_prompt: System role content (rules/instructions).
        user_prompt: User role content (query + excerpts).
        model: Model identifier. Defaults to settings.rag_model.
        temperature: Sampling temperature [0.0, 2.0]. Defaults to settings.rag_temperature.
        top_p: Nucleus sampling [0.0, 1.0]. Defaults to settings.rag_top_p.
        max_tokens: Maximum tokens to generate. Defaults to settings.rag_max_tokens.

    Returns:
        The generated answer string.

    Raises:
        PermanentGenerationError: If OPENROUTER_API_KEY is not set, or on
            any permanent API failure.
        TransientGenerationError: Never raised directly (retried internally),
            but re-raised after exhausting attempts via reraise=True.
    """
    if not settings.openrouter_api_key:
        raise PermanentGenerationError(
            "OPENROUTER_API_KEY is not set. Check your .env file (see .env.example)."
        )

    mdl = model if model is not None else settings.rag_model
    temp = temperature if temperature is not None else settings.rag_temperature
    p = top_p if top_p is not None else settings.rag_top_p
    tokens = max_tokens if max_tokens is not None else settings.rag_max_tokens

    return _post_chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=mdl,
        temperature=temp,
        top_p=p,
        max_tokens=tokens,
        api_key=settings.openrouter_api_key,
    )
