"""
Tests for sanghabot.embeddings.client -- the OpenRouter embedding client.

Covers the retry-classification logic in _post_embeddings()/embed_texts():
HTTP 429/5xx, error payloads smuggled inside HTTP 200, and (the fix this
file specifically regression-tests) raw connection-level failures
(requests.exceptions.Timeout / ConnectionError) that never produce an HTTP
response at all, and so can't be caught by a status-code check.

Regression context: a real `requests.exceptions.ReadTimeout` during a
production ingestion run propagated straight past the old version of this
client (which only retried TransientEmbeddingError, raised solely from
HTTP-level status codes), crashing a long-running script and losing all
in-memory progress since the last per-item DB checkpoint. See
scripts/update_blogs.py's module docstring for the full incident writeup.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from sanghabot.embeddings.client import (
    PermanentEmbeddingError,
    TransientEmbeddingError,
    embed_texts,
)


def _fake_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = str(json_body)
    return resp


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    # embed_texts() raises PermanentEmbeddingError immediately if no API
    # key is configured -- give every test in this module a fake one so
    # that check passes and we reach the actual HTTP-call logic being
    # tested (which is itself mocked, so no real network call happens).
    monkeypatch.setattr("sanghabot.embeddings.client.settings.openrouter_api_key", "fake-key-for-tests")


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # embed_texts() is decorated with tenacity's wait_exponential (min=2s,
    # max=60s) -- without this, exercising its multi-attempt retry paths
    # for real would take over a minute per test. Tenacity calls
    # time.sleep() internally to wait between attempts; patching it out
    # keeps these tests in the fast (`pytest -m "not slow"`) loop while
    # still exercising the exact same retry/attempt-count logic.
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


def test_embed_texts_returns_vectors_on_success():
    ok_response = _fake_response(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    with patch("sanghabot.embeddings.client.requests.post", return_value=ok_response) as mock_post:
        result = embed_texts(["hello"], dimension=3)
    assert len(result) == 1
    assert np.allclose(result[0], [0.1, 0.2, 0.3])
    mock_post.assert_called_once()


def test_embed_texts_retries_then_succeeds_on_read_timeout():
    """
    The core regression test: a raw requests.exceptions.ReadTimeout (no
    HTTP response at all) must be retried, not propagate immediately.
    """
    ok_response = _fake_response(200, {"data": [{"embedding": [1.0, 2.0]}]})
    side_effects = [
        requests.exceptions.ReadTimeout("simulated read timeout"),
        ok_response,
    ]
    with patch("sanghabot.embeddings.client.requests.post", side_effect=side_effects) as mock_post:
        result = embed_texts(["hello"], dimension=2)
    assert len(result) == 1
    assert np.allclose(result[0], [1.0, 2.0])
    assert mock_post.call_count == 2


def test_embed_texts_retries_on_connection_error():
    ok_response = _fake_response(200, {"data": [{"embedding": [5.0]}]})
    side_effects = [
        requests.exceptions.ConnectionError("simulated dropped connection"),
        ok_response,
    ]
    with patch("sanghabot.embeddings.client.requests.post", side_effect=side_effects) as mock_post:
        result = embed_texts(["hello"], dimension=1)
    assert len(result) == 1
    assert mock_post.call_count == 2


def test_embed_texts_gives_up_after_max_attempts_on_persistent_timeout():
    with patch(
        "sanghabot.embeddings.client.requests.post",
        side_effect=requests.exceptions.ReadTimeout("always times out"),
    ) as mock_post:
        with pytest.raises(TransientEmbeddingError):
            embed_texts(["hello"], dimension=2)
    # embed_texts() is decorated with stop_after_attempt(6).
    assert mock_post.call_count == 6


def test_embed_texts_retries_on_http_429():
    ok_response = _fake_response(200, {"data": [{"embedding": [9.0]}]})
    rate_limited = _fake_response(429, {"error": "rate limited"})
    with patch(
        "sanghabot.embeddings.client.requests.post",
        side_effect=[rate_limited, ok_response],
    ) as mock_post:
        result = embed_texts(["hello"], dimension=1)
    assert len(result) == 1
    assert mock_post.call_count == 2


def test_embed_texts_does_not_retry_on_permanent_client_error():
    bad_request = _fake_response(400, {"error": {"message": "bad request"}})
    with patch("sanghabot.embeddings.client.requests.post", return_value=bad_request) as mock_post:
        with pytest.raises(PermanentEmbeddingError):
            embed_texts(["hello"], dimension=1)
    mock_post.assert_called_once()


def test_embed_texts_retries_on_transient_error_payload_inside_http_200():
    busy = _fake_response(200, {"error": {"message": "Model busy, retry later", "code": 429}})
    ok_response = _fake_response(200, {"data": [{"embedding": [7.0]}]})
    with patch(
        "sanghabot.embeddings.client.requests.post",
        side_effect=[busy, ok_response],
    ) as mock_post:
        result = embed_texts(["hello"], dimension=1)
    assert len(result) == 1
    assert mock_post.call_count == 2
