"""
Tests for sanghabot.rag.generator -- the OpenRouter chat-completion client for RAG.

Covers the retry-classification logic in _post_chat_completion()/generate_answer():
HTTP 429/5xx, error payloads smuggled inside HTTP 200, connection-level failures
(requests.exceptions.ConnectionError / Timeout), and malformed response structures.

Mirrors the test patterns from test_embeddings_client.py, using the same
tenacity-based retry decorators and transient/permanent error classification.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from sanghabot.rag.generator import (
    PermanentGenerationError,
    TransientGenerationError,
    _post_chat_completion,
    generate_answer,
)


def _fake_response(status_code: int, json_body: dict) -> MagicMock:
    """Create a mock HTTP response with the given status code and JSON body."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = str(json_body)
    return resp


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    # generate_answer() raises PermanentGenerationError immediately if no API
    # key is configured -- give every test in this module a fake one so
    # that check passes and we reach the actual HTTP-call logic being
    # tested (which is itself mocked, so no real network call happens).
    monkeypatch.setattr("sanghabot.rag.generator.settings.openrouter_api_key", "fake-key-for-tests")


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # generate_answer() is decorated with tenacity's wait_exponential (min=2s,
    # max=60s) -- without this, exercising its multi-attempt retry paths
    # for real would take over a minute per test. Tenacity calls
    # time.sleep() internally to wait between attempts; patching it out
    # keeps these tests in the fast (`pytest -m "not slow"`) loop while
    # still exercising the exact same retry/attempt-count logic.
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


def test_post_chat_completion_returns_content_on_success():
    """Successful 200 response with valid choices[0].message.content structure."""
    ok_response = _fake_response(200, {
        "choices": [
            {
                "message": {
                    "content": "The answer is 42."
                }
            }
        ]
    })
    with patch("sanghabot.rag.generator.requests.post", return_value=ok_response):
        result = _post_chat_completion(
            system_prompt="You are helpful.",
            user_prompt="What is the answer?",
            model="google/gemini-2.5-flash",
            temperature=0.1,
            top_p=0.95,
            max_tokens=1500,
            api_key="fake-key",
        )
    assert result == "The answer is 42."


def test_post_chat_completion_raises_transient_on_http_429():
    """HTTP 429 (rate limited) raises TransientGenerationError."""
    rate_limited = _fake_response(429, {"error": "rate limited"})
    with patch("sanghabot.rag.generator.requests.post", return_value=rate_limited):
        with pytest.raises(TransientGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_transient_on_http_500():
    """HTTP 500 (server error) raises TransientGenerationError."""
    server_error = _fake_response(500, {"error": "internal server error"})
    with patch("sanghabot.rag.generator.requests.post", return_value=server_error):
        with pytest.raises(TransientGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_permanent_on_http_400():
    """HTTP 400 (bad request) raises PermanentGenerationError (not retried)."""
    bad_request = _fake_response(400, {"error": {"message": "bad request"}})
    with patch("sanghabot.rag.generator.requests.post", return_value=bad_request):
        with pytest.raises(PermanentGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_transient_on_connection_error():
    """Connection error (no HTTP response) raises TransientGenerationError."""
    with patch(
        "sanghabot.rag.generator.requests.post",
        side_effect=requests.exceptions.ConnectionError("dropped connection"),
    ):
        with pytest.raises(TransientGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_transient_on_timeout():
    """Timeout error raises TransientGenerationError."""
    with patch(
        "sanghabot.rag.generator.requests.post",
        side_effect=requests.exceptions.Timeout("read timeout"),
    ):
        with pytest.raises(TransientGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_transient_on_error_payload_with_overloaded():
    """200 response with error payload containing 'overloaded' is transient."""
    error_response = _fake_response(200, {
        "error": {
            "message": "Model overloaded, please retry",
            "code": "model_overloaded"
        }
    })
    with patch("sanghabot.rag.generator.requests.post", return_value=error_response):
        with pytest.raises(TransientGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_transient_on_error_payload_with_busy():
    """200 response with error payload containing 'busy' is transient."""
    error_response = _fake_response(200, {
        "error": {
            "message": "Model busy, retry later",
            "code": 429
        }
    })
    with patch("sanghabot.rag.generator.requests.post", return_value=error_response):
        with pytest.raises(TransientGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_transient_on_error_payload_with_429_code():
    """200 response with error payload containing code 429 is transient."""
    error_response = _fake_response(200, {
        "error": {
            "message": "Rate limited",
            "code": 429
        }
    })
    with patch("sanghabot.rag.generator.requests.post", return_value=error_response):
        with pytest.raises(TransientGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_permanent_on_non_transient_error_payload():
    """200 response with error payload that doesn't look transient is permanent."""
    error_response = _fake_response(200, {
        "error": {
            "message": "Invalid API key",
            "code": "invalid_api_key"
        }
    })
    with patch("sanghabot.rag.generator.requests.post", return_value=error_response):
        with pytest.raises(PermanentGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_permanent_on_missing_choices():
    """200 response without 'choices' field raises PermanentGenerationError."""
    bad_response = _fake_response(200, {"data": "something"})
    with patch("sanghabot.rag.generator.requests.post", return_value=bad_response):
        with pytest.raises(PermanentGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_permanent_on_empty_choices():
    """200 response with empty 'choices' array raises PermanentGenerationError."""
    bad_response = _fake_response(200, {"choices": []})
    with patch("sanghabot.rag.generator.requests.post", return_value=bad_response):
        with pytest.raises(PermanentGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_post_chat_completion_raises_permanent_on_missing_content():
    """200 response missing choices[0].message.content raises PermanentGenerationError."""
    bad_response = _fake_response(200, {
        "choices": [
            {
                "message": {
                    # missing 'content' field
                }
            }
        ]
    })
    with patch("sanghabot.rag.generator.requests.post", return_value=bad_response):
        with pytest.raises(PermanentGenerationError):
            _post_chat_completion(
                system_prompt="You are helpful.",
                user_prompt="What?",
                model="google/gemini-2.5-flash",
                temperature=0.1,
                top_p=0.95,
                max_tokens=1500,
                api_key="fake-key",
            )


def test_generate_answer_raises_permanent_if_no_api_key():
    """generate_answer() raises PermanentGenerationError if API key is empty."""
    with patch("sanghabot.rag.generator.settings.openrouter_api_key", ""):
        with pytest.raises(PermanentGenerationError) as exc_info:
            generate_answer(
                system_prompt="You are helpful.",
                user_prompt="What?",
            )
        assert "OPENROUTER_API_KEY is not set" in str(exc_info.value)


def test_generate_answer_retries_on_transient_then_succeeds():
    """
    generate_answer() retries on TransientGenerationError and eventually
    returns the successful result. This test confirms the retry decorator
    is wired correctly.
    """
    error_response = _fake_response(429, {"error": "rate limited"})
    ok_response = _fake_response(200, {
        "choices": [
            {
                "message": {
                    "content": "Successful answer"
                }
            }
        ]
    })
    side_effects = [error_response, ok_response]
    with patch("sanghabot.rag.generator.requests.post", side_effect=side_effects) as mock_post:
        result = generate_answer(
            system_prompt="You are helpful.",
            user_prompt="What?",
        )
    assert result == "Successful answer"
    assert mock_post.call_count == 2


def test_generate_answer_uses_default_settings():
    """generate_answer() uses settings defaults when parameters are None."""
    ok_response = _fake_response(200, {
        "choices": [
            {
                "message": {
                    "content": "Answer"
                }
            }
        ]
    })
    with patch("sanghabot.rag.generator.requests.post", return_value=ok_response) as mock_post:
        with patch("sanghabot.rag.generator.settings") as mock_settings:
            mock_settings.openrouter_api_key = "fake-key"
            mock_settings.rag_model = "test-model"
            mock_settings.rag_temperature = 0.5
            mock_settings.rag_top_p = 0.9
            mock_settings.rag_max_tokens = 2000
            
            result = generate_answer(
                system_prompt="System",
                user_prompt="User",
            )
            
            # Verify the call used the settings defaults
            call_args = mock_post.call_args
            assert call_args[1]["json"]["model"] == "test-model"
            assert call_args[1]["json"]["temperature"] == 0.5
            assert call_args[1]["json"]["top_p"] == 0.9
            assert call_args[1]["json"]["max_tokens"] == 2000
            assert result == "Answer"


def test_generate_answer_uses_provided_parameters():
    """generate_answer() uses provided parameters instead of settings defaults."""
    ok_response = _fake_response(200, {
        "choices": [
            {
                "message": {
                    "content": "Answer"
                }
            }
        ]
    })
    with patch("sanghabot.rag.generator.requests.post", return_value=ok_response) as mock_post:
        result = generate_answer(
            system_prompt="System",
            user_prompt="User",
            model="custom-model",
            temperature=0.7,
            top_p=0.85,
            max_tokens=500,
        )
        
        # Verify the call used the provided parameters, not settings
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "custom-model"
        assert call_args[1]["json"]["temperature"] == 0.7
        assert call_args[1]["json"]["top_p"] == 0.85
        assert call_args[1]["json"]["max_tokens"] == 500
        assert result == "Answer"
