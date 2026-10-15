"""
Tests that the OpenCode session header is stamped automatically on the OpenAI chat
completion path, from the session marker litellm already carries.
"""

import os
import sys

import pytest
import respx
from httpx import Response

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import acompletion, completion


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")


CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-opencode-test",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "glm-5.3-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hey!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


@pytest.fixture
def mock_opencode_endpoint():
    return respx.post("https://opencode.ai/zen/go/v1/chat/completions").mock(
        return_value=Response(200, json=CHAT_COMPLETION_RESPONSE)
    )


@pytest.fixture
def mock_generic_endpoint():
    return respx.post("https://my-llm-gateway.example.com/v1/chat/completions").mock(
        return_value=Response(200, json=CHAT_COMPLETION_RESPONSE)
    )


@respx.mock
def test_completion_to_opencode_gateway_stamps_session_header(mock_opencode_endpoint):
    completion(
        model="openai/glm-5.3-flash",
        api_base="https://opencode.ai/zen/go/v1/",
        api_key="opencode-fake-key",
        messages=[{"role": "user", "content": "Hey"}],
        metadata={"session_id": "sess-123"},
    )

    request_headers = mock_opencode_endpoint.calls.last.request.headers
    assert request_headers["x-opencode-session"] == "sess-123"


@respx.mock
def test_completion_to_non_opencode_gateway_omits_session_header(mock_generic_endpoint):
    completion(
        model="openai/glm-5.3-flash",
        api_base="https://my-llm-gateway.example.com/v1",
        api_key="fake-key",
        messages=[{"role": "user", "content": "Hey"}],
        metadata={"session_id": "sess-123"},
    )

    request_headers = mock_generic_endpoint.calls.last.request.headers
    assert "x-opencode-session" not in request_headers


@respx.mock
def test_caller_supplied_opencode_session_header_is_never_overwritten(mock_opencode_endpoint):
    completion(
        model="openai/glm-5.3-flash",
        api_base="https://opencode.ai/zen/go/v1/",
        api_key="opencode-fake-key",
        messages=[{"role": "user", "content": "Hey"}],
        extra_headers={"x-opencode-session": "caller-session"},
    )

    request_headers = mock_opencode_endpoint.calls.last.request.headers
    assert request_headers["x-opencode-session"] == "caller-session"


@respx.mock
def test_completion_stamps_from_litellm_session_id_kwarg(mock_opencode_endpoint):
    completion(
        model="openai/glm-5.3-flash",
        api_base="https://opencode.ai/zen/go/v1/",
        api_key="opencode-fake-key",
        messages=[{"role": "user", "content": "Hey"}],
        litellm_session_id="chain-456",
    )

    request_headers = mock_opencode_endpoint.calls.last.request.headers
    assert request_headers["x-opencode-session"] == "chain-456"


@respx.mock
async def test_acompletion_to_opencode_gateway_stamps_session_header(mock_opencode_endpoint, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    await acompletion(
        model="openai/glm-5.3-flash",
        api_base="https://opencode.ai/zen/go/v1/",
        api_key="opencode-fake-key",
        messages=[{"role": "user", "content": "Hey"}],
        metadata={"session_id": "sess-789"},
        num_retries=0,
    )

    request_headers = mock_opencode_endpoint.calls.last.request.headers
    assert request_headers["x-opencode-session"] == "sess-789"


@respx.mock
def test_completion_streaming_to_opencode_gateway_stamps_session_header():
    respx.post("https://opencode.ai/zen/go/v1/chat/completions").mock(
        return_value=Response(
            200,
            content=(
                'data: {"id":"chatcmpl-opencode-test","object":"chat.completion.chunk",'
                '"created":1677652288,"model":"glm-5.3-flash","choices":[{"index":0,'
                '"delta":{"role":"assistant","content":"Hey"},"finish_reason":null}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = completion(
        model="openai/glm-5.3-flash",
        api_base="https://opencode.ai/zen/go/v1/",
        api_key="opencode-fake-key",
        messages=[{"role": "user", "content": "Hey"}],
        metadata={"session_id": "sess-456"},
        stream=True,
    )

    request_headers = respx.calls.last.request.headers
    assert request_headers["x-opencode-session"] == "sess-456"
    assert len(list(response)) == 2
