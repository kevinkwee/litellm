"""Tests for the Claude Agent SDK provider.

The provider is exercised through litellm's public completion dispatch, with the
SDK `query` function replaced by fakes that yield real SDK message objects, so
no Claude Code CLI process is ever spawned.
"""

import asyncio
from typing import AsyncIterator

import pytest

import litellm
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock
from litellm.exceptions import APIError
from litellm.integrations import claude_agent_sdk_provider as provider_module
from litellm.utils import ModelResponse


# Fake tokens for testing (not real secrets)
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"


def _assistant_message(text: list[str], error: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=part) for part in text],
        model="claude-sonnet-5",
        error=error,
    )


def _result_message(
    usage: dict[str, int] | None = None, is_error: bool = False, result: str | None = None
) -> ResultMessage:
    return ResultMessage(
        subtype="result_success" if not is_error else "result_error",
        duration_ms=1200,
        duration_api_ms=900,
        is_error=is_error,
        num_turns=1,
        session_id="fake-session",
        total_cost_usd=None,
        usage=usage,
        result=result,
        structured_output=None,
    )


def _install_query(
    monkeypatch: pytest.MonkeyPatch, events: list[AssistantMessage | ResultMessage], captured: dict[str, object]
) -> None:
    async def fake_query(*, prompt: object, options: object) -> AsyncIterator[object]:
        captured["prompt"] = prompt
        captured["options"] = options
        for event in events:
            yield event

    monkeypatch.setattr(provider_module, "query", fake_query)


@pytest.fixture
def provider_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "custom_provider_map",
        [{"provider": "claude-agent-sdk", "custom_handler": provider_module.claude_agent_sdk_provider}],
    )


def _messages() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_completion_maps_sdk_messages_to_response(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello", " world"]), _result_message({"input_tokens": 11, "output_tokens": 4})],
        captured,
    )

    response = litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert response.choices[0].message.content == "Hello world"
    assert response.choices[0].finish_reason == "stop"


def test_completion_maps_result_usage(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"]), _result_message({"input_tokens": 11, "output_tokens": 4})],
        captured,
    )

    response = litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 15


def test_response_model_reports_the_cli_model(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"]), _result_message({"input_tokens": 1, "output_tokens": 1})],
        captured,
    )

    response = litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert response.model == "claude-sonnet-5"


def test_cached_tokens_flow_into_usage_details(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [
            _assistant_message(["Hello world"]),
            _result_message({"input_tokens": 11, "output_tokens": 4, "cache_read_input_tokens": 7}),
        ],
        captured,
    )

    response = litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    details = response.usage.prompt_tokens_details
    assert details is not None
    assert details.cached_tokens == 7


def test_oauth_shaped_api_key_wires_the_cli_token_env(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["ok"]), _result_message()],
        captured,
    )

    litellm.completion(
        model="claude-agent-sdk/claude-sonnet-5",
        messages=_messages(),
        api_key=FAKE_OAUTH_TOKEN,
    )

    options = captured["options"]
    assert options.env == {
        "CLAUDE_CODE_OAUTH_TOKEN": FAKE_OAUTH_TOKEN,
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_AUTH_TOKEN": "",
    }


def test_regular_api_key_wires_the_anthropic_env(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["ok"]), _result_message()],
        captured,
    )

    litellm.completion(
        model="claude-agent-sdk/claude-sonnet-5",
        messages=_messages(),
        api_key="sk-ant-api03-regular-key-for-testing-123456789",
    )

    options = captured["options"]
    assert options.env == {
        "ANTHROPIC_API_KEY": "sk-ant-api03-regular-key-for-testing-123456789",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_AUTH_TOKEN": "",
    }


def test_missing_api_key_leaves_the_cli_env_untouched(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["ok"]), _result_message()],
        captured,
    )

    litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    options = captured["options"]
    assert options.env == {}


def _drain_prompt(prompt: object) -> list[dict[str, object]]:
    async def _collect() -> list[dict[str, object]]:
        if isinstance(prompt, str):
            return [{"type": "user", "message": {"role": "user", "content": prompt}}]
        return [message async for message in prompt]

    return asyncio.run(_collect())


def test_image_messages_go_over_the_stream_json_input(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["I see it"]), _result_message({"input_tokens": 5, "output_tokens": 2})],
        captured,
    )
    data_url = "data:image/png;base64,aGVsbG8="
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    response = litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=messages)

    assert response.choices[0].message.content == "I see it"
    user_inputs = _drain_prompt(captured["prompt"])
    assert len(user_inputs) == 1
    prompt_content = user_inputs[0]["message"]["content"]
    image_blocks = [block for block in prompt_content if block["type"] == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}
    text_blocks = [block for block in prompt_content if block["type"] == "text"]
    assert [block["text"] for block in text_blocks] == ["Human: what is this?"]
    assert user_inputs[0]["message"]["role"] == "user"


def test_https_image_url_is_downloaded_and_inlined(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["ok"]), _result_message()],
        captured,
    )

    def fake_convert_url_to_base64(url: str) -> str:
        return "data:image/png;base64,aGVsbG8="

    monkeypatch.setattr(provider_module, "convert_url_to_base64", fake_convert_url_to_base64)
    messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]}
    ]

    litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=messages)

    prompt_content = _drain_prompt(captured["prompt"])[0]["message"]["content"]
    image_blocks = [block for block in prompt_content if block["type"] == "image"]
    assert image_blocks[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}


@pytest.mark.parametrize(
    "unsupported_type,label",
    [("input_audio", "audio"), ("file", "file"), ("document", "document")],
)
def test_unsupported_content_parts_are_rejected(
    monkeypatch: pytest.MonkeyPatch, provider_map: None, unsupported_type: str, label: str
) -> None:
    captured: dict[str, object] = {}
    _install_query(monkeypatch, [], captured)
    messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": unsupported_type}]},
    ]

    with pytest.raises(APIError) as exc_info:
        litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=messages)

    assert exc_info.value.status_code == 400
    assert label in str(exc_info.value)


def test_completion_attaches_response_cost(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"]), _result_message({"input_tokens": 1000, "output_tokens": 100})],
        captured,
    )

    def fake_completion_cost(completion_response: object) -> float:
        return 0.003

    monkeypatch.setattr(litellm, "completion_cost", fake_completion_cost)

    response = litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    hidden = getattr(response, "_hidden_params", None) or {}
    assert hidden.get("response_cost") == 0.003


async def test_streaming_final_chunk_carries_response_cost(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"]), _result_message({"input_tokens": 1000, "output_tokens": 100})],
        captured,
    )

    def fake_completion_cost(completion_response: object) -> float:
        return 0.003

    monkeypatch.setattr(litellm, "completion_cost", fake_completion_cost)

    chunks = [
        chunk
        for chunk in litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages(), stream=True)
    ]

    costs = [getattr(chunk, "response_cost", None) for chunk in chunks]
    assert 0.003 in costs


async def test_acompletion_awaits_sdk_query(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"]), _result_message({"input_tokens": 2, "output_tokens": 1})],
        captured,
    )

    response = await litellm.acompletion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert response.choices[0].message.content == "Hello world"


def test_prompt_flattens_turns_and_splits_system(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["ok"]), _result_message()],
        captured,
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": " there"}]},
        {"role": "assistant", "content": "earlier reply"},
        {"role": "user", "content": "and now"},
    ]

    litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=messages)

    assert captured["prompt"] == ("Human: hi there\n\nAssistant: earlier reply\n\nHuman: and now")
    assert captured["options"].system_prompt == "be brief"


def test_options_lock_a_bare_completion_shape(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["ok"]), _result_message()],
        captured,
    )

    litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    options = captured["options"]
    assert options.model == "claude-sonnet-5"
    assert options.tools == []
    assert options.setting_sources == []
    assert options.max_turns == 1


def test_fallback_system_prompt_when_caller_sends_none(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["ok"]), _result_message()],
        captured,
    )

    litellm.completion(
        model="claude-agent-sdk/claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert captured["options"].system_prompt == provider_module.FALLBACK_SYSTEM_PROMPT


def test_assistant_rate_limit_error_keeps_429_status(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"], error="rate_limit")],
        captured,
    )

    with pytest.raises(APIError) as exc_info:
        litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert exc_info.value.status_code == 429


def test_assistant_authentication_error_keeps_401_status(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"], error="authentication_failed")],
        captured,
    )

    with pytest.raises(APIError) as exc_info:
        litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert exc_info.value.status_code == 401


def test_error_result_message_maps_to_500(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_result_message(is_error=True)],
        captured,
    )

    with pytest.raises(APIError) as exc_info:
        litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert exc_info.value.status_code == 500


def test_missing_result_message_is_an_error(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"])],
        captured,
    )

    with pytest.raises(APIError) as exc_info:
        litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert exc_info.value.status_code == 500


async def test_async_streaming_yields_deltas_and_final_usage(
    monkeypatch: pytest.MonkeyPatch, provider_map: None
) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello", " world"]), _result_message({"input_tokens": 2, "output_tokens": 1})],
        captured,
    )

    chunks = [
        chunk
        async for chunk in await litellm.acompletion(
            model="claude-agent-sdk/claude-sonnet-5", messages=_messages(), stream=True
        )
    ]

    deltas = [chunk.choices[0].delta.content for chunk in chunks]
    assert "".join(delta for delta in deltas if delta) == "Hello world"
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_astreaming_final_chunk_carries_usage(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello", " world"]), _result_message({"input_tokens": 2, "output_tokens": 1})],
        captured,
    )

    chunks = [
        chunk
        async for chunk in provider_module.claude_agent_sdk_provider.astreaming(
            model="claude-agent-sdk/claude-sonnet-5",
            messages=_messages(),
            api_base="",
            custom_prompt_dict={},
            model_response=ModelResponse(),
            print_verbose=lambda: None,
            encoding=None,
            api_key=None,
            logging_obj=None,
            optional_params={},
        )
    ]

    assert chunks[-1]["is_finished"] is True
    assert chunks[-1]["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


def test_sync_streaming_drains_the_async_generator(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello", " world"]), _result_message({"input_tokens": 2, "output_tokens": 1})],
        captured,
    )

    chunks = [
        chunk
        for chunk in litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages(), stream=True)
    ]

    deltas = [chunk.choices[0].delta.content for chunk in chunks]
    assert "".join(delta for delta in deltas if delta) == "Hello world"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_sync_stream_abandonment_finalizes_sdk_query(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    finalized: list[bool] = []

    async def fake_query(*, prompt: str, options: object) -> AsyncIterator[object]:
        try:
            yield _assistant_message(["Hello"])
            yield _result_message()
        finally:
            finalized.append(True)

    monkeypatch.setattr(provider_module, "query", fake_query)

    stream = provider_module.claude_agent_sdk_provider.streaming(
        model="claude-agent-sdk/claude-sonnet-5",
        messages=_messages(),
        api_base="",
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=lambda: None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params={},
    )
    next(stream)
    stream.close()

    assert finalized == [True]


async def test_completion_pool_path_runs_in_executor(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello world"]), _result_message({"input_tokens": 1, "output_tokens": 1})],
        captured,
    )

    response = provider_module.claude_agent_sdk_provider.completion(
        model="claude-agent-sdk/claude-sonnet-5",
        messages=_messages(),
        api_base="",
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=lambda: None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params={},
    )

    assert response.choices[0].message.content == "Hello world"


async def test_streaming_pool_path_runs_in_executor(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [_assistant_message(["Hello"]), _result_message({"input_tokens": 1, "output_tokens": 1})],
        captured,
    )

    chunks = list(
        provider_module.claude_agent_sdk_provider.streaming(
            model="claude-agent-sdk/claude-sonnet-5",
            messages=_messages(),
            api_base="",
            custom_prompt_dict={},
            model_response=ModelResponse(),
            print_verbose=lambda: None,
            encoding=None,
            api_key=None,
            logging_obj=None,
            optional_params={},
        )
    )

    assert chunks[0]["text"] == "Hello"
    assert chunks[-1]["is_finished"] is True


@pytest.mark.parametrize(
    "sdk_error,expected_status",
    [
        ("billing_error", 402),
        ("invalid_request", 400),
        ("server_error", 500),
        ("unknown", 500),
    ],
)
def test_assistant_error_status_mapping(
    monkeypatch: pytest.MonkeyPatch, provider_map: None, sdk_error: str, expected_status: int
) -> None:
    captured: dict[str, object] = {}
    _install_query(monkeypatch, [_assistant_message(["x"], error=sdk_error)], captured)

    with pytest.raises(APIError) as exc_info:
        litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert exc_info.value.status_code == expected_status


def test_tool_calls_are_rejected(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(monkeypatch, [], captured)

    with pytest.raises(APIError) as exc_info:
        litellm.completion(
            model="claude-agent-sdk/claude-sonnet-5",
            messages=_messages(),
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        )

    assert exc_info.value.status_code == 400


def test_tool_result_messages_are_rejected(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(monkeypatch, [], captured)

    with pytest.raises(APIError) as exc_info:
        litellm.completion(
            model="claude-agent-sdk/claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}, {"role": "tool", "content": "tool output"}],
        )

    assert exc_info.value.status_code == 400


def test_result_text_fallback_when_no_assistant_text(monkeypatch: pytest.MonkeyPatch, provider_map: None) -> None:
    captured: dict[str, object] = {}
    _install_query(
        monkeypatch,
        [
            _assistant_message([]),
            _result_message({"input_tokens": 1, "output_tokens": 1}, result="Hello world"),
        ],
        captured,
    )

    response = litellm.completion(model="claude-agent-sdk/claude-sonnet-5", messages=_messages())

    assert response.choices[0].message.content == "Hello world"
