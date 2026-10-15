"""Tests for the Claude Agent SDK provider.

The provider is exercised through litellm's public completion dispatch, with the
SDK `query` function replaced by fakes that yield real SDK message objects, so
no Claude Code CLI process is ever spawned.
"""

from typing import AsyncIterator

import pytest

import litellm
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock
from litellm.exceptions import APIError
from litellm.integrations import claude_agent_sdk_provider as provider_module
from litellm.utils import ModelResponse


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
    async def fake_query(*, prompt: str, options: object) -> AsyncIterator[object]:
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
