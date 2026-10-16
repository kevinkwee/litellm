"""Claude subscription provider backed by the official Claude Agent SDK.

Serves `claude-agent-sdk/<model>` deployments. The Agent SDK drives the real
Claude Code client, so subscription OAuth, client identity, and first-request
429 retries come from the genuine client. Credentials are set per deployment
like every other provider: the deployment `api_key` is wired into the CLI
process environment (OAuth tokens as `CLAUDE_CODE_OAUTH_TOKEN`, regular keys
as `ANTHROPIC_API_KEY`), with `CLAUDE_CODE_OAUTH_TOKEN` from the proxy
environment as the fallback.

Requests stay bare: built-in tools are disabled, the caller's system prompt
replaces the Claude Code default, and no filesystem settings or MCP servers
are loaded. Text-only conversations go over one flat prompt. Conversations
with images go over the SDK's stream-json input with content blocks, so vision
works through the same single-turn query.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportIncompatibleMethodOverride=false
# The CustomLLM extension contract (litellm.llms.custom_llm) is untyped: the four
# entry points mirror its bare list/dict/Callable signatures, chat messages arrive
# as untyped wire JSON, and the base astreaming contract is async-def-while real
# handlers are async generators. The entries stay one-line delegations into the
# strongly typed internals below, so this boundary is where the unknown-type
# surface is deliberately confined. The attribute-access rule stays on: it catches
# real bugs such as a missing internal method.

import asyncio
import concurrent.futures
import time
import uuid
from types import MappingProxyType
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Final,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import httpx

import litellm
from litellm.exceptions import APIError
from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_anthropic_image_obj,
    convert_url_to_base64,
)
from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key
from litellm.llms.custom_llm import CustomLLM
from litellm.types.llms.openai import ChatCompletionUsageBlock
from litellm.types.utils import Choices, GenericStreamingChunk, Message as LiteLLMMessage
from litellm.utils import ModelResponse, Usage

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

PROVIDER_NAME: Final[str] = "claude-agent-sdk"
FALLBACK_SYSTEM_PROMPT: Final[str] = "You are a helpful assistant."
TURN_PREFIXES: Final[Mapping[str, str]] = MappingProxyType({"user": "Human", "assistant": "Assistant"})
_OAUTH_TOKEN_ENV_VAR: Final[str] = "CLAUDE_CODE_OAUTH_TOKEN"
_API_KEY_ENV_VAR: Final[str] = "ANTHROPIC_API_KEY"
_AUTH_TOKEN_ENV_VAR: Final[str] = "ANTHROPIC_AUTH_TOKEN"
_IMAGE_PART_TYPES: Final[Mapping[str, str]] = MappingProxyType({"image_url": "image_url", "image": "image"})
_UNSUPPORTED_PART_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {"file": "file", "document": "document", "input_audio": "audio"}
)
_ERROR_STATUS_BY_SDK_CODE: Final[Mapping[str, int]] = MappingProxyType(
    {
        "authentication_failed": 401,
        "billing_error": 402,
        "rate_limit": 429,
        "invalid_request": 400,
        "server_error": 500,
        "unknown": 500,
    }
)
_SdkQueryPrompt = Union[str, AsyncGenerator[dict[str, object], None]]


def _credential_env(api_key: Optional[str]) -> dict[str, str]:
    """Build the CLI child environment for a deployment credential.

    Sibling credential variables are shadowed with empty strings because the
    CLI ranks them against the wired token, so an inherited ANTHROPIC_API_KEY
    or ANTHROPIC_AUTH_TOKEN on the proxy process would otherwise win.
    """
    if not isinstance(api_key, str) or not api_key:
        return {}
    if is_anthropic_oauth_key(api_key):
        return {
            _OAUTH_TOKEN_ENV_VAR: api_key,
            _API_KEY_ENV_VAR: "",
            _AUTH_TOKEN_ENV_VAR: "",
        }
    return {
        _API_KEY_ENV_VAR: api_key,
        _OAUTH_TOKEN_ENV_VAR: "",
        _AUTH_TOKEN_ENV_VAR: "",
    }


def _text_part(part: object) -> str:
    if isinstance(part, str):
        return part.strip()
    text: object = part.get("text") if isinstance(part, dict) else None
    return text.strip() if isinstance(text, str) else ""


def _sdk_message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        return " ".join(text for part in content if (text := _text_part(part)))
    return ""


def _content_parts(content: object) -> list[object]:
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [content]


def _content_part_type(part: object) -> Optional[str]:
    if isinstance(part, dict) and isinstance(part.get("type"), str):
        return part["type"]
    return None


def _system_prompt(messages: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("role") == "system":
            text = _sdk_message_text(message.get("content"))
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _split_messages(messages: list[dict[str, object]]) -> Tuple[str, str]:
    turns: list[str] = []
    for message in messages:
        role = message.get("role")
        if not isinstance(role, str):
            continue
        text = _sdk_message_text(message.get("content"))
        if not text:
            continue
        prefix = TURN_PREFIXES.get(role)
        if prefix is not None:
            turns.append(f"{prefix}: {text}")
    return _system_prompt(messages), "\n\n".join(turns)


def _raise_for_unsupported_parts(messages: list[dict[str, object]], model: str) -> None:
    for message in messages:
        for part in _content_parts(message.get("content")):
            part_type = _content_part_type(part)
            if part_type is not None and part_type in _UNSUPPORTED_PART_TYPES:
                label = _UNSUPPORTED_PART_TYPES[part_type]
                raise _provider_error(400, f"{PROVIDER_NAME} does not support {label} parts", model)


def _has_image_parts(messages: list[dict[str, object]]) -> bool:
    return any(
        _content_part_type(part) in _IMAGE_PART_TYPES
        for message in messages
        for part in _content_parts(message.get("content"))
    )


def _image_block(part: object, model: str) -> dict[str, object]:
    if isinstance(part, dict) and part.get("type") == "image" and isinstance(part.get("source"), dict):
        return dict(part)
    image_url_input: object = part.get("image_url") if isinstance(part, dict) else None
    url: object = image_url_input.get("url") if isinstance(image_url_input, dict) else image_url_input
    if not isinstance(url, str) or not url:
        raise _provider_error(400, f"{PROVIDER_NAME} could not read the image part of a message", model)
    if url.startswith("http"):
        url = convert_url_to_base64(url=url)
    image_chunk = convert_to_anthropic_image_obj(openai_image_url=url, format=None)
    return {"type": "image", "source": dict(image_chunk)}


def _build_content_blocks(messages: list[dict[str, object]], model: str) -> list[dict[str, object]]:
    content_blocks: list[dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        if not isinstance(role, str) or role == "system":
            continue
        prefix = TURN_PREFIXES.get(role)
        for part in _content_parts(message.get("content")):
            part_type = _content_part_type(part)
            if part_type in _IMAGE_PART_TYPES:
                content_blocks.append(_image_block(part, model))
            else:
                text = _text_part(part)
                if text and prefix is not None:
                    content_blocks.append({"type": "text", "text": f"{prefix}: {text}"})
                elif text:
                    content_blocks.append({"type": "text", "text": text})
    return content_blocks


async def _single_user_input_stream(content_blocks: list[dict[str, object]]) -> AsyncGenerator[dict[str, object], None]:
    input_message: dict[str, object] = {
        "type": "user",
        "session_id": "",
        "parent_tool_use_id": None,
        "message": {"role": "user", "content": content_blocks},
    }
    yield input_message


def _extract_sdk_model(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def _positive_int(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


class _SdkUsage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


def _usage_from_result(usage: Optional[dict[str, object]]) -> _SdkUsage:
    if not usage:
        return _SdkUsage(0, 0, 0)
    return _SdkUsage(
        _positive_int(usage.get("input_tokens")),
        _positive_int(usage.get("output_tokens")),
        _positive_int(usage.get("cache_read_input_tokens")),
    )


def _usage_block(usage: _SdkUsage) -> ChatCompletionUsageBlock:
    block = ChatCompletionUsageBlock(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.prompt_tokens + usage.completion_tokens,
    )
    if usage.cached_tokens > 0:
        block["prompt_tokens_details"] = {"cached_tokens": usage.cached_tokens}
    return block


def _provider_error(status_code: int, message: str, model: str) -> APIError:
    return APIError(status_code=status_code, message=message, llm_provider=PROVIDER_NAME, model=model)


def _assistant_error(error: Optional[str], model: str) -> APIError:
    code = error or "unknown"
    return _provider_error(_ERROR_STATUS_BY_SDK_CODE.get(code, 500), f"Claude Code error: {code}", model)


def _result_error(result: ResultMessage, model: str) -> APIError:
    detail = result.result or f"{PROVIDER_NAME} returned an error result"
    return _provider_error(500, f"Claude Code error: {detail}", model)


class ClaudeAgentSDKProvider(CustomLLM):
    """LiteLLM provider mapping chat completions onto Claude Agent SDK queries."""

    def _build_options(self, model: str, system_prompt: str, api_key: Optional[str]) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            model=_extract_sdk_model(model),
            system_prompt=system_prompt or FALLBACK_SYSTEM_PROMPT,
            tools=[],
            setting_sources=[],
            max_turns=1,
            env=_credential_env(api_key),
        )

    def _build_prompt(
        self, model: str, messages: list[dict[str, object]], optional_params: dict, api_key: Optional[str]
    ) -> Tuple[ClaudeAgentOptions, _SdkQueryPrompt]:
        if optional_params.get("tools"):
            raise _provider_error(
                400, f"{PROVIDER_NAME} does not support tool calls; send plain chat messages only", model
            )
        if any(message.get("role") in ("tool", "function") for message in messages):
            raise _provider_error(400, f"{PROVIDER_NAME} does not support tool result messages", model)
        _raise_for_unsupported_parts(messages, model)
        if _has_image_parts(messages):
            content_blocks = _build_content_blocks(messages, model)
            if not content_blocks:
                raise _provider_error(400, f"{PROVIDER_NAME} requires at least one non-system message", model)
            system_prompt = _system_prompt(messages)
            prompt: _SdkQueryPrompt = _single_user_input_stream(content_blocks)
        else:
            system_prompt, prompt = _split_messages(messages)
            if not prompt:
                raise _provider_error(400, f"{PROVIDER_NAME} requires at least one non-system message", model)
        return self._build_options(model=model, system_prompt=system_prompt, api_key=api_key), prompt

    async def _generate_sdk_events(
        self, options: ClaudeAgentOptions, prompt: _SdkQueryPrompt, model: str
    ) -> AsyncGenerator[AssistantMessage | ResultMessage, None]:
        saw_result = False
        query_events = query(prompt=prompt, options=options)
        try:
            async for event in query_events:
                if isinstance(event, AssistantMessage):
                    if event.error is not None:
                        raise _assistant_error(event.error, model)
                    yield event
                elif isinstance(event, ResultMessage):
                    saw_result = True
                    if event.is_error:
                        raise _result_error(event, model)
                    yield event
        finally:
            await query_events.aclose()  # pyright: ignore[reportAttributeAccessIssue]  # SDK query() is a runtime async generator despite its AsyncIterator annotation
        if not saw_result:
            raise _provider_error(500, "Claude Code returned no result message", model)

    def _build_response(self, model: str, sdk_model: Optional[str], text: str, usage: _SdkUsage) -> ModelResponse:
        response = ModelResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=sdk_model or model,
            object="chat.completion",
            choices=[Choices(finish_reason="stop", index=0, message=LiteLLMMessage(content=text, role="assistant"))],
            usage=Usage(**_usage_block(usage)),
        )
        self._attach_response_cost(response, sdk_model=sdk_model, usage=usage)
        return response

    @staticmethod
    def _attach_response_cost(response: ModelResponse, sdk_model: Optional[str], usage: _SdkUsage) -> None:
        response_cost = ClaudeAgentSDKProvider._completion_response_cost(sdk_model=sdk_model, usage=usage)
        if response_cost is not None:
            response._hidden_params["response_cost"] = response_cost  # pyright: ignore[reportPrivateUsage]  # ModelResponse private field, codebase-wide convention

    async def _collect_response(
        self,
        model: str,
        messages: list[dict[str, object]],
        optional_params: dict,
        api_key: Optional[str],
    ) -> Tuple[str, Optional[str], _SdkUsage]:
        options, prompt = self._build_prompt(
            model=model, messages=messages, optional_params=optional_params, api_key=api_key
        )
        usage = _SdkUsage(0, 0, 0)
        sdk_model: Optional[str] = None
        result_text: Optional[str] = None
        text_parts: list[str] = []
        events = self._generate_sdk_events(options=options, prompt=prompt, model=model)
        try:
            async for event in events:
                if isinstance(event, AssistantMessage):
                    text_parts.extend(block.text for block in event.content if isinstance(block, TextBlock))
                    sdk_model = event.model or sdk_model
                else:
                    usage = _usage_from_result(event.usage)
                    result_text = event.result or result_text
        finally:
            await events.aclose()
        return "".join(text_parts) or (result_text or ""), sdk_model, usage

    async def _run_completion(
        self, model: str, messages: list[dict[str, object]], optional_params: dict, api_key: Optional[str]
    ) -> ModelResponse:
        text, sdk_model, usage = await self._collect_response(
            model=model, messages=messages, optional_params=optional_params, api_key=api_key
        )
        return self._build_response(model=model, sdk_model=sdk_model, text=text, usage=usage)

    async def _run_streaming(
        self,
        model: str,
        messages: list[dict[str, object]],
        optional_params: dict,
        api_key: Optional[str],
    ) -> AsyncGenerator[GenericStreamingChunk, None]:
        options, prompt = self._build_prompt(
            model=model, messages=messages, optional_params=optional_params, api_key=api_key
        )
        usage = _SdkUsage(0, 0, 0)
        sdk_model: Optional[str] = None
        events = self._generate_sdk_events(options=options, prompt=prompt, model=model)
        try:
            async for event in events:
                if isinstance(event, AssistantMessage):
                    sdk_model = event.model or sdk_model
                    for block in event.content:
                        if isinstance(block, TextBlock) and block.text:
                            yield {
                                "text": block.text,
                                "is_finished": False,
                                "finish_reason": "stop",
                                "tool_use": None,
                                "usage": None,
                                "index": 0,
                            }
                else:
                    usage = _usage_from_result(event.usage)
        finally:
            await events.aclose()
        final_chunk: GenericStreamingChunk = {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "tool_use": None,
            "usage": _usage_block(usage),
            "index": 0,
        }
        response_cost = self._completion_response_cost(sdk_model=sdk_model, usage=usage)
        if response_cost is not None:
            final_chunk["provider_specific_fields"] = {"response_cost": response_cost}
        yield final_chunk

    @staticmethod
    def _completion_response_cost(sdk_model: Optional[str], usage: _SdkUsage) -> Optional[float]:
        model_for_cost = sdk_model or ""
        if not model_for_cost or not usage.prompt_tokens:
            return None
        cost_response = ModelResponse(
            model=f"anthropic/{model_for_cost}",
            usage=Usage(**_usage_block(usage)),
        )
        try:
            response_cost = litellm.completion_cost(completion_response=cost_response)
        except Exception as exc:  # noqa: BLE001  # best-effort cost attachment, logging falls back to its own calculator
            litellm.verbose_logger.debug("claude-agent-sdk cost calculation skipped: %s", exc)
            return None
        if response_cost and response_cost > 0:
            return response_cost
        litellm.verbose_logger.debug("claude-agent-sdk: no cost map entry for CLI model %s", model_for_cost)
        return None

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ModelResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._run_completion(model=model, messages=messages, optional_params=optional_params, api_key=api_key)
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                self._run_completion(model=model, messages=messages, optional_params=optional_params, api_key=api_key),
            ).result()

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ModelResponse:
        return await self._run_completion(
            model=model, messages=messages, optional_params=optional_params, api_key=api_key
        )

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> Iterator[GenericStreamingChunk]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            yield from _drain_async_iterator(
                lambda: self._run_streaming(
                    model=model, messages=messages, optional_params=optional_params, api_key=api_key
                )
            )
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for chunk in pool.submit(
                _drain_list,
                lambda: self._run_streaming(
                    model=model, messages=messages, optional_params=optional_params, api_key=api_key
                ),
            ).result():
                yield chunk

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        async for chunk in self._run_streaming(
            model=model, messages=messages, optional_params=optional_params, api_key=api_key
        ):
            yield chunk


def _drain_list(
    factory: Callable[[], AsyncGenerator[GenericStreamingChunk, None]],
) -> list[GenericStreamingChunk]:
    async def _collect() -> list[GenericStreamingChunk]:
        return [chunk async for chunk in factory()]

    return asyncio.run(_collect())


def _drain_async_iterator(
    factory: Callable[[], AsyncGenerator[GenericStreamingChunk, None]],
) -> Iterator[GenericStreamingChunk]:
    iterator = factory()
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.run_until_complete(iterator.aclose())
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


claude_agent_sdk_provider = ClaudeAgentSDKProvider()
