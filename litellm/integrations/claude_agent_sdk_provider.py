"""Claude subscription provider backed by the official Claude Agent SDK.

Serves `claude-agent-sdk/<model>` deployments. The Agent SDK drives the real
Claude Code client, so subscription OAuth, client identity, and first-request
429 retries come from the genuine client. Requests stay bare: built-in tools
are disabled, the caller's system prompt replaces the Claude Code default, and
no filesystem settings or MCP servers are loaded.
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
from typing import AsyncGenerator, AsyncIterator, Callable, Final, Iterator, Mapping, Optional, Tuple, Union

import httpx

from litellm.exceptions import APIError
from litellm.llms.custom_llm import CustomLLM
from litellm.types.llms.openai import ChatCompletionUsageBlock
from litellm.types.utils import Choices, GenericStreamingChunk, Message as LiteLLMMessage
from litellm.utils import ModelResponse, Usage

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

PROVIDER_NAME: Final[str] = "claude-agent-sdk"
FALLBACK_SYSTEM_PROMPT: Final[str] = "You are a helpful assistant."
TURN_PREFIXES: Final[Mapping[str, str]] = MappingProxyType({"user": "Human", "assistant": "Assistant"})
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


def _split_messages(messages: list[dict[str, object]]) -> Tuple[str, str]:
    system_parts: list[str] = []
    turns: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, (str, list)):
            continue
        text = _sdk_message_text(content)
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
            continue
        prefix = TURN_PREFIXES.get(role)
        if prefix is None:
            continue
        turns.append(f"{prefix}: {text}")
    return "\n\n".join(system_parts), "\n\n".join(turns)


def _extract_sdk_model(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def _positive_int(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def _usage_from_result(usage: Optional[dict[str, object]]) -> Tuple[int, int]:
    if not usage:
        return (0, 0)
    return (_positive_int(usage.get("input_tokens")), _positive_int(usage.get("output_tokens")))


def _stream_usage(prompt_tokens: int, completion_tokens: int) -> ChatCompletionUsageBlock:
    return ChatCompletionUsageBlock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


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

    def _build_options(self, model: str, system_prompt: str) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            model=_extract_sdk_model(model),
            system_prompt=system_prompt or FALLBACK_SYSTEM_PROMPT,
            tools=[],
            setting_sources=[],
            max_turns=1,
        )

    def _build_prompt(
        self, model: str, messages: list[dict[str, object]], optional_params: dict
    ) -> Tuple[ClaudeAgentOptions, str]:
        if optional_params.get("tools"):
            raise _provider_error(
                400, f"{PROVIDER_NAME} does not support tool calls; send plain chat messages only", model
            )
        if any(message.get("role") in ("tool", "function") for message in messages):
            raise _provider_error(400, f"{PROVIDER_NAME} does not support tool result messages", model)
        system_prompt, prompt = _split_messages(messages)
        if not prompt:
            raise _provider_error(400, f"{PROVIDER_NAME} requires at least one non-system message", model)
        return self._build_options(model=model, system_prompt=system_prompt), prompt

    async def _generate_sdk_events(
        self, options: ClaudeAgentOptions, prompt: str, model: str
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

    def _build_response(self, model: str, text: str, usage: Tuple[int, int]) -> ModelResponse:
        prompt_tokens, completion_tokens = usage
        return ModelResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            object="chat.completion",
            choices=[Choices(finish_reason="stop", index=0, message=LiteLLMMessage(content=text, role="assistant"))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _collect_response(
        self, model: str, messages: list[dict[str, object]], optional_params: dict
    ) -> Tuple[str, Tuple[int, int]]:
        options, prompt = self._build_prompt(model=model, messages=messages, optional_params=optional_params)
        prompt_tokens, completion_tokens = 0, 0
        result_text: Optional[str] = None
        text_parts: list[str] = []
        events = self._generate_sdk_events(options=options, prompt=prompt, model=model)
        try:
            async for event in events:
                if isinstance(event, AssistantMessage):
                    text_parts.extend(block.text for block in event.content if isinstance(block, TextBlock))
                else:
                    prompt_tokens, completion_tokens = _usage_from_result(event.usage)
                    result_text = event.result or result_text
        finally:
            await events.aclose()
        return "".join(text_parts) or (result_text or ""), (prompt_tokens, completion_tokens)

    async def _run_completion(
        self, model: str, messages: list[dict[str, object]], optional_params: dict
    ) -> ModelResponse:
        text, usage = await self._collect_response(model=model, messages=messages, optional_params=optional_params)
        return self._build_response(model=model, text=text, usage=usage)

    async def _run_streaming(
        self, model: str, messages: list[dict[str, object]], optional_params: dict
    ) -> AsyncGenerator[GenericStreamingChunk, None]:
        options, prompt = self._build_prompt(model=model, messages=messages, optional_params=optional_params)
        prompt_tokens, completion_tokens = 0, 0
        events = self._generate_sdk_events(options=options, prompt=prompt, model=model)
        try:
            async for event in events:
                if isinstance(event, AssistantMessage):
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
                    prompt_tokens, completion_tokens = _usage_from_result(event.usage)
        finally:
            await events.aclose()
        yield {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "tool_use": None,
            "usage": _stream_usage(prompt_tokens, completion_tokens),
            "index": 0,
        }

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
            return asyncio.run(self._run_completion(model=model, messages=messages, optional_params=optional_params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, self._run_completion(model=model, messages=messages, optional_params=optional_params)
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
        return await self._run_completion(model=model, messages=messages, optional_params=optional_params)

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
                lambda: self._run_streaming(model=model, messages=messages, optional_params=optional_params)
            )
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for chunk in pool.submit(
                _drain_list,
                lambda: self._run_streaming(model=model, messages=messages, optional_params=optional_params),
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
        async for chunk in self._run_streaming(model=model, messages=messages, optional_params=optional_params):
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
