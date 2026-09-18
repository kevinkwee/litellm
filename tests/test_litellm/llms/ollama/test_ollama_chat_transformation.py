import inspect
import os
import sys
import time
from typing import Any, Dict, Optional, Union, cast

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.ollama.chat.transformation import (
    OllamaChatConfig,
    OllamaChatCompletionResponseIterator,
)

from litellm.types.llms.openai import AllMessageValues
from litellm.utils import ModelResponseListIterator, get_optional_params

import json
from unittest.mock import MagicMock

import litellm
from litellm.types.utils import Choices, Message, ModelResponse, ModelResponseStream, StreamingChoices, Delta


class TestEvent(BaseModel):
    name: str
    value: int


class TestOllamaChatConfigResponseFormat:
    def test_get_optional_params_with_pydantic_model(self):
        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format=TestEvent,
            custom_llm_provider="ollama_chat",
        )
        print(f"optional_params: {optional_params}")

        assert "format" in optional_params
        transformed_format = optional_params["format"]

        expected_schema_structure = TestEvent.model_json_schema()
        transformed_format.pop("additionalProperties")

        assert transformed_format == expected_schema_structure, (
            f"Transformed schema does not match expected. Got: {transformed_format}, Expected: {expected_schema_structure}"
        )

    def test_map_openai_params_with_dict_json_schema(self):
        config = OllamaChatConfig()

        direct_schema = TestEvent.model_json_schema()
        response_format_dict = {
            "type": "json_schema",
            "json_schema": {"schema": direct_schema},
        }

        non_default_params = {"response_format": response_format_dict}

        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format=response_format_dict,
            custom_llm_provider="ollama_chat",
        )

        assert "format" in optional_params
        assert optional_params["format"] == direct_schema, (
            f"Schema from dict did not pass through correctly. Got: {optional_params['format']}, Expected: {direct_schema}"
        )

    def test_map_openai_params_with_json_object(self):
        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format={"type": "json_object"},
            custom_llm_provider="ollama_chat",
        )

        assert "format" in optional_params
        assert optional_params["format"] == "json", (
            f"Expected 'json' for type 'json_object', got: {optional_params['format']}"
        )

    def test_transform_request_loads_config_parameters(self):
        """Test that transform_request loads config parameters without overriding existing optional_params"""
        # Set config parameters on the class
        import litellm

        litellm.OllamaChatConfig(num_ctx=8000, temperature=0.0)

        try:
            config = OllamaChatConfig()

            # Initial optional_params with existing temperature (should not be overridden)
            optional_params = {"temperature": 0.3}

            # Transform request
            result = config.transform_request(
                model="llama2",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params=optional_params,
                litellm_params={},
                headers={},
            )

            # Verify config values were loaded but existing optional_params were preserved
            assert result["options"]["temperature"] == 0.3  # Should keep existing value
            assert result["options"]["num_ctx"] == 8000  # Should load from config

        finally:
            # Clean up class attributes
            delattr(litellm.OllamaChatConfig, "num_ctx")
            delattr(litellm.OllamaChatConfig, "temperature")

    def test_transform_request_content_list_to_string(self):
        """Test that content list is properly converted to string in transform_request"""
        config = OllamaChatConfig()

        # Test message with content as list containing text
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world!"},
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify content was converted to string
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello world!"
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_content_string_passthrough(self):
        """Test that string content passes through unchanged in transform_request"""
        config = OllamaChatConfig()

        # Test message with content as string
        messages = cast(list[AllMessageValues], [{"role": "user", "content": "Hello world!"}])

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify string content passes through
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello world!"
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_empty_content_list(self):
        """Test handling of empty content list in transform_request"""
        config = OllamaChatConfig()

        # Test message with empty content list
        messages = cast(list[AllMessageValues], [{"role": "user", "content": []}])

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify empty content becomes empty string
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == ""
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_image_extraction(self):
        """Test that images are properly extracted from messages in transform_request"""
        config = OllamaChatConfig()

        # Test message with images in content list
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."},
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify text content was extracted
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "What's in this image?"
        assert result["messages"][0]["role"] == "user"

        # Verify image was extracted to images list
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 1
        # Ollama expects pure base64 data without the data URL prefix
        assert result["messages"][0]["images"][0] == "/9j/4AAQSkZJRgABAQAAAQ..."

    def test_transform_request_multiple_images_extraction(self):
        """Test extraction of multiple images from a single message"""
        config = OllamaChatConfig()

        # Test message with multiple images
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these images:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,image1data..."},
                        },
                        {"type": "text", "text": " and "},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,image2data..."},
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify text content was combined
        assert result["messages"][0]["content"] == "Compare these images: and "

        # Verify both images were extracted
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 2
        # Ollama expects pure base64 data without the data URL prefix
        assert result["messages"][0]["images"][0] == "image1data..."
        assert result["messages"][0]["images"][1] == "image2data..."

    def test_transform_request_image_url_as_string(self):
        """Test handling of image_url as direct string (edge case)"""
        config = OllamaChatConfig()

        # Test message with image_url as string (edge case from extract_images_from_message)
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this:"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify image URL was extracted
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 1
        assert result["messages"][0]["images"][0] == "https://example.com/image.jpg"

    def test_transform_request_no_images_no_images_key(self):
        """Test that messages without images don't have images key"""
        config = OllamaChatConfig()

        # Test message with no images
        messages = cast(
            list[AllMessageValues],
            [{"role": "user", "content": [{"type": "text", "text": "Just text here"}]}],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify no images key when no images present
        assert result["messages"][0]["content"] == "Just text here"
        # Since extract_images_from_message returns empty list [] when no images found,
        # and the code checks "if images is not None", an empty list will still be set
        assert "images" in result["messages"][0]
        assert result["messages"][0]["images"] == []


class TestOllamaToolCalling:
    """Tests for Ollama tool calling fixes.

    Issue: https://github.com/BerriAI/litellm/issues/18922
    """

    def test_tools_passed_directly_without_capability_check(self):
        """Test that tools are passed directly to Ollama without model capability checks.

        Previously, the code called litellm.get_model_info() which could fail
        when Ollama runs on a remote server, causing a broken fallback.
        Now tools are passed directly - Ollama 0.4+ handles capability detection.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        optional_params = get_optional_params(
            model="ollama_chat/qwen3:14b",
            tools=tools,
            custom_llm_provider="ollama_chat",
        )

        # Tools should be passed through directly
        assert "tools" in optional_params
        assert optional_params["tools"] == tools
        # Should NOT trigger the broken fallback
        assert "functions_unsupported_model" not in optional_params
        assert "format" not in optional_params or optional_params.get("format") != "json"

    def test_finish_reason_tool_calls_non_streaming(self):
        """Test that finish_reason is set to 'tool_calls' when tool_calls present.

        Previously, finish_reason was hardcoded to 'stop' even when tool_calls
        were in the response, causing clients to ignore the tool calls.

        Uses the real Ollama on-wire format (verified against GLM-5.2 via
        Ollama Cloud, 2026-07-26): `id` at top-level, `index` inside
        `function`, `arguments` as a dict. The non-streaming path now
        applies the same strict structural validation as the streaming
        path (`_validate_ollama_tool_call_structure`), so a tool_call
        missing `id` or `function.index` would fail loud instead of
        silently producing wrong results.
        """
        import json
        from unittest.mock import MagicMock

        import litellm
        from litellm.types.utils import Choices, Message, ModelResponse

        config = OllamaChatConfig()

        # Simulated Ollama response with tool_calls (real wire format)
        ollama_response = {
            "model": "qwen3:14b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "function": {
                            "index": 0,
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo"},
                        },
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)

        mock_logging = MagicMock()

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]

        result = config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Weather?"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        # finish_reason should be "tool_calls", not "stop"
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None
        # Ollama's id must be preserved (not overwritten with a random UUID)
        assert result.choices[0].message.tool_calls[0].id == "call_abc123"
        # Dict arguments must be converted to JSON string by Function.__init__
        assert result.choices[0].message.tool_calls[0].function.arguments == '{"location": "Tokyo"}'

    def test_finish_reason_stop_when_no_tool_calls(self):
        """Test that finish_reason remains 'stop' when no tool_calls present."""
        config = OllamaChatConfig()

        # Simulated Ollama response without tool_calls
        ollama_response = {
            "model": "qwen3:14b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?",
            },
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)

        mock_logging = MagicMock()

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]

        result = config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        # finish_reason should be "stop" (default behavior)
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.tool_calls is None

    # ---------- Non-streaming strict validation (fail loud on structural change) ----------
    # These mirror the streaming strict validation tests but exercise the
    # `transform_response` path. They ensure that if Ollama changes its
    # response structure, the non-streaming request also fails loud with
    # an actionable error message (instead of silently producing wrong
    # results or generating replacement UUIDs that break tool_result
    # correlation in the next turn).

    @staticmethod
    def _make_non_streaming_response(tool_calls: list) -> dict:
        """Build a non-streaming Ollama response with the given tool_calls."""
        return {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-26T07:19:54.74373829Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

    @staticmethod
    def _call_transform_response(ollama_response: dict):
        """Helper: invoke transform_response with a mocked raw_response."""
        config = OllamaChatConfig()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]

        return config.transform_response(
            model="glm-5.2:cloud",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

    def test_non_streaming_parallel_tool_calls_preserve_id_and_index(self):
        """Non-streaming with 5 parallel tool_calls (real wire format):
        each tool_call's `id` must be preserved, and `function.index`
        must be accepted (non-streaming doesn't need to promote it to
        top-level, but it must not crash)."""
        ollama_response = self._make_non_streaming_response(
            [
                {"id": "call_a", "function": {"index": 0, "name": "a", "arguments": {"x": 1}}},
                {"id": "call_b", "function": {"index": 1, "name": "b", "arguments": {"x": 2}}},
                {"id": "call_c", "function": {"index": 2, "name": "c", "arguments": {"x": 3}}},
            ]
        )

        result = self._call_transform_response(ollama_response)

        assert result.choices[0].finish_reason == "tool_calls"
        tcs = result.choices[0].message.tool_calls
        assert tcs is not None
        assert len(tcs) == 3
        # Ollama's ids must be preserved exactly.
        assert [tc.id for tc in tcs] == ["call_a", "call_b", "call_c"]
        # Names and arguments must come through correctly (dict → JSON string).
        assert [tc.function.name for tc in tcs] == ["a", "b", "c"]
        assert [tc.function.arguments for tc in tcs] == ['{"x": 1}', '{"x": 2}', '{"x": 3}']

    def test_non_streaming_missing_id_raises(self):
        """Non-streaming: missing top-level `id` must fail loud."""
        from litellm.llms.ollama.common_utils import OllamaError

        ollama_response = self._make_non_streaming_response(
            [
                # no `id` field — strict validation must reject
                {"function": {"index": 0, "name": "get_weather", "arguments": {"location": "Tokyo"}}},
            ]
        )

        with pytest.raises(OllamaError) as exc_info:
            self._call_transform_response(ollama_response)

        msg = str(exc_info.value)
        assert "missing a top-level string `id`" in msg
        assert "litellm/llms/ollama/chat/transformation.py" in msg

    def test_non_streaming_missing_function_raises(self):
        """Non-streaming: missing `function` dict must fail loud."""
        from litellm.llms.ollama.common_utils import OllamaError

        ollama_response = self._make_non_streaming_response(
            [{"id": "call_a", "type": "function"}],  # no `function` key
        )

        with pytest.raises(OllamaError) as exc_info:
            self._call_transform_response(ollama_response)

        assert "missing a `function` dict" in str(exc_info.value)

    def test_non_streaming_missing_function_index_raises(self):
        """Non-streaming: missing `function.index` must fail loud.
        Consistent with streaming path — if Ollama moves/renames
        `function.index`, we want to know immediately."""
        from litellm.llms.ollama.common_utils import OllamaError

        ollama_response = self._make_non_streaming_response(
            [
                {
                    "id": "call_a",
                    "function": {
                        # no `index` field — strict validation must reject
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                    },
                },
            ]
        )

        with pytest.raises(OllamaError) as exc_info:
            self._call_transform_response(ollama_response)

        msg = str(exc_info.value)
        assert "missing an integer `index`" in msg
        assert "litellm/llms/ollama/chat/transformation.py" in msg

    def test_non_streaming_missing_function_name_raises(self):
        """Non-streaming: missing `function.name` must fail loud."""
        from litellm.llms.ollama.common_utils import OllamaError

        ollama_response = self._make_non_streaming_response(
            [
                {
                    "id": "call_a",
                    "function": {
                        "index": 0,
                        # no `name` field
                        "arguments": {"location": "Tokyo"},
                    },
                },
            ]
        )

        with pytest.raises(OllamaError) as exc_info:
            self._call_transform_response(ollama_response)

        assert "missing a non-empty string `name`" in str(exc_info.value)

    def test_non_streaming_missing_function_arguments_raises(self):
        """Non-streaming: missing `function.arguments` must fail loud."""
        from litellm.llms.ollama.common_utils import OllamaError

        ollama_response = self._make_non_streaming_response(
            [
                {
                    "id": "call_a",
                    "function": {
                        "index": 0,
                        "name": "get_weather",
                        # no `arguments` field
                    },
                },
            ]
        )

        with pytest.raises(OllamaError) as exc_info:
            self._call_transform_response(ollama_response)

        assert "missing `arguments`" in str(exc_info.value)

    def test_non_streaming_no_tool_calls_does_not_validate(self):
        """Non-streaming: when there are no tool_calls, validation must
        NOT be triggered (so normal text-only responses still work)."""
        ollama_response = {
            "model": "qwen3:14b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?",
                # no `tool_calls` key at all
            },
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        result = self._call_transform_response(ollama_response)
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.tool_calls is None


class TestOllamaFinishReasonLength:
    """Tests for done_reason 'length' → finish_reason 'length' mapping.

    Ollama returns done_reason='length' when a response is truncated by num_predict
    (max_tokens). Previously finish_reason was hardcoded to 'stop', hiding truncation.
    The Anthropic pass-through adapter then maps OpenAI 'length' → 'max_tokens'.
    """

    def test_finish_reason_length_non_streaming(self):
        """Non-streaming: done_reason='length' must propagate as finish_reason='length'."""
        config = OllamaChatConfig()

        ollama_response = {
            "model": "qwen3:2b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {
                "role": "assistant",
                "content": "A neural network learns through",
            },
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 20,
            "eval_count": 20,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)

        mock_logging = MagicMock()

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]

        result = config.transform_response(
            model="qwen3:2b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "Explain neural networks."}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        assert result.choices[0].finish_reason == "length", (
            f"Expected 'length' when done_reason='length', got '{result.choices[0].finish_reason}'"
        )

    def test_finish_reason_stop_non_streaming(self):
        """Non-streaming: done_reason='stop' (natural finish) must stay 'stop'."""
        config = OllamaChatConfig()

        ollama_response = {
            "model": "qwen3:2b",
            "created_at": "2025-01-11T00:00:00.000000Z",
            "message": {"role": "assistant", "content": "2 + 2 = 4."},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 8,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)

        mock_logging = MagicMock()

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]

        result = config.transform_response(
            model="qwen3:2b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            messages=[{"role": "user", "content": "What is 2+2?"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        assert result.choices[0].finish_reason == "stop", (
            f"Expected 'stop' for natural finish, got '{result.choices[0].finish_reason}'"
        )

    def test_finish_reason_length_streaming(self):
        """Streaming: done_reason='length' in final chunk must produce finish_reason='length'."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        done_chunk = {
            "model": "qwen3:2b",
            "message": {
                "role": "assistant",
                "content": "A neural network learns through",
            },
            "done": True,
            "done_reason": "length",
        }

        result = iterator.chunk_parser(done_chunk)

        assert result.choices[0].finish_reason == "length", (
            f"Expected 'length' when done_reason='length', got '{result.choices[0].finish_reason}'"
        )

    def test_finish_reason_stop_streaming(self):
        """Streaming: done_reason='stop' in final chunk must produce finish_reason='stop'."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        done_chunk = {
            "model": "qwen3:2b",
            "message": {"role": "assistant", "content": "2 + 2 = 4."},
            "done": True,
            "done_reason": "stop",
        }

        result = iterator.chunk_parser(done_chunk)

        assert result.choices[0].finish_reason == "stop", (
            f"Expected 'stop' for natural finish, got '{result.choices[0].finish_reason}'"
        )


class TestOllamaReasoningContentStreaming:
    """Test that reasoning_content is properly extracted from all thinking chunks."""

    def test_multiple_thinking_chunks_all_returned_as_reasoning_content(self):
        """
        Test that more than 2 consecutive thinking chunks are all returned as reasoning_content.

        Previously, the code had a bug where finished_reasoning_content was set to True
        after just 2 chunks with 'thinking', causing subsequent thinking content to be lost.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),  # Not used in chunk_parser
            sync_stream=True,
        )

        # Simulate 5 consecutive chunks with 'thinking' content
        thinking_chunks = [
            {
                "model": "deepseek-r1",
                "message": {"role": "assistant", "thinking": f"Thinking chunk {i}"},
                "done": False,
            }
            for i in range(1, 6)
        ]

        # Process all thinking chunks
        reasoning_contents = []
        for chunk in thinking_chunks:
            result = iterator.chunk_parser(chunk)
            rc = result.choices[0].delta.reasoning_content
            reasoning_contents.append(rc)

        # ALL chunks should have reasoning_content, not just the first 2
        assert len(reasoning_contents) == 5
        assert reasoning_contents[0] == "Thinking chunk 1"
        assert reasoning_contents[1] == "Thinking chunk 2"
        assert reasoning_contents[2] == "Thinking chunk 3"  # This was previously None
        assert reasoning_contents[3] == "Thinking chunk 4"  # This was previously None
        assert reasoning_contents[4] == "Thinking chunk 5"  # This was previously None

        # Verify none of them are None
        for i, rc in enumerate(reasoning_contents):
            assert rc is not None, f"Chunk {i + 1} reasoning_content should not be None"

    def test_thinking_to_content_transition(self):
        """
        Test that transition from thinking to regular content works correctly.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        # First: thinking chunks
        thinking_chunk = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "thinking": "Let me think about this..."},
            "done": False,
        }
        result1 = iterator.chunk_parser(thinking_chunk)
        assert result1.choices[0].delta.reasoning_content == "Let me think about this..."
        assert result1.choices[0].delta.content is None

        # Then: regular content chunk
        content_chunk = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "content": "Here is my answer."},
            "done": False,
        }
        result2 = iterator.chunk_parser(content_chunk)
        assert result2.choices[0].delta.content == "Here is my answer."
        # reasoning_content is not set when there's no thinking in the chunk
        assert getattr(result2.choices[0].delta, "reasoning_content", None) is None

    def test_thinking_and_content_in_same_chunk(self):
        """
        Test that a chunk containing both thinking and content preserves both fields.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        chunk = {
            "model": "deepseek-r1",
            "message": {
                "role": "assistant",
                "thinking": "Let me reason first.",
                "content": "Final answer.",
            },
            "done": False,
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].delta.reasoning_content == "Let me reason first."
        assert result.choices[0].delta.content == "Final answer."

    def test_streaming_chunks_ignore_inactive_empty_reasoning_fields(self):
        """
        Test that Ollama chunks with inactive empty fields stay in the active delta.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        chunk = {
            "model": "deepseek-r1",
            "message": {
                "role": "assistant",
                "thinking": "Let me reason first.",
                "content": "",
            },
            "done": False,
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].delta.reasoning_content == "Let me reason first."
        assert result.choices[0].delta.content is None
        assert iterator.finished_reasoning_content is False

        content_chunk = {
            "model": "deepseek-r1",
            "message": {
                "role": "assistant",
                "thinking": "",
                "content": "Final answer.",
            },
            "done": False,
        }

        result = iterator.chunk_parser(content_chunk)

        assert getattr(result.choices[0].delta, "reasoning_content", None) is None
        assert result.choices[0].delta.content == "Final answer."
        assert iterator.finished_reasoning_content is True

    def test_think_tags_in_content(self):
        """
        Test that <think> tags embedded in content are properly parsed.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        # Content with <think> tag
        chunk1 = {
            "model": "deepseek-r1",
            "message": {
                "role": "assistant",
                "content": "<think>I need to analyze this",
            },
            "done": False,
        }
        result1 = iterator.chunk_parser(chunk1)
        assert result1.choices[0].delta.reasoning_content == "I need to analyze this"
        assert result1.choices[0].delta.content is None

        # Content with </think> tag (end of thinking)
        chunk2 = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "content": "</think>The answer is 42."},
            "done": False,
        }
        result2 = iterator.chunk_parser(chunk2)
        assert result2.choices[0].delta.content == "The answer is 42."
        # reasoning_content is not set when it's regular content
        assert getattr(result2.choices[0].delta, "reasoning_content", None) is None

    def test_done_chunk_with_thinking(self):
        """
        Test that the final chunk with done=True and thinking content works.
        """
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        # Final chunk with thinking
        done_chunk = {
            "model": "deepseek-r1",
            "message": {"role": "assistant", "thinking": "Final thought"},
            "done": True,
            "done_reason": "stop",
        }
        result = iterator.chunk_parser(done_chunk)
        assert result.choices[0].delta.reasoning_content == "Final thought"
        assert result.choices[0].finish_reason == "stop"


class TestOllamaToolCallTransformation:
    def test_transform_request_preserves_tool_calls(self):
        """
        tool_calls on assistant messages must survive transform_request.
        Previously the translated OllamaToolCall list was built but never
        copied into the outgoing OllamaChatCompletionMessage, so Ollama
        received {role: assistant, content: ''} with no tool_calls and
        the model re-issued the same call on every turn.
        Regression: https://github.com/BerriAI/litellm/issues/26094
        """
        config = OllamaChatConfig()
        messages = cast(
            list[AllMessageValues],
            [
                {"role": "user", "content": "What's the weather in SF?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco, CA"}',
                            },
                        }
                    ],
                },
            ],
        )

        result = config.transform_request(
            model="gemma4:27b",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assistant_msg = result["messages"][1]
        assert "tool_calls" in assistant_msg, "tool_calls must be forwarded to Ollama"
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"location": "San Francisco, CA"}

    def test_transform_request_forwards_tool_call_id(self):
        """
        tool_call_id on role:tool messages must be forwarded so Ollama can
        resolve the tool name from the conversation history.
        Regression: https://github.com/BerriAI/litellm/issues/26094
        """
        config = OllamaChatConfig()
        messages = cast(
            list[AllMessageValues],
            [
                {"role": "user", "content": "What's the weather in SF?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco, CA"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "Sunny, 72°F",
                },
            ],
        )

        result = config.transform_request(
            model="gemma4:27b",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == "Sunny, 72°F"
        assert "tool_call_id" in tool_msg, "tool_call_id must be forwarded to Ollama"
        assert tool_msg["tool_call_id"] == "call_abc123"


class TestOllamaReasoningEffortToThinkMapping:
    """Regression tests for reasoning_effort to ollama think parameter translation.

    Ollama's think field accepts booleans or levels (low/medium/high/max).
    See https://docs.ollama.com/capabilities/thinking.

    GPT-OSS only accepts low/medium/high (no max, no boolean).
    """

    @pytest.mark.parametrize(
        "effort,expected_think",
        [
            ("none", False),
            ("minimal", False),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
        ],
    )
    def test_non_gpt_oss_effort_to_think_mapping(self, effort, expected_think):
        """Each reasoning_effort level maps to the correct ollama think value."""
        config = OllamaChatConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params={},
            model="qwen3",
            drop_params=False,
        )
        assert optional_params["think"] == expected_think

    @pytest.mark.parametrize(
        "effort,expected_think",
        [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
            ("none", "low"),
            ("minimal", "low"),
        ],
    )
    def test_gpt_oss_effort_to_think_mapping(self, effort, expected_think):
        """GPT-OSS only accepts low/medium/high; max/xhigh clamp to high, none/minimal to low."""
        config = OllamaChatConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params={},
            model="gpt-oss:120b",
            drop_params=False,
        )
        assert optional_params["think"] == expected_think

    def test_non_string_reasoning_effort_falls_back_to_bool(self):
        """A non-string value (e.g. True) is coerced to a boolean."""
        config = OllamaChatConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": True},
            optional_params={},
            model="qwen3",
            drop_params=False,
        )
        assert optional_params["think"] is True

    def test_reasoning_effort_none_value_does_not_set_think(self):
        """When reasoning_effort is explicitly None, think is not set at all."""
        config = OllamaChatConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": None},
            optional_params={},
            model="qwen3",
            drop_params=False,
        )
        assert "think" not in optional_params


class TestOllamaDurationsSurfaced:
    """Ollama's native GPU-time fields must reach _hidden_params so a custom
    callback can accumulate real GPU time (not wall-clock) per deployment.

    Ollama returns total_duration/load_duration/prompt_eval_duration/eval_duration
    in nanoseconds on the final done:true chunk. Without these surfacing, a
    duration-based balancer would have to fall back to wall-clock request time,
    which over-counts network/overhead and misleads both the protective limit
    and the discovery tracker.
    """

    @staticmethod
    def _done_response_with_durations() -> dict:
        return {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-15T00:00:00Z",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
            "total_duration": 1_500_000_000,
            "load_duration": 200_000_000,
            "prompt_eval_duration": 300_000_000,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 5,
            "eval_count": 10,
        }

    def test_transform_response_stashes_gpu_time_on_hidden_params(self):
        from unittest.mock import MagicMock

        config = OllamaChatConfig()
        ollama_response = self._done_response_with_durations()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)

        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]

        result = config.transform_response(
            model="glm-5.2:cloud",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        psf = result._hidden_params["provider_specific_fields"]
        durations = psf["ollama_durations"]
        assert durations["eval_seconds"] == pytest.approx(1.0)
        assert durations["prompt_eval_seconds"] == pytest.approx(0.3)
        assert psf["gpu_time_seconds"] == pytest.approx(1.3)

    def test_chunk_parser_stashes_gpu_time_on_final_done_chunk(self):
        iterator = OllamaChatCompletionResponseIterator(streaming_response=iter([]), sync_stream=True, json_mode=False)
        final_chunk = self._done_response_with_durations()

        parsed = iterator.chunk_parser(final_chunk)

        psf = parsed._hidden_params["provider_specific_fields"]
        assert psf["gpu_time_seconds"] == pytest.approx(1.3)
        assert psf["ollama_durations"]["eval_seconds"] == pytest.approx(1.0)

    def test_chunk_parser_does_not_attach_on_intermediate_chunk(self):
        iterator = OllamaChatCompletionResponseIterator(streaming_response=iter([]), sync_stream=True, json_mode=False)
        intermediate = {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-15T00:00:00Z",
            "message": {"role": "assistant", "content": "hel"},
            "done": False,
        }

        parsed = iterator.chunk_parser(intermediate)

        psf = parsed._hidden_params.get("provider_specific_fields", {})
        assert "ollama_durations" not in psf
        assert "gpu_time_seconds" not in psf

    def test_apply_assembled_streaming_response_metadata_propagates_durations(self):
        iterator = OllamaChatCompletionResponseIterator(streaming_response=iter([]), sync_stream=True, json_mode=False)
        final_chunk = self._done_response_with_durations()
        parsed_final = iterator.chunk_parser(final_chunk)

        intermediate = {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-15T00:00:00Z",
            "message": {"role": "assistant", "content": "hel"},
            "done": False,
        }
        parsed_intermediate = iterator.chunk_parser(intermediate)

        assembled = ModelResponse(id="asm1", created=1, model="glm-5.2:cloud", object="chat.completion", choices=[])

        config = OllamaChatConfig()
        config.apply_assembled_streaming_response_metadata(
            response=assembled, chunks=[parsed_intermediate, parsed_final]
        )

        psf = assembled._hidden_params["provider_specific_fields"]
        assert psf["gpu_time_seconds"] == pytest.approx(1.3)
        assert psf["ollama_durations"]["eval_seconds"] == pytest.approx(1.0)

    def test_apply_assembled_streaming_response_metadata_noop_without_durations(self):
        chunk_without_durations = ModelResponseStream(
            id="c1",
            created=1,
            model="m",
            object="chat.completion.chunk",
            choices=[StreamingChoices(delta=Delta(content="hi"))],
        )
        assembled = ModelResponse(id="asm1", created=1, model="glm-5.2:cloud", object="chat.completion", choices=[])

        config = OllamaChatConfig()
        config.apply_assembled_streaming_response_metadata(response=assembled, chunks=[chunk_without_durations])

        assert assembled._hidden_params.get("provider_specific_fields") is None


class TestOllamaStreamingParallelToolCalls:
    """Tests for streaming parallel tool_calls from Ollama.

    Ollama's on-wire format (verified against GLM-5.2 via Ollama Cloud,
    captured 2026-07-26) puts `index` INSIDE `function` and `id` at the
    top level of each tool_call:

        {"tool_calls":[{
          "id":"call_2k0nylt2",
          "function":{"index":0,"name":"get_weather","arguments":{"location":"Jakarta"}}
        }]}

    OpenAI's streaming protocol (and LiteLLM's `get_combined_tool_content`
    in `litellm_core_utils/streaming_chunk_builder_utils.py`) keys
    tool_calls by the TOP-LEVEL `index`. Without promotion, every chunk's
    tool_call defaults to `index=0` in `Delta.__init__` (which resets
    `current_index=0` per construction), and all parallel tool_calls
    collapse into one slot with concatenated (broken JSON) arguments.

    The fix in `OllamaChatCompletionResponseIterator.chunk_parser`:
      1. Promotes `function.index` to top-level `tool_call["index"]`.
      2. Preserves Ollama's `id` (does NOT overwrite with a random UUID —
         the id is needed for tool_result correlation in the next turn).
         Only generates a UUID when Ollama did not send one AND arguments
         are complete.

    Design choice: Ollama is expected to always send `id` and `index` for
    tool_calls, so there is NO counter fallback. If `function.index` is
    missing, `Delta.__init__` will assign `index=0` (the original behavior
    for non-parallel calls). Malformed tool_calls (missing/null `function`)
    are NOT silently dropped — they propagate so the request fails loudly
    and the issue is visible.
    """

    @staticmethod
    def _make_chunk(
        *,
        content: str = "",
        tool_calls: Optional[list] = None,
        done: bool = False,
        done_reason: Optional[str] = None,
    ) -> dict:
        """Build an Ollama streaming chunk with arbitrary content/tool_calls."""
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        chunk: Dict[str, Any] = {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-26T00:00:00Z",
            "message": message,
            "done": done,
        }
        if done_reason is not None:
            chunk["done_reason"] = done_reason
        return chunk

    @staticmethod
    def _make_tool_call(
        *,
        name: str,
        arguments: Union[str, dict],
        index: Optional[int] = None,
        function_index: Optional[int] = None,
        id: Optional[str] = None,
        type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a single tool_call dict in Ollama's on-wire shape.

        - `index` → top-level `tool_call["index"]` (rare in real Ollama
          output; OpenAI streaming protocol expects it here).
        - `function_index` → `tool_call["function"]["index"]` (THIS is
          where Ollama Cloud actually puts it).
        - `id` → top-level `tool_call["id"]` (Ollama Cloud sends this
          for parallel tool calls, e.g. "call_2k0nylt2").
        - `type` → top-level `tool_call["type"]` (Ollama Cloud doesn't
          send it; LiteLLM's `Delta.__init__` defaults it to "function").

        Only includes optional fields when explicitly set, so we can
        test the parser's handling of missing keys.
        """
        function: Dict[str, Any] = {"name": name, "arguments": arguments}
        if function_index is not None:
            function["index"] = function_index
        tc: Dict[str, Any] = {"function": function}
        if index is not None:
            tc["index"] = index
        if id is not None:
            tc["id"] = id
        if type is not None:
            tc["type"] = type
        return tc

    # Real captured wire format from Ollama Cloud (2026-07-26).
    # Each chunk is exactly as observed in production, with `id` at top
    # level and `index` inside `function`, and arguments as a dict.
    REAL_CHUNKS = [
        {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-26T07:09:21.682986998Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2k0nylt2",
                        "function": {
                            "index": 0,
                            "name": "get_weather",
                            "arguments": {"location": "Jakarta"},
                        },
                    }
                ],
            },
            "done": False,
        },
        {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-26T07:09:21.727992145Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_gadd4l4t",
                        "function": {
                            "index": 1,
                            "name": "get_stock_price",
                            "arguments": {"symbol": "AAPL"},
                        },
                    }
                ],
            },
            "done": False,
        },
        {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-26T07:09:21.815847849Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_aobtrq8t",
                        "function": {
                            "index": 2,
                            "name": "convert_currency",
                            "arguments": {"from": "USD", "to": "IDR", "amount": 100},
                        },
                    }
                ],
            },
            "done": False,
        },
        {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-26T07:09:21.859707486Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_ken1nf7o",
                        "function": {
                            "index": 3,
                            "name": "get_current_time",
                            "arguments": {"city": "Tokyo"},
                        },
                    }
                ],
            },
            "done": False,
        },
        {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-26T07:09:21.904279913Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_i3cr5cqe",
                        "function": {
                            "index": 4,
                            "name": "search_user",
                            "arguments": {"name": "John"},
                        },
                    }
                ],
            },
            "done": False,
        },
    ]

    # ---------- Promotion of function.index to top-level ----------

    def test_function_index_promoted_to_top_level(self):
        """Ollama puts `index` inside `function`; OpenAI streaming protocol
        expects it at top-level. chunk_parser must promote it."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        parsed = iterator.chunk_parser(self.REAL_CHUNKS[0])
        tc = parsed.choices[0].delta.tool_calls[0]

        # Top-level index must be 0 (promoted from function.index=0).
        assert tc.index == 0, f"Expected top-level index=0 (promoted from function.index), got {tc.index}"

    def test_top_level_index_preserved_when_already_present(self):
        """If a chunk already has a top-level `index` (rare for Ollama but
        valid per OpenAI protocol), it must NOT be overwritten by
        `function.index`."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                self._make_tool_call(
                    name="tool_x",
                    arguments='{"k": 1}',
                    id="call_x",  # required by strict validation
                    index=10,  # top-level
                    function_index=99,  # inside function (should be ignored)
                ),
            ],
        )

        parsed = iterator.chunk_parser(chunk)
        tc = parsed.choices[0].delta.tool_calls[0]
        assert tc.index == 10, f"Top-level index=10 must win over function.index=99, got {tc.index}"

    def test_multiple_tool_calls_in_one_chunk_each_get_their_function_index(self):
        """If Ollama batches multiple tool_calls into ONE chunk, each
        tool_call's `function.index` must be promoted to its own top-level
        `index` — they must NOT all collapse to 0."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                self._make_tool_call(name="a", arguments='{"x": 1}', function_index=0, id="call_a"),
                self._make_tool_call(name="b", arguments='{"x": 2}', function_index=1, id="call_b"),
                self._make_tool_call(name="c", arguments='{"x": 3}', function_index=2, id="call_c"),
            ],
        )

        parsed = iterator.chunk_parser(chunk)
        indices = [tc.index for tc in parsed.choices[0].delta.tool_calls]
        assert indices == [0, 1, 2], f"Expected [0,1,2] (promoted from function.index), got {indices}"

    # ---------- Preservation of Ollama's id ----------

    def test_ollama_id_preserved_not_overwritten_with_uuid(self):
        """Ollama's `id` (e.g. "call_2k0nylt2") must be PRESERVED — not
        overwritten with a random UUID. The id is needed for tool_result
        correlation in the next turn."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        expected_ids = [
            "call_2k0nylt2",
            "call_gadd4l4t",
            "call_aobtrq8t",
            "call_ken1nf7o",
            "call_i3cr5cqe",
        ]
        actual_ids = []
        for chunk in self.REAL_CHUNKS:
            parsed = iterator.chunk_parser(chunk)
            actual_ids.append(parsed.choices[0].delta.tool_calls[0].id)

        assert actual_ids == expected_ids, f"Ollama's ids must be preserved exactly, got {actual_ids}"
        # None of them should be a random UUID (which would have a different
        # format: 36 chars, hyphen-separated, not starting with "call_").
        for actual_id in actual_ids:
            assert actual_id.startswith("call_"), f"Id should be Ollama's 'call_...' format, got {actual_id}"

    def test_missing_id_raises_ollama_error(self):
        """Strict validation: when Ollama did not send a top-level `id`,
        chunk_parser raises OllamaError (HTTP 400) instead of silently
        generating a replacement UUID. The id is required for tool_result
        correlation in the next turn; silently generating one would hide a
        structural change in Ollama's response format."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                self._make_tool_call(
                    name="get_weather",
                    arguments='{"location": "Jakarta"}',
                    function_index=0,
                    # no `id` provided — strict validation must reject this
                ),
            ],
        )

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        # Error message must name the missing field and point to the file
        # to update, so a structural change is immediately actionable.
        msg = str(exc_info.value)
        assert "missing a top-level string `id`" in msg, f"Error message must name the missing `id` field, got: {msg}"
        assert "litellm/llms/ollama/chat/transformation.py" in msg, (
            f"Error message must point to the file to update, got: {msg}"
        )

    def test_missing_id_with_incomplete_args_also_raises(self):
        """Strict validation rejects missing `id` regardless of whether
        arguments are complete or partial. The id is required structurally,
        not conditionally on argument completeness."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                self._make_tool_call(
                    name="partial",
                    arguments='{"location": "Ja',  # incomplete JSON
                    function_index=0,
                    # no `id` provided
                ),
            ],
        )

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        assert "missing a top-level string `id`" in str(exc_info.value)

    def test_dict_style_arguments_preserved_as_id_anchor(self):
        """Ollama natively sends `arguments` as a dict (not JSON string).
        Strict validation accepts dict arguments, and `Function.__init__`
        converts the dict to a JSON string for the final response."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                self._make_tool_call(
                    name="get_weather",
                    arguments={"location": "Tokyo"},  # dict, not JSON string
                    function_index=0,
                    id="call_dict_args",
                ),
            ],
        )

        parsed = iterator.chunk_parser(chunk)
        tc = parsed.choices[0].delta.tool_calls[0]
        assert tc.id == "call_dict_args", "Ollama's id must be preserved"
        # Function.__init__ converts dict to JSON string.
        assert tc.function.arguments == '{"location": "Tokyo"}'

    # ---------- End-to-end aggregation via ChunkProcessor ----------

    def test_real_wire_format_aggregates_to_5_separate_tool_calls(self):
        """End-to-end with the REAL Ollama Cloud wire format: 5 chunks must
        aggregate to 5 separate tool_calls, each with the original Ollama id,
        correct name, and intact (un-concatenated) arguments."""
        from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
            ChunkProcessor,
        )

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        streamed = [iterator.chunk_parser(c).model_dump() for c in self.REAL_CHUNKS]
        processor = ChunkProcessor(streamed)
        processor.build_base_response(streamed)

        tc_chunks = [
            c
            for c in streamed
            if len(c["choices"]) > 0
            and "tool_calls" in c["choices"][0]["delta"]
            and c["choices"][0]["delta"]["tool_calls"] is not None
        ]
        tcs = processor.get_combined_tool_content(tc_chunks)

        assert len(tcs) == 5, f"Real wire format must produce 5 tool_calls, got {len(tcs)}"

        # Order follows promoted index: 0, 1, 2, 3, 4 → tool names in order.
        expected = [
            ("call_2k0nylt2", "get_weather", '{"location": "Jakarta"}'),
            ("call_gadd4l4t", "get_stock_price", '{"symbol": "AAPL"}'),
            ("call_aobtrq8t", "convert_currency", '{"from": "USD", "to": "IDR", "amount": 100}'),
            ("call_ken1nf7o", "get_current_time", '{"city": "Tokyo"}'),
            ("call_i3cr5cqe", "search_user", '{"name": "John"}'),
        ]
        for i, (exp_id, exp_name, exp_args) in enumerate(expected):
            assert tcs[i].id == exp_id, f"tool_call[{i}].id = {tcs[i].id!r}, want {exp_id!r}"
            assert tcs[i].function.name == exp_name, f"tool_call[{i}].name = {tcs[i].function.name!r}"
            assert tcs[i].function.arguments == exp_args, (
                f"tool_call[{i}].args = {tcs[i].function.arguments!r}, want {exp_args!r}"
            )

    def test_aggregated_mixed_chunk_sizes_produce_correct_tool_calls(self):
        """End-to-end via ChunkProcessor: mix of 1-tool-chunks and
        2-tools-per-chunk. The aggregated response must contain all
        tool_calls with correct names and arguments, in index order."""
        from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
            ChunkProcessor,
        )

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunks = [
            # chunk 1: 2 tool_calls (function.index 0, 1)
            self._make_chunk(
                tool_calls=[
                    self._make_tool_call(name="a", arguments='{"x": 1}', function_index=0, id="call_a"),
                    self._make_tool_call(name="b", arguments='{"x": 2}', function_index=1, id="call_b"),
                ],
            ),
            # chunk 2: 1 tool_call (function.index 2)
            self._make_chunk(
                tool_calls=[self._make_tool_call(name="c", arguments='{"x": 3}', function_index=2, id="call_c")],
            ),
            # chunk 3: 2 tool_calls (function.index 3, 4)
            self._make_chunk(
                tool_calls=[
                    self._make_tool_call(name="d", arguments='{"x": 4}', function_index=3, id="call_d"),
                    self._make_tool_call(name="e", arguments='{"x": 5}', function_index=4, id="call_e"),
                ],
            ),
        ]

        streamed = [iterator.chunk_parser(c).model_dump() for c in chunks]
        processor = ChunkProcessor(streamed)
        processor.build_base_response(streamed)

        tc_chunks = [
            c
            for c in streamed
            if len(c["choices"]) > 0
            and "tool_calls" in c["choices"][0]["delta"]
            and c["choices"][0]["delta"]["tool_calls"] is not None
        ]
        tcs = processor.get_combined_tool_content(tc_chunks)

        assert len(tcs) == 5
        names = [tc.function.name for tc in tcs]
        assert names == ["a", "b", "c", "d", "e"], f"Got {names}"
        args = [tc.function.arguments for tc in tcs]
        assert args == ['{"x": 1}', '{"x": 2}', '{"x": 3}', '{"x": 4}', '{"x": 5}']
        ids = [tc.id for tc in tcs]
        assert ids == ["call_a", "call_b", "call_c", "call_d", "call_e"]

    # ---------- done chunk + finish_reason ----------

    def test_done_chunk_with_tool_calls_sets_finish_reason_tool_calls(self):
        """The final `done: True` chunk with tool_calls present must set
        finish_reason="tool_calls" (existing behavior, must not regress).
        function.index must still be promoted on the done chunk."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # First: 2 tool_calls in earlier chunks
        c1 = self._make_chunk(
            tool_calls=[self._make_tool_call(name="a", arguments='{"x": 1}', function_index=0, id="call_a")],
        )
        c2 = self._make_chunk(
            tool_calls=[self._make_tool_call(name="b", arguments='{"x": 2}', function_index=1, id="call_b")],
        )
        # Final done chunk with another tool_call (some models do this)
        done_chunk = self._make_chunk(
            tool_calls=[self._make_tool_call(name="c", arguments='{"x": 3}', function_index=2, id="call_c")],
            done=True,
            done_reason="stop",
        )

        p1 = iterator.chunk_parser(c1)
        p2 = iterator.chunk_parser(c2)
        p_done = iterator.chunk_parser(done_chunk)

        assert p1.choices[0].delta.tool_calls[0].index == 0
        assert p2.choices[0].delta.tool_calls[0].index == 1
        assert p_done.choices[0].delta.tool_calls[0].index == 2
        # Done chunk with tool_calls → finish_reason must be "tool_calls"
        # (override of done_reason="stop"), per existing fix for
        # https://github.com/BerriAI/litellm/issues/18922
        assert p_done.choices[0].finish_reason == "tool_calls"

    def test_done_chunk_without_tool_calls_keeps_finish_reason_from_provider(self):
        """If the done chunk has NO tool_calls, finish_reason must come
        from done_reason (or default "stop"), NOT be overridden to
        "tool_calls". This is the existing behavior and must not regress."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # Tool calls in earlier chunks, then a done chunk with no tool_calls.
        c1 = self._make_chunk(
            tool_calls=[self._make_tool_call(name="a", arguments='{"x": 1}', function_index=0, id="call_a")],
        )
        done_chunk = self._make_chunk(content="", done=True, done_reason="stop")

        p1 = iterator.chunk_parser(c1)
        p_done = iterator.chunk_parser(done_chunk)

        assert p1.choices[0].delta.tool_calls[0].index == 0
        # Done chunk has no tool_calls → finish_reason comes from done_reason.
        assert p_done.choices[0].finish_reason == "stop"

    # ---------- Interleaved content + thinking + tool_calls ----------

    def test_interleaved_content_and_tool_call_chunks(self):
        """Real streaming: model may emit content chunks interleaved with
        tool_call chunks. function.index promotion must work on each
        tool_call chunk regardless of intervening content chunks."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # Chunk 1: content "Thinking..."
        c1 = self._make_chunk(content="Thinking...")
        # Chunk 2: tool_call #1 (function.index=0)
        c2 = self._make_chunk(
            tool_calls=[self._make_tool_call(name="tool_a", arguments='{"x": 1}', function_index=0, id="call_a")],
        )
        # Chunk 3: content "Now calling..."
        c3 = self._make_chunk(content="Now calling...")
        # Chunk 4: tool_call #2 (function.index=1)
        c4 = self._make_chunk(
            tool_calls=[self._make_tool_call(name="tool_b", arguments='{"x": 2}', function_index=1, id="call_b")],
        )

        iterator.chunk_parser(c1)
        p2 = iterator.chunk_parser(c2)
        iterator.chunk_parser(c3)
        p4 = iterator.chunk_parser(c4)

        assert p2.choices[0].delta.tool_calls[0].index == 0
        assert p4.choices[0].delta.tool_calls[0].index == 1

    def test_thinking_chunks_then_real_ollama_cloud_tool_calls(self):
        """Reproduces the EXACT production scenario: many `thinking` chunks
        first, then 5 parallel tool_call chunks. function.index promotion
        and id preservation must work after the thinking phase."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # A few thinking chunks (real Ollama emits ~30 of these; we use 3).
        thinking_chunks = [
            {
                "model": "glm-5.2:cloud",
                "created_at": "2026-07-26T07:09:21.072076096Z",
                "message": {"role": "assistant", "content": "", "thinking": "The user"},
                "done": False,
            },
            {
                "model": "glm-5.2:cloud",
                "created_at": "2026-07-26T07:09:21.08807962Z",
                "message": {"role": "assistant", "content": "", "thinking": " wants"},
                "done": False,
            },
            {
                "model": "glm-5.2:cloud",
                "created_at": "2026-07-26T07:09:21.088121889Z",
                "message": {"role": "assistant", "content": "", "thinking": " parallel calls."},
                "done": False,
            },
        ]

        for tc_chunk in thinking_chunks:
            iterator.chunk_parser(tc_chunk)

        # Now feed the real tool_call chunks.
        results = [iterator.chunk_parser(c) for c in self.REAL_CHUNKS]
        indices = [r.choices[0].delta.tool_calls[0].index for r in results]
        ids = [r.choices[0].delta.tool_calls[0].id for r in results]

        assert indices == [0, 1, 2, 3, 4], f"Promoted function.index must survive thinking phase, got {indices}"
        assert ids == [
            "call_2k0nylt2",
            "call_gadd4l4t",
            "call_aobtrq8t",
            "call_ken1nf7o",
            "call_i3cr5cqe",
        ], f"Ollama ids must survive thinking phase, got {ids}"

    # ---------- Stress test ----------

    def test_many_tool_calls_across_many_chunks_indices_match_function_index(self):
        """Stress test: 50 tool_calls across 50 chunks. Each chunk's
        promoted top-level index must match its `function.index` exactly
        — no duplicates, no gaps, no off-by-one."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        n = 50
        chunks = [
            self._make_chunk(
                tool_calls=[
                    self._make_tool_call(
                        name=f"tool_{i}",
                        arguments=f'{{"i": {i}}}',
                        function_index=i,
                        id=f"call_{i}",
                    ),
                ],
            )
            for i in range(n)
        ]

        indices = []
        ids = []
        for c in chunks:
            parsed = iterator.chunk_parser(c)
            indices.append(parsed.choices[0].delta.tool_calls[0].index)
            ids.append(parsed.choices[0].delta.tool_calls[0].id)

        assert indices == list(range(n)), (
            f"Expected indices [0..{n - 1}] matching function.index, got {indices[:10]}...{indices[-10:] if n > 20 else indices}"
        )
        assert ids == [f"call_{i}" for i in range(n)], f"Ids must be preserved"
        assert len(set(indices)) == n, "Indices must be unique"

    def test_many_tool_calls_batched_in_one_chunk_indices_match_function_index(self):
        """Stress test variant: 50 tool_calls ALL in one chunk. Each
        tool_call's promoted top-level index must match its `function.index`."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        n = 50
        chunk = self._make_chunk(
            tool_calls=[
                self._make_tool_call(
                    name=f"tool_{i}",
                    arguments=f'{{"i": {i}}}',
                    function_index=i,
                    id=f"call_{i}",
                )
                for i in range(n)
            ],
        )

        parsed = iterator.chunk_parser(chunk)
        indices = [tc.index for tc in parsed.choices[0].delta.tool_calls]
        assert indices == list(range(n))
        ids = [tc.id for tc in parsed.choices[0].delta.tool_calls]
        assert ids == [f"call_{i}" for i in range(n)]

    # ---------- Non-parallel baseline ----------

    def test_single_tool_call_gets_index_zero(self):
        """Non-parallel baseline: a single tool_call with function.index=0
        must get top-level index=0. Regression guard for the common
        single-tool-call case."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                self._make_tool_call(
                    name="only_tool",
                    arguments='{"x": 1}',
                    function_index=0,
                    id="call_only",
                ),
            ],
        )

        parsed = iterator.chunk_parser(chunk)
        tc = parsed.choices[0].delta.tool_calls[0]
        assert tc.index == 0
        assert tc.id == "call_only"

    # ---------- Malformed tool_calls: fail loud, do NOT silently drop ----------

    def test_tool_call_with_no_function_key_raises(self):
        """A malformed tool_call missing the `function` key must raise
        OllamaError (NOT be silently dropped). The request fails so the
        issue is visible. Ollama is expected to always send `function`,
        so this is a real error condition, not a recoverable one."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[{"type": "function", "id": "call_x"}],  # no `function` key
        )

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        msg = str(exc_info.value)
        assert "missing a `function` dict" in msg, f"Error message must name the missing `function` field, got: {msg}"
        assert "litellm/llms/ollama/chat/transformation.py" in msg

    def test_tool_call_with_null_function_raises(self):
        """A tool_call with `"function": null` must raise OllamaError."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[{"function": None, "id": "call_x"}],
        )

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        assert "missing a `function` dict" in str(exc_info.value)

    def test_tool_call_with_missing_function_index_raises(self):
        """If Ollama moves/renames `function.index` (e.g. to `function.idx`),
        strict validation must fail loud — otherwise parallel tool_calls
        would silently collapse to index=0 and merge into one broken entry
        (the original bug)."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # Simulate Ollama renaming `index` to `idx` inside function.
        chunk = self._make_chunk(
            tool_calls=[
                {
                    "id": "call_a",
                    "function": {
                        "idx": 0,  # renamed from `index` — validation must catch
                        "name": "get_weather",
                        "arguments": '{"location": "Jakarta"}',
                    },
                },
            ],
        )

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        msg = str(exc_info.value)
        assert "missing an integer `index`" in msg, (
            f"Error message must name the missing `function.index` field, got: {msg}"
        )
        assert "litellm/llms/ollama/chat/transformation.py" in msg

    def test_tool_call_with_string_function_index_raises(self):
        """If Ollama sends `function.index` as a string (e.g. "0" instead
        of int 0), strict validation must fail loud — type coercion could
        mask a structural change."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                self._make_tool_call(
                    name="get_weather",
                    arguments='{"location": "Jakarta"}',
                    function_index="0",  # string, not int
                    id="call_a",
                ),
            ],
        )

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        assert "missing an integer `index`" in str(exc_info.value)

    def test_tool_call_with_missing_function_name_raises(self):
        """If Ollama removes `function.name`, strict validation must fail
        loud — without a name, the downstream aggregator can't build a
        meaningful tool_call."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                {
                    "id": "call_a",
                    "function": {
                        "index": 0,
                        # no `name` field
                        "arguments": '{"location": "Jakarta"}',
                    },
                },
            ],
        )

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        msg = str(exc_info.value)
        assert "missing a non-empty string `name`" in msg, (
            f"Error message must name the missing `function.name` field, got: {msg}"
        )

    def test_tool_call_with_missing_function_arguments_raises(self):
        """If Ollama removes `function.arguments`, strict validation must
        fail loud — without arguments, the tool_call is meaningless."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(
            tool_calls=[
                {
                    "id": "call_a",
                    "function": {
                        "index": 0,
                        "name": "get_weather",
                        # no `arguments` field
                    },
                },
            ],
        )

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        msg = str(exc_info.value)
        assert "missing `arguments`" in msg, (
            f"Error message must name the missing `function.arguments` field, got: {msg}"
        )

    def test_tool_call_with_non_string_id_raises(self):
        """If Ollama sends `id` as a non-string (e.g. int), strict
        validation must fail loud — type coercion could mask a structural
        change."""
        from litellm.llms.ollama.common_utils import OllamaError

        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = {
            "model": "glm-5.2:cloud",
            "created_at": "2026-07-26T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": 12345,  # int, not string
                        "function": {
                            "index": 0,
                            "name": "get_weather",
                            "arguments": '{"location": "Jakarta"}',
                        },
                    }
                ],
            },
            "done": False,
        }

        with pytest.raises(OllamaError) as exc_info:
            iterator.chunk_parser(chunk)

        assert "missing a top-level string `id`" in str(exc_info.value)

    # ---------- Empty tool_calls list ----------

    def test_empty_tool_calls_list_is_no_op(self):
        """An empty `tool_calls: []` list is a no-op — the delta carries
        an empty list (matching what `Delta.__init__` does with an empty
        list input). No error, no index assignment."""
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        chunk = self._make_chunk(tool_calls=[])
        parsed = iterator.chunk_parser(chunk)

        delta = parsed.choices[0].delta
        assert delta.tool_calls is not None
        assert len(delta.tool_calls) == 0


class TestOllamaPromptEvalCachedCount:
    """Ollama's prompt_eval_cached_count is the cache-hit subset of the full
    prompt_eval_count. It must surface as usage.prompt_tokens_details.cached_tokens
    while prompt_tokens keeps reporting the full prompt_eval_count total."""

    def _call_transform_response(self, ollama_response: dict):
        config = OllamaChatConfig()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_response
        mock_response.text = json.dumps(ollama_response)
        model_response = ModelResponse()
        model_response.choices = [Choices(message=Message(content=""), index=0)]
        return config.transform_response(
            model="qwen3:14b",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

    def test_transform_response_maps_prompt_eval_cached_count(self):
        result = self._call_transform_response(
            {
                "model": "qwen3:14b",
                "created_at": "2025-01-11T00:00:00.000000Z",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
                "prompt_eval_count": 100,
                "prompt_eval_cached_count": 40,
                "eval_count": 50,
            }
        )

        assert result.usage.prompt_tokens == 100
        assert result.usage.prompt_tokens_details is not None
        assert result.usage.prompt_tokens_details.cached_tokens == 40
        assert result.usage.total_tokens == 150

    def test_transform_response_without_prompt_eval_cached_count(self):
        result = self._call_transform_response(
            {
                "model": "qwen3:14b",
                "created_at": "2025-01-11T00:00:00.000000Z",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
                "prompt_eval_count": 100,
                "eval_count": 50,
            }
        )

        assert result.usage.prompt_tokens == 100
        assert result.usage.prompt_tokens_details is None

    def test_chunk_parser_maps_prompt_eval_cached_count(self):
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        done_chunk = {
            "model": "qwen3:14b",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 100,
            "prompt_eval_cached_count": 40,
            "eval_count": 50,
        }

        result = iterator.chunk_parser(done_chunk)

        assert result.usage.prompt_tokens == 100
        assert result.usage.prompt_tokens_details is not None
        assert result.usage.prompt_tokens_details.cached_tokens == 40

    def test_chunk_parser_without_prompt_eval_cached_count(self):
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        chunk = {
            "model": "qwen3:14b",
            "message": {"role": "assistant", "content": "He"},
            "done": False,
        }

        result = iterator.chunk_parser(chunk)

        assert result.usage.prompt_tokens == 0
        assert result.usage.prompt_tokens_details is None

    def test_streaming_final_usage_keeps_cached_tokens(self):
        iterator = OllamaChatCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )
        content_chunk = iterator.chunk_parser(
            {
                "model": "qwen3:14b",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": False,
            }
        )
        done_chunk = iterator.chunk_parser(
            {
                "model": "qwen3:14b",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 100,
                "prompt_eval_cached_count": 40,
                "eval_count": 50,
            }
        )

        completion_stream = ModelResponseListIterator(model_responses=[content_chunk, done_chunk])
        response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model="ollama_chat/qwen3:14b",
            custom_llm_provider="ollama_chat",
            logging_obj=Logging(
                model="ollama_chat/qwen3:14b",
                messages=[{"role": "user", "content": "Hey"}],
                stream=True,
                call_type="completion",
                start_time=time.time(),
                litellm_call_id="12345",
                function_id="1245",
            ),
            stream_options={"include_usage": True},
        )

        final_usage = None
        for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                final_usage = chunk.usage

        assert final_usage is not None
        assert final_usage.prompt_tokens == 100
        assert final_usage.completion_tokens == 50
        assert final_usage.prompt_tokens_details is not None
        assert final_usage.prompt_tokens_details.cached_tokens == 40
