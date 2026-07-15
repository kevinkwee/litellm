import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from litellm.llms.ollama.duration_utils import (
    OLLAMA_DURATIONS_KEY,
    extract_ollama_durations,
    gpu_time_seconds,
    attach_durations_to_response,
    attach_durations_to_chunk,
)
from litellm.types.utils import ModelResponse, ModelResponseStream, StreamingChoices, Delta


def _ollama_done_response() -> dict:
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


class TestExtractOllamaDurations:
    def test_converts_nanoseconds_to_seconds(self):
        durations = extract_ollama_durations(_ollama_done_response())
        assert durations["total_seconds"] == pytest.approx(1.5)
        assert durations["load_seconds"] == pytest.approx(0.2)
        assert durations["prompt_eval_seconds"] == pytest.approx(0.3)
        assert durations["eval_seconds"] == pytest.approx(1.0)

    def test_gpu_time_is_prompt_eval_plus_eval(self):
        durations = extract_ollama_durations(_ollama_done_response())
        assert gpu_time_seconds(durations) == pytest.approx(1.3)

    def test_missing_duration_fields_yield_zero(self):
        durations = extract_ollama_durations({"model": "x", "done": True})
        assert durations["total_seconds"] == 0.0
        assert gpu_time_seconds(durations) == 0.0

    def test_non_numeric_duration_fields_yield_zero(self):
        durations = extract_ollama_durations({"total_duration": "oops", "eval_duration": None, "done": True})
        assert durations["total_seconds"] == 0.0
        assert durations["eval_seconds"] == 0.0


class TestAttachDurationsToResponse:
    def test_stashes_under_provider_specific_fields(self):
        response = ModelResponse(id="r1", created=1, model="m", object="chat.completion", choices=[])
        durations = extract_ollama_durations(_ollama_done_response())
        attach_durations_to_response(response, durations)
        psf = response._hidden_params["provider_specific_fields"]
        assert psf[OLLAMA_DURATIONS_KEY] is durations
        assert psf["gpu_time_seconds"] == pytest.approx(1.3)

    def test_preserves_existing_provider_specific_fields(self):
        response = ModelResponse(id="r1", created=1, model="m", object="chat.completion", choices=[])
        response._hidden_params["provider_specific_fields"] = {"existing": "kept"}
        durations = extract_ollama_durations(_ollama_done_response())
        attach_durations_to_response(response, durations)
        psf = response._hidden_params["provider_specific_fields"]
        assert psf["existing"] == "kept"
        assert OLLAMA_DURATIONS_KEY in psf


class TestAttachDurationsToChunk:
    def test_stashes_on_chunk_hidden_params(self):
        chunk = ModelResponseStream(
            id="c1",
            created=1,
            model="m",
            object="chat.completion.chunk",
            choices=[StreamingChoices(delta=Delta(content="hi"))],
        )
        durations = extract_ollama_durations(_ollama_done_response())
        attach_durations_to_chunk(chunk, durations)
        psf = chunk._hidden_params["provider_specific_fields"]
        assert psf[OLLAMA_DURATIONS_KEY] is durations
        assert psf["gpu_time_seconds"] == pytest.approx(1.3)
