from typing import TypedDict

from litellm.types.utils import ModelResponse, ModelResponseStream


class OllamaDurations(TypedDict):
    total_seconds: float
    load_seconds: float
    prompt_eval_seconds: float
    eval_seconds: float


OLLAMA_DURATIONS_KEY = "ollama_durations"
_GPU_TIME_KEY = "gpu_time_seconds"

_NANOSECOND_CONVERSION = 1_000_000_000


def _nanos_to_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) / _NANOSECOND_CONVERSION


def extract_ollama_durations(response_json: dict) -> OllamaDurations:
    return OllamaDurations(
        total_seconds=_nanos_to_seconds(response_json.get("total_duration")),
        load_seconds=_nanos_to_seconds(response_json.get("load_duration")),
        prompt_eval_seconds=_nanos_to_seconds(response_json.get("prompt_eval_duration")),
        eval_seconds=_nanos_to_seconds(response_json.get("eval_duration")),
    )


def gpu_time_seconds(durations: OllamaDurations) -> float:
    breakdown = durations["prompt_eval_seconds"] + durations["eval_seconds"]
    if breakdown > 0:
        return breakdown
    return max(durations["total_seconds"] - durations["load_seconds"], 0.0)


def attach_durations_to_response(model_response: ModelResponse, durations: OllamaDurations) -> None:
    provider_specific = dict(model_response._hidden_params.get("provider_specific_fields") or {})
    provider_specific[OLLAMA_DURATIONS_KEY] = durations
    provider_specific[_GPU_TIME_KEY] = gpu_time_seconds(durations)
    model_response._hidden_params["provider_specific_fields"] = provider_specific


def attach_durations_to_chunk(chunk: ModelResponseStream, durations: OllamaDurations) -> None:
    provider_specific = dict(chunk._hidden_params.get("provider_specific_fields") or {})
    provider_specific[OLLAMA_DURATIONS_KEY] = durations
    provider_specific[_GPU_TIME_KEY] = gpu_time_seconds(durations)
    chunk._hidden_params["provider_specific_fields"] = provider_specific
    chunk.provider_specific_fields = provider_specific
