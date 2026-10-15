"""Unit tests for the OpenCode session helpers in litellm/llms/opencode/common_utils.py."""

import pytest

from litellm.llms.opencode.common_utils import (
    OPENCODE_SESSION_HEADER,
    apply_opencode_session_headers,
    is_opencode_api_base,
    resolve_opencode_session_id,
)
from pydantic import ValidationError


class TestIsOpencodeApiBase:
    def test_gateway_urls_are_opencode(self):
        assert is_opencode_api_base("https://opencode.ai/zen/go/v1/") is True
        assert is_opencode_api_base("https://opencode.ai/zen/v1") is True
        assert is_opencode_api_base("https://gateway.opencode.ai/v1") is True

    def test_hostname_casing_is_normalized(self):
        assert is_opencode_api_base("https://OPencode.AI/zen/go/v1/") is True

    def test_other_hosts_are_not_opencode(self):
        assert is_opencode_api_base("https://api.openai.com/v1") is False
        assert is_opencode_api_base("http://localhost:4000/v1") is False

    def test_lookalike_hosts_are_not_opencode(self):
        assert is_opencode_api_base("https://evil-opencode.ai/v1") is False
        assert is_opencode_api_base("https://opencode.ai.evil.com/v1") is False

    @pytest.mark.parametrize("api_base", [None, "", "opencode.ai/zen/go/v1"])
    def test_unparseable_api_bases_are_not_opencode(self, api_base):
        assert is_opencode_api_base(api_base) is False


class TestResolveOpencodeSessionId:
    def test_litellm_session_id_wins(self):
        litellm_params = {
            "litellm_session_id": "sess-1",
            "metadata": {"session_id": "sess-2"},
            "litellm_trace_id": "trace-1",
            "litellm_call_id": "call-1",
        }
        assert resolve_opencode_session_id(litellm_params) == "sess-1"

    def test_metadata_session_id_beats_trace_and_call_id(self):
        litellm_params = {
            "metadata": {"session_id": "sess-2"},
            "litellm_trace_id": "trace-1",
            "litellm_call_id": "call-1",
        }
        assert resolve_opencode_session_id(litellm_params) == "sess-2"

    def test_trace_id_beats_call_id(self):
        litellm_params = {"litellm_trace_id": "trace-1", "litellm_call_id": "call-1"}
        assert resolve_opencode_session_id(litellm_params) == "trace-1"

    def test_call_id_is_the_last_fallback(self):
        assert resolve_opencode_session_id({"litellm_call_id": "call-1"}) == "call-1"

    def test_top_level_session_id_marker_is_honored(self):
        litellm_params = {"session_id": "sess-3", "litellm_call_id": "call-1"}
        assert resolve_opencode_session_id(litellm_params) == "sess-3"

    def test_truthy_non_string_markers_are_coerced_by_the_shared_contract(self):
        litellm_params = {"litellm_session_id": 7, "litellm_call_id": "call-1"}
        assert resolve_opencode_session_id(litellm_params) == "7"

    def test_empty_strings_are_skipped(self):
        litellm_params = {"litellm_session_id": "", "litellm_call_id": "call-1"}
        assert resolve_opencode_session_id(litellm_params) == "call-1"

    def test_no_marker_returns_none(self):
        assert resolve_opencode_session_id({}) is None

    def test_none_params_returns_none(self):
        assert resolve_opencode_session_id(None) is None

    def test_non_mapping_metadata_is_ignored(self):
        litellm_params = {"metadata": "not-a-dict", "litellm_call_id": "call-1"}
        assert resolve_opencode_session_id(litellm_params) == "call-1"


class TestApplyOpencodeSessionHeaders:
    def test_opencode_host_without_headers_gets_only_the_session_header(self):
        headers = apply_opencode_session_headers(
            headers=None,
            api_base="https://opencode.ai/zen/go/v1/",
            litellm_params={"metadata": {"session_id": "sess-123"}},
        )
        assert headers == {OPENCODE_SESSION_HEADER: "sess-123"}

    def test_unrelated_headers_are_preserved(self):
        headers = apply_opencode_session_headers(
            headers={"x-foo": "bar", "Authorization": "Bearer opencode-fake-key"},
            api_base="https://opencode.ai/zen/go/v1/",
            litellm_params={"litellm_session_id": "sess-123"},
        )
        assert headers == {
            "x-foo": "bar",
            "Authorization": "Bearer opencode-fake-key",
            OPENCODE_SESSION_HEADER: "sess-123",
        }

    def test_caller_supplied_session_header_wins_case_insensitively(self):
        headers = apply_opencode_session_headers(
            headers={"X-OpenCode-Session": "caller-session"},
            api_base="https://opencode.ai/zen/go/v1/",
            litellm_params={"metadata": {"session_id": "derived-session"}},
        )
        assert headers is None

    def test_empty_caller_session_header_is_replaced(self):
        headers = apply_opencode_session_headers(
            headers={OPENCODE_SESSION_HEADER: ""},
            api_base="https://opencode.ai/zen/go/v1/",
            litellm_params={"metadata": {"session_id": "sess-123"}},
        )
        assert headers == {OPENCODE_SESSION_HEADER: "sess-123"}

    def test_empty_caller_session_header_in_other_casing_is_replaced_without_duplicates(self):
        headers = apply_opencode_session_headers(
            headers={"X-OpenCode-Session": ""},
            api_base="https://opencode.ai/zen/go/v1/",
            litellm_params={"metadata": {"session_id": "sess-123"}},
        )
        assert headers == {OPENCODE_SESSION_HEADER: "sess-123"}

    def test_no_session_marker_leaves_headers_unchanged(self):
        headers = apply_opencode_session_headers(
            headers={"x-foo": "bar"},
            api_base="https://opencode.ai/zen/go/v1/",
            litellm_params=None,
        )
        assert headers is None

    def test_no_headers_and_no_session_marker_returns_none(self):
        headers = apply_opencode_session_headers(
            headers=None,
            api_base="https://opencode.ai/zen/go/v1/",
            litellm_params={},
        )
        assert headers is None

    def test_non_opencode_host_returns_none(self):
        headers = apply_opencode_session_headers(
            headers=None,
            api_base="https://api.openai.com/v1",
            litellm_params={"metadata": {"session_id": "sess-123"}},
        )
        assert headers is None

    def test_non_mapping_header_shapes_fail_into_the_caller_error_handling(self):
        with pytest.raises(ValidationError):
            apply_opencode_session_headers(
                headers="not-a-mapping",
                api_base="https://opencode.ai/zen/go/v1/",
                litellm_params={"metadata": {"session_id": "sess-123"}},
            )
