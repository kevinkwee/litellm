"""Shared helpers for calls routed to OpenCode's managed inference gateways (Zen and Go)."""

from collections.abc import Iterable, Mapping
from itertools import chain
from typing import Final
from urllib.parse import urlparse

from pydantic import TypeAdapter

from litellm.llms.chatgpt.common_utils import get_chatgpt_session_id

OPENCODE_SESSION_HEADER: Final[str] = "x-opencode-session"
_OPENCODE_HOST: Final[str] = "opencode.ai"

_HEADERS_VALIDATOR: Final = TypeAdapter(Mapping[str, str])


def is_opencode_api_base(api_base: str | None) -> bool:
    if not api_base:
        return False
    host: Final[str | None] = urlparse(api_base).hostname
    if host is None:
        return False
    return host == _OPENCODE_HOST or host.endswith(f".{_OPENCODE_HOST}")


def resolve_opencode_session_id(litellm_params: Mapping[str, object] | None) -> str | None:
    """OpenCode tracks prompt-cache affinity per session id, but only callers know where one
    conversation ends and the next begins. Delegating resolution to the shared litellm
    session-marker contract keeps every provider reading the same precedence."""
    return get_chatgpt_session_id(litellm_params)


def apply_opencode_session_headers(
    headers: object,
    api_base: str | None,
    litellm_params: Mapping[str, object] | None,
) -> Mapping[str, str] | None:
    """Return the request headers extended with the OpenCode session header, or None when no
    stamping is needed: a non-OpenCode destination, a non-empty caller-supplied session
    header (any casing), or no usable session marker. Empty caller session headers, in any
    casing, are dropped before the stamp is appended. Non-mapping header shapes fail into
    the caller's error handling, matching how the OpenAI SDK rejects them at the client."""
    if not is_opencode_api_base(api_base):
        return None
    existing: Final[Iterable[tuple[str, str]]] = (
        _HEADERS_VALIDATOR.validate_python(headers).items() if headers is not None else ()
    )
    if any(name.lower() == OPENCODE_SESSION_HEADER and value for name, value in existing):
        return None
    session_id: Final[str | None] = resolve_opencode_session_id(litellm_params)
    if session_id is None:
        return None
    return dict(  # mutable-ok: extras flow into the legacy optional_params and SDK headers as a dict
        chain(
            (item for item in existing if item[0].lower() != OPENCODE_SESSION_HEADER),
            ((OPENCODE_SESSION_HEADER, session_id),),
        ),
    )
