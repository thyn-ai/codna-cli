"""_structured_error must recognize decision_engine SDK exceptions (`.error_code`), not just
codna's own internal error types (`.code`) — see cli.py's _structured_error for why: a missing
fallback here means main()'s outer except re-raises the exception bare, leaking a raw Python
traceback to stderr instead of codna's documented `{"error": {"code", "message", ...}}` JSON
contract (docs/troubleshooting.md). Caught by a 100-repo soak test: every `codna_error` case
was exactly this — an unhandled decision_engine.exceptions.ServerError reaching main() unwrapped.
"""
from __future__ import annotations

from codna.cli import _structured_error
from codna.local_client import _error_code


class _FakeSdkError(Exception):
    """Stands in for decision_engine.exceptions.ServerError without needing the engine import."""

    def __init__(self, message: str, *, error_code: str = "unknown_error", request_id=None, details=None):
        super().__init__(message)
        self.error_code = error_code
        self.request_id = request_id
        self.details = details


class _FakeCodnaError(Exception):
    """Stands in for codna's own internal error types (LocalRuntimeError, etc.)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def test_structured_error_recognizes_decision_engine_error_code():
    exc = _FakeSdkError("An unexpected error occurred.")
    payload = _structured_error(exc)
    assert payload is not None, "an SDK ServerError must not fall through to a bare re-raise"
    assert payload["error"]["code"] == "unknown_error"
    assert payload["error"]["message"] == "An unexpected error occurred."


def test_structured_error_still_recognizes_codnas_own_code_attribute():
    exc = _FakeCodnaError("local_mojo_pool_unavailable", "pool didn't start")
    payload = _structured_error(exc)
    assert payload is not None
    assert payload["error"]["code"] == "local_mojo_pool_unavailable"


def test_structured_error_includes_request_id_when_present():
    exc = _FakeSdkError("boom", error_code="server_error", request_id="req_abc123")
    payload = _structured_error(exc)
    assert payload["error"]["request_id"] == "req_abc123"


def test_structured_error_returns_none_for_truly_unstructured_exceptions():
    # Preserves the existing behavior: an exception with neither .code nor .error_code
    # still can't be rendered structurally, so main() correctly lets it propagate.
    assert _structured_error(ValueError("plain old bug")) is None


def test_local_client_error_code_prefers_decision_engine_error_code():
    exc = _FakeSdkError("boom", error_code="server_error")
    assert _error_code(exc) == "server_error"


def test_local_client_error_code_falls_back_to_class_name_when_unstructured():
    assert _error_code(ValueError("plain old bug")) == "ValueError"
