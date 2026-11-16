"""The two invariants a payload entering workflow history must satisfy — size, and no secrets.

Split out of `state.py` at the repository's 1000-line ceiling (`test_modularity.py`), on the same
boundary `state_coercion.py` and `publish.py` already sit on: these are pure functions over an
already-decoded payload that need exactly one thing from their caller, the state module's error type.
So the error type is INJECTED (`fail=...`) rather than imported, and this module never imports
`state.py` back. That is what keeps it a LEAF — stdlib plus `..sandbox`'s token patterns — and
`state.py` is imported by the workflow sandbox, which refuses anything reaching `urllib.request`.

WHY THEY STILL BELONG TOGETHER. A size check that can be run without the secret check is a size check
that will be run without it, so `assert_state_encodable` performs both and is the only entry point.
The two rules are also the same rule from a defender's point of view: both answer "may this payload
enter history at all", and the answer must be decided at ONE chokepoint rather than at each field.

The checks run on the way in (`TaskState.from_payload`, used when decoding a workflow input) and on
the way out (`TaskState.to_payload`, used before any activity call), so a field added later cannot
bypass them by following a different path — the guard counts the whole encoding, not a named set of
fields.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Final, Mapping

from ..sandbox import WRITE_TOKEN_NAMES, WRITE_TOKEN_VALUE

__all__ = [
    "MAX_STATE_BYTES",
    "assert_state_encodable",
    "find_secret_like_strings",
    "redact_secret_like",
]

MAX_STATE_BYTES: Final = 2 * 1024 * 1024

# Beyond the GitHub token patterns codna.sandbox already owns, a state can also leak a provider
# key. These are the shapes that actually get pasted into prompts and env files:
#   sk-...            OpenAI/Anthropic-style (Anthropic uses sk-ant-..., OpenAI sk-proj-...)
#   sk-ant-api03-...  Anthropic
#   AIza...           Google
#   xai-...           xAI
#   hf_...            Hugging Face
#   glpat-...         GitLab
#   AKIA...           AWS access key id
_SECRET_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    WRITE_TOKEN_VALUE,
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\bxai-[A-Za-z0-9]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*", re.IGNORECASE),
)

# Field NAMES that must never appear in a state payload at all — a caller passing one is not
# smuggling a value, it is telling us the design was violated. Kept separate from the value
# patterns because the name alone is the error, whatever the value looks like.
_FORBIDDEN_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    name.lower()
    for name in (
        *WRITE_TOKEN_NAMES,
        "api_key",
        "apikey",
        "provider_api_key",
        "authorization",
        "auth_token",
        "bearer",
        "password",
        "secret",
        "private_key",
        "client_secret",
    )
)


def find_secret_like_strings(payload: Any, *, path: str = "") -> list[dict[str, str]]:
    """Return every secret-looking finding in an already-decoded payload, with its JSON path.

    Walks dicts, lists and strings. Returning the path (not the value) keeps the report itself
    safe to log — the whole point is that this text is already too sensitive to repeat.
    """
    findings: list[dict[str, str]] = []

    def walk(node: Any, where: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_text = str(key)
                child = f"{where}.{key_text}" if where else key_text
                if key_text.lower() in _FORBIDDEN_FIELD_NAMES:
                    findings.append({"path": child, "reason": "forbidden_field_name"})
                walk(value, child)
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")
            return
        if isinstance(node, str):
            for pattern in _SECRET_VALUE_PATTERNS:
                if pattern.search(node):
                    findings.append({"path": where or "$", "reason": "secret_value_pattern"})
                    return

    walk(payload, path)
    return findings


#: What a redacted span becomes. Deliberately a phrase rather than asterisks, so a reader of the
#: prompt understands that something was removed instead of reading it as a strange literal.
_REDACTION_MARKER: Final = "[redacted: looked like a credential]"


def redact_secret_like(text: str) -> str:
    """Replace every secret-shaped substring with a fixed marker. Total: returns a string, always.

    WHY THIS EXISTS. `assert_state_encodable` REFUSES a payload carrying a token, which is right for
    a field the product fills in (a key must never be written, not even scrubbed). But a REVIEWER'S
    FINDING is text about a diff, and a diff can legitimately CONTAIN a live credential — that is
    frequently the finding ("a token is hardcoded here"). Refusing to encode the state would turn
    "the reviewer found a leaked secret" into a workflow failure at the exact moment the user most
    needs to be told; storing it would publish the secret into workflow history, which is the thing
    the guard exists to prevent. So the finding's text is REDACTED at record time, and this function
    is the one place that happens.

    Fixed marker, not a partial mask: the length of a token is a hint about which provider issued it,
    and keeping a prefix is how a "redacted" secret ends up identifying itself.
    """
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub(_REDACTION_MARKER, text)
    return text


def assert_state_encodable(
    payload: Mapping[str, Any], *, limit: int, fail: Callable[..., Exception]
) -> int:
    """Enforce the size + secret invariants on a state payload. Returns the serialized size.

    ``fail`` builds the state module's own error type; it is injected for the reason this module's
    docstring gives (the error class stays defined in exactly one place, and this stays a leaf).

    ``limit`` is a PARAMETER rather than this module's own constant read at call time so that
    `state.py`'s thin wrapper can pass ITS module attribute. That keeps the documented test seam —
    ``monkeypatch.setattr(state_module, "MAX_STATE_BYTES", …)`` — working, which is what proves the
    bound is enforced on the whole encoding rather than on a chosen field.
    """
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_reject_non_json)
    except TypeError as exc:
        raise fail(
            f"task state is not JSON-serializable: {exc}",
            code="state_not_serializable",
            details={"error": str(exc)},
        ) from exc
    size = len(encoded.encode("utf-8"))
    if size > limit:
        raise fail(
            f"task state is {size} bytes, over the {limit}-byte limit: store a reference, "
            "not the payload.",
            code="state_too_large",
            details={"bytes": size, "limit": limit},
        )
    findings = find_secret_like_strings(payload)
    if findings:
        raise fail(
            "task state contains secret-like data and may not enter workflow history: "
            + ", ".join(sorted({finding["path"] for finding in findings}))
            + ".",
            code="state_contains_secret",
            details={"findings": findings},
        )
    return size


def _reject_non_json(value: Any) -> Any:
    """`json.dumps` default hook: refuse rather than stringify an unexpected object.

    Stringifying would let a `Path`, a model object or a bytes blob slip into history in a form
    nobody can decode back, which is worse than a loud failure here.
    """
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")
