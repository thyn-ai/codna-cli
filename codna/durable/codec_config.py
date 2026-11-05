"""DECISION: whether a Temporal client carries the AES-GCM payload codec, and why.

The codec is a POLICY decision, so it is made in ONE place, reported everywhere, and never inferred
silently. Three modes, selected by `CODNA_TEMPORAL_CODEC`:

* ``auto`` (default) — install the codec when a key is present in the keystore; otherwise run
  PLAINTEXT and say so. "Say so" is load-bearing: `codec_mode` is reported in the worker's startup
  banner and in `codna durable status`, so an operator can see that history is unencrypted. This is
  not a silent fallback — it is a reported mode with a documented way to close it.
* ``required`` — a missing key is a STARTUP FAILURE (`codec_key_missing`). Use this on a hosted
  worker, where nobody is watching a banner.
* ``off`` — the explicit opt-out (`NullPayloadCodec`). Never selected because a key is missing.

Why not "always required": a plain `pip install codna` on a machine with no OS keychain and no
`codna durable codec keygen` yet would be unable to start a dev server at all. The default has to be
usable, so it is usable AND visible.

The key is NEVER read at import time. `resolve_codec()` runs when a client is built, which is why
importing the durable package needs no keychain and the base CLI keeps working without the extra.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from .codec import (
    CodecKeyError,
    NullPayloadCodec,
    payload_codec_from_keystore,
)

__all__ = [
    "CODEC_ENV",
    "CodecResolution",
    "CodecUnavailable",
    "resolve_codec",
]

CODEC_ENV = "CODNA_TEMPORAL_CODEC"
VALID_MODES = ("auto", "required", "off")


class CodecUnavailable(RuntimeError):
    """The codec is REQUIRED and cannot be built. Always carries a stable code."""

    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class CodecResolution:
    """What the client should install, and what to TELL the operator about it."""

    codec: Any
    #: "aes-256-gcm" when encrypting, "plaintext" when not. Reported, never inferred downstream.
    mode: str
    key_id: str | None
    requested: str
    reason: str

    @property
    def encrypting(self) -> bool:
        return self.mode == "aes-256-gcm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "key_id": self.key_id,
            "requested": self.requested,
            "reason": self.reason,
        }


def resolve_codec(
    env: Mapping[str, str] | None = None,
    *,
    lookup: Any | None = None,
    retired_names: Any | None = None,
) -> CodecResolution:
    """Decide the codec for a client about to be built.

    Raises `CodecUnavailable` in ``required`` mode when no key exists — the caller turns that into
    its own startup error so the message names the command that fixes it.
    """
    source = os.environ if env is None else env
    requested = (source.get(CODEC_ENV) or "auto").strip().lower()
    if requested not in VALID_MODES:
        raise CodecUnavailable(
            f"unknown {CODEC_ENV} value {requested!r}; expected one of {', '.join(VALID_MODES)}",
            code="codec_mode_invalid",
            details={"received": requested, "valid": list(VALID_MODES)},
        )
    if requested == "off":
        return CodecResolution(
            codec=NullPayloadCodec(),
            mode="plaintext",
            key_id=None,
            requested=requested,
            reason=f"{CODEC_ENV}=off (explicit opt-out)",
        )
    try:
        codec = payload_codec_from_keystore(lookup=lookup, retired_names=retired_names)
    except CodecKeyError as exc:
        if requested == "required":
            raise CodecUnavailable(
                str(exc),
                code=exc.code,
                details={
                    **exc.details,
                    "hint": (
                        "run `codna durable codec keygen` to create a key on this device, or set "
                        f"{CODEC_ENV}=auto to run plaintext with that fact reported"
                    ),
                },
            ) from exc
        # `auto`: no key installed. Report PLAINTEXT — the caller must surface this, which is what
        # makes it a reported mode rather than a silent fallback.
        return CodecResolution(
            codec=NullPayloadCodec(),
            mode="plaintext",
            key_id=None,
            requested=requested,
            reason=f"no codec key in the keystore ({exc.code}); set {CODEC_ENV}=required to refuse",
        )
    material_id = getattr(codec, "current_key_id", None)
    return CodecResolution(
        codec=codec,
        mode="aes-256-gcm",
        key_id=material_id,
        requested=requested,
        reason="codec key present in the keystore",
    )


def data_converter_kwargs(resolution: CodecResolution) -> dict[str, Any]:
    """`Client.connect`/`Worker` kwargs carrying the codec.

    Imported lazily so the base CLI never needs `temporalio`. An explicit opt-out still installs the
    `NullPayloadCodec` rather than omitting the converter: that keeps the shape of every call site
    identical whether the codec is on or off, so enabling it later cannot depend on a code path that
    was never exercised.
    """
    from temporalio.converter import DataConverter

    return {"data_converter": DataConverter(payload_codec=resolution.codec)}
