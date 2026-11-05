"""D6: the E3 codec through the WORKER's payload path, and the Cloud credential shapes.

Two separate claims, each tested where it can fail:

**A. The codec round-trips through the object the worker actually installs.** E3's codec is already
tested as a codec (`test_codec.py`). What D6 adds is that the duration the WORKER builds — the
`DataConverter` handed to `Client.connect` by `worker.run_worker` — encrypts a real workflow payload
and decrypts it back, so "the worker encrypts history" is a property of the worker's construction
rather than of a class it happens to import. The payloads are REAL `temporalio.api.common.v1.Payload`
messages, so a passing test also proves the duck-typed codec is structurally compatible with the SDK
(E3 could only prove that against a stand-in).

**B. Cloud accepts TWO credential shapes, and the worker picks the configured one.** An API key and
an mTLS client certificate are both credentials; a config that only understood the key would refuse a
correctly configured certificate fleet. The material is resolved from the environment or the keystore,
never from a field, and neither shape can be formatted into a log line.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from codna.durable.codec import generate_key_material, encode_key_for_keystore
from codna.durable.codec_config import CODEC_ENV, resolve_codec

HAS_EXTRA = importlib.util.find_spec("temporalio") is not None

CURRENT_KEY_ENTRY = "CODNA_TEMPORAL_CODEC_KEY"


# -- A. the codec the worker installs ------------------------------------------------------


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra (temporalio) is not installed")
def test_the_worker_data_converter_round_trips_a_real_payload():
    """`data_converter_kwargs` -> a real `Payload`, encrypted and decrypted, with the SDK's types."""
    import asyncio

    from temporalio.api.common.v1 import Payload

    from codna.durable.codec_config import data_converter_kwargs

    material = generate_key_material()
    entries = {CURRENT_KEY_ENTRY: encode_key_for_keystore(material)}
    resolution = resolve_codec({CODEC_ENV: "required"}, lookup=entries.get, retired_names=[])
    assert resolution.mode == "aes-256-gcm"

    # The EXACT object `run_worker` hands to `Client.connect`.
    converter = data_converter_kwargs(resolution)["data_converter"]

    payload = Payload(metadata={"encoding": b"json/plain"}, data=b'{"prompt":"fix the retry"}')
    encoded = asyncio.run(converter.payload_codec.encode([payload]))
    plaintext = b'{"prompt":"fix the retry"}'
    assert encoded[0].data != plaintext, "the worker's converter did not encrypt the payload"
    assert encoded[0].metadata["encoding"] == b"codna:enc:v1"
    # The key id travels in metadata; the KEY does not, anywhere.
    assert material.key not in encoded[0].data
    assert material.key not in b"".join(encoded[0].metadata.values())

    decoded = asyncio.run(converter.payload_codec.decode(encoded))
    assert decoded[0].data == plaintext
    # The payload's OWN encoding is restored, which is what makes the SDK's converter able to read it
    # back. Before this was preserved, a payload travelling through a workflow raised
    # `KeyError: Unknown payload encoding <unknown>` on decode — see
    # `test_an_sdk_encoded_payload_survives_a_codec_round_trip`.
    assert decoded[0].metadata["encoding"] == b"json/plain"


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra (temporalio) is not installed")
def test_a_required_codec_with_no_key_REFUSES_before_a_worker_can_connect():
    """The hosted-worker setting: a missing codec key is a startup failure, not a plaintext banner.

    This is the mode `Dockerfile.worker` sets (`CODNA_TEMPORAL_CODEC=required`), and the refusal is
    what `run_worker` turns into a `WorkerError` before `Client.connect`.
    """
    from codna.durable.codec_config import CodecUnavailable

    with pytest.raises(CodecUnavailable) as caught:
        resolve_codec({CODEC_ENV: "required"}, lookup=lambda _name: None, retired_names=[])
    assert caught.value.code == "codec_key_missing"


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra (temporalio) is not installed")
def test_the_codec_survives_a_real_history_payload_from_the_sdk_converter():
    """A SECOND round trip through the SDK's own `DataConverter`, not the codec directly.

    `DataConverter` encrypts on the way out and decrypts on the way in, and the SDK calls it with
    payloads the codec did not create (a `json/plain` envelope with `temporal.io/encoding` metadata).
    A codec that only tolerated its own output would pass the test above and fail here.
    """
    import asyncio

    from codna.durable.codec_config import data_converter_kwargs

    material = generate_key_material()
    entries = {CURRENT_KEY_ENTRY: encode_key_for_keystore(material)}
    resolution = resolve_codec({CODEC_ENV: "required"}, lookup=entries.get, retired_names=[])
    converter = data_converter_kwargs(resolution)["data_converter"]

    value = {"run_id": "run-1", "task_id": "task-1", "brief": "fix the flaky retry"}
    payloads = asyncio.run(converter.encode([value]))
    assert len(payloads) == 1
    assert material.key not in payloads[0].data
    restored = asyncio.run(converter.decode(payloads))
    assert restored == [value]


# -- B. Cloud credential shapes ------------------------------------------------------------

_PEM_CERT = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBkTCB+wIJAOexampleexampleexampleexampleexampleexample\n"
    "-----END CERTIFICATE-----\n"
)
_PEM_KEY = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIBexampleexampleexampleexampleexampleexampleexample\n"
    "-----END PRIVATE KEY-----\n"
)


def test_an_api_key_cloud_config_still_connects_with_tls_and_the_key():
    from codna.durable.config import load_durable_config

    env = {
        "TEMPORAL_ENDPOINT": "codna.cihxb.tmprl.cloud:7233",
        "TEMPORAL_NAMESPACE": "codna.cihxb",
        "TEMPORAL_API_KEY": "a-fake-key",
    }
    config = load_durable_config(env=env)
    options = config.connect_options(env=env)
    assert options["tls"] is True
    assert options["api_key"] == "a-fake-key"
    assert "a-fake-key" not in repr(options)


@pytest.mark.skipif(importlib.util.find_spec("temporalio") is None, reason="the `durable` extra is not installed")
def test_an_mtls_cloud_config_is_a_credential_with_no_api_key():
    """The other shape. Requiring the KEY would refuse this fleet outright."""
    from codna.durable.config import load_durable_config
    from codna.durable.credentials import resolve_temporal_credentials

    env = {
        "TEMPORAL_ENDPOINT": "codna.cihxb.tmprl.cloud:7233",
        "TEMPORAL_NAMESPACE": "codna.cihxb",
        "CODNA_TEMPORAL_CLIENT_CERT_PEM": _PEM_CERT,
        "CODNA_TEMPORAL_CLIENT_KEY_PEM": _PEM_KEY,
    }
    config = load_durable_config(env=env)
    assert config.is_cloud is True
    credentials = resolve_temporal_credentials(env)
    assert credentials.mode == "mtls"
    # No key means `temporal_api_key` reports None rather than a credential it does not have.
    assert config.temporal_api_key(env=env) is None
    options = config.connect_options(env=env)
    assert "api_key" not in options
    # A TLSConfig, not a bool: the client certificate is how the tenant knows who is calling.
    assert options["tls"] is not False


def test_half_an_mtls_pair_is_refused_and_names_the_missing_variable():
    from codna.durable.config import DurableConfigError, load_durable_config

    env = {
        "TEMPORAL_ENDPOINT": "codna.cihxb.tmprl.cloud:7233",
        "TEMPORAL_NAMESPACE": "codna.cihxb",
        "CODNA_TEMPORAL_CLIENT_CERT_PEM": _PEM_CERT,
    }
    with pytest.raises(DurableConfigError) as caught:
        load_durable_config(env=env)
    assert caught.value.code == "temporal_cloud_config_incomplete"
    assert "CODNA_TEMPORAL_CLIENT_KEY_PEM" in caught.value.details["missing"]
    # The detail map is printed; it must name variables, never values.
    assert "BEGIN CERTIFICATE" not in str(caught.value.details)


def test_the_mtls_material_is_never_formatted_and_the_source_is_reported():
    from codna.durable.credentials import resolve_temporal_credentials

    env = {
        "CODNA_TEMPORAL_CLIENT_CERT_PEM": _PEM_CERT,
        "CODNA_TEMPORAL_CLIENT_KEY_PEM": _PEM_KEY,
    }
    credentials = resolve_temporal_credentials(env)
    for formatted in (repr(credentials), str(credentials), f"{credentials}"):
        assert "BEGIN" not in formatted, "credential material must never be formatted"
        assert _PEM_KEY.splitlines()[1] not in formatted
    # Provenance is reported, because "which cert/key is loaded" is what an operator asks.
    assert credentials.sources == {
        "client_cert": "environment",
        "client_key": "environment",
        "server_ca": "missing",
    }
    assert credentials.to_payload()["mtls"] is True


def test_a_credential_can_come_from_the_keystore_when_the_environment_has_none():
    """The desktop/developer path: the keychain, read at connect time, never a config field."""
    from codna.durable.credentials import resolve_temporal_credentials

    stored = {"TEMPORAL_API_KEY": "  from-the-keychain  "}
    credentials = resolve_temporal_credentials({}, lookup=stored.get)
    assert credentials.mode == "api_key"
    assert credentials.api_key == "from-the-keychain"  # trimmed
    assert credentials.sources == {"api_key": "keystore"}


def test_the_environment_wins_over_the_keystore():
    """A platform secret is the operator's statement about THIS container; the keychain is not."""
    from codna.durable.credentials import resolve_temporal_credentials

    credentials = resolve_temporal_credentials(
        {"TEMPORAL_API_KEY": "from-env"},
        lookup=lambda name: "from-keychain" if name == "TEMPORAL_API_KEY" else None,
    )
    assert credentials.api_key == "from-env"
    assert credentials.sources == {"api_key": "environment"}


def test_a_blank_env_value_falls_through_to_the_keystore():
    """A platform that rendered an empty secret must not shadow a usable one."""
    from codna.durable.credentials import resolve_temporal_credentials

    credentials = resolve_temporal_credentials(
        {"TEMPORAL_API_KEY": "   "},
        lookup=lambda name: "from-keychain" if name == "TEMPORAL_API_KEY" else None,
    )
    assert credentials.api_key == "from-keychain"


def test_no_credential_at_all_is_refused_naming_the_variables():
    from codna.durable.credentials import TemporalAuthError, resolve_temporal_credentials

    with pytest.raises(TemporalAuthError) as caught:
        resolve_temporal_credentials({}, lookup=lambda _n: None)
    assert caught.value.code == "temporal_cloud_config_incomplete"
    assert "TEMPORAL_API_KEY" in caught.value.details["missing"]
    # The alternative shape is offered, so an operator knows mTLS is an option.
    assert "CODNA_TEMPORAL_CLIENT_CERT_PEM" in caught.value.details["alternatives"]


@pytest.mark.parametrize("mode", ["mtls", "api_key"])
def test_an_explicit_auth_mode_refuses_a_missing_shape(mode: str):
    """An operator who pinned a shape gets a refusal, not a silent switch to the other one."""
    from codna.durable.credentials import TemporalAuthError, resolve_temporal_credentials

    with pytest.raises(TemporalAuthError):
        resolve_temporal_credentials({"CODNA_TEMPORAL_AUTH": mode}, lookup=lambda _n: None)


def test_a_non_pem_certificate_is_refused_by_name():
    """A pasted path or a bare base64 body otherwise fails deep inside gRPC, naming nothing."""
    from codna.durable.credentials import TemporalAuthError, resolve_temporal_credentials

    with pytest.raises(TemporalAuthError) as caught:
        resolve_temporal_credentials(
            {
                "CODNA_TEMPORAL_CLIENT_CERT_PEM": "/etc/certs/client.pem",
                "CODNA_TEMPORAL_CLIENT_KEY_PEM": _PEM_KEY,
            }
        )
    assert caught.value.code == "temporal_mtls_material_invalid"
    assert caught.value.details["variable"] == "CODNA_TEMPORAL_CLIENT_CERT_PEM"
    assert "/etc/certs" not in str(caught.value.details)


def test_a_certificate_only_setup_is_inferred_as_cloud():
    """Without this, a certificate-only fleet would resolve `local` and poll a server nobody started."""
    from codna.durable.config import load_durable_config

    with pytest.raises(Exception) as caught:
        load_durable_config(
            env={
                "CODNA_TEMPORAL_CLIENT_CERT_PEM": _PEM_CERT,
                "CODNA_TEMPORAL_CLIENT_KEY_PEM": _PEM_KEY,
            }
        )
    # It is a CLOUD attempt, so the refusal is about the missing endpoint, not a silent local fallback.
    assert "TEMPORAL_ENDPOINT" in str(caught.value)


def test_connect_options_is_one_class_shared_with_the_config_alias():
    """Two names for one type, asserted rather than assumed (mTLS build options go through it)."""
    from codna.durable import config as config_module
    from codna.durable.credentials import ConnectOptions

    assert config_module.ConnectOptions is ConnectOptions


def _unused(*_args: Any) -> None:  # pragma: no cover - keeps the import surface honest
    return None
