"""`codna.mcp_bridge` — the IDE chat path's MCP config, judged by the ONE durable validator.

The bridge exists so the IDE does not grow a second MCP parser (see the module docstring). These tests
therefore do two jobs:

* prove the FORMAT ADAPTERS (Cursor/Theia shapes → the canonical shape) are right for the files real
  clients write, including the migration case that must need no edits; and
* prove that the VALIDATION is still `durable.mcp`'s — every refusal asserted here is the durable
  module's own code, reached through `normalize_mcp_servers` rather than restated.

The secret rule is tested in both directions: a Codna literal is refused (the durable contract), and a
Cursor literal is converted to a same-name environment reference and never returned.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codna import mcp_bridge
from codna.durable.mcp import MAX_MCP_SERVERS


def write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def codna_source(path: Path) -> dict:
    return {"path": str(path), "kind": "codna"}


def cursor_source(path: Path) -> dict:
    return {"path": str(path), "kind": "cursor"}


def theia_source(path: Path) -> dict:
    return {"path": str(path), "kind": "theia"}


# ── the adapter: Cursor's own shape, unmodified ────────────────────────────────────────────────


def test_cursor_stdio_maps_without_any_discriminant(tmp_path: Path) -> None:
    """Cursor writes `command`/`args` with no `type`. The file must work exactly as written."""
    path = write(
        tmp_path / ".cursor" / "mcp.json",
        {"mcpServers": {"filesystem": {"command": "npx", "args": ["-y", "server-filesystem", "/tmp"]}}},
    )
    response = mcp_bridge.handle({"sources": [cursor_source(path)]})
    assert response["ok"] is True
    assert response["servers"] == [
        {
            "name": "filesystem",
            "transport": {"type": "stdio", "command": "npx", "args": ["-y", "server-filesystem", "/tmp"]},
        }
    ]
    # No warning: there was no credential to convert.
    assert response["warnings"] == []


def test_cursor_remote_url_becomes_a_transport_that_needs_no_auth(tmp_path: Path) -> None:
    path = write(
        tmp_path / ".cursor" / "mcp.json",
        {"mcpServers": {"docs": {"url": "https://mcp.example.com/mcp"}}},
    )
    response = mcp_bridge.handle({"sources": [cursor_source(path)]})
    assert response["servers"][0]["transport"] == {
        "type": "streamableHttp",
        "url": "https://mcp.example.com/mcp",
    }


def test_a_cursor_LITERAL_becomes_a_same_name_reference_and_the_value_never_returns(tmp_path: Path) -> None:
    """The migration case. A user's file is not edited; the credential simply stops travelling.

    The value is deliberately NOT credential-shaped: what this test asserts is that the adapter is
    value-agnostic (any literal is replaced by a reference and never returned), and writing a
    secret-shaped string into a fixture would make the suite itself look like clear-text credential
    storage to a scanner — a finding this file does not need in order to prove its point.
    """
    literal = "migrated-literal-value"
    path = write(
        tmp_path / ".cursor" / "mcp.json",
        {"mcpServers": {"linear": {"command": "npx", "env": {"LINEAR_API_KEY": literal}}}},
    )
    response = mcp_bridge.handle({"sources": [cursor_source(path)]})
    assert response["ok"] is True
    assert response["servers"][0]["transport"]["env"] == {"LINEAR_API_KEY": {"from_env": "LINEAR_API_KEY"}}
    # The value appears NOWHERE in the answer — not in a warning, not in a report.
    assert literal not in json.dumps(response)
    # …and the substitution is reported, so a user knows why the value is no longer read from the file.
    assert any("LINEAR_API_KEY" in warning for warning in response["warnings"])


def test_a_cursor_dollar_brace_reference_is_normalized_by_the_validator(tmp_path: Path) -> None:
    path = write(
        tmp_path / ".cursor" / "mcp.json",
        {"mcpServers": {"docs": {"command": "npx", "env": {"DOCS_TOKEN": "${DOCS_SOURCE}"}}}},
    )
    response = mcp_bridge.handle({"sources": [cursor_source(path)]})
    assert response["servers"][0]["transport"]["env"] == {"DOCS_TOKEN": {"from_env": "DOCS_SOURCE"}}


def test_theia_registry_is_read_from_its_own_preference_key(tmp_path: Path) -> None:
    path = write(
        tmp_path / ".theia" / "settings.json",
        {
            "ai-features.mcp.mcpServers": {"fs": {"command": "npx", "args": ["-y", "server-filesystem"]}},
            "unrelated.preference": True,
        },
    )
    response = mcp_bridge.handle({"sources": [theia_source(path)]})
    assert response["ok"] is True
    assert [server["name"] for server in response["servers"]] == ["fs"]


# ── precedence ──────────────────────────────────────────────────────────────────────────────────


def test_the_first_source_wins_for_a_shared_name(tmp_path: Path) -> None:
    codna = write(
        tmp_path / ".codna" / "mcp.json",
        {"mcpServers": {"shared": {"transport": {"type": "stdio", "command": "codna-owned"}}}},
    )
    cursor = write(tmp_path / ".cursor" / "mcp.json", {"mcpServers": {"shared": {"command": "cursor-owned"}}})
    response = mcp_bridge.handle({"sources": [codna_source(codna), cursor_source(cursor)]})
    assert len(response["servers"]) == 1
    assert response["servers"][0]["transport"]["command"] == "codna-owned"
    # The losing file is named, so nobody has to guess which config took effect.
    assert any("higher-precedence" in warning for warning in response["warnings"])


def test_source_order_is_the_precedence_rule_not_a_condition() -> None:
    """Two files with the SAME name: swapping the order swaps the winner, nothing else changes."""
    # Exercised through `handle` rather than a helper, because the rule is "first in the list".
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        first = write(root / "a.json", {"mcpServers": {"n": {"command": "from-a"}}})
        second = write(root / "b.json", {"mcpServers": {"n": {"command": "from-b"}}})
        one = mcp_bridge.handle({"sources": [cursor_source(first), cursor_source(second)]})
        two = mcp_bridge.handle({"sources": [cursor_source(second), cursor_source(first)]})
        assert one["servers"][0]["transport"]["command"] == "from-a"
        assert two["servers"][0]["transport"]["command"] == "from-b"


# ── the ONE validator still decides ────────────────────────────────────────────────────────────


def test_a_codna_LITERAL_env_value_is_REFUSED_because_it_would_enter_history(tmp_path: Path) -> None:
    """Codna's own format is the durable format: the durable rule applies unchanged."""
    path = write(
        tmp_path / ".codna" / "mcp.json",
        {"mcpServers": {"linear": {"transport": {"type": "stdio", "command": "npx", "env": {"K": "a-value"}}}}},
    )
    response = mcp_bridge.handle({"sources": [codna_source(path)]})
    assert response["ok"] is False
    assert response["error"]["code"] == "mcp_config_invalid"
    assert "from_env" in response["error"]["message"]


def test_a_bad_transport_is_refused_with_the_durable_modules_own_message(tmp_path: Path) -> None:
    path = write(
        tmp_path / ".codna" / "mcp.json",
        {"mcpServers": {"bad": {"transport": {"type": "carrier-pigeon", "url": "https://x"}}}},
    )
    response = mcp_bridge.handle({"sources": [codna_source(path)]})
    assert response["ok"] is False
    assert "stdio, sse, streamableHttp" in response["error"]["message"]


def test_more_than_the_cap_is_REFUSED_not_truncated(tmp_path: Path) -> None:
    servers = {f"s{i}": {"transport": {"type": "stdio", "command": "true"}} for i in range(MAX_MCP_SERVERS + 1)}
    path = write(tmp_path / ".codna" / "mcp.json", {"mcpServers": servers})
    response = mcp_bridge.handle({"sources": [codna_source(path)]})
    assert response["ok"] is False
    assert response["error"]["code"] == "mcp_config_invalid"
    assert str(MAX_MCP_SERVERS) in response["error"]["message"]


def test_exactly_the_cap_is_accepted(tmp_path: Path) -> None:
    servers = {f"s{i}": {"transport": {"type": "stdio", "command": "true"}} for i in range(MAX_MCP_SERVERS)}
    path = write(tmp_path / ".codna" / "mcp.json", {"mcpServers": servers})
    response = mcp_bridge.handle({"sources": [codna_source(path)]})
    assert response["ok"] is True
    assert len(response["servers"]) == MAX_MCP_SERVERS


def test_a_disabled_server_is_kept_by_the_validator_and_carried_as_disabled(tmp_path: Path) -> None:
    path = write(
        tmp_path / ".codna" / "mcp.json",
        {"mcpServers": {"off": {"transport": {"type": "stdio", "command": "true"}, "disabled": True}}},
    )
    response = mcp_bridge.handle({"sources": [codna_source(path)]})
    assert response["ok"] is True
    assert response["servers"][0]["disabled"] is True


def test_a_duplicate_name_across_files_is_merged_not_refused(tmp_path: Path) -> None:
    """Precedence is a merge, so a name in two files is ONE server, not a duplicate error."""
    a = write(tmp_path / "a.json", {"mcpServers": {"n": {"command": "one"}}})
    b = write(tmp_path / "b.json", {"mcpServers": {"n": {"command": "two"}}})
    response = mcp_bridge.handle({"sources": [cursor_source(a), cursor_source(b)]})
    assert response["ok"] is True
    assert len(response["servers"]) == 1


# ── what cannot work, stated instead of silently dropped ──────────────────────────────────────


def test_a_remote_entry_with_no_url_is_refused_with_the_field_named(tmp_path: Path) -> None:
    """Neither `url` nor `serverUrl` present: the validator names the field, not an internal error.

    This is a regression pin. The adapter derives a remote transport whenever `command` is absent, so a
    bare `{"args": [...]}` entry reaches `normalize_mcp_servers` with `url: None` — and the answer must be
    the validator's own "url must be a non-empty string", not a traceback or a misleading message.
    """
    path = write(tmp_path / ".cursor" / "mcp.json", {"mcpServers": {"bad": {"args": ["x"]}}})
    response = mcp_bridge.handle({"sources": [cursor_source(path)]})
    assert response["ok"] is False
    assert response["error"]["code"] == "mcp_config_invalid"
    assert "url" in response["error"]["message"]
    assert response["error"]["details"]["field"] == "mcp_servers.bad.transport.url"


def test_a_remote_server_with_a_bearer_token_is_REFUSED_with_a_named_code(tmp_path: Path) -> None:
    """A remote credential has no reference form, so it cannot travel; the refusal says so."""
    path = write(
        tmp_path / ".theia" / "settings.json",
        {
            "ai-features.mcp.mcpServers": {
                "jira": {"serverUrl": "https://jira.example.com/mcp", "serverAuthToken": "a-remote-header-value"},
            }
        },
    )
    response = mcp_bridge.handle({"sources": [theia_source(path)]})
    assert response["ok"] is False
    assert response["error"]["code"] == "mcp_remote_auth_unsupported"
    assert "jira" in response["error"]["message"]
    # The token is not echoed anywhere in the refusal.
    assert "a-remote-header-value" not in json.dumps(response)


def test_a_remote_server_with_headers_is_REFUSED_by_the_same_rule(tmp_path: Path) -> None:
    path = write(
        tmp_path / ".codna" / "mcp.json",
        {
            "mcpServers": {
                "remote": {
                    "transport": {
                        "type": "streamableHttp",
                        "url": "https://x.example.com/mcp",
                        "headers": {"Authorization": "Bearer abc"},
                    }
                }
            }
        },
    )
    response = mcp_bridge.handle({"sources": [codna_source(path)]})
    assert response["ok"] is False
    assert response["error"]["code"] == "mcp_remote_auth_unsupported"


# ── absent / broken files ──────────────────────────────────────────────────────────────────────


def test_a_missing_file_is_skipped_and_reported_not_an_error(tmp_path: Path) -> None:
    response = mcp_bridge.handle({"sources": [codna_source(tmp_path / "nope.json")]})
    assert response["ok"] is True
    assert response["servers"] == []
    assert response["sources"][0]["found"] is False
    assert "skipped" in response["sources"][0]


def test_a_file_without_the_key_is_treated_as_no_servers(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    response = mcp_bridge.handle({"sources": [codna_source(path)]})
    assert response["ok"] is True
    assert response["servers"] == []


def test_malformed_json_is_REFUSED_and_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    response = mcp_bridge.handle({"sources": [codna_source(path)]})
    assert response["ok"] is False
    assert response["error"]["code"] == "mcp_config_unreadable"
    assert str(path) in response["error"]["message"]


@pytest.mark.parametrize(
    "request_body",
    [
        [],
        {"sources": "not-a-list"},
        {"sources": [{"path": 3, "kind": "codna"}]},
        {"sources": [{"path": "/x", "kind": "vim"}]},
    ],
)
def test_a_malformed_request_is_refused_rather_than_guessed(request_body: object) -> None:
    response = mcp_bridge.handle(request_body)
    assert response["ok"] is False
    assert response["error"]["code"] == "mcp_bridge_bad_request"


# ── the subprocess protocol ────────────────────────────────────────────────────────────────────


def test_main_reads_one_request_and_writes_one_answer(tmp_path: Path, monkeypatch, capsys) -> None:
    path = write(tmp_path / ".codna" / "mcp.json", {"mcpServers": {"x": {"transport": {"type": "stdio", "command": "true"}}}})
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps({"sources": [codna_source(path)]})))
    assert mcp_bridge.main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert [server["name"] for server in out["servers"]] == ["x"]


def test_main_exits_non_zero_on_a_refusal(tmp_path: Path, monkeypatch, capsys) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps({"sources": [codna_source(path)]})))
    assert mcp_bridge.main([]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False


def test_main_answers_bad_json_instead_of_tracebacking(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{not json"))
    assert mcp_bridge.main([]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"]["code"] == "mcp_bridge_bad_json"
