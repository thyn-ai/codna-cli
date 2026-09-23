from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from codna.runtime.config import ConfigValue, RuntimeConfigError, resolve_runtime_config
import codna.runtime.config as config_module
from codna.runtime.health import ProbeResult
import codna.runtime.health as health_module
from codna.runtime.local_stack import LocalRuntimeError, ensure_running, inspect_runtime
from codna.runtime.ports import ListenerInfo
import codna.runtime.local_stack as local_stack_module


@pytest.fixture(autouse=True)
def _clear_runtime_override_env(monkeypatch) -> None:
    for key in (
        "CODNA_ENGINE_URL",
        "ALGENTA_ENGINE_URL",
        "ALGENTA_BASE_URL",
        "CODNA_ALLOW_LOOPBACK_ENGINE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_fake_engine_checkout(tmp_path: Path) -> Path:
    engine_dir = tmp_path / "decision-engine"
    api_server_dir = engine_dir / "apps" / "api_server"
    api_server_dir.mkdir(parents=True)
    (api_server_dir / "main.py").write_text("app = object()\n", encoding="utf-8")
    return engine_dir

def test_resolve_runtime_config_defaults_to_fixed_ports(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.delenv("CODNA_ENGINE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_ENGINE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_BASE_URL", raising=False)

    config = resolve_runtime_config()

    assert config.port_base == 18600
    assert config.engine_url == "http://127.0.0.1:18600"
    assert config.sidecar_url == "http://127.0.0.1:18601"
    assert config.paths.state_path == tmp_path / ".codna" / "runtime" / "local-stack.json"


def test_probe_engine_ready_uses_offline_health_without_db_readiness(monkeypatch) -> None:
    requested: list[str] = []

    def fake_request_json(url: str, *, timeout_s: float) -> ProbeResult:
        requested.append(url)
        assert timeout_s == 2.0
        return ProbeResult(True, url, 200, {"status": "ok"})

    monkeypatch.setattr(health_module, "request_json", fake_request_json)

    result = health_module.probe_engine_ready("http://127.0.0.1:18600", timeout_s=2.0)

    assert requested == ["http://127.0.0.1:18600/v1/health"]
    assert result.ok is True
    assert result.payload is not None
    assert result.payload["readiness"] == "local_offline"
    assert result.payload["db"] == "not_required"


def test_resolve_runtime_config_resolves_relative_runtime_root(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", "build/local-runtime-relative")

    config = resolve_runtime_config()

    assert config.paths.root.is_absolute()
    assert config.paths.root == (Path.cwd() / "build" / "local-runtime-relative").resolve()


def test_resolve_runtime_config_hash_tracks_runtime_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "runtime-a"))
    first = resolve_runtime_config()
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "runtime-b"))
    second = resolve_runtime_config()

    assert first.runtime_config_hash != second.runtime_config_hash


def test_resolve_runtime_config_uses_runtime_root_for_cline_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    config = resolve_runtime_config()

    assert config.cline_data_dir == tmp_path / ".codna" / "sidecar-state" / "cline-data"


def test_resolve_runtime_config_missing_sidecar_points_to_package_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    checkout_root = tmp_path / "checkout"
    package_root = tmp_path / "site-packages" / "codna"
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(config_module, "_codna_home", lambda: checkout_root)
    monkeypatch.setattr(config_module, "_codna_package_root", lambda: package_root)

    config = resolve_runtime_config()

    assert config.sidecar_dir == package_root / "agent-core"


def test_resolve_runtime_config_accepts_agent_core_vendor_dir_alias(tmp_path, monkeypatch) -> None:
    sidecar_dir = tmp_path / "agent-core"
    vendor_cline = sidecar_dir / "vendor" / "cline"
    (vendor_cline / "algenta").mkdir(parents=True)
    (vendor_cline / "algenta" / "run-server.ts").write_text("console.log('ok')\n", encoding="utf-8")
    (sidecar_dir / "run-server.mjs").write_text("console.log('launcher')\n", encoding="utf-8")
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_AGENT_CORE_DIR", str(vendor_cline))

    config = resolve_runtime_config()

    assert config.sidecar_dir == sidecar_dir


def test_resolve_runtime_config_rejects_loopback_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_ENGINE_URL", "http://127.0.0.1:9000")

    with pytest.raises(RuntimeConfigError) as excinfo:
        resolve_runtime_config()

    assert excinfo.value.code == "loopback_override_rejected"


def test_resolve_runtime_config_accepts_loopback_env_override_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_ENGINE_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("CODNA_ALLOW_LOOPBACK_ENGINE_URL", "1")

    config = resolve_runtime_config()

    assert config.remote_engine_url == "http://127.0.0.1:9000"


def test_resolve_runtime_config_ignores_loopback_keys_file_override(monkeypatch) -> None:
    monkeypatch.delenv("CODNA_ENGINE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_ENGINE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_BASE_URL", raising=False)
    config = resolve_runtime_config(
        keys={
            "CODNA_ENGINE_URL": ConfigValue(
                key="CODNA_ENGINE_URL",
                value="http://127.0.0.1:8002",
                source="keys_txt",
            )
        }
    )

    assert config.remote_engine_url is None
    assert config.ignored_engine_url is not None
    assert config.ignored_engine_url.value == "http://127.0.0.1:8002"


def test_resolve_runtime_config_accepts_non_loopback_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_ENGINE_URL", "https://api.codna.ai/")

    config = resolve_runtime_config()

    assert config.remote_engine_url == "https://api.codna.ai"
    assert config.engine_url == "http://127.0.0.1:18600"


def test_resolve_runtime_config_validates_port_base(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_PORT_BASE", "1023")

    with pytest.raises(RuntimeConfigError) as excinfo:
        resolve_runtime_config()

    assert excinfo.value.code == "invalid_port_base"


def test_resolve_runtime_config_sidecar_build_id_tracks_imported_sources(tmp_path, monkeypatch) -> None:
    sidecar_dir = tmp_path / "agent-core"
    algenta_dir = sidecar_dir / "vendor" / "cline" / "algenta" / "server"
    sdk_tools_dir = sidecar_dir / "vendor" / "cline" / "sdk" / "packages" / "core" / "src" / "extensions" / "tools"
    algenta_dir.mkdir(parents=True)
    sdk_tools_dir.mkdir(parents=True)
    (sidecar_dir / "run-server.mjs").write_text('console.log("launcher");\n', encoding="utf-8")
    (sidecar_dir / "package.json").write_text('{"name":"@algenta/agent-core"}\n', encoding="utf-8")
    (sidecar_dir / "vendor" / "cline" / "algenta" / "run-server.ts").write_text(
        'console.log("sidecar");\n',
        encoding="utf-8",
    )
    artifacts_path = algenta_dir / "artifacts.ts"
    artifacts_path.write_text("export const gate = 'v1';\n", encoding="utf-8")
    definitions_path = sdk_tools_dir / "definitions.ts"
    definitions_path.write_text("export const runCommands = 'parallel';\n", encoding="utf-8")
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_SIDECAR_DIR", str(sidecar_dir))

    first = resolve_runtime_config()

    definitions_path.write_text("export const runCommands = 'sequential';\n", encoding="utf-8")

    second = resolve_runtime_config()

    assert first.sidecar_build_id != second.sidecar_build_id


def test_engine_spawn_uses_codna_owned_algenta_runtime_dirs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("ALGENTA_ENGINE_DIR", str(_write_fake_engine_checkout(tmp_path)))
    engine_env_file = tmp_path / "engine.env"
    engine_env_file.write_text(
        "DATABASE_URL=should-not-leak\n"
        "ASYNC_DATABASE_URL=should-not-leak\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALGENTA_ENGINE_ENV_FILE", str(engine_env_file))
    monkeypatch.setenv("DATABASE_URL", "should-not-leak")
    monkeypatch.setenv("ASYNC_DATABASE_URL", "should-not-leak")
    config = resolve_runtime_config()

    command, cwd, env = local_stack_module._engine_spawn(config, "runtime-1")

    assert command[:3] == [config.engine_python, "-m", "uvicorn"]
    assert cwd == config.engine_dir
    assert env["ALGENTA_RUNTIME_DIR"] == str((tmp_path / ".codna" / "algenta-runtime").resolve())
    assert env["ALGENTA_REPOSITORY_INTELLIGENCE_SHARED_RUNTIME_DIR"] == str(
        (tmp_path / ".codna" / "algenta-runtime-shared").resolve()
    )
    assert env["ALGENTA_SOURCES_DIR"] == str((tmp_path / ".codna" / "algenta-sources").resolve())
    assert env["ALGENTA_SOURCE_ARTIFACT_CACHE_DIR"] == str(
        (tmp_path / ".codna" / "algenta-source-artifact-cache").resolve()
    )
    assert env["RUNTIME_FALLBACK_MODE"] == "deny"
    assert env["ALGENTA_ENGINE_URL"] == config.engine_url
    assert env["ALGENTA_AGENT_CORE_URL"] == config.sidecar_url
    assert "DATABASE_URL" not in env
    assert "ASYNC_DATABASE_URL" not in env


def test_sidecar_spawn_does_not_inherit_local_database_urls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("DATABASE_URL", "should-not-leak")
    monkeypatch.setenv("ASYNC_DATABASE_URL", "should-not-leak")
    monkeypatch.setattr(
        local_stack_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"node", "bun"} else None,
    )
    config = resolve_runtime_config()

    _command, _cwd, env = local_stack_module._sidecar_spawn(config, "runtime-1")

    expected_tmp_dir = str((tmp_path / ".codna" / "sidecar-runtime" / "tmp").resolve())
    expected_bun_cache_dir = str((tmp_path / ".codna" / "sidecar-runtime" / "bun-cache").resolve())
    assert env["RUNTIME_FALLBACK_MODE"] == "deny"
    assert env["TMPDIR"] == expected_tmp_dir
    assert env["TMP"] == expected_tmp_dir
    assert env["TEMP"] == expected_tmp_dir
    assert env["BUN_INSTALL_CACHE_DIR"] == expected_bun_cache_dir
    assert Path(expected_tmp_dir).is_dir()
    assert Path(expected_bun_cache_dir).is_dir()
    assert "DATABASE_URL" not in env
    assert "ASYNC_DATABASE_URL" not in env


def test_sidecar_spawn_requires_bun_for_real_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(
        local_stack_module.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )
    config = resolve_runtime_config()

    with pytest.raises(LocalRuntimeError) as excinfo:
        local_stack_module._sidecar_spawn(config, "runtime-1")

    assert excinfo.value.code == "bun_runtime_not_found"
    assert excinfo.value.details["expected_entrypoint"].endswith("run-server.mjs")


def test_sidecar_spawn_allows_node_only_for_contract_stub_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("ALGENTA_ALLOW_STUB_RUNTIME", "1")
    monkeypatch.setenv("ALGENTA_RUN_CLASS", "offline_contract_only")
    monkeypatch.setenv("CI_OFFLINE_CONTRACT_TEST", "1")
    monkeypatch.setattr(
        local_stack_module.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )
    config = resolve_runtime_config()

    command, _cwd, env = local_stack_module._sidecar_spawn(config, "runtime-1")

    assert command == ["/usr/bin/node", "run-server.mjs"]
    assert env["ALGENTA_ALLOW_STUB_RUNTIME"] == "1"


def test_sidecar_spawn_bundled_binary_redirects_proofs_to_writable_root(tmp_path, monkeypatch) -> None:
    # The bundled `bun build --compile` sidecar runs with NO node/bun, and its import.meta resolves
    # under the read-only `/$bunfs` embedded FS — so proof writes must be redirected to a writable
    # dir via AGENT_CORE_ROOT_DIR, or a fix run fails with `EROFS ... mkdir '/$bunfs'`.
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    binary = tmp_path / "codna-sidecar"
    binary.write_bytes(b"\x7fELF fake sidecar")
    binary.chmod(0o755)
    monkeypatch.setenv("CODNA_SIDECAR_BINARY", str(binary))
    # node and bun are deliberately absent — the bundled binary needs neither.
    monkeypatch.setattr(local_stack_module.shutil, "which", lambda name: None)
    config = resolve_runtime_config()

    assert config.sidecar_binary == binary
    command, _cwd, env = local_stack_module._sidecar_spawn(config, "runtime-1")

    assert command == [str(binary)]
    assert env["AGENT_CORE_ROOT_DIR"] == str(config.cline_data_dir)


def test_runtime_child_env_uses_keys_file_values_without_runtime_url_authority(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("ALGENTA_ENGINE_DIR", str(_write_fake_engine_checkout(tmp_path)))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        local_stack_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"node", "bun"} else None,
    )
    config = resolve_runtime_config()
    keys = {
        "OPENAI_API_KEY": ConfigValue(
            key="OPENAI_API_KEY",
            value="openai-from-keys",
            source="keys_txt",
        ),
        "TELYS_KERNEL": ConfigValue(
            key="TELYS_KERNEL",
            value="/tmp/libtelys.dylib",
            source="keys_txt",
        ),
        "CODNA_ENGINE_URL": ConfigValue(
            key="CODNA_ENGINE_URL",
            value="http://127.0.0.1:8002",
            source="keys_txt",
        ),
    }

    _sidecar_command, _sidecar_cwd, sidecar_env = local_stack_module._sidecar_spawn(
        config,
        "runtime-1",
        keys=keys,
    )
    _engine_command, _engine_cwd, engine_env = local_stack_module._engine_spawn(
        config,
        "runtime-1",
        keys=keys,
    )

    assert sidecar_env["OPENAI_API_KEY"] == "openai-from-keys"
    assert sidecar_env["TELYS_KERNEL"] == "/tmp/libtelys.dylib"
    assert "CODNA_ENGINE_URL" not in sidecar_env
    assert engine_env["OPENAI_API_KEY"] == "openai-from-keys"
    assert engine_env["TELYS_KERNEL"] == "/tmp/libtelys.dylib"
    assert engine_env["ALGENTA_ENGINE_URL"] == config.engine_url


def test_runtime_child_env_preserves_shell_provider_key_over_keys_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-from-env")
    monkeypatch.setattr(
        local_stack_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"node", "bun"} else None,
    )
    config = resolve_runtime_config()
    keys = {
        "OPENAI_API_KEY": ConfigValue(
            key="OPENAI_API_KEY",
            value="openai-from-keys",
            source="keys_txt",
        ),
    }

    _command, _cwd, sidecar_env = local_stack_module._sidecar_spawn(
        config,
        "runtime-1",
        keys=keys,
    )

    assert sidecar_env["OPENAI_API_KEY"] == "openai-from-env"


def test_inspect_runtime_reports_healthy_owned_pair(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "engine": {"pid": 101, "pid_create_time": 1001.0},
        "sidecar": {"pid": 102, "pid_create_time": 1002.0},
        "codna_cli_version": config.codna_cli_version,
        "engine_build_id": config.engine_build_id,
        "sidecar_build_id": config.sidecar_build_id,
        "python_executable": config.engine_python,
        "runtime_config_hash": config.runtime_config_hash,
    }

    monkeypatch.setattr(local_stack_module, "_load_state", lambda _: state)
    monkeypatch.setattr(
        local_stack_module,
        "_single_listener",
        lambda port: ListenerInfo(
            pid=101 if port == config.engine_port else 102,
            command="uvicorn apps.api_server.main:app" if port == config.engine_port else "node run-server.mjs",
            host="127.0.0.1",
            port=port,
            raw_name=f"127.0.0.1:{port}",
        ),
    )
    monkeypatch.setattr(local_stack_module, "pid_create_time", lambda pid: 1001.0 if pid == 101 else 1002.0)
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_health",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ok"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ok"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_version",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/version", 200, {"engine_version": "1.0.0"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_health",
        lambda *_args, **_kwargs: ProbeResult(
            True,
            "http://127.0.0.1:18601/health",
            200,
            {
                "service": "codna-sidecar",
                "status": "ok",
                "runtime_id": "runtime-1",
                "build_id": config.sidecar_build_id,
            },
        ),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18601/ready", 200, {"ready": True}),
    )

    payload = inspect_runtime()

    assert payload["status"] == "healthy_owned"
    assert payload["runtime_id"] == "runtime-1"
    assert payload["engine_listener"]["host"] == "127.0.0.1"


def test_inspect_runtime_rejects_stale_sidecar_build_health(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "engine": {"pid": 101, "pid_create_time": 1001.0},
        "sidecar": {"pid": 102, "pid_create_time": 1002.0},
        "codna_cli_version": config.codna_cli_version,
        "engine_build_id": config.engine_build_id,
        "sidecar_build_id": config.sidecar_build_id,
        "python_executable": config.engine_python,
        "runtime_config_hash": config.runtime_config_hash,
    }

    monkeypatch.setattr(local_stack_module, "_load_state", lambda _: state)
    monkeypatch.setattr(
        local_stack_module,
        "_single_listener",
        lambda port: ListenerInfo(
            pid=101 if port == config.engine_port else 102,
            command="uvicorn apps.api_server.main:app" if port == config.engine_port else "node run-server.mjs",
            host="127.0.0.1",
            port=port,
            raw_name=f"127.0.0.1:{port}",
        ),
    )
    monkeypatch.setattr(local_stack_module, "pid_create_time", lambda pid: 1001.0 if pid == 101 else 1002.0)
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_health",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ok"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ok"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_version",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/version", 200, {"engine_version": "1.0.0"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_health",
        lambda *_args, **_kwargs: ProbeResult(
            True,
            "http://127.0.0.1:18601/health",
            200,
            {
                "service": "codna-sidecar",
                "status": "ok",
                "runtime_id": "runtime-1",
                "build_id": "stale-build",
            },
        ),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18601/ready", 200, {"ready": True}),
    )

    payload = inspect_runtime()

    assert payload["status"] == "owned_partial_or_stale"


def test_inspect_runtime_rejects_sidecar_identity_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "engine": {"pid": 101, "pid_create_time": 1001.0},
        "sidecar": {"pid": 102, "pid_create_time": 1002.0},
        "codna_cli_version": config.codna_cli_version,
        "engine_build_id": config.engine_build_id,
        "sidecar_build_id": config.sidecar_build_id,
        "python_executable": config.engine_python,
        "runtime_config_hash": config.runtime_config_hash,
    }

    monkeypatch.setattr(local_stack_module, "_load_state", lambda _: state)
    monkeypatch.setattr(
        local_stack_module,
        "_single_listener",
        lambda port: ListenerInfo(
            pid=101 if port == config.engine_port else 102,
            command="uvicorn apps.api_server.main:app" if port == config.engine_port else "node run-server.mjs",
            host="127.0.0.1",
            port=port,
            raw_name=f"127.0.0.1:{port}",
        ),
    )
    monkeypatch.setattr(local_stack_module, "pid_create_time", lambda pid: 1001.0 if pid == 101 else 1002.0)
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_health",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ok"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ok"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_version",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/version", 200, {"engine_version": "1.0.0"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_health",
        lambda *_args, **_kwargs: ProbeResult(
            True,
            "http://127.0.0.1:18601/health",
            200,
            {"service": "other-sidecar", "status": "ok", "runtime_id": "runtime-1"},
        ),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18601/ready", 200, {"ready": True}),
    )

    payload = inspect_runtime()

    assert payload["status"] == "owned_partial_or_stale"


def test_inspect_runtime_reports_collision_for_foreign_listener(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    monkeypatch.setattr(local_stack_module, "_load_state", lambda _: None)
    monkeypatch.setattr(
        local_stack_module,
        "_single_listener",
        lambda port: ListenerInfo(
            pid=777,
            command="python -m other_app" if port == config.engine_port else None,
            host="0.0.0.0" if port == config.engine_port else "127.0.0.1",
            port=port,
            raw_name=f"0.0.0.0:{port}" if port == config.engine_port else f"127.0.0.1:{port}",
        )
        if port == config.engine_port
        else None,
    )

    payload = inspect_runtime()

    assert payload["status"] == "collision"
    assert payload["engine_listener"]["pid"] == 777


def test_inspect_runtime_classifies_health_proven_legacy_codna_pair(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    monkeypatch.setattr(local_stack_module, "_load_state", lambda _: None)
    monkeypatch.setattr(
        local_stack_module,
        "_single_listener",
        lambda port: ListenerInfo(
            pid=101 if port == config.engine_port else 102,
            command="Python" if port == config.engine_port else "bun.exe",
            host="127.0.0.1",
            port=port,
            raw_name=f"127.0.0.1:{port}",
        ),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_health",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ok"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ready"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_version",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/version", 200, {"engine_version": "1.0.0"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_health",
        lambda *_args, **_kwargs: ProbeResult(
            True,
            "http://127.0.0.1:18601/health",
            200,
            {"service": "codna-sidecar", "status": "ok", "runtime_id": "runtime-legacy"},
        ),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18601/ready", 200, {"ready": True}),
    )

    payload = inspect_runtime()

    assert payload["status"] == "legacy_codna_fixed_pair"
    assert payload["engine_listener"]["pid"] == 101
    assert payload["sidecar_listener"]["pid"] == 102


def test_inspect_runtime_keeps_foreign_pair_as_collision_without_sidecar_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    monkeypatch.setattr(local_stack_module, "_load_state", lambda _: None)
    monkeypatch.setattr(
        local_stack_module,
        "_single_listener",
        lambda port: ListenerInfo(
            pid=201 if port == config.engine_port else 202,
            command="python",
            host="127.0.0.1",
            port=port,
            raw_name=f"127.0.0.1:{port}",
        ),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_health",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ok"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/health", 200, {"status": "ready"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_engine_version",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18600/v1/version", 200, {"engine_version": "1.0.0"}),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_health",
        lambda *_args, **_kwargs: ProbeResult(
            True,
            "http://127.0.0.1:18601/health",
            200,
            {"service": "other-sidecar", "status": "ok", "runtime_id": "runtime-foreign"},
        ),
    )
    monkeypatch.setattr(
        local_stack_module,
        "probe_sidecar_ready",
        lambda *_args, **_kwargs: ProbeResult(True, "http://127.0.0.1:18601/ready", 200, {"ready": True}),
    )

    payload = inspect_runtime()

    assert payload["status"] == "collision"


def test_ensure_running_reuses_healthy_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        assert timeout_s > 0
        yield

    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)
    monkeypatch.setattr(
        local_stack_module,
        "inspect_runtime",
        lambda **_kwargs: {
            "status": "healthy_owned",
            "runtime_id": "runtime-42",
        },
    )
    monkeypatch.setattr(
        local_stack_module,
        "_start_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not start")),
    )

    endpoint = ensure_running()

    assert endpoint.engine_url == "http://127.0.0.1:18600"
    assert endpoint.sidecar_url == "http://127.0.0.1:18601"
    assert endpoint.runtime_id == "runtime-42"


def test_sidecar_spawn_source_checkout_also_redirects_proofs_to_writable_root(
    tmp_path, monkeypatch
) -> None:
    """The SOURCE/run-server.mjs branch must redirect proof writes too, not just the compiled binary.

    This is the shape a packaged deployment actually runs (codna-webhook has run-server.mjs and no
    sidecar binary). Left unset, run-server resolves ROOT_DIR to the install dir -- on codna-webhook
    that is `/app`, root-owned while the service runs unprivileged, so
    finalizeSessionArtifacts' unguarded mkdir raised EACCES on /app/build. That happens AFTER the
    model turn, so every fix run failed after paying for it and the patch was discarded, with no PR.
    """
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    sidecar_dir = tmp_path / "agent-core"
    sidecar_dir.mkdir()
    (sidecar_dir / "run-server.mjs").write_text("// fake supervisor\n")
    monkeypatch.setenv("CODNA_SIDECAR_DIR", str(sidecar_dir))
    monkeypatch.setattr(
        local_stack_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"node", "bun"} else None,
    )
    config = resolve_runtime_config()
    assert config.sidecar_binary is None  # the source branch is the one under test

    command, cwd, env = local_stack_module._sidecar_spawn(config, "runtime-src")

    assert command == ["/usr/bin/node", "run-server.mjs"]
    assert cwd == sidecar_dir
    assert env["AGENT_CORE_ROOT_DIR"] == str(config.cline_data_dir)
    # Never the install/checkout dir -- that is precisely what was unwritable in the container.
    assert env["AGENT_CORE_ROOT_DIR"] != str(sidecar_dir)
