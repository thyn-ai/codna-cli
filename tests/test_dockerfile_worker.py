"""Guards on the hosted durable-worker image (D6): no secret, the SANITIZED sidecar, pinned bases.

A Dockerfile is not executable by the test suite, so these are STRUCTURAL assertions about the text —
which is the right level for the properties that matter, because each one is a property of what the
file SAYS:

* **No credential.** A key in an image is a key in every registry copy, every `docker history` and
  every builder cache. The check is a DENYLIST of secret-shaped names plus a positive assertion that
  every `ENV` line is a path or a non-secret knob, so an added `ENV TEMPORAL_API_KEY=...` fails here
  rather than surviving to a registry.
* **The sanitized sidecar.** The unsanitized build carries 20 `cline.bot` literals; shipping it undoes
  V0. The Dockerfile must compile the REWRITTEN bundle (never the source), and must GREP the compiled
  binary and `exit 1` — a comment promising sanitization is not sanitization.
* **Pinned bases.** A tag is a moving target: the same `python:3.12-slim` built twice is two images.
  Both `FROM` lines must carry a digest.
* **Ordering.** `agent-core/` must be copied before `pip install /app/cli` (the wheel build bundles it
  and fails closed without it — the ordering that broke the Fly build once), and the sidecar must be
  staged into `codna/_agent_core_runtime/` BEFORE the install so the wheel has it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "infra" / "docker" / "Dockerfile.worker"
FLY = ROOT / "infra" / "fly" / "codna-durable-worker.fly.toml"
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy_durable_worker.sh"
README = ROOT / "infra" / "README-durable-worker.md"


@pytest.fixture(scope="module")
def text() -> str:
    assert DOCKERFILE.is_file(), "infra/docker/Dockerfile.worker is required for the hosted worker"
    return DOCKERFILE.read_text(encoding="utf-8")


# -- 1. no secret may be in the image ------------------------------------------------------


#: Names that must never appear on an `ENV`/`ARG` line. Deliberately broad: a miss here is a leaked
#: credential in a published artifact, so the cost of a false positive (a rename) is the cheap side.
SECRET_SHAPED = (
    "TEMPORAL_API_KEY",
    "CODNA_TEMPORAL_CODEC_KEY",
    "CODNA_DURABLE_DEPLOY_KEY",
    "CODNA_TEMPORAL_CLIENT_KEY_PEM",
    "CODNA_TEMPORAL_CLIENT_CERT_PEM",
    "CODNA_GITHUB_TOKEN",
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "CODNA_API_KEY",
    "SECRET",
    "PASSWORD",
    "PRIVATE_KEY",
    "DEPLOY_KEY",
    "APIKEY",
)


def _directive_lines(text: str, directive: str) -> list[str]:
    """The `ENV`/`ARG` instructions, with line continuations joined.

    Joined because a multi-line `ENV a=1 \\` + `b=${SECRET}` is one instruction, and a per-line check
    would see only the harmless half.
    """
    joined = re.sub(r"\\\n", " ", text)
    return [
        line.strip()
        for line in joined.splitlines()
        if line.strip().upper().startswith(f"{directive} ")
    ]


def test_no_env_or_arg_line_names_a_credential(text: str):
    offending: list[str] = []
    for line in _directive_lines(text, "ENV") + _directive_lines(text, "ARG"):
        upper = line.upper()
        for name in SECRET_SHAPED:
            if name.upper() in upper:
                offending.append(line)
                break
    assert not offending, (
        "the worker image must not set a credential — a key in an image is a key in every registry "
        f"copy. Offending instructions: {offending}"
    )


def test_no_build_arg_can_carry_a_credential(text: str):
    """`ARG` is the subtle leak: a build arg is recorded in the image's history even if unused."""
    args = _directive_lines(text, "ARG")
    assert args, "expected the base-image ARGs"
    # The only ARGs permitted are the pinned base image references and the Telys staging flag.
    for line in args:
        assert re.match(r"ARG\s+(PYTHON_IMAGE|BUN_IMAGE|TELYS_STAGED)=", line), (
            f"only the base-image pins and TELYS_STAGED may be build args; found {line!r}"
        )


def test_every_env_value_is_a_path_or_a_non_secret_knob(text: str):
    """Positive complement to the denylist: a NEW `ENV` has to look like configuration, not a key."""
    allowed_names = {
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        "CODNA_DURABLE_RUNS_ROOT",
        "CODNA_RUNTIME_ROOT",
        "TMPDIR",
        "CODNA_TEMPORAL_CODEC",
        "CODNA_WORKER_MAX_CONCURRENT_ACTIVITIES",
    }
    assigned = set()
    for line in _directive_lines(text, "ENV"):
        body = line[len("ENV ") :]
        for token in re.finditer(r"([A-Z_][A-Z0-9_]*)\s*=", body):
            assigned.add(token.group(1))
    unknown = sorted(assigned - allowed_names)
    assert not unknown, (
        f"the worker image sets an ENV this guard does not recognise: {unknown}. If it is a real "
        "non-secret knob, add it to `allowed_names`; if it is a credential, it does not belong here."
    )


def test_no_copy_reads_a_dev_credential_file(text: str):
    """No `COPY` may pull in a keychain, an env file, a PEM or a key material file."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        for forbidden in (".env", ".pem", ".key", "keys.txt", ".keys.txt", ".codna/", "*.jwt"):
            assert forbidden not in stripped, f"COPY must not read {forbidden}: {stripped!r}"


def test_the_dockerignore_keeps_credentials_out_of_the_context():
    """Belt and braces: the build CONTEXT must not carry a key either."""
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in (".env", ".env.*", "keys.txt", ".keys.txt", "*.pem", "*.key", ".codna"):
        assert required in patterns, f".dockerignore must exclude {required!r} from every build"


# -- 2. the SANITIZED sidecar, proven not promised -----------------------------------------


def _instructions(text: str) -> list[str]:
    """The Dockerfile's INSTRUCTIONS: comments stripped, continuations joined.

    Comments are dropped because a comment is prose about the build, not part of it — matching one is
    how a guard passes or fails for a reason unrelated to what the image does.
    """
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    joined = re.sub(r"\\\n", " ", body)
    return [line.strip() for line in joined.splitlines() if line.strip()]


def test_the_sidecar_is_built_from_the_SANITIZED_bundle(text: str):
    """`bun build run-server.bundle.mjs --compile` — the rewritten bundle, never the source.

    The sanitize-then-compile steps share ONE `RUN` layer (they must: the rewrite happens between
    them), so the commands are split on `&&` and the COMPILE command is the one asserted.
    """
    commands: list[str] = []
    for line in _instructions(text):
        if not line.startswith("RUN "):
            continue
        body = line[len("RUN ") :] if line.startswith("RUN ") else line[3:]
        commands.extend(part for part in body.split("&&") if part.strip())
    compile_commands = [part.strip() for part in commands if "--compile" in part]
    assert compile_commands, "the image must compile a standalone sidecar"
    for command in compile_commands:
        assert command.startswith("bun build run-server.bundle.mjs"), (
            "the standalone must be compiled from the SANITIZED bundle (`bun build "
            f"run-server.bundle.mjs --compile`), not from the TS source: {command!r}"
        )
        assert "run-server.ts" not in command, f"compiling the unsanitized source: {command!r}"
    # The bundle must be produced from the TS ENTRYPOINT in an EARLIER command in the same layer,
    # and the sanitize step must sit between the bundle and the compile — that ORDER is the
    # guarantee, so it is asserted rather than assumed from the presence of both strings.
    bundle_index = next(
        (index for index, part in enumerate(commands) if "--outfile run-server.bundle.mjs" in part),
        None,
    )
    sanitize_index = next(
        (index for index, part in enumerate(commands) if "cline.invalid" in part), None
    )
    compile_index = next(
        (index for index, part in enumerate(commands) if "--compile" in part), None
    )
    assert bundle_index is not None and sanitize_index is not None and compile_index is not None
    assert bundle_index < sanitize_index < compile_index, (
        "order must be: build the bundle -> rewrite it -> compile the REWRITTEN bundle. "
        f"got bundle={bundle_index}, sanitize={sanitize_index}, compile={compile_index}"
    )


def test_the_sanitization_rewrite_is_present_and_self_checking(text: str):
    """The rewrite must replace both literals AND fail the build if either survives in the bundle."""
    assert 'replaceAll("cline.bot","cline.invalid")' in text
    assert 'replace(/phc_[A-Za-z0-9]{20,}/g,"phc_redacted")' in text
    # The bun -e script exits non-zero when a literal remains, so a failed rewrite fails the LAYER.
    assert "SANITIZE FAILED" in text
    assert "process.exit(1)" in text


def test_the_COMPILED_binary_is_grepped_and_the_build_fails_on_a_match(text: str):
    joined = re.sub(r"\\\n", " ", text)
    # A build-stage gate and an installed-image gate: both must exist, and both must `exit 1`.
    gates = [
        block
        for block in joined.splitlines()
        if "cline.bot" in block and "grep" in block
    ]
    assert len(gates) >= 2, (
        "expected a gate in the build stage AND one on the installed package; a single grep can be "
        f"bypassed by a later layer: {gates}"
    )
    assert joined.count("SANITIZE FAILED") >= 2, (
        "each gate must fail the build with a message naming the problem"
    )


def test_the_installed_sidecar_is_the_one_the_package_launches(text: str):
    """`runtime/config.py` looks for `codna/_agent_core_runtime/codna-sidecar` — that exact path."""
    from codna.durable import __doc__ as _  # noqa: F401 - the package must import without the extra
    from codna.runtime.config import SIDECAR_RUNTIME_DIRNAME, _sidecar_binary_name

    assert "_agent_core_runtime" in text
    assert "codna-sidecar" in text
    # The names the runtime hard-codes must be the ones the Dockerfile stages.
    assert SIDECAR_RUNTIME_DIRNAME == "_agent_core_runtime"
    assert _sidecar_binary_name() == "codna-sidecar"


def test_agent_core_is_copied_before_pip_install_and_the_sidecar_is_staged_before_it(text: str):
    """Both orderings fail the wheel build if reversed — one broke the Fly build once."""
    lines = text.splitlines()
    copy_agent_core = next(
        (index for index, line in enumerate(lines) if "COPY" in line and "agent-core" in line), None
    )
    pip_install = next(
        (index for index, line in enumerate(lines) if "pip install" in line and "/app/cli" in line),
        None,
    )
    stage_sidecar = next(
        (
            index
            for index, line in enumerate(lines)
            if "cp " in line and "_agent_core_runtime/codna-sidecar" in line
        ),
        None,
    )
    assert copy_agent_core is not None and pip_install is not None and stage_sidecar is not None
    assert copy_agent_core < pip_install, "agent-core must be copied before the wheel build"
    assert stage_sidecar < pip_install, (
        "the compiled sidecar must be staged into the package BEFORE `pip install`, or the wheel "
        "does not contain it"
    )


# -- 3. deliberate pins and package coordinates --------------------------------------------


def test_both_base_images_are_pinned_by_digest(text: str):
    """A tag alone is a moving target: the same tag built twice is two different images.

    The `FROM`s interpolate ARGs, so the ASSERTION is on the ARG defaults — that is where a digest
    can be silently dropped, and the only place it is written.
    """
    from_lines = [
        line for line in _instructions(text) if line.upper().startswith("FROM ")
    ]
    assert len(from_lines) >= 2, "expected a build stage and a runtime stage"
    arg_defaults = [
        line for line in _instructions(text) if line.startswith("ARG PYTHON_IMAGE=")
        or line.startswith("ARG BUN_IMAGE=")
    ]
    assert len(arg_defaults) == 2, f"expected the two base-image ARGs, found {arg_defaults}"
    for line in arg_defaults:
        assert "@sha256:" in line, (
            f"base image is not pinned by digest, so it can change under us: {line!r}"
        )
    # Every `FROM` must be one of those two pinned values, never a bare tag.
    for line in from_lines:
        assert "${PYTHON_IMAGE}" in line or "${BUN_IMAGE}" in line, (
            f"a FROM bypasses the digest-pinned ARGs: {line!r}"
        )


def test_the_runtime_python_is_3_12_and_the_durable_extra_is_pinned(text: str):
    assert "python:3.12" in text
    assert "'/app/cli[durable]'" in text
    # The constraint file is the exact set the durable soak job tests, so an image and a green soak
    # cannot disagree about the Public-Preview Temporal/LangGraph pair.
    assert "constraints-durable.txt" in text
    assert "-c /tmp/constraints-durable.txt" in text


def test_the_image_is_non_root_and_declares_a_healthcheck(text: str):
    assert "USER app" in text
    assert "useradd" in text and "--uid 10001" in text
    assert "HEALTHCHECK" in text
    # The health check must ask the DURABLE stack, not merely that python starts.
    assert "require_durable" in text


def test_the_entrypoint_is_the_worker_and_execs_for_signal_delivery(text: str):
    assert 'ENTRYPOINT ["codna", "worker"]' in text
    # `--no-promote` must NOT be baked in: it is a deploy-time decision (see the README), and a baked
    # one would stop the first worker on a fresh deployment from ever receiving runs.
    entrypoint = next(line for line in text.splitlines() if line.strip().startswith("ENTRYPOINT"))
    assert "--no-promote" not in entrypoint


def test_the_image_does_NOT_set_CODNA_SIDECAR_DIR(text: str):
    """A measured trap: `CODNA_SIDECAR_DIR` forces the Node+Bun dev path and IGNORES the packaged
    standalone (`runtime/config.py::resolve_runtime_config`), so the worker fails with
    `bun_runtime_not_found` in an image that has no Bun. The standalone must be found by its own
    default lookup instead."""
    for line in _directive_lines(text, "ENV"):
        assert "CODNA_SIDECAR_DIR" not in line, (
            "CODNA_SIDECAR_DIR makes codna ignore the packaged self-contained sidecar and look for "
            "node+bun, which this image does not have. Remove it; the default lookup finds "
            "codna/_agent_core_runtime/codna-sidecar."
        )


def test_the_codec_is_required_in_the_image(text: str):
    """A hosted worker has no operator reading its banner, so plaintext history must fail startup."""
    assert 'CODNA_TEMPORAL_CODEC=required' in text


# -- 4. the deploy surface ----------------------------------------------------------------


def test_the_fly_config_matches_the_image_and_declares_the_pool():
    import tomllib

    config = tomllib.loads(FLY.read_text(encoding="utf-8"))
    assert config["app"] == "codna-durable-worker"
    assert config["build"]["dockerfile"] == "../docker/Dockerfile.worker"
    env = config["env"]
    assert env["CODNA_DURABLE_POOL"], "a hosted worker without a pool polls its hostname queue"
    assert env["CODNA_TEMPORAL_CODEC"] == "required"
    # The safety pair: whatever the counts, they must be SELF-CONSISTENT with pool.py's rule.
    from codna.durable.pool import pool_settings_from_env

    settings = pool_settings_from_env(env)
    assert settings is not None, "the Fly config's own env must be a legal pool configuration"
    assert settings.task_queue == f"codna.pool.{env['CODNA_DURABLE_POOL']}"
    # No `[http_service]`: a worker takes no proxy traffic, so there is nothing to unroute.
    assert "http_service" not in config


def test_the_fly_config_sets_no_secret():
    import tomllib

    config = tomllib.loads(FLY.read_text(encoding="utf-8"))
    for name in config.get("env", {}):
        for secret in SECRET_SHAPED:
            assert secret.upper() not in name.upper(), (
                f"{name} looks like a credential; Fly secrets are set with `fly secrets set`, not "
                "in this file"
            )


def test_the_deploy_script_is_executable_and_covered_by_its_own_guards():
    assert DEPLOY_SCRIPT.is_file()
    body = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    # Presence-only secret checks: the script must never read a value.
    assert "flyctl secrets list" in body
    assert "--json" in body
    # The pool safety gate must be in the script as well as in the worker's startup.
    assert "pool_state_not_shared" in body or "shared" in body
    # The sanitized-sidecar check on the DEPLOYED image.
    assert "cline.bot" in body
    # A successful deploy is verified by the worker's own lifecycle event, not by "the machine runs".
    assert "worker.starting" in body


def test_the_operator_readme_exists_and_names_the_unrun_half():
    """The exit criterion is a nightly run with the laptop closed; it is UNRUN, and saying so is part
    of the deliverable rather than an omission."""
    assert README.is_file(), "the deploy procedure must ship with the image"
    body = README.read_text(encoding="utf-8")
    for needle in (
        "fly secrets set",
        "CODNA_DURABLE_POOL",
        "pool_state_not_shared",
        "cline.bot",
        "NOT PROVEN",
    ):
        assert needle in body, f"the README must document {needle!r}"
