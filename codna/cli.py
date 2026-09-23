"""The `codna` command. Codna understands and fixes your repository.

Normal local topology: codna calls the local Algenta repository-intelligence SDK/core
in-process, writes local Arrow/Parquet artifacts, and starts the sidecar-only agent
core when verified planning needs model execution. Remote engine URLs remain supported
only when explicitly provided through the real shell environment.
"""
from __future__ import annotations

import argparse
from itertools import count
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import httpx

from . import __version__
from .focus_paths import _issue_focus_paths  # focus-path extraction (split out to keep cli.py small)
from .agent_core_runtime import ensure_agent_core_running, inspect_agent_core_runtime, stop_agent_core_runtime
from .local_mojo_pool import legacy_global_mojo_worker_diagnostics, mojo_worker_diagnostics
from .ci_cli import register as _register_ci
from .report_cli import register as _register_report
from .mcp_cli import cmd_mcp as _cmd_mcp
from .webhook_cli import cmd_webhook as _cmd_webhook
from .remote_http import RemoteHttpTimeoutError, remote_http_timeout
from .secure_open_pr_cli import (
    SecureOpenPrCliError,
    cmd_secure_open_pr as _cmd_secure_open_pr,
    write_secure_evidence,
)
from .runtime import LocalRuntimeError, ensure_running
from .runtime.config import API_KEY_KEYS, RuntimeConfigError, parse_keys_file_values, resolve_runtime_config

_CONNECTOR_NAME_COUNTER = count()


class CodnaError(Exception):
    """User-facing CLI error that can be surfaced without terminating embedded callers."""


def _die(msg: str) -> None:
    raise CodnaError(msg)


def _structured_error(exc: BaseException) -> dict[str, object] | None:
    # decision_engine's SDK exceptions (raised by the local Algenta engine bridge, e.g.
    # ServerError on a 5xx) expose their code as `.error_code`, not `.code` — codna's own
    # internal error types use `.code`. Missing the `.error_code` fallback meant any
    # unwrapped SDK exception that reached here returned None and got re-raised bare,
    # leaking a raw Python traceback to stderr instead of codna's documented
    # `{"error": {"code": ..., "message": ...}}` JSON contract (see troubleshooting.md).
    code = getattr(exc, "code", None) or getattr(exc, "error_code", None)
    if not code:
        return None
    details = getattr(exc, "details", {})
    payload: dict[str, object] = {
        "code": str(code),
        "message": str(exc),
        "details": details if isinstance(details, dict) else {},
    }
    request_id = getattr(exc, "request_id", None)
    if request_id:
        payload["request_id"] = str(request_id)
    return {"error": payload}


def _keys_file_path() -> Path | None:
    """The legacy DEV ``keys.txt`` (repo-root config for dev/CI), or ``None`` for an installed package.

    keys.txt is purely a source-checkout convenience. An END USER installs a wheel and must never see
    codna hunting for a ``keys.txt`` inside ``site-packages`` — so for an installed package we return
    ``None`` (no path computed, no stat, nothing surfaced in diagnostics). An explicit ``CODNA_KEYS_FILE``
    always wins (dev/CI); otherwise the default is used only from a source checkout.
    """
    override = os.environ.get("CODNA_KEYS_FILE")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    if any(part in ("site-packages", "dist-packages") for part in here.parts):
        return None  # installed wheel — keys.txt is a dev-only artifact, never looked for here
    return here.parents[2] / "keys.txt"


def _runtime_keys(*, include_keychain: bool = True):
    merged = {}
    path = _keys_file_path()
    if path is not None:
        path = path.expanduser()
        if path.is_file():
            merged.update(parse_keys_file_values(path))
    if not include_keychain:
        return merged
    from . import keystore
    merged.update(keystore.config_values())
    return merged


def _api_key(keys) -> str | None:
    for name in API_KEY_KEYS:
        value = os.environ.get(name)
        if value:
            return value
    for name in API_KEY_KEYS:
        config_value = keys.get(name)
        if config_value and config_value.value:
            return config_value.value
    return None


class _HttpCodnaClient:
    """Explicit remote/compatibility client for the repository-intelligence API."""

    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    _DEFAULT_MAX_RETRIES = 2
    _BACKOFF_BASE_S = 0.25

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        max_retries: int | None = None,
    ) -> None:
        if not api_key:
            _die("no API key — set CODNA_API_KEY (your codna key).")
        if not base_url:
            _die("no engine URL resolved — run `codna doctor --start-stack` for local runtime diagnostics.")
        self._max_retries = self._resolve_max_retries(max_retries)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"codna-cli/{__version__}",
            },
        )

    @classmethod
    def _resolve_max_retries(cls, max_retries: int | None) -> int:
        raw = str(
            max_retries
            if max_retries is not None
            else os.environ.get("CODNA_HTTP_MAX_RETRIES", cls._DEFAULT_MAX_RETRIES)
        )
        try:
            value = int(raw)
        except ValueError as exc:
            raise CodnaError("CODNA_HTTP_MAX_RETRIES must be an integer.") from exc
        if value < 0:
            raise CodnaError("CODNA_HTTP_MAX_RETRIES must be >= 0.")
        return value

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if json_payload is not None and json is not None:
            raise CodnaError("internal client received both json_payload and json.")
        body = json_payload if json_payload is not None else json
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(method, path, json=body, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise CodnaError(f"engine {method} {path} failed: {exc}") from exc
            if response.status_code in self._RETRYABLE_STATUS and attempt < self._max_retries:
                self._sleep_before_retry(attempt)
                continue
            return self._parse_response(method, path, response)
        raise CodnaError(f"engine {method} {path} failed: {last_error}")

    def _sleep_before_retry(self, attempt: int) -> None:
        time.sleep(self._BACKOFF_BASE_S * (attempt + 1))

    @staticmethod
    def _parse_response(method: str, path: str, response: httpx.Response) -> dict[str, Any]:
        payload: Any = {}
        if response.content:
            try:
                payload = response.json()
            except ValueError:
                payload = {"raw": response.text[:1000]}
        if 200 <= response.status_code < 300:
            if isinstance(payload, dict):
                return payload
            raise CodnaError(f"engine {method} {path} returned non-object JSON.")
        message, details = _extract_engine_error(payload)
        request_id = response.headers.get("x-request-id") or response.headers.get("x-codna-request-id")
        suffix = f" request_id={request_id}" if request_id else ""
        detail_text = f" details={json.dumps(details, sort_keys=True)}" if details else ""
        raise CodnaError(
            f"engine {method} {path} failed: status={response.status_code} "
            f"message={message}{suffix}{detail_text}"
        )

    def create_connector(
        self,
        *,
        name: str,
        connector_type: str,
        config: dict[str, Any] | None = None,
        description: str | None = None,
        visibility: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name, "connector_type": connector_type, "config": config or {}}
        if description is not None:
            payload["description"] = description
        if visibility is not None:
            payload["visibility"] = visibility
        return self._request("POST", "/v1/connectors", json_payload=payload)

    def get_repository_intelligence_capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/v1/repositories/capabilities")

    def create_repository_snapshot(self, repository_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/repositories/{repository_id}/snapshots", json_payload=request)

    def get_repository_snapshot(self, repository_id: str, snapshot_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/repositories/{repository_id}/snapshots/{snapshot_id}")

    def triage_repository(self, repository_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/repositories/{repository_id}/triage", json_payload=request)

    def create_repository_decision_plan(self, repository_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/repositories/{repository_id}/decision-plans", json_payload=request)

    def query_repository_graph(self, repository_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/repositories/{repository_id}/graph-query", json_payload=request)

    def simulate_repository(self, repository_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/repositories/{repository_id}/simulate", json_payload=request)

    def apply_repository(self, repository_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/repositories/{repository_id}/apply", json_payload=request)


def _extract_engine_error(payload: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return "engine returned a non-object error response", {}
    error = payload.get("error")
    if isinstance(error, dict):
        msg = error.get("message") or error.get("code") or "engine request failed"
        details = error.get("details") if isinstance(error.get("details"), dict) else {}
        return str(msg), details
    msg = payload.get("message") or payload.get("detail") or payload.get("raw") or "engine request failed"
    return str(msg), {}


def _engine_url_key():
    """Resolve engine URL + API key.

    Remote explicit env wins. Otherwise the owned local runtime is ensured for
    legacy HTTP-only surfaces. Normal repository triage/fix uses _client(), which
    defaults to the in-process local SDK path and does not call this function.
    """
    keys = _runtime_keys(include_keychain=False)
    config = resolve_runtime_config(keys=keys)
    if config.remote_engine_url:
        return config.remote_engine_url.rstrip("/"), _api_key(keys)
    endpoint = ensure_running(keys=keys)
    url = endpoint.engine_url
    return url.rstrip("/"), _api_key(keys)


def _remote_http_timeout() -> httpx.Timeout:
    try:
        return remote_http_timeout(os.environ)
    except RemoteHttpTimeoutError as exc:
        raise CodnaError(str(exc)) from exc


def _client(*, include_keychain: bool = False):
    keys = _runtime_keys(include_keychain=include_keychain)
    config = resolve_runtime_config(keys=keys)
    if not config.remote_engine_url:
        from .local_client import LocalCodnaRuntimeClient

        return LocalCodnaRuntimeClient(keys=keys)

    url = config.remote_engine_url
    key = _api_key(keys)
    if not key:
        _die("no API key — set CODNA_API_KEY (your codna key).")
    return _HttpCodnaClient(api_key=key, base_url=url, timeout=_remote_http_timeout())


def _dump(x):
    return x.model_dump() if hasattr(x, "model_dump") else x


def _connector_name(prefix: str = "codna") -> str:
    return f"{prefix}-{time.time_ns()}-{next(_CONNECTOR_NAME_COUNTER)}"


def _clean_git_head(local_path: str) -> str | None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=local_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if head.returncode != 0:
            return None
        revision = head.stdout.strip().lower()
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=local_path,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if status.returncode != 0 or status.stdout:
            return None
        return revision
    except (OSError, subprocess.SubprocessError):
        return None


def _local_repo_connector_config(local_path: str) -> dict[str, object]:
    config: dict[str, object] = {"path": local_path}
    clean_head = _clean_git_head(local_path)
    if clean_head is not None:
        config["assume_clean_git_clone"] = True
        config["resolved_revision"] = clean_head
    return config


def _register(
    c,
    repo: str,
    ref: str | None,
    github_token: str | None = None,
    focus_paths: list[str] | None = None,
):
    local = os.path.abspath(os.path.expanduser(repo))
    if os.path.isdir(local) and not github_token:
        # the user's own repo on this machine (local SDK/core reads it directly)
        conn = _dump(c.create_connector(name=_connector_name(),
                                        connector_type="local_repo",
                                        config=_local_repo_connector_config(local)))
    elif repo.startswith(("http://", "https://")) or repo.endswith(".git"):
        cfg = {"repository_url": repo}
        if github_token:  # write scope, needed to push a branch + open a PR
            cfg["access_token"] = github_token
        conn = _dump(c.create_connector(name=_connector_name(),
                                        connector_type="github_repo", config=cfg))
    else:
        _die(f"'{repo}': not a local directory or a git URL.")
    rid = conn.get("id") or conn.get("connector_id")
    req = {"focus_paths": list(dict.fromkeys(focus_paths or []))}
    if ref:
        req["ref"] = ref
    snap = _dump(c.create_repository_snapshot(rid, req))
    return rid, snap


def cmd_triage(args) -> None:
    as_json = getattr(args, "as_json", False)
    c = _client()
    if not as_json:
        print(f"codna: understanding {args.repo} …")
    t = time.perf_counter()
    local = os.path.abspath(os.path.expanduser(args.repo))
    issue = args.issue or "Map this repository and locate its most relevant code."
    focus_paths = _issue_focus_paths(local, issue) if os.path.isdir(local) else []
    rid, snap = _register(c, args.repo, args.ref, focus_paths=focus_paths)
    sig = {"issue_text": issue}
    if focus_paths:
        sig["changed_files"] = focus_paths
    tri = _dump(c.triage_repository(rid, {"snapshot_id": snap["snapshot_id"], "signals": sig}))
    dt = time.perf_counter() - t
    files = snap.get("snapshot_file_count") or snap.get("file_count")
    raw = tri.get("raw_repo_token_estimate") or 0
    bundle = tri.get("evidence_bundle_token_count") or 0
    rr = tri.get("reduction_ratio")
    if as_json:
        # Stable machine-readable triage result (stdout stays pure JSON; errors go to stderr in main()).
        print(json.dumps({
            "repository_id": rid,
            "snapshot_id": snap.get("snapshot_id"),
            "snapshot_file_count": files,
            "suspect_files": tri.get("suspect_files") or [],
            "suspect_symbols": tri.get("suspect_symbols") or [],
            "raw_repo_token_estimate": raw,
            "evidence_bundle_token_count": bundle,
            "reduction_ratio": rr,
            "workspace_evidence_bundle_ref": tri.get("workspace_evidence_bundle_ref"),
            "elapsed_s": round(dt, 3),
        }, indent=2))
        return
    print(f"\n✓ codna understood {files} files in {dt:.1f}s")
    print(f"  suspect files : {', '.join(tri.get('suspect_files') or []) or '(none)'}")
    if rr:
        print(f"  context        : {raw:,} → {bundle:,} tokens  ({rr:.0f}× smaller for the agent)")


# Fix-input helpers (from_junit / resolve_issue / sim_id / pr_body) live in fix_inputs.py to keep
# cli.py under the modularity ceiling. Aliased to the historical private names so cmd_fix call
# sites and tests are unchanged.


def cmd_fix(args) -> int:
    """Find + fix a bug. With --apply runs the risk gate + applies to a local branch; with --tests it
    then re-runs the repo's tests and, up to --max-iterations, re-fixes until green. --json for scripts."""
    from . import fix_run
    c = _client(include_keychain=True)
    as_json = getattr(args, "as_json", False)
    if as_json:
        import contextlib

        # Keep stdout machine-parseable even if the local engine/agent stack logs to stdout.
        with contextlib.redirect_stdout(sys.stderr):
            result = fix_run.run_fix(c, args)
        print(json.dumps(result, indent=2))
    else:
        print(f"codna: fixing {args.repo} …")
        result = fix_run.run_fix(c, args)
        print("\n".join(fix_run.render_human(result)))
    # non-zero exit when verification ran and did not reach green (scriptable signal)
    return 1 if result.get("verified") is False else 0


def cmd_secure(args) -> None:
    """Ingest a scanner's SARIF, prove reachability via the engine, and report which findings
    are exploitable/reachable and autofix-eligible (Tier-1: read-only, 0 LLM tokens). With
    --fix/--open-pr, runs the sandboxed worker + PR writer to remediate eligible findings."""
    from .engine import EngineAdapter, http_post_factory
    from .policy import Policy
    from .sarif import SarifError, ingest_sarif

    if not args.from_sarif:
        _die("`codna secure` needs --from-sarif <results.sarif>.")
    if args.open_pr and not args.verification:
        _die("--open-pr needs --verification <codna-security.yaml> (pinned scanner + build/test commands).")
    if args.fix and args.engine != "local" and not args.verification:
        _die("--fix needs --verification <codna-security.yaml> (scanner + build/test commands); "
             "or use --engine local for a scanner-less local fix.")
    evidence_dir = getattr(args, "evidence_dir", None)
    if evidence_dir:
        if not args.fix:
            _die("--evidence-dir needs --fix so the worker can generate and verify a patch.")
        if args.open_pr:
            _die("--evidence-dir is for the two-job handoff; remove --open-pr from the analyze job.")
        if args.engine == "local":
            _die("--evidence-dir currently requires --engine remote; use local --fix for report-only local fixes.")
        if not args.verification:
            _die("--evidence-dir needs --verification <codna-security.yaml>.")
    if getattr(args, "as_json", False) and (args.fix or args.open_pr):
        _die("`codna secure --json` is read-only; remove --json to run --fix/--open-pr.")

    policy = Policy()
    config_digest = None
    manifest = None
    if args.verification:
        from .manifest import load_manifest, ManifestError
        try:
            manifest = load_manifest(args.verification)
        except ManifestError as exc:
            _die(f"invalid verification manifest {args.verification}: {exc}")
        policy, config_digest = manifest.policy, manifest.scanner.configuration_digest
        if args.open_pr:
            try:
                manifest.require_resolved_for_pr()
            except ManifestError as exc:
                _die(str(exc))

    if not os.path.isfile(args.from_sarif):
        _die(f"could not read SARIF {args.from_sarif}: file not found")
    try:
        ingest = ingest_sarif(args.from_sarif)
    except SarifError as exc:
        _die(f"could not ingest SARIF {args.from_sarif}: {exc}")
    if not ingest.provenance_valid:
        _die("SARIF provenance incomplete: " + "; ".join(ingest.provenance_errors))

    rid = snap = url = key = None
    if args.engine == "local":
        # Bounded, self-hostable reference engine — never claims 'exploitable'.
        from .refengine import LocalReferenceEngine
        engine = LocalReferenceEngine()
    else:
        c = _client()
        if not getattr(args, "as_json", False):
            print(f"codna: understanding {args.repo} for security analysis …")
        rid, snap = _register(c, args.repo, args.ref)
        url, key = _engine_url_key()
        model_pack = os.environ.get("CODNA_MODEL_PACK_DIGEST", "sha256:" + "0" * 64)
        engine = EngineAdapter(
            http_post_factory(url, key),
            repository_id=rid, snapshot_id=snap["snapshot_id"],
            policy_digest=policy.digest, model_pack_digest=model_pack,
            configuration_digest=config_digest,
        )

    from .secure import classify_only
    report = classify_only(ingest, engine=engine, policy=policy)
    if getattr(args, "as_json", False):
        # Structured Tier-1 reachability result (read-only; --fix/--open-pr remediation stays human).
        print(json.dumps({
            "repository": args.repo,
            "sarif": args.from_sarif,
            "findings": [{"classification": str(r.classification), "rule_id": r.rule_id,
                          "finding_kind": r.finding_kind, "eligible": bool(r.eligible),
                          "reason": None if r.eligible else r.reason} for r in report.rows],
            "summary": report.counts(),
            "autofix_eligible": report.eligible,
        }, indent=2))
        return
    print(f"\n✓ analyzed {len(report.rows)} finding(s) from {args.from_sarif}")
    for r in report.rows:
        mark = "→" if r.eligible else "·"
        note = "autofix-eligible" if r.eligible else r.reason
        print(f"  {mark} [{r.classification}] {r.rule_id} ({r.finding_kind})  {note}")
    counts = ", ".join(f"{k}={v}" for k, v in sorted(report.counts().items()))
    print(f"  summary: {counts}  ·  autofix-eligible: {report.eligible}")

    if not (args.open_pr or args.fix):
        return
    if args.engine == "local":
        # Self-hosted detect→fix→verify→close via the agentic Cline SDK; local checkout apply only.
        if args.open_pr:
            _die("--open-pr needs the privilege-separated writer + a scoped token; "
                 "`--engine local --fix` applies to the local checkout but opens no PR.")
        return _run_local_fix(args, ingest=ingest, engine=engine, policy=policy, manifest=manifest)
    else:
        return _run_remediation(args, ingest=ingest, rid=rid, snapshot_id=snap["snapshot_id"],
                                url=url, key=key, policy=policy, manifest=manifest)


def _repo_slug(repo_dir: str, override: str | None) -> str:
    if override:
        return override
    import re
    import subprocess
    out = subprocess.run(["git", "-C", repo_dir, "remote", "get-url", "origin"],
                         capture_output=True, text=True).stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", out)
    if not m:
        _die("could not derive owner/repo from the git remote; pass --repo-slug owner/repo.")
    return m.group(1)


class _NoWriter:
    """Report-only writer for `--fix` (no PR): local tree, never opens anything."""

    def base_unchanged(self, ingest) -> bool:
        return True

    def open_draft_pr(self, finding, patch, attestation):  # pragma: no cover - never called
        raise RuntimeError("report-only mode does not open PRs")


def _secure_fix_exit_code(*, eligible: int, remediated: int) -> int:
    """Signal CI failure when requested eligible fixes did not all verify."""
    return 1 if eligible > remediated else 0


def _run_local_fix(args, *, ingest, engine, policy, manifest) -> int:
    """Local secure fix: prove, verify, close, then apply the patch to the checkout."""
    import os

    from .localfix import run_local_fix
    from .patchgen import ClinePatchGenerator
    from .sarif import validate_commit_binding

    repo_dir = os.path.abspath(os.path.expanduser(args.repo))
    if not os.path.isdir(repo_dir):
        _die("--fix needs a local repo checkout (pass a local path, not a URL).")
    commit = validate_commit_binding(ingest)
    test_cmds = (manifest.build + manifest.tests) if manifest else []
    model = os.environ.get("CODNA_FIX_MODEL")

    print("\ncodna: remediating eligible findings with the Codna agent (local engine) …")
    report = run_local_fix(
        ingest, engine=engine, patch_generator=ClinePatchGenerator(model=model, keys=_runtime_keys()),
        repo_dir=repo_dir, commit=commit, policy=policy, test_cmds=test_cmds, apply_to_repo=True,
    )

    print(f"\n✓ local fix: {report.eligible} eligible finding(s) processed")
    for o in report.outcomes:
        if not o.eligible:
            continue
        if o.remediated:
            tag = "tests pass" if o.tests_passed else "no tests configured" if o.tests_passed is None else ""
            status = f"REMEDIATED — closure={o.closure}" + (f", {tag}" if tag else "")
            status += ", applied to checkout" if o.target_applied is True else ""
        elif not o.integrity_ok:
            status = f"blocked: patch integrity ({o.integrity_reason})"
        elif o.tests_passed is False:
            status = f"blocked: tests failed (closure={o.closure})"
        elif o.target_applied is False:
            status = f"blocked: target apply failed ({o.target_apply_reason})"
        elif o.target_tests_passed is False:
            status = f"blocked: target tests failed ({o.target_apply_reason})"
        else:
            status = f"blocked: obligation not closed (closure={o.closure})"
        print(f"  [{o.classification}] {o.rule_id}  {status}")
    remediated = len(report.remediated)
    print(f"  remediated: {remediated}/{report.eligible}")
    return _secure_fix_exit_code(eligible=report.eligible, remediated=remediated)


def _run_remediation(args, *, ingest, rid, snapshot_id, url, key, policy, manifest) -> None:
    import os

    from .engine import EngineAdapter, http_post_factory
    from .evidence import HmacSigner
    from .patchgen import EnginePatchGenerator
    from .sarif import validate_commit_binding
    from .secure import run_secure
    from .worker import LocalWorker
    from .writer import GitHubWriter, gh_head_resolver_factory, git_push_create_pr_factory

    repo_dir = os.path.abspath(os.path.expanduser(args.repo))
    if not os.path.isdir(repo_dir):
        _die("remediation needs a local repo checkout (pass a local path, not a URL).")
    commit = validate_commit_binding(ingest)
    post = http_post_factory(url, key)
    attestation_key = os.environ.get("CODNA_ATTESTATION_KEY") or ""
    evidence_dir = getattr(args, "evidence_dir", None)
    if evidence_dir and not attestation_key:
        _die("set CODNA_ATTESTATION_KEY in the analyze job before using --evidence-dir.")
    signer = HmacSigner(attestation_key.encode() if attestation_key else os.urandom(32), key_id="codna-worker")
    worker = LocalWorker(
        repo_dir=repo_dir, commit=commit, manifest=manifest, signer=signer,
        patch_generator=EnginePatchGenerator(post, repository_id=rid, snapshot_id=snapshot_id),
        base_env=dict(os.environ),
    )
    engine = EngineAdapter(post, repository_id=rid, snapshot_id=snapshot_id,
                           policy_digest=policy.digest,
                           model_pack_digest=os.environ.get("CODNA_MODEL_PACK_DIGEST", "sha256:" + "0" * 64),
                           configuration_digest=manifest.scanner.configuration_digest)
    if args.open_pr:
        gh = args.github_token or os.environ.get("CODNA_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not gh:
            _die("--open-pr needs a write token: --github-token or $GITHUB_TOKEN.")
        slug = _repo_slug(repo_dir, getattr(args, "repo_slug", None))
        writer = GitHubWriter(trusted_signers={signer.key_id: signer},
                              create_pr=git_push_create_pr_factory(repo_dir, slug, gh),
                              head_resolver=gh_head_resolver_factory(slug, gh),
                              base_branch=args.base_branch or "main")
    else:
        writer = _NoWriter()

    try:
        report = run_secure(ingest, engine=engine, worker=worker, writer=writer,
                            policy=policy, open_pr=args.open_pr)
    finally:
        worker.close()

    print(f"\n✓ remediation: {report.eligible} eligible finding(s) processed")
    for d in report.decisions:
        if d.gate is None:
            continue
        if d.opened:
            status = f"opened {d.pr_url}"
        elif d.gate.should_open_pr:
            status = "would open (report-only)"
        else:
            status = "blocked: " + ",".join(d.gate.failed)
        print(f"  [{d.classification.value}] {d.canonical_id[:20]}…  {status}")
    if evidence_dir:
        try:
            write_secure_evidence(evidence_dir, report.evidence_bundles)
        except SecureOpenPrCliError as exc:
            _die(str(exc))


def cmd_secure_open_pr(args) -> None:
    return _cmd_secure_open_pr(args, die=_die)

def cmd_webhook(args):
    return _cmd_webhook(args)


def cmd_mcp(args):
    return _cmd_mcp(args, die=_die)


def cmd_review(args) -> int:
    """Read-only review of a change (findings by default, ``--triage`` for engine risk-triage)."""
    from . import review
    return review.dispatch(args, _client)


def cmd_init(args) -> int:
    from .scaffold import init_project
    result = init_project(force=getattr(args, "force", False), with_agents=not getattr(args, "no_agents", False))
    print(json.dumps({"ok": True, "files": result}, indent=2))
    return 0


def cmd_status(args) -> int:
    """Concise health: engine · telys runtime · license · keys. Degrades gracefully (no spawn)."""
    from .status import build_status_lines
    keys = _runtime_keys(include_keychain=False)
    print("\n".join(build_status_lines(keys, api_key_present=bool(_api_key(keys)))))
    return 0


def cmd_login(args) -> int:
    """One-time device authorization that also installs the on-device runtime — one command.

    The body lives in ``codna.login.run`` (cli.py is at the module-size ceiling). Exit codes:
    0 = signed in AND provisioned (or already provisioned); 1 = sign-in failed; 2 = signed in but
    runtime provisioning incomplete (needs network on first run — re-run).
    """
    from .login import run as login_run
    return login_run(args, runtime_keys=_runtime_keys)


def cmd_doctor(args) -> int:
    try:
        start_stack = bool(getattr(args, "start_stack", False))
        stop_stack = bool(getattr(args, "stop_stack", False))
        if start_stack and stop_stack:
            _die("--start-stack and --stop-stack are mutually exclusive")
        keys = _runtime_keys(include_keychain=False)
        if stop_stack:
            payload = stop_agent_core_runtime(keys=keys)
            payload["mojo_workers"] = _doctor_mojo_worker_diagnostics(keys)
            print(json.dumps(payload, indent=2))
            return 0
        if start_stack:
            endpoint = ensure_agent_core_running(keys=keys)
            payload = inspect_agent_core_runtime(keys=keys)
            payload["started"] = {
                "url": endpoint.url,
                "port": endpoint.port,
                "runtime_id": endpoint.runtime_id,
                "pid": endpoint.pid,
            }
        else:
            payload = inspect_agent_core_runtime(keys=keys)
    except LocalRuntimeError as exc:
        print(json.dumps(exc.to_dict(), indent=2))
        return 1
    payload["mojo_workers"] = _doctor_mojo_worker_diagnostics(keys)
    print(json.dumps(payload, indent=2))
    return 0


def _doctor_mojo_worker_diagnostics(keys) -> dict[str, object]:
    try:
        config = resolve_runtime_config(keys=keys)
    except RuntimeConfigError as exc:
        owned_local = {
            "status": "unavailable",
            "owned_by_current_runtime": True,
            "reason": str(exc),
            "error_code": exc.code,
            "details": exc.details,
        }
    else:
        return mojo_worker_diagnostics(config)
    return {
        "owned_local": owned_local,
        "legacy_global": legacy_global_mojo_worker_diagnostics(),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codna", description="Codna — understand, fix & secure your repo.")
    p.add_argument("--version", action="version", version=f"codna {__version__}")
    p.add_argument("--config", help="path to codna.yaml (default: ./codna.yaml or ./.codna.yaml)")
    # `webhook` is the hosted GitHub App server (operator-only) — kept functional but hidden from
    # `--help`, so the public metavar lists only user-facing commands.
    sub = p.add_subparsers(
        dest="cmd",
        metavar="{triage,fix,doctor,secure,secure-open-pr,mcp,review,init,status,login,key}",
    )

    pt = sub.add_parser("triage", help="Understand a repo + locate relevant code (fast, deterministic).")
    pt.add_argument("repo", nargs="?", default=".", help="local path or git URL (default: current dir)")
    pt.add_argument("--ref", help="branch/tag/commit")
    pt.add_argument("--issue", help="optional: what you're looking for")
    pt.add_argument("--json", dest="as_json", action="store_true", help="emit the triage result as JSON")
    pt.set_defaults(func=cmd_triage, uses_config=True)

    pf = sub.add_parser("fix", help="Find and fix a bug in a repo.")
    pf.add_argument("repo", nargs="?", default=".", help="local path or git URL (default: current dir)")
    pf.add_argument("--issue", required=False, help="what's broken (e.g. the failing test)")
    pf.add_argument("--failing-test", action="append", help="failing test id (repeatable)")
    pf.add_argument("--from-junit", help="read failing tests from a JUnit/pytest XML report")
    pf.add_argument("--tests", action="store_true",
                    help="auto-discover failing tests by running the repo's tests (sandboxed) and fix them")
    pf.add_argument("--test-cmd", dest="test_cmd",
                    help="custom test command for --tests (writes JUnit to $CODNA_JUNIT); default is pytest")
    pf.add_argument("--ref", help="branch/tag/commit")
    pf.add_argument("--model", default="repository.verified_agentic_v1",
                    help="planner model (default: the codna agent planner)")
    pf.add_argument("--apply", action="store_true", help="apply the patch to a local branch")
    pf.add_argument("--open-pr", dest="open_pr", action="store_true",
                    help="push a branch and open a pull request (git URL + write token)")
    pf.add_argument("--github-token", help="write token for --open-pr (or env GITHUB_TOKEN)")
    pf.add_argument("--base-branch", help="PR base branch (default: repo default)")
    pf.add_argument("--pr-title", help="pull request title")
    pf.add_argument("--pr-body", help="pull request body")
    pf.add_argument("--json", dest="as_json", action="store_true", help="emit the fix result as JSON")
    pf.add_argument("--max-iterations", dest="max_iterations", type=int, default=1,
                    help="with --tests --apply: re-fix up to N times until the tests pass (default: 1)")
    pf.set_defaults(func=cmd_fix, uses_config=True, uses_model_config=True)

    pd = sub.add_parser("doctor", help="Inspect the owned local Codna runtime and optionally start agent-core.")
    pd.add_argument("--start-stack", action="store_true", help="Ensure the fixed local agent-core runtime is running.")
    pd.add_argument("--stop-stack", action="store_true", help="Stop the fixed local agent-core runtime.")
    pd.set_defaults(func=cmd_doctor)

    ps = sub.add_parser("secure", help="Prove which scanner findings are reachable (ingest SARIF, 0 LLM tokens).")
    ps.add_argument("repo", nargs="?", default=".", help="local path or git URL (default: current dir)")
    ps.add_argument("--from-sarif", dest="from_sarif", required=True,
                    help="path to the scanner's SARIF output (CodeQL/Semgrep/Snyk/Trivy)")
    ps.add_argument("--ref", help="branch/tag/commit")
    ps.add_argument("--engine", choices=["remote", "local"], default="local",
                    help="reachability engine: 'local' (bounded self-host reference) or explicit 'remote'")
    ps.add_argument("--verification", help="codna-security.yaml manifest (required for --open-pr)")
    ps.add_argument("--open-pr", dest="open_pr", action="store_true",
                    help="open a fix PR per reachable finding (requires a resolved manifest)")
    ps.add_argument("--fix", action="store_true", help="remediate eligible findings in the local checkout (no PR)")
    ps.add_argument("--github-token", help="write token for --open-pr (or env GITHUB_TOKEN)")
    ps.add_argument("--base-branch", help="PR base branch (default: repo default)")
    ps.add_argument("--repo-slug", dest="repo_slug", help="owner/repo for --open-pr (else derived from the git remote)")
    ps.add_argument("--evidence-dir", dest="evidence_dir",
                    help="write signed evidence bundles for secure-open-pr handoff")
    ps.add_argument("--json", dest="as_json", action="store_true",
                    help="emit the Tier-1 reachability result as JSON (read-only)")
    ps.set_defaults(func=cmd_secure, uses_config=True)

    po = sub.add_parser("secure-open-pr", help="Writer domain: verify a signed evidence bundle and open the draft PR.")
    po.add_argument("--evidence", required=True, help="evidence dir (attestation.json + patch.diff + finding.json)")
    po.add_argument("--repo-slug", dest="repo_slug", required=True, help="owner/repo")
    po.add_argument("--github-token", help="write token (or env GITHUB_TOKEN)")
    po.add_argument("--base-branch", help="PR base branch (default: main)")
    po.set_defaults(func=cmd_secure_open_pr)

    pw = sub.add_parser("webhook")  # operator-only (hosted App server); omit help= to hide from --help
    pw_sub = pw.add_subparsers(dest="action")
    pw_serve = pw_sub.add_parser("serve", help="Serve the webhook: verify → classify → run codna fix/secure.")
    pw_serve.add_argument("--host", default="0.0.0.0", help="bind host (default: 0.0.0.0)")
    pw_serve.add_argument("--port", type=int, default=8080, help="bind port (default: 8080)")
    pw_serve.add_argument("--role", choices=["all", "ingress", "worker"], default=None,
                          help="process role (default: $CODNA_WEBHOOK_ROLE or all)")
    pw_serve.set_defaults(func=cmd_webhook, action="serve")
    from .webhook_ops import add_ops_parser

    add_ops_parser(pw_sub)  # `codna webhook ops|migrate|queue` (Postgres queue backend)
    pw.set_defaults(func=cmd_webhook, action="serve")

    pm = sub.add_parser("mcp", help="Run codna as an MCP server, or install it into a client (Cursor / Claude).")
    pm.add_argument("action", nargs="?", choices=["start", "install"], default="start",
                    help="start the MCP server (default), or install it into a client config")
    pm.add_argument("--repo", help="default repo for MCP tools that accept one (local path or git URL)")
    pm.add_argument("--client", choices=["cursor", "claude"], help="for `install`: which MCP client to configure")
    pm.add_argument("--project", action="store_true",
                    help="for `install`: write the project-local config (Cursor) instead of the user config")
    pm.set_defaults(func=cmd_mcp)

    prv = sub.add_parser("review", help="Read-only review of a change (diff → structured bug findings; optionally post to a PR).")
    prv.add_argument("repo", nargs="?", default=".", help="local repo path (default: current dir)")
    prv.add_argument("--base", help="diff against this ref (default: HEAD — all uncommitted changes)")
    prv.add_argument("--diff", help="diff range to review, e.g. origin/main...HEAD (overrides --base)")
    prv.add_argument("--pr", help="PR to review/post to: 123, owner/repo#123, or a PR URL")
    prv.add_argument("--post", action="store_true", help="post the findings to the PR (one review + a 'codna review' check); needs --pr + a write token")
    prv.add_argument("--github-token", dest="github_token", help="write token for --post (else $GITHUB_TOKEN / $CODNA_GITHUB_TOKEN)")
    prv.add_argument("--min-confidence", dest="min_confidence", type=float, help="only report findings at/above this confidence (default from codna.yaml or 0.75)")
    prv.add_argument("--max-findings", dest="max_findings", type=int, help="cap the number of findings (default from codna.yaml or 10)")
    prv.add_argument("--blocking", action="store_true", help="make the 'codna review' check fail on a blocking-severity finding (default: non-blocking)")
    prv.add_argument("--effort", choices=("low", "medium", "high"), help="review depth: high = more thorough (more agent iterations), low = cheaper (default: medium)")
    prv.add_argument("--full", action="store_true", help="review the WHOLE PR, not only commits pushed since Codna's last review (default: incremental)")
    prv.add_argument("--model", help="planner/review model (default: provider default)")
    prv.add_argument("--triage", action="store_true", help="legacy: engine-backed risk triage (suspect files/symbols) instead of findings")
    prv.add_argument("--issue", help="triage mode: focus the review (e.g. 'check the auth refactor')")
    prv.add_argument("--ref", help="branch/tag/commit to register (triage mode)")
    prv.add_argument("--json", dest="as_json", action="store_true", help="emit the review result as JSON")
    prv.set_defaults(func=cmd_review, uses_config=True)

    pin = sub.add_parser("init", help="Scaffold codna.yaml (+ AGENTS.md) in the current directory.")
    pin.add_argument("--force", action="store_true", help="overwrite existing files")
    pin.add_argument("--no-agents", dest="no_agents", action="store_true", help="do not write AGENTS.md")
    pin.set_defaults(func=cmd_init)

    pstat = sub.add_parser("status", help="Concise health: engine · telys · keys (vs the verbose `doctor`).")
    pstat.set_defaults(func=cmd_status)

    _register_ci(sub)
    _register_report(sub)

    pl = sub.add_parser("login", help="Authorize this device and install the on-device runtime (one-time, free).")
    pl.add_argument("--token", help="headless access token (or env CODNA_TOKEN) — skips the browser step")
    pl.add_argument("--no-browser", dest="no_browser", action="store_true",
                    help="do not open a browser; print the verification URL + code to authorize elsewhere")
    pl.set_defaults(func=cmd_login)

    from .byok_cli import register as _register_key
    _register_key(sub)
    from .impact_cli import register_cli as _register_impact_cli  # impact + memory commands
    _register_impact_cli(sub)
    return p


def _restore_environ(snapshot: dict[str, str] | None) -> None:
    if snapshot is None:
        return
    os.environ.clear()
    os.environ.update(snapshot)


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 0
    # Apply codna.yaml only at the scope a command consumes. Deterministic commands load privacy
    # posture but must not require model keys; model-backed fix actions load the full model config.
    # Snapshot/restore keeps embedded callers and tests from inheriting command-local config state.
    env_snapshot: dict[str, str] | None = None
    if getattr(args, "uses_config", False) or getattr(args, "config", None):
        from .config_file import ConfigError, load_and_apply
        env_snapshot = dict(os.environ)
        try:
            load_and_apply(
                getattr(args, "config", None),
                include_model=bool(getattr(args, "uses_model_config", False)),
            )
        except ConfigError as exc:
            _restore_environ(env_snapshot)
            print(json.dumps({"error": {"code": "config_error", "message": str(exc)}}, indent=2), file=sys.stderr)
            return 1
    try:
        result = args.func(args)
    except CodnaError as exc:
        print(json.dumps({"error": {"code": "cli_error", "message": str(exc), "details": {}}}, indent=2), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        payload = _structured_error(exc)
        if payload is None:
            raise
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1
    finally:
        _restore_environ(env_snapshot)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
