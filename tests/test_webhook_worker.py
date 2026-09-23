"""Worker: keyless env, per-installation metering, fail-closed when unlinked."""
from __future__ import annotations

import json
import threading
import time

import pytest

import codna.webhook_pool as pool_module
import codna.webhook_worker as worker_module
from codna.webhook import WebhookError, WebhookJob
from codna.webhook_queue import QueuedJob
from codna.webhook_pool import WorkerPool
from codna.webhook_worker import JobResult, process_job, run_codna_job


def _qjob(kind="fix", ref="sha1", installation_id=42, attempts=1):
    return QueuedJob(row_id=1, delivery_id="d1", attempts=attempts,
                     job=WebhookJob(kind, "acme/app", ref=ref, installation_id=installation_id, reason="test"))


# --- env: per-job runtime isolation (the P0 fix -- concurrent jobs never share a sidecar) -----
def test_free_port_pair_returns_two_adjacent_ports():
    pair = worker_module._free_port_pair()
    assert pair is not None
    base, sidecar = pair
    assert sidecar == base + 1


def test_free_port_pair_returns_the_exact_ports_it_actually_probed(monkeypatch):
    """REGRESSION-shaped: the returned tuple must be the SAME ports that were bind-tested, not
    merely two numbers one-apart -- a probe of the wrong port (e.g. base+2 while still returning
    base+1) would pass a check that only looks at the return value's arithmetic, while actually
    verifying nothing about whether the port it hands back is free."""
    bound: list[int] = []
    real_socket = worker_module.socket.socket

    class _RecordingSocket:
        def __init__(self, *a, **k):
            self._s = real_socket(*a, **k)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._s.close()
            return False

        def bind(self, addr):
            bound.append(addr[1])
            self._s.bind(addr)

        def getsockname(self):
            return self._s.getsockname()

    monkeypatch.setattr(worker_module.socket, "socket", lambda *a, **k: _RecordingSocket(*a, **k))
    pair = worker_module._free_port_pair()
    assert pair is not None
    assert bound == [0, pair[0] + 1]  # first bind is ephemeral (port 0); second checks base+1 EXACTLY


def test_free_port_pair_does_not_repeat_across_calls():
    """Not a hash of job identity -- if it were, two different jobs could derive the SAME port by
    coincidence, reintroducing the exact class of collision this exists to eliminate."""
    seen = {worker_module._free_port_pair() for _ in range(5)}
    assert len(seen) == 5


def test_isolated_runtime_env_nests_inside_the_jobs_own_scratch_dir(tmp_path):
    env = worker_module._isolated_runtime_env(str(tmp_path))
    assert env["CODNA_RUNTIME_ROOT"] == str(tmp_path / ".codna-runtime")
    assert "CODNA_PORT_BASE" in env


def test_isolated_runtime_env_degrades_without_a_port_when_probing_fails(monkeypatch, tmp_path):
    """A probe failure must shrink the isolation, never fail the job -- CODNA_RUNTIME_ROOT alone
    still keeps two jobs' runtime STATE apart even if they end up sharing the default ports."""
    monkeypatch.setattr(worker_module, "_free_port_pair", lambda: None)
    env = worker_module._isolated_runtime_env(str(tmp_path))
    assert "CODNA_RUNTIME_ROOT" in env
    assert "CODNA_PORT_BASE" not in env


def test_job_env_without_tmp_is_unaffected_by_isolation(monkeypatch):
    """Back-compat: callers that don't pass tmp= (existing tests, any future direct caller) get
    exactly the old behavior -- isolation is opt-in via the keyword, not a silent default change
    that could surprise a caller who isn't expecting CODNA_RUNTIME_ROOT/CODNA_PORT_BASE to appear.

    Explicitly clears both from the ambient env first: whether they happen to be set in the
    environment this suite runs in is not this test's concern -- an inherited value passing
    through the allowlist (pre-existing behavior) is not the same claim as "this call injects
    nothing new," which is the one actually being tested here.
    """
    monkeypatch.delenv("CODNA_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("CODNA_PORT_BASE", raising=False)
    env = worker_module._job_env("tok", "ekey")
    assert "CODNA_RUNTIME_ROOT" not in env
    assert "CODNA_PORT_BASE" not in env


def test_job_env_with_tmp_overrides_the_hosts_own_shared_runtime_root(monkeypatch, tmp_path):
    """REGRESSION-shaped: the whole point is defeated if a job silently inherits the WEBHOOK
    PROCESS's own CODNA_RUNTIME_ROOT instead of getting its own -- that's the exact shared state
    that let two jobs discover and tear down each other's sidecar."""
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", "/shared/host/runtime")
    env = worker_module._job_env("tok", "ekey", tmp=str(tmp_path))
    assert env["CODNA_RUNTIME_ROOT"] != "/shared/host/runtime"
    assert env["CODNA_RUNTIME_ROOT"].startswith(str(tmp_path))


def test_job_env_with_tmp_points_the_jobs_temp_dir_into_its_own_scratch_dir(monkeypatch, tmp_path):
    """REGRESSION: scratch dirs the job's own tools create with tempfile (the remote-review clone,
    secure-pr checkouts, junit files) followed the HOST's TMPDIR into the machine's shared /tmp and
    outlived the job. With TMPDIR/TMP/TEMP inside the job dir they die with it."""
    monkeypatch.setenv("TMPDIR", "/host/shared/tmp")
    env = worker_module._job_env("tok", "ekey", tmp=str(tmp_path))
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == str(tmp_path)


def test_job_env_without_tmp_leaves_the_hosts_temp_dir_alone(monkeypatch):
    monkeypatch.setenv("TMPDIR", "/host/shared/tmp")
    monkeypatch.delenv("TMP", raising=False)
    env = worker_module._job_env("tok", "ekey")
    assert env["TMPDIR"] == "/host/shared/tmp"
    assert "TMP" not in env


def test_job_env_with_tmp_gives_two_concurrent_jobs_different_runtime_roots_and_ports(tmp_path):
    a = worker_module._job_env("tok-a", "ekey-a", tmp=str(tmp_path / "job-a"))
    b = worker_module._job_env("tok-b", "ekey-b", tmp=str(tmp_path / "job-b"))
    assert a["CODNA_RUNTIME_ROOT"] != b["CODNA_RUNTIME_ROOT"]
    assert a["CODNA_PORT_BASE"] != b["CODNA_PORT_BASE"]


def test_run_codna_job_passes_its_own_scratch_dir_to_job_env(monkeypatch):
    """run_codna_job must actually WIRE tmp through, not just leave the isolation code unreachable
    dead weight -- confirms the argument that reaches subprocess.run's env came from _job_env(tmp=...)."""
    captured = {}

    def _fake_job_env(*args, **kwargs):
        captured["tmp"] = kwargs.get("tmp")
        return {}

    def _fake_run(argv, **kwargs):
        class _P:
            returncode = 0
            stdout = ""
            stderr = ""
        return _P()

    monkeypatch.setattr(worker_module, "_job_env", _fake_job_env)
    monkeypatch.setattr(worker_module, "_run_job_process", _fake_run)
    job = WebhookJob("fix", "acme/app", ref="sha1")
    run_codna_job(job, token="t", engine_key="k")

    assert captured["tmp"] is not None
    import tempfile
    assert captured["tmp"].startswith(tempfile.gettempdir()) or "codna-webhook-job-" in captured["tmp"]


# --- env: keyless (Codna credential injected, no raw provider key, no host secrets) ----------
def test_job_env_blocks_arbitrary_and_app_secrets_and_raw_provider_keys(monkeypatch):
    for k in ("AWS_SECRET_ACCESS_KEY", "GH_PAT", "STRIPE_SECRET_KEY", "SUPABASE_TOKEN", "PGPASSWORD",
              "GITHUB_APP_PRIVATE_KEY", "CODNA_GITHUB_WEBHOOK_SECRET",
              "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):  # raw provider keys are NOT forwarded (keyless)
        monkeypatch.setenv(k, "leak-me")
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-broad-token")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = worker_module._job_env("scoped-token", "org-codna-key")
    for k in ("AWS_SECRET_ACCESS_KEY", "GH_PAT", "STRIPE_SECRET_KEY", "SUPABASE_TOKEN", "PGPASSWORD",
              "GITHUB_APP_PRIVATE_KEY", "CODNA_GITHUB_WEBHOOK_SECRET", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        assert k not in env, f"{k} leaked into the untrusted job env"
    assert env["GITHUB_TOKEN"] == "scoped-token"       # scoped GitHub token (branch/PR)
    assert env["CODNA_API_KEY"] == "org-codna-key"     # per-org Codna credential meters this org
    assert env["PATH"] == "/usr/bin"


def test_job_env_forwards_telys_but_blocks_engine_url_and_provider_keys(monkeypatch):
    monkeypatch.setenv("CODNA_ENGINE_URL", "https://api.codna.ai")
    monkeypatch.setenv("ALGENTA_ENGINE_URL", "https://api.codna.ai")
    monkeypatch.setenv("CODNA_TELYS_MEMORY_ROOT", "/telys/mem")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-forward")
    env = worker_module._job_env("t", "org-key")
    assert env["CODNA_TELYS_MEMORY_ROOT"] == "/telys/mem"
    assert "CODNA_ENGINE_URL" not in env
    assert "ALGENTA_ENGINE_URL" not in env
    assert "ANTHROPIC_API_KEY" not in env  # managed/BYOK model routing stays outside the untrusted job env


# --- env: the org's own BYOK provider key, set consistently regardless of provider -----------
def test_job_env_sets_the_matching_key_for_each_byok_provider(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    env = worker_module._job_env("t", "org-key", "anthropic", "sk-ant-org")
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-org"
    assert "OPENAI_API_KEY" not in env
    assert env["ALGENTA_AGENT_PROVIDER"] == "anthropic"

    env = worker_module._job_env("t", "org-key", "openai", "sk-oa-org")
    assert env["OPENAI_API_KEY"] == "sk-oa-org"
    assert "ANTHROPIC_API_KEY" not in env
    assert env["ALGENTA_AGENT_PROVIDER"] == "openai"

    # Google is dual-set: the packaged sidecar's SDKs vary in which env var they read.
    env = worker_module._job_env("t", "org-key", "google", "gk-org")
    assert env["GEMINI_API_KEY"] == "gk-org"
    assert env["GOOGLE_API_KEY"] == "gk-org"
    assert env["ALGENTA_AGENT_PROVIDER"] == "google"


def test_job_env_sets_no_provider_key_when_org_has_no_byok():
    env = worker_module._job_env("t", "org-key")  # provider/provider_key default to None
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ALGENTA_AGENT_PROVIDER"):
        assert k not in env


class _FakeGitHub:
    def __init__(self, existing_pr=None, issue_comment_result="https://github.com/acme/app/issues/12#issuecomment-1"):
        self.calls = []
        self._existing_pr = existing_pr
        self._issue_comment_result = issue_comment_result

    def installation_token(self, app_id, private_key, installation_id, *, repo_full_name, kind):
        self.calls.append(("token", installation_id, kind))
        return f"scoped-{kind}-token"

    def find_open_pr_by_marker(self, repo_full_name, token, marker, *, open_only=False):
        self.calls.append(("find_pr", repo_full_name, open_only))
        return self._existing_pr

    def create_check_run(self, repo, token, *, name, head_sha, summary):
        self.calls.append(("create_check", head_sha, summary[:24]))
        return 9001

    def update_check_run(self, repo, token, check_run_id, *, conclusion, summary, name="codna"):
        self.calls.append(("update_check", conclusion))

    def post_issue_comment(self, repo, token, issue_number, body):
        self.calls.append(("issue_comment", issue_number, body[:80]))
        return self._issue_comment_result


# --- per-installation metering + fail-closed -----------------------------------------------
def test_process_job_meters_per_installation_and_runs():
    gh = _FakeGitHub()
    runs = []
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: runs.append(kw) or JobResult(True, "done"),
                         resolve_engine_key=lambda iid: f"engine-key-for-{iid}")
    assert result.ok
    # the resolved per-org Codna key is what the fix runs with.
    assert runs[0]["engine_key"] == "engine-key-for-42"
    assert ("token", 42, "fix") in gh.calls
    assert ("update_check", "success") in gh.calls


def test_process_job_threads_the_orgs_byok_provider_into_the_runner():
    gh = _FakeGitHub()
    runs = []
    result = process_job(
        _qjob(kind="fix"), app_id="a", private_key="p", github=gh,
        runner=lambda job, **kw: runs.append(kw) or JobResult(True, "done"),
        resolve_engine_key=lambda iid: f"engine-key-for-{iid}",
        resolve_provider_credentials=lambda iid: ("openai", "sk-org-key"),
    )
    assert result.ok
    assert runs[0]["provider"] == "openai"
    assert runs[0]["provider_key"] == "sk-org-key"


def test_process_job_survives_a_byok_lookup_crash():
    """A BYOK-store hiccup must never fail the whole job -- the fix still runs, just without a
    local provider key (house-metered), same as before this feature existed."""
    gh = _FakeGitHub()
    runs = []

    def _boom(_iid):
        raise RuntimeError("bridge hiccup")

    result = process_job(
        _qjob(kind="fix"), app_id="a", private_key="p", github=gh,
        runner=lambda job, **kw: runs.append(kw) or JobResult(True, "done"),
        resolve_engine_key=lambda iid: f"engine-key-for-{iid}",
        resolve_provider_credentials=_boom,
    )
    assert result.ok
    assert runs[0]["provider"] is None
    assert runs[0]["provider_key"] is None


def test_process_job_fails_closed_when_installation_is_not_linked():
    """No per-org engine credential -> do NOT run a cloud fix on a house key; prompt to link."""
    gh = _FakeGitHub()
    ran = []
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: None)  # unlinked
    assert result.ok
    assert ran == []                                   # NOTHING ran -> zero spend
    assert ("update_check", "neutral") in gh.calls     # posted the "link your account" Check Run
    assert "link" in result.summary.lower() or "not linked" in result.summary.lower()


def test_issue_label_job_unlinked_installation_posts_issue_comment_and_never_spends():
    """An issue-label trigger has no commit SHA, so fail-closed feedback must be an issue comment."""
    gh = _FakeGitHub()
    ran = []
    qjob = QueuedJob(row_id=1, delivery_id="d1", attempts=1,
                     job=WebhookJob("fix", "acme/app", ref=None, installation_id=42, issue_number=12,
                                    reason="labeled_codna_fix"))
    result = process_job(qjob, app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: None)
    assert result.ok
    assert ran == []                                   # zero spend
    assert not any(call[0] == "create_check" for call in gh.calls)
    assert any(call[0] == "issue_comment" and call[1] == 12 for call in gh.calls)
    assert any("isn't linked" in call[2] for call in gh.calls if call[0] == "issue_comment")


def test_issue_label_job_unlinked_prompt_comment_failure_is_not_silent():
    gh = _FakeGitHub(issue_comment_result=None)
    ran = []
    qjob = QueuedJob(row_id=1, delivery_id="d1", attempts=1,
                     job=WebhookJob("fix", "acme/app", ref=None, installation_id=42, issue_number=12,
                                    reason="labeled_codna_fix"))
    result = process_job(qjob, app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: None)
    assert result.ok is False
    assert ran == []                                   # still zero spend
    assert any(call[0] == "issue_comment" and call[1] == 12 for call in gh.calls)
    assert "failed to post" in result.summary


# --- the account bridge gave NO answer (control plane mid-redeploy) --------------------------
class _RecordingGitHub(_FakeGitHub):
    def update_check_run(self, repo, token, check_run_id, *, conclusion, summary, name="codna"):
        self.calls.append(("update_check", conclusion))
        self.summaries = getattr(self, "summaries", []) + [summary]


def _bridge_down(_iid):
    from codna.webhook_metering import BridgeUnavailable
    raise BridgeUnavailable("HTTP 503")


def test_process_job_retries_instead_of_saying_not_linked_when_the_bridge_is_unavailable(monkeypatch):
    """REGRESSION (2026-09-18 01:44Z): the control plane was redeploying, the credential lookup got
    no answer, and a linked org's review was replaced by a neutral "link your account" check."""
    from codna import webhook_metering

    gh = _RecordingGitHub()
    ran = []
    result = process_job(_qjob(kind="review", attempts=1), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=_bridge_down)
    assert not result.ok and result.retryable
    assert result.retry_after_s == webhook_metering.BRIDGE_RETRY_WAIT_S  # the QUEUE holds it, no sleep
    assert "bridge unavailable" in result.summary
    assert ran == []                                                   # nothing ran, nothing spent
    assert not any(c[0] in ("create_check", "update_check") for c in gh.calls)  # no false prompt


def test_process_job_last_attempt_with_the_bridge_down_posts_an_honest_neutral_check():
    gh = _RecordingGitHub()
    ran = []
    result = process_job(_qjob(kind="review", attempts=3), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=_bridge_down)
    assert not result.ok and not result.retryable                      # terminal: no 4th attempt
    assert ran == []
    assert ("update_check", "neutral") in gh.calls
    posted = gh.summaries[-1]
    assert "could not reach its account service" in posted and "@codna review" in posted
    assert "Link this GitHub install" not in posted                    # never the false link prompt


# --- fix_enabled: convenience kill switch, fails OPEN (contrast with resolve_engine_key) ----
def test_process_job_skips_when_fix_disabled_for_org():
    """An org that turned automatic fixes off gets the same non-silent treatment as "not
    linked" -- a neutral Check Run, zero spend, and the runner is never called."""
    gh = _FakeGitHub()
    ran = []
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: "org-key",
                         resolve_fix_enabled=lambda iid: False)
    assert result.ok
    assert ran == []                                     # NOTHING ran -> zero spend
    assert ("update_check", "neutral") in gh.calls
    assert "disabled" in result.summary.lower()


def test_issue_label_job_skips_when_fix_disabled_for_org():
    """An issue-label trigger has no commit SHA, so the disabled notice must be an issue comment."""
    gh = _FakeGitHub()
    ran = []
    qjob = QueuedJob(row_id=1, delivery_id="d1", attempts=1,
                     job=WebhookJob("fix", "acme/app", ref=None, installation_id=42, issue_number=12,
                                    reason="labeled_codna_fix"))
    result = process_job(qjob, app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: "org-key",
                         resolve_fix_enabled=lambda iid: False)
    assert result.ok
    assert ran == []
    assert not any(call[0] == "create_check" for call in gh.calls)
    assert any(call[0] == "issue_comment" and call[1] == 12 for call in gh.calls)
    assert any("turned off" in call[2] for call in gh.calls if call[0] == "issue_comment")


def test_process_job_runs_when_fix_enabled_is_true():
    """fix_enabled=True (the explicit default) lets the job proceed to the runner as before."""
    gh = _FakeGitHub()
    ran = []
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: ran.append(1) or JobResult(True, "done"),
                         resolve_engine_key=lambda iid: "org-key",
                         resolve_fix_enabled=lambda iid: True)
    assert result.ok
    assert ran == [1]


def test_process_job_defaults_to_fix_enabled_when_bridge_unconfigured(monkeypatch):
    """No resolve_fix_enabled override -> falls through to the real webhook_metering function,
    which FAILS OPEN when the live bridge isn't configured (no engine URL / secret) -- a
    convenience kill switch must never silently disable every org's fixes on its own."""
    monkeypatch.delenv("CODNA_ENGINE_URL", raising=False)
    monkeypatch.delenv("CODNA_WEBHOOK_INTERNAL_SECRET", raising=False)
    gh = _FakeGitHub()
    ran = []
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: ran.append(1) or JobResult(True, "done"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok
    assert ran == [1]


def test_process_job_reuses_existing_pr_and_does_not_re_run():
    gh = _FakeGitHub(existing_pr="https://github.com/acme/app/pull/7")
    ran = []
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and "pull/7" in result.summary and ran == []


def test_process_job_reports_failure_conclusion_when_run_fails():
    gh = _FakeGitHub()
    result = process_job(_qjob(kind="secure"), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: JobResult(False, "boom"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok is False
    assert ("update_check", "failure") in gh.calls


def _issue_label_qjob():
    # labeled_codna_fix: no ref (no commit to check-run against), no context (not a comment
    # reply) -- the one trigger shape with neither existing reporting channel.
    return QueuedJob(row_id=1, delivery_id="d1", attempts=1,
                     job=WebhookJob("fix", "acme/app", ref=None, installation_id=42, issue_number=12,
                                    reason="labeled_codna_fix"))


def test_process_job_issue_label_fix_failure_posts_issue_comment():
    gh = _FakeGitHub()
    result = process_job(_issue_label_qjob(), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: JobResult(False, "could not fetch issue #12 text"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok is False
    assert not any(c[0] == "create_check" for c in gh.calls)  # no ref -> no check-run possible
    assert any(c[0] == "issue_comment" and c[1] == 12 and "could not fetch issue #12" in c[2]
              for c in gh.calls)


def test_process_job_issue_label_fix_success_comments_success_never_a_failure():
    """This used to assert success posted NO comment, on the theory that "the opened PR speaks for
    itself". It does not: the PR body carries only the codna-webhook-id marker, which is not a
    GitHub reference, so the PR and the issue were never cross-linked. Live, that meant a correct
    fix PR existed while the issue's newest comment was an older failure. Success now comments too
    -- what must never happen is a FAILURE comment on a successful run."""
    gh = _FakeGitHub()
    result = process_job(_issue_label_qjob(), app_id="a", private_key="p",
                         github=gh, runner=lambda job, **kw: JobResult(True, "pull/9"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok is True
    comments = [c for c in gh.calls if c[0] == "issue_comment"]
    assert len(comments) == 1
    assert "✅" in comments[0][2]
    assert "⚠️" not in comments[0][2] and "couldn't finish" not in comments[0][2]


def test_process_job_still_completes_check_run_when_runner_raises(monkeypatch):
    """Regression: a runner crash used to leave the Check Run permanently "queued" on the PR/
    commit (create_check_run ran, but the matching update_check_run was never reached) — the
    exact bug behind codna-app-smoke's check-suite, stuck since 2026-07-08. The worker's outer
    per-job try/except (_loop) marks the internal queue row "failed", but that's invisible on
    GitHub — only completing the Check Run itself closes the loop for anyone looking at the PR."""
    gh = _FakeGitHub()

    def _boom(job, **kw):
        raise RuntimeError("subprocess bookkeeping blew up, has a token in it: ghs_should_not_leak")

    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p",
                         github=gh, runner=_boom,
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok is False
    assert ("update_check", "failure") in gh.calls  # the check-run was completed, not left hanging
    assert "RuntimeError" in result.summary
    # the raw exception text (and anything it might contain) is never surfaced
    assert "ghs_should_not_leak" not in result.summary
    assert "subprocess bookkeeping" not in result.summary


# --- runner: packaged codna subprocess -----------------------------------------------------
def test_run_codna_job_invokes_the_binary_and_maps_exit_code(monkeypatch):
    monkeypatch.setattr(worker_module, "_JOB_TIMEOUT_S", 30)
    assert run_codna_job(_qjob(kind="fix").job, token="t", engine_key="k", codna_bin="true").ok is True
    assert run_codna_job(_qjob(kind="fix").job, token="t", engine_key="k", codna_bin="false").ok is False


def test_run_codna_secure_without_token_fails_closed():
    res = run_codna_job(_qjob(kind="secure").job, token=None, engine_key="k", codna_bin="true")
    assert res.ok is False and "token" in res.summary.lower()


# --- runner: a "review" job's summary must be human text, never the raw --json stdout ------
# codna_command's --json flag exists ONLY for the "review" kind (so review_github.py's own
# posting logic can parse the subprocess's result); before this fix, run_codna_job reused that
# same raw JSON verbatim as the wrapper Check Run's summary. Live evidence: thyn-ai/algenta run
# 105075740163, whose Check Run summary was a multi-KB wall of unformatted JSON braces.
def test_review_job_summary_reformats_valid_json_into_a_short_sentence():
    stdout = json.dumps({
        "findings": [{"path": "a.py", "line": 1}, {"path": "b.py", "line": 2}],
        "inline_count": 2, "summary_count": 0, "conclusion": "neutral",
        "posted": {"posted_review": True, "inline_posted": 2, "skipped_duplicates": 0},
    })
    summary = worker_module._review_job_summary(stdout)
    assert summary is not None
    assert "{" not in summary and "}" not in summary  # never raw JSON
    assert "2 finding(s)" in summary and "2 inline" in summary and "neutral" in summary
    assert "codna review" in summary  # points at the check that has the real detail


def test_review_job_summary_no_findings():
    stdout = json.dumps({"findings": [], "inline_count": 0, "summary_count": 0, "conclusion": "success"})
    summary = worker_module._review_job_summary(stdout)
    assert summary == "codna review: no high-confidence issues found ✅"


def test_review_job_summary_falls_back_to_none_on_anything_else():
    # Not JSON at all, JSON that isn't a dict, and a dict missing the review-result shape --
    # every one must return None so the caller keeps its existing raw-tail fallback rather
    # than fabricating a summary from something that isn't actually a review result.
    assert worker_module._review_job_summary("not json at all") is None
    assert worker_module._review_job_summary("[1, 2, 3]") is None
    assert worker_module._review_job_summary(json.dumps({"ok": True})) is None
    assert worker_module._review_job_summary("") is None


def test_run_codna_job_review_kind_uses_the_clean_summary_not_raw_stdout(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_module, "_JOB_TIMEOUT_S", 30)
    fake_result = {
        "findings": [{"path": "x.py", "line": 9}], "inline_count": 1, "summary_count": 0,
        "conclusion": "neutral", "posted": {"posted_review": True, "inline_posted": 1},
    }
    fake_bin = tmp_path / "fake-codna"
    fake_bin.write_text("#!/bin/sh\ncat <<'EOF'\n" + json.dumps(fake_result) + "\nEOF\n")
    fake_bin.chmod(0o755)
    job = WebhookJob("review", "acme/app", ref="sha1", installation_id=42, pr_number=7, reason="test")
    res = run_codna_job(job, token="t", engine_key="k", codna_bin=str(fake_bin))
    assert res.ok is True
    assert "{" not in res.summary
    assert "1 finding(s)" in res.summary


# --- runner: a failed fix/secure job's summary must be the CLI's error message, not its JSON ---
# cli.py prints `{"error": {"code": ..., "message": ...}}` (indent=2) to stderr on every CodnaError;
# the raw-tail fallback pasted that into the "codna fix" Check Run summary verbatim. Two live
# examples on thyn-ai/algenta (2026-09-17): "pytest is not installed" and a clone timeout, both
# rendered as a wall of braces.
def test_error_json_summary_renders_the_documented_cli_error_as_a_sentence():
    stderr = 'codna: fixing https://github.com/acme/app.git …\n' + json.dumps(
        {"error": {"code": "cli_error", "message": "could not prepare remote checkout for --tests: timed out", "details": {}}},
        indent=2)
    s = worker_module._error_json_summary("fix", "", stderr)
    assert s == "codna fix failed (cli_error): could not prepare remote checkout for --tests: timed out"


def test_error_json_summary_picks_the_last_error_object_and_ignores_noise():
    text = 'garbage {"not": "json"\n' + json.dumps({"error": {"code": "a", "message": "first"}}) + "\nmore\n" \
           + json.dumps({"error": {"code": "b", "message": "second"}}) + "\ntrailing"
    assert worker_module._error_json_summary("secure", text) == "codna secure failed (b): second"


def test_error_json_summary_is_none_without_a_structured_error():
    assert worker_module._error_json_summary("fix", "plain failure text", "") is None
    assert worker_module._error_json_summary("fix", None, None) is None
    assert worker_module._error_json_summary("fix", json.dumps({"error": "just a string"})) is None


def test_run_codna_job_fix_kind_failure_uses_the_error_message_not_raw_stderr(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_module, "_JOB_TIMEOUT_S", 30)
    err = json.dumps({"error": {"code": "cli_error", "message": "pytest is not installed", "details": {}}}, indent=2)
    fake_bin = tmp_path / "fake-codna"
    fake_bin.write_text("#!/bin/sh\necho 'codna: fixing …'\ncat >&2 <<'EOF'\n" + err + "\nEOF\nexit 1\n")
    fake_bin.chmod(0o755)
    job = WebhookJob("fix", "acme/app", ref="sha1", installation_id=42, reason="check_suite_failure")
    res = run_codna_job(job, token="t", engine_key="k", codna_bin=str(fake_bin))
    assert res.ok is False
    assert res.summary == "codna fix failed (cli_error): pytest is not installed"


# --- check runs: a retry must supersede a dead in_progress run, not stack a second one ----------
class _StaleAwareGitHub(_FakeGitHub):
    def complete_stale_check_runs(self, repo, token, *, head_sha, name, summary):
        self.calls.append(("complete_stale", head_sha, name))
        return 1


def test_process_job_completes_stale_in_progress_runs_before_opening_a_new_one():
    gh = _StaleAwareGitHub()
    process_job(_qjob(kind="fix", attempts=2), app_id="a", private_key="p", github=gh,
                runner=lambda job, **kw: JobResult(True, "ok"),
                resolve_engine_key=lambda iid: "org-key")
    names = [c[0] for c in gh.calls]
    assert "complete_stale" in names and "create_check" in names
    assert names.index("complete_stale") < names.index("create_check")
    stale = next(c for c in gh.calls if c[0] == "complete_stale")
    assert stale[1] == "sha1" and stale[2] == "codna fix"


def test_process_job_tolerates_a_github_client_without_stale_cleanup():
    # older fakes / clients without the helper keep working: the cleanup is opportunistic
    gh = _FakeGitHub()
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(True, "ok"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and ("update_check", "success") in gh.calls


# --- a `@codna fix` reply must never go silent, even when process_job itself raises ------------
# process_job replies in-thread on every outcome it can reach; a WebhookError raised before its
# first reply (token minting, usually) unwinds to _run_claimed and the retries burn out in
# seconds with only a stderr line. Live on 2026-09-17: `@codna fix` replies with no ack, no check,
# no PR, no error. The worker now speaks on the LAST attempt.
class _RecordingQueue:
    def __init__(self):
        self.completed = []

    def complete(self, row_id, *, status, result=None, retry=True, retry_after_s=None):
        self.completed.append((row_id, status, result))


def _terminal_comment_fix_qjob(attempts):
    return _comment_fix_qjob(attempts=attempts)


def test_terminal_comment_fix_failure_is_reported_in_thread(monkeypatch):
    from codna.webhook_queue import _MAX_ATTEMPTS
    posted = []
    monkeypatch.setattr(worker_module.webhook_github, "installation_token",
                        lambda *a, **k: "minted-tok")
    monkeypatch.setattr(worker_module.webhook_github, "post_review_comment_reply",
                        lambda repo, token, pr, reply_to, body: posted.append((repo, token, pr, reply_to, body)) or "url")

    def boom(_qjob):
        raise WebhookError("installation_token_failed", "installation token: 422 …")

    pool = WorkerPool(_RecordingQueue(), process=boom, app_id="APP", private_key="pem")
    pool._run_claimed(_terminal_comment_fix_qjob(attempts=_MAX_ATTEMPTS))
    assert len(posted) == 1
    repo, token, pr, reply_to, body = posted[0]
    assert (repo, token, pr, reply_to) == ("acme/app", "minted-tok", 7, 555)
    assert "installation_token_failed" in body and "logged the details" in body


def test_non_terminal_comment_fix_failure_stays_quiet_so_the_retry_can_speak(monkeypatch):
    posted = []
    monkeypatch.setattr(worker_module.webhook_github, "installation_token", lambda *a, **k: "tok")
    monkeypatch.setattr(worker_module.webhook_github, "post_review_comment_reply",
                        lambda *a, **k: posted.append(a) or "url")

    def boom(_qjob):
        raise WebhookError("installation_token_failed", "…")

    pool = WorkerPool(_RecordingQueue(), process=boom, app_id="APP", private_key="pem")
    pool._run_claimed(_terminal_comment_fix_qjob(attempts=1))
    assert posted == []


def test_terminal_failure_reply_falls_back_to_github_token_when_minting_is_the_failure(monkeypatch):
    from codna.webhook_queue import _MAX_ATTEMPTS
    posted = []

    def mint_fails(*a, **k):
        raise WebhookError("installation_token_failed", "…")

    monkeypatch.setattr(worker_module.webhook_github, "installation_token", mint_fails)
    monkeypatch.setattr(worker_module.webhook_github, "post_review_comment_reply",
                        lambda repo, token, pr, reply_to, body: posted.append(token) or "url")
    monkeypatch.setenv("GITHUB_TOKEN", "env-tok")

    def boom(_qjob):
        raise RuntimeError("worker exploded")

    pool = WorkerPool(_RecordingQueue(), process=boom, app_id="APP", private_key="pem")
    pool._run_claimed(_terminal_comment_fix_qjob(attempts=_MAX_ATTEMPTS))
    assert posted == ["env-tok"]


def test_terminal_failure_reply_is_skipped_for_jobs_without_a_thread(monkeypatch):
    from codna.webhook_queue import _MAX_ATTEMPTS
    posted = []
    monkeypatch.setattr(worker_module.webhook_github, "installation_token", lambda *a, **k: "tok")
    monkeypatch.setattr(worker_module.webhook_github, "post_review_comment_reply",
                        lambda *a, **k: posted.append(a) or "url")

    def boom(_qjob):
        raise WebhookError("x", "…")

    pool = WorkerPool(_RecordingQueue(), process=boom, app_id="APP", private_key="pem")
    pool._run_claimed(QueuedJob(row_id=1, delivery_id="d", attempts=_MAX_ATTEMPTS,
                                job=WebhookJob("fix", "acme/app", ref="sha1", installation_id=42, reason="check_suite_failure")))
    assert posted == []  # a CI-failure fix has no thread to speak in; the check run + log carry it


# --- phase logging: a job that stalls must be locatable from the machine's stderr -------------
def test_run_claimed_logs_claimed_and_terminal_phases(capsys):
    pool = WorkerPool(_RecordingQueue(), process=lambda _q: JobResult(True, "ok"))
    pool._run_claimed(_qjob(kind="review"))
    lines = [json.loads(raw) for raw in capsys.readouterr().err.splitlines() if raw.startswith("{")]
    phases = [entry["phase"] for entry in lines if entry.get("event") == "job_phase"]
    assert phases == ["claimed", "done"]
    assert lines[0]["kind"] == "review" and lines[0]["delivery_id"] == "d1"


def test_run_claimed_logs_failed_phase_and_the_failure_line(capsys):
    pool = WorkerPool(_RecordingQueue(), process=lambda _q: JobResult(False, "risk gate rejected"))
    pool._run_claimed(_qjob(kind="fix"))
    lines = [json.loads(raw) for raw in capsys.readouterr().err.splitlines() if raw.startswith("{")]
    assert [entry["phase"] for entry in lines if entry.get("event") == "job_phase"] == ["claimed", "failed"]
    assert any(entry.get("event") == "job_failed" and entry["code"] == "job_failed" for entry in lines)


def _issue_label_job():
    return WebhookJob("fix", "acme/app", ref=None, installation_id=42, issue_number=12,
                      reason="labeled_codna_fix")


def test_run_codna_job_issue_label_fix_fetches_issue_text_before_running(monkeypatch):
    monkeypatch.setattr(worker_module, "_JOB_TIMEOUT_S", 30)
    monkeypatch.setattr(worker_module.webhook_github, "fetch_issue_text",
                        lambda repo, token, n: "Login crashes on empty password")
    captured = {}

    def _fake_run(argv, **kw):
        captured["argv"] = list(argv)
        return worker_module.subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(worker_module, "_run_job_process", _fake_run)
    result = run_codna_job(_issue_label_job(), token="t", engine_key="k")
    assert result.ok is True
    assert "--issue" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--issue") + 1] == "Login crashes on empty password"


def test_run_codna_job_issue_label_fix_without_token_fails_closed_before_running(monkeypatch):
    def _must_not_run(*a, **k):
        raise AssertionError("must not shell out without a token")

    monkeypatch.setattr(worker_module, "_run_job_process", _must_not_run)
    result = run_codna_job(_issue_label_job(), token=None, engine_key="k")
    assert result.ok is False
    assert "token" in result.summary.lower()


def test_run_codna_job_issue_label_fix_fetch_failure_fails_closed_before_running(monkeypatch):
    monkeypatch.setattr(worker_module.webhook_github, "fetch_issue_text", lambda repo, token, n: None)

    def _must_not_run(*a, **k):
        raise AssertionError("must not shell out with nothing to describe the fix")

    monkeypatch.setattr(worker_module, "_run_job_process", _must_not_run)
    result = run_codna_job(_issue_label_job(), token="t", engine_key="k")
    assert result.ok is False
    assert "12" in result.summary  # names the issue it couldn't fetch


class _OneJobQueue:
    def __init__(self, qjob: QueuedJob):
        self.qjob = qjob
        self.completed = []

    def recover_stale(self):
        return {"requeued": 0, "failed": 0}

    def claim(self):
        qjob = self.qjob
        self.qjob = None
        return qjob

    def complete(self, row_id, *, status, result=None, retry=True, retry_after_s=None):
        self.completed.append((row_id, status, result))


def test_worker_logs_structured_failure_without_secrets(capsys):
    qjob = _qjob(kind="fix", ref="sha1", installation_id=42)
    queue = _OneJobQueue(qjob)
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01,
                      process=lambda _qjob: (_ for _ in ()).throw(
                          WebhookError("installation_token_failed", "installation token: 401 bad credentials")
                      ))
    pool.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not queue.completed:
        time.sleep(0.01)
    pool.stop()
    assert queue.completed and queue.completed[0][1] == "failed"
    entries = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.strip()]
    logged = next(entry for entry in entries if entry.get("event") == "job_failed")
    assert logged["service"] == "codna-webhook-worker"
    assert logged["code"] == "installation_token_failed"
    assert logged["repo"] == "acme/app"
    assert logged["installation_id"] == 42
    assert "bad credentials" in logged["message"]
    assert "token=" not in logged["message"]


# --- comment-triggered verified fix (@codna fix wedge) -------------------------------------
from codna.review_findings import CodnaReviewFinding  # noqa: E402
from codna.review_github import comment_body  # noqa: E402


def _codna_finding_comment_body():
    f = CodnaReviewFinding(path="src/a.py", line=10, severity="high", category="security",
                           title="SQL injection", explanation="user input reaches the query unescaped",
                           confidence=0.95, fingerprint="abcdef012345")
    return comment_body(f)


class _CommentFixGitHub(_FakeGitHub):
    def __init__(self, *, parent_body, authored=True, existing_pr=None):
        super().__init__(existing_pr=existing_pr)
        self._parent = {"body": parent_body, "performed_via_github_app": {"id": "APP"} if authored else None,
                        "user": {"type": "Bot" if authored else "User"}}
        self.replies = []

    def get_pull_review_comment(self, repo, token, comment_id):
        self.calls.append(("get_comment", comment_id))
        return self._parent

    def comment_authored_by_app(self, comment, app_id):
        from codna.webhook_github import comment_authored_by_app
        return comment_authored_by_app(comment, app_id)

    def post_review_comment_reply(self, repo, token, pr_number, in_reply_to_id, body):
        self.replies.append((pr_number, in_reply_to_id, body))
        return "https://github.com/acme/app/pull/7#discussion_r1"


def _comment_fix_qjob(attempts=1, **ctx_over):
    ctx = {"in_reply_to_id": 555, "path": "src/a.py", "line": 10, "head_sha": "sha1",
           "head_ref": "feature", "base_ref": "main", "is_fork": False}
    ctx.update(ctx_over)
    return QueuedJob(row_id=1, delivery_id="d1", attempts=attempts,
                     job=WebhookJob("fix", "acme/app", ref="sha1", installation_id=42, pr_number=7,
                                    context=ctx, reason="review_comment_codna_fix"))


def test_comment_fix_acks_enriches_and_replies_with_pr_on_success():
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body(),
                           existing_pr=None)
    seen = {}
    result = process_job(_comment_fix_qjob(), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: seen.update(ctx=dict(job.context)) or JobResult(True, "opened"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok
    # enrichment: the runner saw the finding fields folded into context (drives --issue)
    assert seen["ctx"]["severity"] == "high" and seen["ctx"]["title"] == "SQL injection"
    assert "unescaped" in seen["ctx"]["explanation"]
    # in-thread: an ack, then a success reply. Success is checked AFTER the run.
    bodies = [b for (_pr, _rt, b) in gh.replies]
    assert any("On it" in b for b in bodies)                      # ack
    # find_open_pr_by_marker returns None here → generic success reply (still non-error)
    assert gh.replies[-1][1] == 555                                # replied to the finding thread


def test_comment_fix_replies_fail_closed_when_verification_fails():
    from codna.webhook_queue import _MAX_ATTEMPTS
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    # final attempt (out of retries) → the single fail-closed reply is posted
    result = process_job(_comment_fix_qjob(attempts=_MAX_ATTEMPTS), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(False, "risk gate rejected"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok is False
    last = gh.replies[-1][2]
    assert "couldn't open a verified fix" in last.lower() and "fail-closed" in last.lower()


def test_comment_fix_skips_forked_pr_without_running():
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    ran = []
    result = process_job(_comment_fix_qjob(is_fork=True), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "x"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and ran == []                                 # never spent
    assert any("forked" in b.lower() for (_p, _r, b) in gh.replies)


def test_comment_fix_skips_when_parent_is_not_codna_authored():
    # Body has a valid Codna marker, but the App did NOT author it (a human pasted the marker).
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body(), authored=False)
    ran = []
    result = process_job(_comment_fix_qjob(), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "x"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and ran == []
    assert any("only works as a reply to a codna review finding" in b for (_p, _r, b) in gh.replies)


def test_comment_fix_unlinked_org_replies_in_thread_and_never_spends():
    # Regression: an unlinked org must NOT leave the @codna fix thread silent (it used to return
    # before any reply). It replies in-thread with the link prompt and runs nothing.
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    ran = []
    result = process_job(_comment_fix_qjob(), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "x"),
                         resolve_engine_key=lambda iid: None)  # unlinked
    assert result.ok and ran == []                              # zero spend
    assert gh.replies and gh.replies[-1][1] == 555              # replied in the finding thread
    assert "isn't linked" in gh.replies[-1][2]


def test_comment_fix_skips_and_replies_when_fix_disabled_for_org():
    # A comment trigger must never go silent, same as the "not linked" case: reply in-thread
    # with the disabled notice and run nothing.
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    ran = []
    result = process_job(_comment_fix_qjob(), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "x"),
                         resolve_engine_key=lambda iid: "org-key",
                         resolve_fix_enabled=lambda iid: False)
    assert result.ok and ran == []                              # zero spend
    assert gh.replies and gh.replies[-1][1] == 555               # replied in the finding thread
    assert "turned off" in gh.replies[-1][2]


def test_comment_fix_reuse_matches_open_only_and_skips_the_ack():
    # An already-OPEN fix PR → reply once with the reuse link, NO "On it" ack, and don't re-run.
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body(),
                           existing_pr="https://github.com/acme/app/pull/10")
    ran = []
    result = process_job(_comment_fix_qjob(), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "x"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and ran == []
    bodies = [b for (_p, _r, b) in gh.replies]
    assert any("Already opened" in b for b in bodies)
    assert not any("On it" in b for b in bodies)               # no spurious "analyzing…" ack on reuse
    # the comment-fix reuse check must query OPEN PRs only (a closed PR shouldn't block a re-request)
    assert ("find_pr", "acme/app", True) in gh.calls


# ---- retry hygiene: no duplicate acks / fail-closed spam across retries ----------------------

def test_comment_fix_retry_does_not_reack():
    # A retried job (attempts>1) must NOT post another "On it…" ack on the finding thread.
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    process_job(_comment_fix_qjob(attempts=2), app_id="APP", private_key="p", github=gh,
                runner=lambda job, **kw: JobResult(True, "opened"),
                resolve_engine_key=lambda iid: "org-key")
    bodies = [b for (_p, _r, b) in gh.replies]
    assert not any("On it" in b for b in bodies)  # no duplicate ack on retry


def test_failing_fix_replies_only_on_the_final_attempt():
    # Early failing attempts stay quiet (may still succeed on retry); the fail-closed reply lands once,
    # on the last attempt — not once per attempt.
    early = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    process_job(_comment_fix_qjob(attempts=1), app_id="APP", private_key="p", github=early,
                runner=lambda job, **kw: JobResult(False, "engine 401"),
                resolve_engine_key=lambda iid: "org-key")
    assert not any("couldn't open a verified fix" in b for (_p, _r, b) in early.replies)

    final = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    from codna.webhook_queue import _MAX_ATTEMPTS
    process_job(_comment_fix_qjob(attempts=_MAX_ATTEMPTS), app_id="APP", private_key="p", github=final,
                runner=lambda job, **kw: JobResult(False, "engine 401"),
                resolve_engine_key=lambda iid: "org-key")
    assert sum("couldn't open a verified fix" in b for (_p, _r, b) in final.replies) == 1


# --- issue-label fix: the issue is the ONLY surface, so BOTH outcomes must land there ---------
def _label_qjob(issue_number=8, installation_id=42):
    """A label-triggered fix: no ref (so no check run) and no comment context (so no thread)."""
    return QueuedJob(
        row_id=1, delivery_id="d-label", attempts=1,
        job=WebhookJob("fix", "acme/app", ref=None, issue_number=issue_number,
                       installation_id=installation_id, reason="labeled_codna_fix"),
    )


def test_label_fix_success_comments_the_pr_link_on_the_issue():
    """Success used to be SILENT here ("a success speaks for itself via the opened PR"), but the PR
    body carries only the codna-webhook-id marker -- not a GitHub reference -- so nothing ever
    cross-linked them. Observed live: a correct fix PR was opened while the issue's newest comment
    was still an older failure, making the issue look broken."""
    gh = _FakeGitHub()
    # find_open_pr_by_marker is consulted twice: the idempotency pre-check (must be None so the job
    # actually runs) and again after success to link the PR.
    seq = [None, "https://github.com/acme/app/pull/9"]
    gh.find_open_pr_by_marker = lambda repo, token, marker, *, open_only=False: (
        gh.calls.append(("find_pr", repo, open_only)) or seq.pop(0)
    )
    result = process_job(_label_qjob(), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(True, "opened"),
                         resolve_engine_key=lambda iid: "engine-key")
    assert result.ok
    comments = [c for c in gh.calls if c[0] == "issue_comment"]
    assert len(comments) == 1, f"expected exactly one issue comment, got {comments}"
    assert comments[0][1] == 8
    assert "✅" in comments[0][2]
    assert "https://github.com/acme/app/pull/9" in comments[0][2]


def test_label_fix_success_still_comments_when_the_pr_lookup_fails():
    """A lookup hiccup must not make a successful fix go silent, and must not fail the job."""
    gh = _FakeGitHub()
    seq = [None]

    def _find(repo, token, marker, *, open_only=False):
        gh.calls.append(("find_pr", repo, open_only))
        if seq:
            return seq.pop(0)
        raise RuntimeError("github hiccup")

    gh.find_open_pr_by_marker = _find
    result = process_job(_label_qjob(), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(True, "opened"),
                         resolve_engine_key=lambda iid: "engine-key")
    assert result.ok  # the fix succeeded; a comment-link failure never changes that
    comments = [c for c in gh.calls if c[0] == "issue_comment"]
    assert len(comments) == 1 and "✅" in comments[0][2]
    # Pins the FALLBACK arm. Without asserting "None" is absent, deleting the pr_url guard would
    # still pass while production posted "✅ Codna opened a fix for this issue: None".
    assert "opened a pull request" in comments[0][2]
    assert "None" not in comments[0][2]


def test_label_fix_failure_still_comments_the_reason_on_the_issue():
    gh = _FakeGitHub()
    result = process_job(_label_qjob(), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(False, "boom: something broke"),
                         resolve_engine_key=lambda iid: "engine-key")
    assert not result.ok
    comments = [c for c in gh.calls if c[0] == "issue_comment"]
    assert len(comments) == 1
    assert "⚠️" in comments[0][2] and "boom" in comments[0][2]
    assert "✅" not in comments[0][2]


def test_secure_label_success_never_claims_a_pull_request_was_opened():
    """`codna-secure` on an issue also yields an issue-triggered job with no ref and no thread, but
    it is a READ-ONLY classification pass that never opens a PR. The first cut of the success comment
    was kind-agnostic and told those users "opened a pull request for it", sending them hunting for a
    PR that never existed."""
    gh = _FakeGitHub()
    qjob = QueuedJob(
        row_id=1, delivery_id="d-sec", attempts=1,
        job=WebhookJob("secure", "acme/app", ref=None, issue_number=8,
                       installation_id=42, reason="labeled_codna_secure"),
    )
    result = process_job(qjob, app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(True, "3 findings classified"),
                         resolve_engine_key=lambda iid: "engine-key")
    assert result.ok
    comments = [c for c in gh.calls if c[0] == "issue_comment"]
    assert len(comments) == 1
    body = comments[0][2]
    assert "pull request" not in body, f"secure pass must not claim a PR: {body!r}"
    assert "opened a fix" not in body
    assert "secure" in body and "✅" in body


# --- admission gating: threads are the claimer ceiling, the admitter is the governor ------------
class _ManyJobQueue:
    """Hands out `n` jobs then None. Tracks how many are claimed but not yet completed."""

    def __init__(self, n: int, kind: str = "fix"):
        self._left = n
        self._kind = kind
        self.completed = []
        self._lock = threading.Lock()
        self.claimed = 0

    def recover_stale(self):
        return {"requeued": 0, "failed": 0}

    def claim(self):
        with self._lock:
            if self._left <= 0:
                return None
            self._left -= 1
            self.claimed += 1
            row = self.claimed
        return QueuedJob(row_id=row, delivery_id=f"d{row}", attempts=1,
                         job=WebhookJob(self._kind, "acme/app", ref=f"sha{row}", installation_id=42, reason="test"))

    def complete(self, row_id, *, status, result=None, retry=True, retry_after_s=None):
        with self._lock:
            self.completed.append((row_id, status))


def test_worker_pool_never_exceeds_the_admitter_ceiling(monkeypatch):
    """The pool may run MORE threads than the admitter's ceiling — threads are only the claimer
    ceiling. What must hold is that concurrent FIXES never exceed what the admitter allows, since
    beyond it the sidecar sheds load with 429 agent_core_saturated.

    Before the gate existed, in-flight fixes == CODNA_WEBHOOK_CONCURRENCY exactly, with nothing
    between claim and subprocess spawn.
    """
    import codna.admission_control as ac

    ceiling = 2
    admitter = ac.MCAdmissionDecider(
        ac.AdmissionConfig(ceiling=ceiling, model_rate_budget=ceiling, sla_s=900.0),
        mc_fn=lambda *a: 0.0,
    )
    monkeypatch.setattr(pool_module, "get_admitter", lambda: admitter)

    peak, lock = [0], threading.Lock()
    live = [0]

    def _slow_fix(_qjob):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return JobResult(True, "ok")

    queue = _ManyJobQueue(12)
    pool = WorkerPool(queue, concurrency=8, poll_interval=0.01, process=_slow_fix)
    pool.start()
    deadline = time.monotonic() + 15
    while len(queue.completed) < 12 and time.monotonic() < deadline:
        time.sleep(0.02)
    pool.stop(timeout=5)

    assert len(queue.completed) == 12, f"only {len(queue.completed)} of 12 completed"
    assert peak[0] <= ceiling, f"ran {peak[0]} concurrent fixes with an admitter ceiling of {ceiling}"
    assert admitter.in_flight == 0, "a reserved slot leaked"


def test_worker_releases_the_slot_when_the_queue_is_empty(monkeypatch):
    """A reserved slot must be freed when the claim finds nothing, or idle polling would permanently
    consume capacity and the pool would wedge after `ceiling` empty polls."""
    import codna.admission_control as ac

    admitter = ac.MCAdmissionDecider(ac.AdmissionConfig(ceiling=2, model_rate_budget=2), mc_fn=lambda *a: 0.0)
    monkeypatch.setattr(pool_module, "get_admitter", lambda: admitter)

    queue = _ManyJobQueue(0)          # always empty
    pool = WorkerPool(queue, concurrency=2, poll_interval=0.01, process=lambda _q: JobResult(True, "ok"))
    pool.start()
    time.sleep(0.2)                   # several empty polls per thread
    pool.stop(timeout=5)

    assert admitter.in_flight == 0, "empty polls leaked reserved slots"
    assert admitter.snapshot()["samples"] == 0, "an idle poll polluted the duration window"


@pytest.mark.parametrize("kind, learned", [("fix", 1), ("review", 0), ("secure", 0)])
def test_only_a_fix_teaches_the_fix_admitter_its_duration(monkeypatch, kind, learned):
    """The admitter forecasts FIX admission against the durations it has seen, and the loop used to
    hand it every claimed job's wall-clock -- a ten-minute review (review_budget's adaptive turn) would
    stretch the fix distribution and throttle fixes that were never at risk. Reviews have their own
    series (review_budget.observe_review_turn, recorded inside the job); the admitter learns only
    from fixes. The slot is still released for every kind."""
    import codna.admission_control as ac

    admitter = ac.MCAdmissionDecider(ac.AdmissionConfig(ceiling=2, model_rate_budget=2), mc_fn=lambda *a: 0.0)
    monkeypatch.setattr(pool_module, "get_admitter", lambda: admitter)

    def _job(_qjob):
        time.sleep(0.02)
        return JobResult(True, "ok")

    queue = _ManyJobQueue(1, kind=kind)
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01, process=_job)
    pool.start()
    deadline = time.monotonic() + 5
    while len(queue.completed) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    pool.stop(timeout=5)

    assert queue.completed == [(1, "done")]
    assert admitter.in_flight == 0, "the slot must be released for every job kind"
    assert admitter.snapshot()["samples"] == learned


def test_diagnostics_reports_admission_state():
    queue = _ManyJobQueue(0)
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01, process=lambda _q: JobResult(True, "ok"))
    diag = pool.diagnostics()
    assert diag["configured_threads"] == 1
    assert "admission" in diag
    assert set(diag["admission"]) >= {"in_flight", "samples", "ceiling", "det_cap"}


# --- bounded job subprocess + wedge visibility -----------------------------------------------
def test_job_process_timeout_kills_the_whole_process_group():
    # A backgrounded grandchild keeps the pipes open; killing only the direct child (what
    # subprocess.run does) would leave communicate() blocked until it exits ~30 s later.
    import os
    import subprocess

    argv = ["sh", "-c", "sleep 30 & exec sleep 30"]
    started = time.monotonic()
    raised = False
    try:
        worker_module._run_job_process(argv, env=os.environ.copy(), cwd=None, timeout=0.5)
    except subprocess.TimeoutExpired:
        raised = True
    assert raised
    assert time.monotonic() - started < 10


def test_job_process_returns_output_and_exit_code_when_it_finishes_in_time():
    import os

    proc = worker_module._run_job_process(
        ["sh", "-c", "echo out; echo err >&2; exit 3"], env=os.environ.copy(), cwd=None, timeout=10
    )
    assert (proc.returncode, proc.stdout.strip(), proc.stderr.strip()) == (3, "out", "err")


def test_diagnostics_flag_a_thread_wedged_past_the_job_bound(monkeypatch):
    import threading

    pool = WorkerPool(_RecordingQueue(), concurrency=1)
    pool._threads.append(threading.current_thread())  # alive, by construction
    monkeypatch.setattr(worker_module, "_JOB_TIMEOUT_S", 10)
    with pool._busy_lock:
        pool._busy_since["worker-0"] = time.monotonic() - 1000
    diag = pool.diagnostics()
    assert diag["busy_threads"] == 1 and diag["longest_running_s"] >= 999
    assert diag["ready"] is False
    with pool._busy_lock:
        pool._busy_since.clear()
    assert pool.diagnostics()["ready"] is True and pool.diagnostics()["busy_threads"] == 0


def test_run_claimed_clears_busy_marker_even_when_the_job_raises():
    pool = WorkerPool(_RecordingQueue(), process=lambda _q: (_ for _ in ()).throw(RuntimeError("boom")))
    pool._run_claimed(_qjob())
    with pool._busy_lock:
        assert pool._busy_since == {}



# --- post-incident hardening (2026-09-17 16:17Z outage) ---------------------------------------
def test_first_attempt_never_cancels_check_runs_it_did_not_create():
    gh = _StaleAwareGitHub()
    process_job(_qjob(kind="fix", attempts=1), app_id="a", private_key="p", github=gh,
                runner=lambda job, **kw: JobResult(True, "ok"),
                resolve_engine_key=lambda iid: "org-key")
    names = [c[0] for c in gh.calls]
    assert "complete_stale" not in names and "create_check" in names


def test_stale_cleanup_raising_never_blocks_the_jobs_own_check_run():
    class _Exploding(_StaleAwareGitHub):
        def complete_stale_check_runs(self, *a, **k):
            raise RuntimeError("github 502")

    gh = _Exploding()
    result = process_job(_qjob(kind="fix", attempts=3), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(True, "ok"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and "create_check" in [c[0] for c in gh.calls]


def test_teardown_kills_detached_processes_recorded_in_runtime_state(tmp_path):
    import json as _json
    import os
    import subprocess

    root = tmp_path / ".codna-runtime"
    root.mkdir()
    env = dict(os.environ, CODNA_RUNTIME_ROOT=str(root))
    child = subprocess.Popen(["sleep", "60"], env=env, start_new_session=True)
    try:
        (root / "local-stack.json").write_text(_json.dumps({"engine": {"pid": child.pid}, "listener_pid": 1}))
        assert worker_module._teardown_job_runtime(str(tmp_path)) >= 1
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.05)
        assert child.poll() is not None, "detached child survived teardown"
    finally:
        if child.poll() is None:
            child.kill()
    assert worker_module._teardown_job_runtime(str(tmp_path / "nowhere")) == 0


def test_teardown_via_proc_environ_finds_children_without_state_files(tmp_path):
    import os
    import subprocess
    import sys as _sys

    if not _sys.platform.startswith("linux"):
        import pytest
        pytest.skip("/proc scan is Linux-only; state-file path is covered above")
    root = tmp_path / ".codna-runtime"
    root.mkdir()
    child = subprocess.Popen(["sleep", "60"], env=dict(os.environ, CODNA_RUNTIME_ROOT=str(root)), start_new_session=True)
    try:
        time.sleep(0.2)
        assert child.pid in worker_module._pids_with_runtime_root(root)
        worker_module._teardown_job_runtime(str(tmp_path))
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.05)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()


def test_worker_loop_survives_a_claim_exception(capsys):
    class _FlakyQueue(_RecordingQueue):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def recover_stale(self, *a, **k):
            return 0

        def claim(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("database is locked")
            return None

    queue = _FlakyQueue()
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01)
    pool.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and queue.calls < 3:
        time.sleep(0.01)
    alive = pool.diagnostics()["alive_threads"]
    pool.stop()
    assert queue.calls >= 3 and alive == 1
    err = capsys.readouterr().err
    assert '"event": "worker_loop_error"' in err and "database is locked" in err


def test_terminal_reply_no_longer_claims_nothing_was_opened():
    import inspect

    src = inspect.getsource(worker_module.WorkerPool._notify_terminal_comment_fix_failure)
    assert "nothing was pushed or opened" not in src



# --- deterministic CLI failures are not retried ------------------------------------------------
def test_summary_code_decides_retryability():
    assert worker_module._summary_is_retryable("codna fix failed (cli_error): no test runner") is False
    assert worker_module._summary_is_retryable("codna review failed (review_error): sidecar") is True
    assert worker_module._summary_is_retryable("some raw tail") is True


def test_run_codna_job_marks_a_cli_error_as_non_retryable(tmp_path):
    import json as _json

    err = _json.dumps({"error": {"code": "cli_error", "message": "`codna fix --tests` could not detect a test runner"}})
    fake_bin = tmp_path / "codna"
    fake_bin.write_text("#!/bin/sh\ncat >&2 <<'EOF'\n" + err + "\nEOF\nexit 1\n")
    fake_bin.chmod(0o755)
    res = run_codna_job(_qjob(kind="fix").job, token="t", engine_key="k", codna_bin=str(fake_bin))
    assert res.ok is False and res.retryable is False and "cli_error" in res.summary


def test_run_claimed_passes_retryability_to_the_queue():
    seen = {}

    class _Q(_RecordingQueue):
        def complete(self, row_id, *, status, result=None, retry=True, retry_after_s=None):
            seen["retry"] = retry
            super().complete(row_id, status=status, result=result, retry=retry)

    pool = WorkerPool(_Q(), process=lambda _q: JobResult(False, "codna fix failed (cli_error): bad input", retryable=False))
    pool._run_claimed(_qjob())
    assert seen["retry"] is False



# --- a terminal (non-retryable) comment-fix failure replies on its only attempt --------------
def test_comment_fix_replies_immediately_when_the_failure_is_not_retryable():
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    result = process_job(_comment_fix_qjob(attempts=1), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(False, "codna fix failed (cli_error): git clone timed out after 300 seconds", retryable=False),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok is False
    bodies = [b for (_pr, _rt, b) in gh.replies]
    assert any("On it" in b for b in bodies)
    assert "couldn't complete this fix" in bodies[-1] and "git clone timed out" in bodies[-1]
    assert "test + risk gate" not in bodies[-1]


def test_comment_fix_retryable_failure_before_the_last_attempt_stays_quiet():
    gh = _CommentFixGitHub(parent_body=_codna_finding_comment_body())
    process_job(_comment_fix_qjob(attempts=1), app_id="APP", private_key="p", github=gh,
                runner=lambda job, **kw: JobResult(False, "risk gate rejected"),
                resolve_engine_key=lambda iid: "org-key")
    bodies = [b for (_pr, _rt, b) in gh.replies]
    assert bodies and all("On it" in b for b in bodies)  # ack only; the retry may still succeed



def test_terse_cli_failure_drops_paths_and_keeps_gits_reason():
    live = ('codna fix failed (cli_error): apply failed: Git command failed while applying the packaged local patch. '
            '| code=git_command_failed | details={"args": ["apply", "--check", "--whitespace=nowarn", "-"], '
            '"cwd": "/tmp/codna-webhook-job-xsyylfg6/.codna-runtime/repository-intelligence/packaged/checkouts/4e80", '
            '"stderr": "error: corrupt patch at line 99\\n"}  (patch ref: packaged-local-patch://patch_87368')
    terse = worker_module._terse_cli_failure(live)
    assert terse == ("apply failed: Git command failed while applying the packaged local patch "
                     "(git: error: corrupt patch at line 99) [git_command_failed]")
    assert "/tmp/" not in terse and "details=" not in terse
    # a summary cut mid-JSON (the live 17:37Z push failure) still yields the rejection line, no payload
    truncated = ('codna fix failed (cli_error): apply failed: Git command failed while preparing packaged repository '
                 'access. | code=git_command_failed | details={"args": ["push", "origin", "HEAD:refs/heads/codna/c90d"], '
                 '"cwd": "/tmp/codna-webhook-job-i1ga45ok/.codna-runtime/x", "stderr": "To https://github.com/thyn-ai/algenta.git'
                 '\\n ! [remote rejected] HEAD -> codna/c90d (push declined due to repository rule violations)')
    t2 = worker_module._terse_cli_failure(truncated)
    assert t2 == ("apply failed: Git command failed while preparing packaged repository access "
                  "(git: ! [remote rejected] HEAD -> codna/c90d (push declined due to repository rule violations)) [git_command_failed]")
    assert "/tmp/" not in t2
    assert worker_module._terse_cli_failure("codna fix failed (cli_error): no test runner") == "no test runner [cli_error]"
    assert worker_module._terse_cli_failure("plain text") == "plain text"



def test_terse_cli_failure_prefers_the_rejection_line_over_gits_generic_error_line():
    live = ('codna fix failed (cli_error): apply failed: Git command failed while preparing packaged repository access. '
            '| code=git_command_failed | details={"args": ["push", "origin", "HEAD:refs/heads/codna/6e4b"], "cwd": "/tmp/x", '
            '"stderr": "To https://github.com/thyn-ai/algenta.git\\n ! [remote rejected] HEAD -> codna/6e4b (refusing to allow a '
            'GitHub App to create or update workflow `.github/workflows/security.yml` without `workflows` permission)\\nerror: '
            'failed to push some refs to \'https://github.com/thyn-ai/algenta.git\'\\n"}  (patch ref: packaged-local-patch://p1)')
    terse = worker_module._terse_cli_failure(live)
    assert "refusing to allow a GitHub App" in terse and "failed to push some refs" not in terse
    assert terse.endswith("[git_command_failed]") and "/tmp/" not in terse



# --- @codna fix authorization is the repository's collaborator permission, not the payload ------
class _PermissionGitHub(_CommentFixGitHub):
    def __init__(self, permission, **kw):
        super().__init__(**kw)
        self._permission = permission
        self.permission_lookups = []

    def collaborator_permission(self, repo, token, username):
        self.permission_lookups.append(username)
        return self._permission


def _contributor_qjob(**over):
    return _comment_fix_qjob(commenter="0xamlab", author_association="CONTRIBUTOR", **over)


def test_contributor_with_write_permission_gets_the_fix():
    gh = _PermissionGitHub("admin", parent_body=_codna_finding_comment_body())
    ran = []
    result = process_job(_contributor_qjob(), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "opened"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and ran and gh.permission_lookups == ["0xamlab"]
    assert any("On it" in b for (_p, _r, b) in gh.replies)


def test_read_only_commenter_is_refused_before_any_spend():
    gh = _PermissionGitHub("read", parent_body=_codna_finding_comment_body())
    ran = []
    result = process_job(_contributor_qjob(), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "opened"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and result.retryable is False and "needs write" in result.summary
    assert not ran
    bodies = [b for (_p, _r, b) in gh.replies]
    assert len(bodies) == 1 and "write access" in bodies[0] and "On it" not in bodies[0]
    assert not any(c[0] == "create_check" for c in gh.calls)


def test_permission_lookup_failure_falls_back_to_the_payload_association():
    refused = process_job(_contributor_qjob(), app_id="APP", private_key="p",
                          github=_PermissionGitHub(None, parent_body=_codna_finding_comment_body()),
                          runner=lambda job, **kw: JobResult(True, "opened"), resolve_engine_key=lambda iid: "org-key")
    assert "permission lookup was unavailable" in refused.summary
    allowed = process_job(_comment_fix_qjob(commenter="0xamlab", author_association="MEMBER"), app_id="APP",
                          private_key="p", github=_PermissionGitHub(None, parent_body=_codna_finding_comment_body()),
                          runner=lambda job, **kw: JobResult(True, "opened"), resolve_engine_key=lambda iid: "org-key")
    assert allowed.ok and allowed.summary == "opened"



# --- pre-flight: missing `workflows` permission + stale workflow files -> remedy, no spend ------
class _WorkflowsGitHub(_PermissionGitHub):
    def __init__(self, *, granted, differs, **kw):
        super().__init__("admin", **kw)
        self._granted, self._differs, self.differ_lookups = granted, differs, []

    def installation_has_workflows(self, installation_id):
        return self._granted

    def workflows_differ_from_default(self, repo, token, head_sha):
        self.differ_lookups.append(head_sha)
        return self._differs


def test_fix_on_stale_branch_without_workflows_permission_replies_with_the_remedy_and_spends_nothing():
    gh = _WorkflowsGitHub(granted=False, differs=True, parent_body=_codna_finding_comment_body())
    ran = []
    result = process_job(_contributor_qjob(), app_id="APP", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "opened"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and result.retryable is False and "workflows permission missing" in result.summary
    assert not ran and gh.differ_lookups == ["sha1"]
    bodies = [b for (_p, _r, b) in gh.replies]
    assert len(bodies) == 1 and "Workflows: Read and write" in bodies[0] and "nothing was spent" in bodies[0]
    assert not any(c[0] == "create_check" for c in gh.calls)


def test_fix_proceeds_when_workflows_is_granted_or_the_branch_matches_default():
    for granted, differs in ((True, True), (False, False), (None, True)):
        gh = _WorkflowsGitHub(granted=granted, differs=differs, parent_body=_codna_finding_comment_body())
        result = process_job(_contributor_qjob(), app_id="APP", private_key="p", github=gh,
                             runner=lambda job, **kw: JobResult(True, "opened"),
                             resolve_engine_key=lambda iid: "org-key")
        assert result.ok and result.summary == "opened", (granted, differs)



# --- fix commits are authored as the App's bot account -----------------------------------------
def test_job_env_carries_the_git_identity_when_given():
    env = worker_module._job_env("tok", "ekey", git_identity=("codna-ai[bot]", "1+codna-ai[bot]@users.noreply.github.com"))
    assert env["CODNA_GIT_USER_NAME"] == "codna-ai[bot]" and env["CODNA_GIT_USER_EMAIL"].endswith("@users.noreply.github.com")
    assert "CODNA_GIT_USER_NAME" not in worker_module._job_env("tok", "ekey")


def test_process_job_hands_the_app_bot_identity_to_the_runner():
    class _IdentityGitHub(_FakeGitHub):
        def app_bot_identity(self, app_id, private_key):
            return ("codna-ai[bot]", "1+codna-ai[bot]@users.noreply.github.com")

    seen = {}
    process_job(_qjob(kind="fix"), app_id="a", private_key="p", github=_IdentityGitHub(),
                runner=lambda job, **kw: seen.update(kw) or JobResult(True, "ok"),
                resolve_engine_key=lambda iid: "org-key")
    assert seen["git_identity"] == ("codna-ai[bot]", "1+codna-ai[bot]@users.noreply.github.com")


# ---- one Title-Case Check Run per job (was: lowercase wrapper + a second capitalised CLI check) -----

def test_check_run_name_is_title_case_and_matches_the_ruleset():
    from codna.webhook import check_run_name

    assert check_run_name("review") == "codna review"   # the exact name the org ruleset requires
    assert check_run_name("fix") == "codna fix"
    assert check_run_name("secure") == "codna secure"
    assert check_run_name("odd_kind") == "codna odd kind"


def test_job_env_hands_the_jobs_check_run_id_to_the_cli():
    env = worker_module._job_env("tok", "ekey", check_run_id=9001)
    assert env["CODNA_CHECK_RUN_ID"] == "9001"
    assert "CODNA_CHECK_RUN_ID" not in worker_module._job_env("tok", "ekey")


def test_cli_completed_check_detects_only_this_jobs_run():
    ok = json.dumps({"conclusion": "neutral", "findings": [], "posted": {"check": {"id": 9001, "updated_existing": True}}})
    assert worker_module._cli_completed_check(ok, 9001) is True
    assert worker_module._cli_completed_check(ok, 9002) is False          # a different run
    created = json.dumps({"conclusion": "neutral", "findings": [], "posted": {"check": {"id": 9001, "updated_existing": False}}})
    assert worker_module._cli_completed_check(created, 9001) is False     # the CLI opened its own
    assert worker_module._cli_completed_check("not json", 9001) is False
    assert worker_module._cli_completed_check(ok, None) is False


def test_process_job_opens_one_title_case_check_and_leaves_it_to_the_cli_when_it_completed_it():
    """REGRESSION (2026-09-18, owner: "one is capital and the other all lowercase, not professional"):
    a review used to show TWO checks -- the worker's lowercase `codna review` wrapper and the CLI's
    `codna review`. Now the worker opens `codna review`, hands its id to the CLI, and does not
    overwrite the findings the CLI completed it with."""
    class _Gh(_FakeGitHub):
        def create_check_run(self, repo, token, *, name, head_sha, summary):
            self.calls.append(("create_check", name, summary))
            return 9001

    gh = _Gh()
    seen = {}

    def runner(job, **kw):
        seen.update(kw)
        return JobResult(True, "codna review: 2 finding(s)", check_completed=True)

    result = process_job(_qjob(kind="review"), app_id="a", private_key="p", github=gh, runner=runner,
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok
    assert seen["check_run_id"] == 9001                                   # the CLI got the run to complete
    assert ("create_check", "codna review", "codna review started (test)") in gh.calls
    assert not any(c[0] == "update_check" for c in gh.calls)              # findings left untouched


def test_process_job_still_completes_the_check_itself_when_the_cli_did_not():
    class _Gh(_FakeGitHub):
        def create_check_run(self, repo, token, *, name, head_sha, summary):
            self.calls.append(("create_check", name, summary))
            return 9001

    gh = _Gh()
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(True, "opened PR #5"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok
    assert ("create_check", "codna fix", "codna fix started (test)") in gh.calls
    assert ("update_check", "success") in gh.calls


def test_terse_cli_failure_handles_the_title_case_summary():
    """REGRESSION (Codna review of #539): the terse-clause regex matched only `codna fix failed (`;
    with Title-Case summaries it fell through and pasted the raw paths/argv into the PR thread."""
    raw = ('codna fix failed (cli_error): apply failed: Git command failed | code=git_command_failed | '
           'details={"args": ["git", "apply"], "cwd": "/tmp/x", "stderr": "error: corrupt patch at line 99"}')
    terse = worker_module._terse_cli_failure(raw)
    assert "/tmp/x" not in terse and '"args"' not in terse
    assert "corrupt patch" in terse and "[git_command_failed]" in terse


def test_run_codna_job_keeps_check_completed_when_the_review_summary_is_unparsable(monkeypatch):
    """The CLI completed the job's run (posted.check.updated_existing) but its JSON lacks the keys the
    summary formatter wants: the flag must still reach process_job, or it overwrites the findings."""
    stdout = json.dumps({"posted": {"check": {"id": 9001, "updated_existing": True}}})  # no conclusion/findings

    class _P:
        returncode = 0
        stderr = ""

    _P.stdout = stdout
    monkeypatch.setattr(worker_module, "_run_job_process", lambda *a, **k: _P())
    monkeypatch.setattr(worker_module, "_teardown_job_runtime", lambda tmp: 0)
    job = WebhookJob("review", "acme/app", ref="sha1", installation_id=42, pr_number=7, reason="test")
    res = worker_module.run_codna_job(job, token="t", engine_key="k", check_run_id=9001)
    assert res.ok and res.check_completed is True



# ---- merge queue: the group commit inherits the PR head's review -------------------------------
class _QueueGitHub(_FakeGitHub):
    def __init__(self, head="a" * 40, prior=None):
        super().__init__()
        self._head, self._prior, self.summaries = head, prior, []

    def pull_request_head_sha(self, repo, token, number):
        self.calls.append(("pr_head", number))
        return self._head

    def latest_completed_check_run(self, repo, token, *, head_sha, name):
        self.calls.append(("latest_run", head_sha, name))
        return self._prior

    def create_check_run(self, repo, token, *, name, head_sha, summary):
        self.calls.append(("create_check", name, head_sha))
        return 4242

    def update_check_run(self, repo, token, check_run_id, *, conclusion, summary, name="codna"):
        self.calls.append(("update_check", conclusion))
        self.summaries.append(summary)


def _queue_job(attempts=1):
    return QueuedJob(row_id=9, delivery_id="d9", attempts=attempts,
                     job=WebhookJob("queue", "acme/app", ref="f" * 40, installation_id=42, pr_number=48,
                                    reason="merge_group_checks_requested"))


def test_queue_job_copies_the_pr_heads_review_verdict_onto_the_group_commit():
    """algenta-sdk gates main with a merge queue: the required `codna review` must appear on the
    queue's group commit or the queue times out and kicks the PR."""
    gh = _QueueGitHub(prior={"conclusion": "neutral", "summary": "**codna review** — 2 finding(s)\nmore", "html_url": "u"})
    ran = []
    result = process_job(_queue_job(), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: (_ for _ in ()).throw(AssertionError("no metering for queue jobs")))
    assert result.ok and ran == []                                        # no CLI, no metering, no spend
    assert ("token", 42, "queue") in gh.calls                             # least-privilege scope for the kind
    assert ("create_check", "codna review", "f" * 40) in gh.calls         # on the GROUP commit
    assert ("update_check", "neutral") in gh.calls                        # the PR head's verdict, verbatim
    assert "inherited from PR #48" in gh.summaries[-1] and "2 finding(s)" in gh.summaries[-1]


def test_queue_job_fails_closed_when_the_pr_head_has_no_completed_review():
    gh = _QueueGitHub(prior=None)
    result = process_job(_queue_job(), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(True, "ran"), resolve_engine_key=lambda iid: "unused")
    assert not result.ok and not result.retryable
    assert ("update_check", "failure") in gh.calls                        # never a pass-through neutral
    assert "@codna review" in gh.summaries[-1]


def test_queue_token_scope_is_read_pr_write_checks_only():
    from codna.webhook_github import token_permissions_for

    assert token_permissions_for("queue") == {"pull_requests": "read", "checks": "write"}


# --- CI-failure triage before a check_suite fix spends anything ------------------------------
def _ci_qjob(attempts=1):
    return QueuedJob(row_id=1, delivery_id="d1", attempts=attempts,
                     job=WebhookJob("fix", "acme/app", ref="sha1", pr_number=7, installation_id=42,
                                    context={"head_ref": "feature", "check_suite_id": 555},
                                    reason="check_suite_failure"))


def test_check_suite_fix_ends_neutral_without_running_when_triage_is_terminal(monkeypatch):
    gh = _RecordingGitHub()
    ran = []
    monkeypatch.setattr(worker_module.webhook_ci_triage, "triage",
                        lambda job, token, github, **kw: worker_module.webhook_ci_triage.CITriage(
                            "infrastructure", summary="**codna fix** — not a code defect"))
    result = process_job(_ci_qjob(), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: ran.append(1) or JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and ran == []                                  # nothing ran -> zero spend
    assert ("update_check", "neutral") in gh.calls
    assert gh.summaries[-1].startswith("**codna fix** — not a code defect")


def test_check_suite_fix_passes_ci_evidence_to_the_runner_and_appends_the_note(monkeypatch):
    gh = _RecordingGitHub()
    seen = []
    monkeypatch.setattr(worker_module.webhook_ci_triage, "triage",
                        lambda job, token, github, **kw: worker_module.webhook_ci_triage.CITriage(
                            "code", issue_text="CI failed: FAILED tests/test_x.py", note="ℹ️ grant Actions: read"))
    result = process_job(_ci_qjob(), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: seen.append(job) or JobResult(True, "opened PR"),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok
    assert seen[0].context["ci_failure"] == "CI failed: FAILED tests/test_x.py"
    assert ("update_check", "success") in gh.calls
    assert gh.summaries[-1].endswith("ℹ️ grant Actions: read")


def test_a_fix_that_is_not_a_check_suite_failure_is_never_triaged(monkeypatch):
    called = []
    monkeypatch.setattr(worker_module.webhook_ci_triage, "triage", lambda *a, **k: called.append(1))
    gh = _FakeGitHub()
    process_job(_qjob(kind="fix"), app_id="a", private_key="p", github=gh,
                runner=lambda job, **kw: JobResult(True, "done"), resolve_engine_key=lambda iid: "k")
    assert called == []


# --- a structured CLI error keeps its diagnosis in the Check Run --------------------------------
def test_error_summary_shows_the_review_agents_actual_output_when_it_was_not_json():
    """'did not return parseable findings JSON' three times in 35 s on a sandbox PR, and nothing to
    read: the CLI attached raw_head/raw_tail, the worker rendered only the sentence."""
    from codna.webhook_summaries import _error_json_summary, _terse_cli_failure
    err = {"error": {"code": "review_error",
                     "message": "review agent failed: review agent did not return parseable findings JSON",
                     "details": {"raw_chars": 812,
                                 "raw_head": "I reviewed the workflow change. This PR intentionally breaks CI, so ```no findings```",
                                 "raw_tail": "",
                                 "looked_for": ["the whole message as JSON", "a ```json fence"]}}}
    summary = _error_json_summary("review", None, json.dumps(err, indent=2))
    assert summary.startswith("codna review failed (review_error): review agent failed")
    assert "raw_head: I reviewed the workflow change." in summary
    assert "raw_chars: 812" in summary
    assert "looked_for" not in summary and "raw_tail" not in summary     # only informative keys
    assert summary.count("```") == 2                                    # the agent's own fence is defanged
    # the in-thread reply stays one clause: the block is for the Check Run
    assert "raw_head" not in _terse_cli_failure(summary)
    assert _terse_cli_failure(summary).endswith("[review_error]")


def test_error_summary_without_details_is_unchanged():
    from codna.webhook_summaries import _error_json_summary
    err = {"error": {"code": "cli_error", "message": "no test runner detected", "details": {}}}
    assert _error_json_summary("fix", json.dumps(err)) == "codna fix failed (cli_error): no test runner detected"


def test_error_summary_keeps_a_multiline_agent_output_under_its_own_key():
    """raw_head is prose with paragraphs: every continuation line is indented, so nothing in the
    block reads as a new `key:` entry (review finding on #549)."""
    from codna.webhook_summaries import _error_json_summary
    err = {"error": {"code": "review_error", "message": "review agent failed: not JSON",
                     "details": {"raw_head": "Line one.\nraw_tail: not a key\r\nLine three.", "raw_chars": 40}}}
    summary = _error_json_summary("review", json.dumps(err))
    block = summary.split("```")[1]
    lines = block.strip("\n").split("\n")
    assert lines[0] == "raw_chars: 40"                     # _DETAIL_KEYS order: counts before text
    assert lines[1] == "raw_head: Line one."
    assert lines[2] == "    raw_tail: not a key" and lines[3] == "    Line three."
    assert [ln for ln in lines if not ln.startswith(" ")] == ["raw_chars: 40", "raw_head: Line one."]


def test_detail_value_cut_on_a_crlf_boundary_leaves_no_bare_carriage_return():
    from codna.webhook_summaries import _DETAIL_VALUE_CHARS, _details_block
    value = "x" * (_DETAIL_VALUE_CHARS - 1) + "\r\nsecond line"
    block = _details_block({"raw_head": value})
    assert "\r" not in block and block.endswith("…\n```")


def test_review_job_summary_mentions_the_approval():
    from codna.webhook_summaries import _review_job_summary
    clean = json.dumps({"conclusion": "success", "findings": [], "posted": {"event": "APPROVE"}})
    assert _review_job_summary(clean).endswith("✅ · approved")
    with_findings = json.dumps({"conclusion": "neutral", "findings": [{"x": 1}], "summary_count": 0,
                                "posted": {"event": "COMMENT", "inline_posted": 1}})
    assert "approved" not in _review_job_summary(with_findings)
    noted = json.dumps({"conclusion": "success", "findings": [], "note": "no changed files vs main",
                        "posted": {"event": "APPROVE"}})
    assert _review_job_summary(noted) == "codna review: no changed files vs main · approved"


# --- the sandbox cannot run the repo's tests: the check ends neutral, nothing to retry ---------
def _fake_codna_bin(tmp_path, code, message):
    err = json.dumps({"error": {"code": code, "message": message, "details": {}}})
    fake_bin = tmp_path / "codna"
    fake_bin.write_text("#!/bin/sh\ncat >&2 <<'EOF'\n" + err + "\nEOF\nexit 1\n")
    fake_bin.chmod(0o755)
    return str(fake_bin)


def test_run_codna_job_ends_neutral_when_the_sandbox_cannot_run_the_repos_tests(tmp_path):
    """thyn-ai/mojo-kernels#1 (2026-09-19): pytest could not import a pixi-managed suite and the
    `codna fix` check went red over a pull request with nothing wrong in it."""
    msg = ("Codna could not run this repository's tests in its sandbox: pytest could not import the "
           "test modules. Tell Codna how this repository's tests run with `fix.test_command` in its codna.yaml.")
    res = run_codna_job(_qjob(kind="fix").job, token="t", engine_key="k",
                        codna_bin=_fake_codna_bin(tmp_path, "test_environment_unavailable", msg))
    assert res.ok is True and res.neutral is True and res.conclusion == "neutral"
    assert res.summary.startswith("**codna fix** — skipped: Codna could not run")
    assert "fix.test_command" in res.summary and "failed (" not in res.summary
    assert res.summary.endswith("No fix was attempted.")


def test_run_codna_job_keeps_every_other_cli_error_red(tmp_path):
    res = run_codna_job(_qjob(kind="fix").job, token="t", engine_key="k",
                        codna_bin=_fake_codna_bin(tmp_path, "cli_error", "apply failed"))
    assert res.ok is False and res.neutral is False and res.conclusion == "failure"


def test_process_job_completes_the_check_neutral_for_a_neutral_result():
    gh = _RecordingGitHub()
    result = process_job(_qjob(kind="fix"), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(True, "**codna fix** — skipped: no pixi", neutral=True),
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok
    assert ("update_check", "neutral") in gh.calls and ("update_check", "success") not in gh.calls
    assert gh.summaries[-1].startswith("**codna fix** — skipped")


def test_job_result_conclusion_follows_ok_and_neutral():
    assert JobResult(True, "x").conclusion == "success"
    assert JobResult(False, "x").conclusion == "failure"
    assert JobResult(True, "x", neutral=True).conclusion == "neutral"


def test_neutral_summary_helpers_recognise_only_the_environment_code():
    from codna.webhook_summaries import _neutral_check_summary, _summary_is_neutral

    assert _summary_is_neutral("codna fix failed (test_environment_unavailable): no pixi") is True
    assert _summary_is_neutral("codna fix failed (cli_error): bad input") is False
    assert _summary_is_neutral("raw tail") is False
    rendered = _neutral_check_summary(
        "codna fix failed (test_environment_unavailable): no pixi here.\n\n```\ntail: x\n```")
    assert rendered == "**codna fix** — skipped: no pixi here.\n\nNo fix was attempted."


# ---- #569: a review's Check Run is anchored to the head the review READS, not the event SHA -------
# thyn-ai/algenta#1085: the PR opened at 7679ecbe, was rebased to 5a3ba5ff while the queue was
# saturated, and the review turn (which fetches pull/N/head) reviewed 5a3ba5ff -- its body marker said
# so -- yet created the required `codna review` check on 7679ecbe. statusCheckRollup saw the check as
# ABSENT on the head, and an APPROVED pull request stayed BLOCKED.
_EVENT_HEAD = "7679ecbe" + "0" * 32
_MOVED_HEAD = "5a3ba5ff" + "0" * 32


class _MovingHeadGitHub(_FakeGitHub):
    """`pull_request_head_sha` answers from ``heads`` in order (the last answer repeats), so a test
    states what the pull request's head was at turn start and at turn end. An Exception in the list
    is raised in its turn (an API hiccup)."""

    def __init__(self, *heads):
        super().__init__()
        self._heads = list(heads)
        self.created: list[tuple[str, str]] = []   # (head_sha, summary) of every Check Run created
        self.stale: list[dict] = []                 # every complete_stale_check_runs call
        self.updates: list[tuple[int, str, str]] = []

    def pull_request_head_sha(self, repo, token, number):
        self.calls.append(("pr_head", number))
        head = self._heads.pop(0) if len(self._heads) > 1 else self._heads[0]
        if isinstance(head, Exception):
            raise head
        return head

    def complete_stale_check_runs(self, repo, token, *, head_sha, name, summary, conclusion="cancelled"):
        self.stale.append({"head_sha": head_sha, "name": name, "summary": summary, "conclusion": conclusion})
        return 1

    def create_check_run(self, repo, token, *, name, head_sha, summary):
        self.created.append((head_sha, summary))
        self.calls.append(("create_check", head_sha, summary[:24]))
        return 9000 + len(self.created)

    def update_check_run(self, repo, token, check_run_id, *, conclusion, summary, name="codna"):
        self.updates.append((check_run_id, conclusion, summary))
        self.calls.append(("update_check", conclusion))


def _review_qjob(ref=_EVENT_HEAD, attempts=1, pr=1085, row_id=3):
    return QueuedJob(row_id=row_id, delivery_id=f"d{row_id}", attempts=attempts,
                     job=WebhookJob("review", "thyn-ai/algenta", ref=ref, installation_id=42, pr_number=pr,
                                    reason="pull_request_opened"))


def _phase_lines(capsys, phase):
    return [json.loads(line) for line in capsys.readouterr().err.splitlines()
            if line.startswith("{") and json.loads(line).get("phase") == phase]


def _cli_completed(job, **kw):
    return JobResult(True, "codna review: no high-confidence issues found ✅ · approved", check_completed=True)


def test_review_check_run_is_anchored_to_the_head_the_review_reads_not_the_event_sha(capsys):
    """The head moved BEFORE the job started and no event for the new head arrived (delivered late, or
    while the pool was saturated): the check goes on the head the review will read, the event SHA's
    open run is completed neutral so nothing dangles `in_progress` there, and the review runs once."""
    from codna.webhook_control import JobControl

    gh = _MovingHeadGitHub(_MOVED_HEAD)
    control = JobControl().register(row_id=3, kind="review", repo="thyn-ai/algenta", pr_number=1085,
                                    ref=_EVENT_HEAD, deadline_s=1800.0)
    runs = []

    def runner(job, **kw):
        runs.append(kw)
        return _cli_completed(job)

    result = process_job(_review_qjob(), app_id="a", private_key="p", github=gh, runner=runner,
                         resolve_engine_key=lambda iid: "org-key", control=control)
    assert result.ok and result.requeue_head is None
    assert [c[0] for c in gh.created] == [_MOVED_HEAD]                     # ONE check, on the reviewed head
    assert len(runs) == 1 and runs[0]["check_run_id"] == 9001 and runs[0]["check_head_sha"] == _MOVED_HEAD
    [stale] = gh.stale
    assert stale["head_sha"] == _EVENT_HEAD and stale["name"] == "codna review" and stale["conclusion"] == "neutral"
    assert f"head moved to {_MOVED_HEAD[:8]} before the review started" in stale["summary"]
    assert "see the check on that commit" in stale["summary"]
    assert gh.updates == []                                                 # the CLI's findings stand
    assert control.ref == _MOVED_HEAD                                       # a late event for THIS head no longer supersedes it
    [moved] = _phase_lines(capsys, "head_moved_before_start")
    assert moved["event_head"] == _EVENT_HEAD and moved["head"] == _MOVED_HEAD and moved["row_id"] == 3


def test_review_head_moving_during_the_turn_keeps_the_check_on_the_reviewed_sha_and_requeues_the_new_head(capsys):
    """The head was current at start and moved while the review ran: the check completed by the CLI on
    the reviewed commit stands (the review is evidence for THAT commit), nothing is opened or posted
    for the new head here, and the pool is told to queue it."""
    gh = _MovingHeadGitHub(_EVENT_HEAD, _MOVED_HEAD)
    result = process_job(_review_qjob(), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and result.requeue_head == _MOVED_HEAD
    assert [c[0] for c in gh.created] == [_EVENT_HEAD]
    assert gh.stale == [] and gh.updates == []                              # no second check, no second post
    [moved] = _phase_lines(capsys, "head_moved_during_turn")
    assert moved["reviewed_head"] == _EVENT_HEAD and moved["head"] == _MOVED_HEAD


def test_a_worker_completed_review_check_stays_on_the_reviewed_sha_and_says_the_head_moved():
    """When the CLI did not complete the run (it failed), the worker does -- on the reviewed commit,
    with the move on record, never on the new head."""
    gh = _MovingHeadGitHub(_EVENT_HEAD, _MOVED_HEAD)
    result = process_job(_review_qjob(), app_id="a", private_key="p", github=gh,
                         runner=lambda job, **kw: JobResult(False, "codna review failed (review_error): agent died"),
                         resolve_engine_key=lambda iid: "org-key")
    assert not result.ok and result.requeue_head == _MOVED_HEAD
    [(check_id, conclusion, summary)] = gh.updates
    assert check_id == 9001 and conclusion == "failure"
    assert summary.startswith("codna review failed") and f"moved to {_MOVED_HEAD[:8]}" in summary and "queued" in summary
    assert [c[0] for c in gh.created] == [_EVENT_HEAD]


def test_a_review_whose_head_did_not_move_behaves_exactly_as_before():
    """No movement: the only difference from a client without the head lookup is the lookup itself."""
    plain, aware = _FakeGitHub(), _MovingHeadGitHub(_EVENT_HEAD)
    for gh in (plain, aware):
        result = process_job(_review_qjob(), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                             resolve_engine_key=lambda iid: "org-key")
        assert result.ok and result.requeue_head is None
    assert [c for c in aware.calls if c[0] != "pr_head"] == plain.calls
    assert aware.stale == [] and [c[0] for c in aware.created] == [_EVENT_HEAD]


@pytest.mark.parametrize("answer", [None, RuntimeError("github is down")], ids=["none", "raises"])
def test_an_unreadable_pr_head_falls_back_to_the_event_sha_and_never_reports_a_move(answer, capsys):
    gh = _MovingHeadGitHub(answer)
    result = process_job(_review_qjob(), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and result.requeue_head is None
    assert [c[0] for c in gh.created] == [_EVENT_HEAD] and gh.stale == []
    assert _phase_lines(capsys, "head_moved_before_start") == [] and _phase_lines(capsys, "head_moved_during_turn") == []


def test_a_comment_triggered_review_without_an_event_sha_anchors_its_check_to_the_resolved_head():
    """`@codna review` carries no head: the worker used to open no check at all and the CLI opened its
    own late. Now the check opens on the resolved head up front, and there is no event SHA to tidy."""
    gh = _MovingHeadGitHub(_MOVED_HEAD)
    runs = []

    def runner(job, **kw):
        runs.append(kw)
        return _cli_completed(job)

    result = process_job(_review_qjob(ref=None), app_id="a", private_key="p", github=gh, runner=runner,
                         resolve_engine_key=lambda iid: "org-key")
    assert result.ok and result.requeue_head is None
    assert [c[0] for c in gh.created] == [_MOVED_HEAD] and gh.stale == []
    assert runs[0]["check_run_id"] == 9001 and runs[0]["check_head_sha"] == _MOVED_HEAD


def test_a_retry_supersedes_its_predecessors_run_on_the_head_the_check_is_anchored_to():
    """Attempt 2 after a crash: the interrupted run to close is on the commit this attempt's check
    goes on (the resolved head), while the event SHA's run is closed neutral for the move."""
    gh = _MovingHeadGitHub(_MOVED_HEAD)
    process_job(_review_qjob(attempts=2), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key")
    assert [(s["head_sha"], s["conclusion"]) for s in gh.stale] == [(_EVENT_HEAD, "neutral"), (_MOVED_HEAD, "cancelled")]
    assert [c[0] for c in gh.created] == [_MOVED_HEAD]


def test_a_superseded_review_never_requeues_the_head_that_superseded_it():
    """A cancelled job is left alone: whatever superseded it is already driving the new head."""
    from codna.webhook_control import JobControl

    control = JobControl().register(row_id=3, kind="review", repo="thyn-ai/algenta", pr_number=1085,
                                    ref=_EVENT_HEAD, deadline_s=1800.0)
    gh = _MovingHeadGitHub(_EVENT_HEAD, _MOVED_HEAD)

    def runner(job, **kw):
        control.cancel(reason="superseded", detail=f"superseded by {_MOVED_HEAD[:8]}")
        return worker_module._cancelled_result(job, control)

    result = process_job(_review_qjob(), app_id="a", private_key="p", github=gh, runner=runner,
                         resolve_engine_key=lambda iid: "org-key", control=control)
    assert result.cancelled and result.requeue_head is None
    assert [u[1] for u in gh.updates] == ["neutral"]                        # closed by the cancellation, nothing else


def test_only_a_review_resolves_the_head_before_its_check():
    """A `codna fix` on a pull request keeps its event head: it analyses and patches that commit.
    (Not a `check_suite_failure` fix: CI triage has a head check of its own, after the run opens.)"""
    gh = _MovingHeadGitHub(_MOVED_HEAD)
    qjob = QueuedJob(row_id=4, delivery_id="d4", attempts=1,
                     job=WebhookJob("fix", "thyn-ai/algenta", ref=_EVENT_HEAD, installation_id=42, pr_number=1085,
                                    reason="pull_request_labeled", context={"head_ref": "feat"}))
    process_job(qjob, app_id="a", private_key="p", github=gh, runner=lambda job, **kw: JobResult(True, "ok"),
                resolve_engine_key=lambda iid: "org-key")
    assert [c[0] for c in gh.created] == [_EVENT_HEAD]
    assert ("pr_head", 1085) not in gh.calls and gh.stale == []


# The merge-group payload exactly as test_webhook.py's _merge_group_payload builds it (classify_event
# is the fixture: the job is derived from the event, never hand-built).
def _merge_group_payload(pr=48, base="main"):
    return {
        "action": "checks_requested",
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 42},
        "merge_group": {
            "head_sha": "feedfacefeedfacefeedfacefeedfacefeedface",
            "head_ref": f"refs/heads/gh-readonly-queue/{base}/pr-{pr}-0123456789abcdef0123456789abcdef01234567",
            "base_ref": f"refs/heads/{base}",
        },
    }


class _StaleAwareQueueGitHub(_QueueGitHub):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.stale = []

    def complete_stale_check_runs(self, repo, token, *, head_sha, name, summary, conclusion="cancelled"):
        self.stale.append(head_sha)
        return 1


def test_merge_group_inheritance_still_targets_the_group_commit_not_the_pr_head():
    """A merge-group job's `ref` IS the queue's own commit; re-anchoring it to the pull request's head
    would leave the group without its required check and GitHub would kick the PR out of the queue.
    The verdict is READ from the PR head and WRITTEN on merge_group.head_sha, exactly as before."""
    from codna.webhook import classify_event

    payload = _merge_group_payload(pr=48)
    job = classify_event("merge_group", payload)
    pr_head = "a" * 40
    gh = _StaleAwareQueueGitHub(head=pr_head, prior={"conclusion": "success", "summary": "**codna review** — ok", "html_url": "u"})
    result = process_job(QueuedJob(row_id=9, delivery_id="d9", attempts=1, job=job), app_id="a", private_key="p",
                         github=gh, runner=lambda j, **kw: JobResult(True, "ran"),
                         resolve_engine_key=lambda iid: (_ for _ in ()).throw(AssertionError("no metering for queue jobs")))
    assert result.ok and result.requeue_head is None
    assert job.ref == payload["merge_group"]["head_sha"]
    assert [c for c in gh.calls if c[0] == "create_check"] == [("create_check", "codna review", job.ref)]
    assert ("latest_run", pr_head, "codna review") in gh.calls              # the verdict comes from the PR head
    assert gh.stale == []                                                    # nothing re-anchored, nothing tidied
    assert ("update_check", "success") in gh.calls


def test_job_env_hands_the_checks_anchor_to_the_cli_only_with_a_run():
    env = worker_module._job_env("tok", "ekey", check_run_id=9001, check_head_sha=_MOVED_HEAD)
    assert env["CODNA_CHECK_RUN_ID"] == "9001" and env["CODNA_CHECK_RUN_HEAD_SHA"] == _MOVED_HEAD
    assert "CODNA_CHECK_RUN_HEAD_SHA" not in worker_module._job_env("tok", "ekey", check_run_id=9001)


def test_run_codna_job_passes_the_checks_anchor_through_to_the_cli_env(monkeypatch):
    seen = {}

    class _P:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(argv, *, env, cwd, timeout, on_process=None):
        seen.update(env)
        return _P()

    monkeypatch.setattr(worker_module, "_run_job_process", fake_run)
    monkeypatch.setattr(worker_module, "_teardown_job_runtime", lambda tmp: 0)
    worker_module.run_codna_job(_review_qjob().job, token="t", engine_key="k", check_run_id=9001, check_head_sha=_MOVED_HEAD)
    assert seen["CODNA_CHECK_RUN_ID"] == "9001" and seen["CODNA_CHECK_RUN_HEAD_SHA"] == _MOVED_HEAD


# ---- the pool queues the head a review found had moved under it -----------------------------------
def _moved(result_ok=True):
    return JobResult(result_ok, "ok", check_completed=True, requeue_head=_MOVED_HEAD)


def test_requeue_moved_head_queues_the_new_head_once_ahead_of_its_class_and_retires_the_stale_row(tmp_path):
    from codna.webhook_queue import WebhookQueue

    q = WebhookQueue(tmp_path / "q.db")
    # queued earlier: a review of ANOTHER pull request (priority 0) and a re-delivered review of the
    # stale event head of THIS one; the requeued head must go ahead of the first and retire the second.
    assert q.enqueue(WebhookJob("review", "thyn-ai/algenta", ref="c" * 40, pr_number=1086, installation_id=42,
                                reason="pull_request_opened"), delivery_id="other-pr")
    assert q.enqueue(WebhookJob("review", "thyn-ai/algenta", ref=_EVENT_HEAD, pr_number=1085, installation_id=42,
                                reason="pull_request_synchronize"), delivery_id="stale-redelivery")
    assert worker_module.requeue_moved_head(_review_qjob(), _moved(), q) is True
    assert worker_module.requeue_moved_head(_review_qjob(), _moved(), q) is False   # already queued: once is enough
    assert q.counts().get("superseded") == 1                                       # the stale head's row is retired
    first = q.claim()
    assert first.job.kind == "review" and first.job.ref == _MOVED_HEAD and first.job.pr_number == 1085
    assert first.job.installation_id == 42 and first.job.reason == "head_moved_during_review"
    assert worker_module.requeue_moved_head(_review_qjob(), _moved(), q) is False   # running counts as pending too
    assert q.claim().job.pr_number == 1086
    assert q.claim() is None


def test_requeue_moved_head_ignores_results_that_did_not_move_and_queues_it_cannot_use(tmp_path):
    from codna.webhook_queue import WebhookQueue

    q = WebhookQueue(tmp_path / "q.db")
    assert worker_module.requeue_moved_head(_review_qjob(), JobResult(True, "ok"), q) is False
    assert worker_module.requeue_moved_head(_qjob(kind="fix"), _moved(), q) is False        # a fix never re-queues
    assert worker_module.requeue_moved_head(_review_qjob(), _moved(), None) is False
    assert worker_module.requeue_moved_head(_review_qjob(), _moved(), object()) is False      # a queue without enqueue
    assert q.claim() is None


def test_the_pool_queues_the_moved_head_after_the_reviewed_rows_terminal_write(tmp_path):
    from codna.webhook_queue import WebhookQueue

    q = WebhookQueue(tmp_path / "q.db")
    assert q.enqueue(_review_qjob().job, delivery_id="d3")
    qjob = q.claim()
    pool = WorkerPool(q, concurrency=1, process=lambda qj: _moved())
    pool._run_claimed(qjob)
    assert q.row_state(qjob.row_id) == ("done", 1)
    nxt = q.claim()
    assert nxt is not None and nxt.job.kind == "review" and nxt.job.ref == _MOVED_HEAD and nxt.job.pr_number == 1085
    assert q.claim() is None


# --- the row's own Check Run: opened `queued` by the ingress, UPDATED by the worker -----------------
# webhook_queued_check opens a review's run at enqueue on the Postgres backend and the row is born
# owning it (QueuedJob.check_run_id). The worker flips THAT run to in_progress and completes it: one
# `codna review` per commit, visible from the moment the delivery was accepted. Rows without one (the
# SQLite queue, an older ingress, a requeue) keep today's create-at-claim.

class _OwnedRunGitHub(_MovingHeadGitHub):
    def __init__(self, *heads, status="queued"):
        super().__init__(*heads)
        self.started: list[tuple[int, str]] = []   # (check_run_id, summary) of every start_check_run
        self.probed: list[int] = []
        self._status = status

    def start_check_run(self, repo, token, check_run_id, *, name, summary):
        self.started.append((check_run_id, summary))
        self.calls.append(("start_check", check_run_id))

    def check_run_status(self, repo, token, check_run_id):
        self.probed.append(check_run_id)
        return self._status

    def complete_check_run_if_open(self, repo, token, check_run_id, *, conclusion, summary, name):
        self.closed_if_open = getattr(self, "closed_if_open", []) + [(check_run_id, conclusion, summary)]
        return self._status in ("queued", "in_progress", None)


def _owned_qjob(check_run_id=5150, **kw):
    q = _review_qjob(**kw)
    return QueuedJob(row_id=q.row_id, delivery_id=q.delivery_id, attempts=q.attempts, job=q.job,
                     check_run_id=check_run_id)


def test_the_worker_updates_the_run_the_ingress_opened_instead_of_creating_a_second_one(capsys):
    """(e) The row carries the run the ingress opened `queued` on the event head, the head has not
    moved: the worker flips it to in_progress, hands it to the CLI, records it, attaches it to the
    cancellation handle -- and never calls create."""
    from codna.webhook_control import JobControl

    gh = _OwnedRunGitHub(_EVENT_HEAD)
    control = JobControl().register(row_id=3, kind="review", repo="thyn-ai/algenta", pr_number=1085,
                                    ref=_EVENT_HEAD, deadline_s=1800.0)
    recorded, runs = [], []

    def runner(job, **kw):
        runs.append(kw)
        return _cli_completed(job)

    result = process_job(_owned_qjob(), app_id="a", private_key="p", github=gh, runner=runner,
                         resolve_engine_key=lambda iid: "org-key", control=control, on_check_run=recorded.append)
    assert result.ok
    assert gh.created == []                                                 # never a second run
    assert gh.started == [(5150, "codna review started (pull_request_opened)")]
    assert gh.probed == [5150]                                              # asked GitHub it is open before starting it
    assert runs[0]["check_run_id"] == 5150 and runs[0]["check_head_sha"] == _EVENT_HEAD
    assert recorded == [5150] and control.check_run_id == 5150
    assert gh.stale == [] and gh.updates == []                              # the CLI's findings stand on THAT run
    [started] = _phase_lines(capsys, "check_run_started")
    assert started["check_run_id"] == 5150 and started["head"] == _EVENT_HEAD


def test_a_row_without_a_run_still_creates_one_at_claim():
    """The fallback (e): rows an older ingress wrote during a rolling deploy, the SQLite queue, a
    worker's or the reaper's requeue -- QueuedJob.check_run_id is None and the worker creates."""
    gh = _OwnedRunGitHub(_EVENT_HEAD)
    process_job(_review_qjob(), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key")
    assert [c[0] for c in gh.created] == [_EVENT_HEAD] and gh.started == []


def test_head_moved_before_start_completes_the_queued_run_neutral_and_anchors_a_new_one(capsys):
    """(b) The ingress opened the run on the event head; by the time a worker claims the row the
    pull request head has moved. The row's run is completed neutral BY ID (plus the listing sweep for
    anything else on that commit), a fresh run is created on the head the review reads, and the row
    is re-pointed at it (on_check_run) so a restart or the reaper closes the right one."""
    gh = _OwnedRunGitHub(_MOVED_HEAD)
    recorded, runs = [], []

    def runner(job, **kw):
        runs.append(kw)
        return _cli_completed(job)

    result = process_job(_owned_qjob(), app_id="a", private_key="p", github=gh, runner=runner,
                         resolve_engine_key=lambda iid: "org-key", on_check_run=recorded.append)
    assert result.ok
    [(cid, conclusion, summary)] = gh.updates
    assert cid == 5150 and conclusion == "neutral"
    assert f"head moved to {_MOVED_HEAD[:8]} before the review started" in summary
    assert gh.started == []                                                 # the old run is not "started"
    assert [c[0] for c in gh.created] == [_MOVED_HEAD]                     # one fresh run, on the reviewed head
    assert runs[0]["check_run_id"] == 9001 and runs[0]["check_head_sha"] == _MOVED_HEAD
    assert recorded == [9001]                                               # the row now owns the new run
    [stale] = gh.stale
    assert stale["head_sha"] == _EVENT_HEAD and stale["conclusion"] == "neutral"
    [moved] = _phase_lines(capsys, "head_moved_before_start")
    assert moved["check_run_id"] == 5150


def test_the_rows_run_is_reused_only_while_github_says_it_is_still_open(capsys):
    """Every claim asks GitHub before reusing: a run a previous attempt completed (or the sweep / the
    dead-letter notice closed before an operator's retry-now) is not reopened -- a fresh run replaces
    it (today's retry path, predecessor swept) -- while one still in_progress (a worker died mid-job
    and recover_stale re-queued the row) is simply continued."""
    gh = _OwnedRunGitHub(_EVENT_HEAD, status="completed")
    process_job(_owned_qjob(attempts=2), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key")
    assert gh.probed == [5150] and gh.started == []
    assert [c[0] for c in gh.created] == [_EVENT_HEAD]
    assert [(s["head_sha"], s["conclusion"]) for s in gh.stale] == [(_EVENT_HEAD, "cancelled")]
    [line] = _phase_lines(capsys, "check_run_not_reusable")
    assert line["check_run_id"] == 5150 and line["status"] == "completed"

    gh = _OwnedRunGitHub(_EVENT_HEAD, status="completed")                   # retry-now: attempts back to 0
    process_job(_owned_qjob(attempts=1), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key")
    assert gh.probed == [5150] and gh.started == [] and [c[0] for c in gh.created] == [_EVENT_HEAD]
    assert gh.stale == []                                                   # a first attempt sweeps nothing
    assert getattr(gh, "closed_if_open", []) == []                          # known completed: nothing to close

    gh = _OwnedRunGitHub(_EVENT_HEAD, status="in_progress")
    process_job(_owned_qjob(attempts=2), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key")
    assert gh.probed == [5150] and gh.started == [(5150, "codna review started (pull_request_opened)")]
    assert gh.created == [] and gh.stale == []                              # nothing to supersede: it IS the run


def test_a_run_that_cannot_be_read_is_replaced_never_started_blind_and_closed_if_open(capsys):
    """An unreadable answer (a 404, a bad body, a client without the helper, a transport error) is
    never started blind: the run is REPLACED -- and, because it may well still be the `queued` run
    the ingress opened seconds ago (attempt 1!), it is completed `cancelled` first when it is open,
    so nothing is left stranded on the commit (codna review of #596, round 2)."""
    class _Unreadable(_OwnedRunGitHub):
        def check_run_status(self, repo, token, check_run_id):
            raise ConnectionError("github is down")

    gh = _Unreadable(_EVENT_HEAD, status=None)
    process_job(_owned_qjob(attempts=3), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key")
    assert gh.started == [] and [c[0] for c in gh.created] == [_EVENT_HEAD]
    [(cid, conclusion, summary)] = gh.closed_if_open
    assert (cid, conclusion) == (5150, "cancelled") and "replaced by a fresh run" in summary
    [line] = _phase_lines(capsys, "check_run_not_reusable")
    assert line["status"] is None and line["closed"] is True

    class _NoProbe(_OwnedRunGitHub):                                        # an older client without the helper
        check_run_status = None

    gh = _NoProbe(_EVENT_HEAD, status=None)                                  # ... on the FIRST attempt
    process_job(_owned_qjob(), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key")
    assert gh.started == [] and [c[0] for c in gh.created] == [_EVENT_HEAD]
    assert [(c[0], c[1]) for c in gh.closed_if_open] == [(5150, "cancelled")]   # the ingress's run is not stranded

    class _NoCloser(_NoProbe):                                              # neither helper: today's client
        complete_check_run_if_open = None

    gh = _NoCloser(_EVENT_HEAD)
    process_job(_owned_qjob(), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key")
    assert gh.started == [] and [c[0] for c in gh.created] == [_EVENT_HEAD]
    lines = _phase_lines(capsys, "check_run_not_reusable")                # the _NoProbe case's line, then this one
    assert [ln["closed"] for ln in lines] == [True, None]


def test_a_verdict_reached_before_running_completes_the_rows_own_run():
    """(d) The org is not linked: the fail-closed prompt lands on the run the ingress opened, not on
    a second one -- and the same for the bridge being down on the last attempt and automation off."""
    gh = _OwnedRunGitHub(_EVENT_HEAD)
    result = process_job(_owned_qjob(), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                         resolve_engine_key=lambda iid: None)
    assert result.ok and gh.created == [] and gh.started == []
    assert [(u[0], u[1]) for u in gh.updates] == [(5150, "neutral")]

    gh = _OwnedRunGitHub(_EVENT_HEAD)
    process_job(_owned_qjob(), app_id="a", private_key="p", github=gh, runner=_cli_completed,
                resolve_engine_key=lambda iid: "org-key", resolve_fix_enabled=lambda iid: False)
    assert gh.created == [] and [(u[0], u[1]) for u in gh.updates] == [(5150, "neutral")]

    from codna import webhook_metering

    def bridge_down(iid):
        raise webhook_metering.BridgeUnavailable("502")

    gh = _OwnedRunGitHub(_EVENT_HEAD)
    process_job(_owned_qjob(attempts=worker_module._MAX_ATTEMPTS), app_id="a", private_key="p", github=gh,
                runner=_cli_completed, resolve_engine_key=bridge_down)
    assert gh.created == [] and [(u[0], u[1]) for u in gh.updates] == [(5150, "neutral")]


def test_a_job_that_raises_before_touching_its_run_closes_it_when_the_row_ends_failed(monkeypatch):
    """(d) Token minting raises out of process_job: the pool marks the row failed. With the ingress's
    run on the row and no further attempt coming, the pool completes that run (failure, with the code)
    -- a required check must not sit `queued` forever. A retry keeps it for the next attempt; a `dead`
    row is the reaper's dead-letter notice."""
    closed, minted = [], []

    class _Queue:
        def __init__(self, final):
            self.final = final
            self.completed = []

        def complete(self, row_id, *, status, result=None, retry=True, retry_after_s=None):
            self.completed.append((row_id, status))

        def row_state(self, row_id):
            return (self.final, 1)

    def raising_process(qjob):
        raise WebhookError("installation_token_failed", "installation token: 401")

    monkeypatch.setattr(pool_module.webhook_github, "installation_token",
                        lambda *a, **k: minted.append(k["kind"]) or "tok-for-close")
    monkeypatch.setattr(pool_module.webhook_github, "complete_check_run_if_open",
                        lambda repo, token, cid, *, conclusion, summary, name: closed.append((cid, conclusion, summary, token)) or True)
    for final in ("retry", "dead"):
        queue = _Queue(final)
        WorkerPool(queue, process=raising_process, app_id="a", private_key="p")._run_claimed_inner(_owned_qjob())
        assert queue.completed == [(3, "failed")] and closed == [], final
    queue = _Queue("failed")
    WorkerPool(queue, process=raising_process, app_id="a", private_key="p")._run_claimed_inner(_owned_qjob())
    assert queue.completed == [(3, "failed")]
    [(cid, conclusion, summary, token)] = closed
    assert (cid, conclusion, token) == (5150, "failure", "tok-for-close") and minted == ["review"]
    assert "installation_token_failed" in summary and "codna review" in summary
    # a row without an ingress-opened run: nothing to close (today's path)
    closed.clear()
    WorkerPool(_Queue("failed"), process=raising_process, app_id="a", private_key="p")._run_claimed_inner(_review_qjob())
    assert closed == []
