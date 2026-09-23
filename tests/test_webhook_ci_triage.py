"""CI-failure triage: a red check suite is inspected before a fix spends anything."""
from __future__ import annotations

from codna.webhook import WebhookJob
from codna.webhook_ci_triage import TERMINAL, classify_job, focus_log, triage

BUN_LOG = "\n".join([
    "2026-09-19T04:38:20.1234567Z ##[group]Run bun install --frozen-lockfile",
    "2026-09-19T04:38:20.1234567Z \x1b[36;1mbun install --frozen-lockfile\x1b[0m",
    "2026-09-19T04:38:20.1234567Z ##[endgroup]",
    "2026-09-19T04:38:21.0000000Z bun install v1.3.14 (0d9b296a)",
    '2026-09-19T04:38:22.0000000Z error: Fail extracting tarball for "mermaid"',
    "2026-09-19T04:38:22.0000000Z error: Fail extracting tarball from mermaid",
    "2026-09-19T04:38:22.0000000Z ##[error]Process completed with exit code 1.",
    "2026-09-19T04:38:23.0000000Z ##[group]Run bun run bin/compliance-test.ts",
    "2026-09-19T04:38:23.0000000Z Error: --api-key is required",
    "2026-09-19T04:38:23.0000000Z ##[error]Process completed with exit code 1.",
])
PYTEST_LOG = "\n".join([
    "2026-09-19T04:38:20.0000000Z ##[group]Run pytest -q",
    "2026-09-19T04:38:20.0000000Z ##[endgroup]",
    "2026-09-19T04:38:30.0000000Z FAILED tests/test_mcp.py::test_server_info_version - AssertionError: assert '0.4.2' == '0.4.3'",
    "2026-09-19T04:38:30.0000000Z 1 failed, 211 passed in 9.81s",
    "2026-09-19T04:38:30.0000000Z ##[error]Process completed with exit code 1.",
])


# --- reading a job log ---------------------------------------------------------------------------
def test_focus_log_centres_on_the_first_error_marker_and_cleans_the_lines():
    window = focus_log(BUN_LOG)
    assert window.startswith("bun install --frozen-lockfile")   # timestamps and colour codes gone
    assert "2026-09-19T" not in window and "\x1b" not in window
    assert "##[group]" not in window
    assert "Fail extracting tarball" in window
    assert window.endswith("##[error]Process completed with exit code 1.")
    assert "--api-key" not in window   # the `if: always()` step that ran afterwards is not the failure


def test_focus_log_without_a_marker_keeps_the_tail():
    assert focus_log("a\nb\nc", before=2) == "b\nc"
    assert focus_log(None) is None and focus_log("") is None


# --- classifying one failed job --------------------------------------------------------------------
def test_bun_tarball_failure_in_an_install_step_is_infrastructure():
    assert classify_job("bun install (openresponses CLI)", focus_log(BUN_LOG)) == "infrastructure"


def test_frozen_lockfile_drift_in_an_install_step_is_code():
    log = "error: lockfile had changes, but lockfile is frozen\n##[error]Process completed with exit code 1."
    assert classify_job("bun install", log) == "code"


def test_failing_tests_are_code_even_when_the_step_name_sounds_like_setup():
    assert classify_job("Install + run parity/proof tests", focus_log(PYTEST_LOG)) == "code"


def test_a_registry_503_in_a_setup_step_is_infrastructure():
    log = "npm ERR! 503 Service Unavailable - GET https://registry.npmjs.org/left-pad\n##[error]Process completed with exit code 1."
    assert classify_job("Setup node deps", log) == "infrastructure"


def test_a_network_error_in_a_test_step_is_still_code():
    """Only install/setup-shaped steps get the transient reading: a test that hits the network and
    fails is the test's problem until a human says otherwise."""
    log = "requests.exceptions.ConnectionError: ECONNRESET\n##[error]Process completed with exit code 1."
    assert classify_job("Run integration tests", log) == "code"


def test_runner_lost_communication_is_infrastructure_in_any_step():
    log = ("The runner has received a shutdown signal. This can happen when the runner service is "
           "stopped, or a manually started runner is canceled.")
    assert classify_job("Run tests", log) == "infrastructure"


def test_no_log_decides_nothing():
    assert classify_job("bun install", None) == "unknown"
    assert classify_job(None, "") == "unknown"


# --- the whole decision ----------------------------------------------------------------------------
class _GitHub:
    def __init__(self, *, head="abc123", failing=None, jobs=None, logs=None,
                 grant=("ci_rerun", "ci_triage"), rerun_ok=True):
        self.head, self.failing = head, failing
        self.jobs, self.logs = jobs or {}, logs or {}
        self.grant, self.rerun_ok, self.calls = set(grant), rerun_ok, []

    def installation_token(self, app_id, private_key, installation_id, *, repo_full_name, kind):
        self.calls.append(("token", kind))
        if kind in ("ci_rerun", "ci_triage") and kind not in self.grant:
            raise RuntimeError("installation token: 422 permission not granted")
        return f"tok-{kind}"

    def pull_request_head_sha(self, repo, token, number):
        return self.head

    def failing_check_runs_in_suite(self, repo, token, suite_id):
        self.calls.append(("suite", suite_id))
        return self.failing

    def actions_job(self, repo, token, job_id):
        self.calls.append(("job", job_id, token))
        return self.jobs.get(job_id)

    def actions_job_log_tail(self, repo, token, job_id):
        self.calls.append(("log", job_id, token))
        return self.logs.get(job_id)

    def rerun_failed_jobs(self, repo, token, run_id):
        self.calls.append(("rerun", run_id, token))
        return self.rerun_ok


def _job(**ctx):
    return WebhookJob("fix", "acme/app", ref="abc123", pr_number=7, installation_id=42,
                      context={"head_ref": "feature", "check_suite_id": 555, **ctx},
                      reason="check_suite_failure")


RUN = {"id": 9001, "name": "OpenResponses conformance",
       "html_url": "https://github.com/acme/app/actions/runs/1/job/9001", "app_slug": "github-actions"}
JOB = {"run_id": 1, "run_attempt": 1, "name": "OpenResponses conformance",
       "steps": [{"name": "Checkout", "conclusion": "success"},
                 {"name": "bun install (openresponses CLI)", "conclusion": "failure"}]}


def test_head_moved_is_terminal_and_reads_nothing_else():
    gh = _GitHub(head="def456", failing=[RUN])
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "head_moved" and t.verdict in TERMINAL
    assert "abc123" in t.summary and "def456" in t.summary
    assert not any(c[0] in ("suite", "job", "log", "rerun") for c in gh.calls)


def test_a_suite_with_no_failing_job_left_is_green():
    gh = _GitHub(failing=[])
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "green" and "re-run" in t.summary
    assert ("suite", 555) in gh.calls


def test_infrastructure_failure_is_terminal_and_rerun_once():
    gh = _GitHub(failing=[RUN], jobs={9001: JOB}, logs={9001: BUN_LOG})
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "infrastructure" and t.rerun
    assert ("rerun", 1, "tok-ci_rerun") in gh.calls
    assert "bun install (openresponses CLI)" in t.summary
    assert "Fail extracting tarball" in t.summary
    assert "attempt 2" in t.summary and "nothing was spent" in t.summary


def test_second_attempt_failing_the_same_way_is_left_to_a_human():
    gh = _GitHub(failing=[RUN], jobs={9001: {**JOB, "run_attempt": 2}}, logs={9001: BUN_LOG})
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "infrastructure" and not t.rerun
    assert not any(c[0] == "rerun" for c in gh.calls)
    assert "already attempt 2" in t.summary and "human" in t.summary


def test_without_actions_write_the_summary_asks_for_it_instead_of_rerunning():
    gh = _GitHub(failing=[RUN], jobs={9001: JOB}, logs={9001: BUN_LOG}, grant=("ci_triage",))
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "infrastructure" and not t.rerun
    assert "Actions: write" in t.summary
    assert ("job", 9001, "tok-ci_triage") in gh.calls          # read with the read-only token


def test_a_refused_rerun_is_reported_not_claimed():
    gh = _GitHub(failing=[RUN], jobs={9001: JOB}, logs={9001: BUN_LOG}, rerun_ok=False)
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "infrastructure" and not t.rerun
    assert "refused" in t.summary and "Re-ran" not in t.summary


def test_without_any_actions_permission_the_fix_runs_as_before_with_a_note():
    gh = _GitHub(failing=[RUN], grant=())
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "unknown" and t.verdict not in TERMINAL
    assert t.note and "Actions: read" in t.note
    assert not any(c[0] in ("job", "log") for c in gh.calls)


def test_code_failure_hands_the_agent_the_ci_evidence():
    gh = _GitHub(failing=[RUN], jobs={9001: {**JOB, "steps": [{"name": "Run tests", "conclusion": "failure"}]}},
                 logs={9001: PYTEST_LOG})
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "code" and t.summary == "" and t.note is None
    assert "failing step: Run tests" in t.issue_text
    assert "test_server_info_version" in t.issue_text
    assert "abc123" in t.issue_text and "do not weaken" in t.issue_text


def test_mixed_failures_fix_the_code_and_name_the_infrastructure_ones():
    infra_run = {**RUN, "id": 9002, "name": "Install deps"}
    gh = _GitHub(failing=[RUN, infra_run],
                 jobs={9001: {**JOB, "steps": [{"name": "Run tests", "conclusion": "failure"}]}, 9002: JOB},
                 logs={9001: PYTEST_LOG, 9002: BUN_LOG})
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "code"
    assert "Ignore these jobs" in t.issue_text and "Install deps" in t.issue_text
    assert not any(c[0] == "rerun" for c in gh.calls)


def test_an_unreadable_suite_runs_as_before():
    gh = _GitHub(failing=None)
    assert triage(_job(), "fix-token", gh, app_id="a", private_key="p").verdict == "unknown"


def test_a_non_actions_ci_system_is_no_evidence():
    gh = _GitHub(failing=[{**RUN, "app_slug": "circleci"}])
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "unknown" and t.note is None
    assert not any(c[0] in ("job", "log") for c in gh.calls)


def test_dev_mode_without_app_auth_uses_the_ambient_token():
    gh = _GitHub(failing=[RUN], jobs={9001: JOB}, logs={9001: BUN_LOG})
    job = WebhookJob("fix", "acme/app", ref="abc123", pr_number=7, installation_id=None,
                     context={"check_suite_id": 555}, reason="check_suite_failure")
    t = triage(job, "ghp-dev", gh)
    assert t.verdict == "infrastructure"
    assert ("job", 9001, "ghp-dev") in gh.calls and not any(c[0] == "token" for c in gh.calls)


def test_triage_never_raises_on_a_broken_client():
    class _Broken:
        def __getattr__(self, name):
            def boom(*a, **k):
                raise RuntimeError("boom")
            return boom

    t = triage(_job(), "fix-token", _Broken(), app_id="a", private_key="p")
    assert t.verdict == "unknown"


def test_every_workflow_run_is_rerun_even_when_the_first_is_refused():
    """`all()` over a lazy generator stopped at the first refusal (review finding on #546)."""
    class _Picky(_GitHub):
        def rerun_failed_jobs(self, repo, token, run_id):
            self.calls.append(("rerun", run_id, token))
            return run_id != 1                       # run 1 refused, run 2 fine

    run_b = {**RUN, "id": 9002, "name": "Other install"}
    gh = _Picky(failing=[RUN, run_b], jobs={9001: JOB, 9002: {**JOB, "run_id": 2}},
                logs={9001: BUN_LOG, 9002: BUN_LOG})
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert t.verdict == "infrastructure" and not t.rerun
    assert [c for c in gh.calls if c[0] == "rerun"] == [("rerun", 1, "tok-ci_rerun"), ("rerun", 2, "tok-ci_rerun")]


def test_a_partial_rerun_says_exactly_which_runs_were_refused():
    class _Picky(_GitHub):
        def rerun_failed_jobs(self, repo, token, run_id):
            self.calls.append(("rerun", run_id, token))
            return run_id != 1

    run_b = {**RUN, "id": 9002, "name": "Other install"}
    gh = _Picky(failing=[RUN, run_b], jobs={9001: JOB, 9002: {**JOB, "run_id": 2}},
                logs={9001: BUN_LOG, 9002: BUN_LOG})
    t = triage(_job(), "fix-token", gh, app_id="a", private_key="p")
    assert not t.rerun
    assert "Re-ran the failed jobs of workflow run 2" in t.summary
    assert "refused to re-run 1" in t.summary
    assert "refused by the API; re-run them manually" not in t.summary   # that line is for zero successes
