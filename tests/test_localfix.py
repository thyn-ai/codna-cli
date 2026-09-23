"""run_local_fix tests — the self-hosted detect→fix→verify→close loop behind
`codna secure --fix --engine local`. Real worktree + real git diff/apply; the engine and the
agentic patch generator are stubbed (no LLM, no network) so the loop discipline is exercised
deterministically offline."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import codna.localfix as localfix_module
from codna.findings import Classification, ClosureStatus
from codna.localfix import run_local_fix
from codna.policy import Policy
from codna.sarif import ingest_sarif
from codna.secure import ClosureVerdict, ReachVerdict

PROD_POLICY = Policy(autofix_classifications=("exploitable", "production-reachable"))


def _run(argv, cwd):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True)


def _doc():
    return json.dumps({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0",
                                "rules": [{"id": "py/sqli", "properties": {"security-severity": "9.1", "tags": ["security"]}}]}},
            "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://x/y"}],
            "results": [{"ruleId": "py/sqli", "message": {"text": "sqli"},
                         "locations": [{"physicalLocation": {"artifactLocation": {"uri": "src/app.py"},
                                                             "region": {"startLine": 2}}}]}],
        }],
    })


INGEST = ingest_sarif(_doc())


def _init_repo(tmp_path):
    d = tmp_path / "repo"
    (d / "src").mkdir(parents=True)
    (d / "src" / "app.py").write_text('def q(cur, u):\n    return cur.execute("SELECT * FROM t WHERE id=\'" + u + "\'")\n')
    _run(["git", "init", "-q"], d)
    _run(["git", "config", "user.email", "a@b.c"], d)
    _run(["git", "config", "user.name", "t"], d)
    _run(["git", "add", "-A"], d)
    _run(["git", "commit", "-q", "-m", "init"], d)
    return str(d), _run(["git", "rev-parse", "HEAD"], d).stdout.strip()


class StubEngine:
    def __init__(self, classification=Classification.PRODUCTION_REACHABLE, closure=ClosureStatus.CLOSED):
        self.classification, self.closure = classification, closure

    def analyze(self, ingest, finding):
        return ReachVerdict(self.classification)

    def reprove_closure(self, finding, patch):
        return ClosureVerdict(self.closure)


def gen_parameterize(finding, cwd):
    p = os.path.join(cwd, "src", "app.py")
    open(p, "w").write('def q(cur, u):\n    return cur.execute("SELECT * FROM t WHERE id = ?", (u,))\n')
    diff = _run(["git", "diff"], cwd).stdout
    _run(["git", "checkout", "--", "."], cwd)
    return diff


def gen_evasion(finding, cwd):
    path = ".github/workflows/ci.yml"
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n on: push\n+    # tampered\n"


def test_happy_path_remediates(tmp_path):
    repo, sha = _init_repo(tmp_path)
    report = run_local_fix(
        INGEST, engine=StubEngine(), patch_generator=gen_parameterize,
        repo_dir=repo, commit=sha, policy=PROD_POLICY, test_cmds=[["true"]],
    )
    assert report.eligible == 1
    o = report.outcomes[0]
    assert o.fix_applied and o.integrity_ok and o.tests_passed is True
    assert o.closure == "closed" and o.remediated is True
    assert len(report.remediated) == 1
    # the user's checkout is untouched (all work happened in an ephemeral worktree)
    assert "?" not in open(os.path.join(repo, "src", "app.py")).read()


def test_python_verification_command_uses_current_interpreter_when_absent(tmp_path, monkeypatch):
    repo, sha = _init_repo(tmp_path)
    monkeypatch.setattr(localfix_module.shutil, "which", lambda name: None if name == "python" else name)
    report = run_local_fix(
        INGEST, engine=StubEngine(), patch_generator=gen_parameterize,
        repo_dir=repo, commit=sha, policy=PROD_POLICY,
        test_cmds=[["python", "-c", "import sys; raise SystemExit(0 if sys.executable else 1)"]],
    )

    assert report.remediated
    assert report.outcomes[0].tests_passed is True


def test_apply_to_repo_applies_verified_patch_to_checkout(tmp_path):
    repo, sha = _init_repo(tmp_path)
    report = run_local_fix(
        INGEST, engine=StubEngine(), patch_generator=gen_parameterize,
        repo_dir=repo, commit=sha, policy=PROD_POLICY, test_cmds=[["true"]], apply_to_repo=True,
    )

    o = report.outcomes[0]
    assert o.remediated is True
    assert o.target_applied is True
    assert o.target_tests_passed is True
    assert "?" in open(os.path.join(repo, "src", "app.py")).read()


def test_apply_to_repo_blocks_when_target_patch_no_longer_applies(tmp_path):
    repo, sha = _init_repo(tmp_path)
    app = os.path.join(repo, "src", "app.py")
    open(app, "w").write('def q(cur, u):\n    return cur.execute("changed")\n')

    report = run_local_fix(
        INGEST, engine=StubEngine(), patch_generator=gen_parameterize,
        repo_dir=repo, commit=sha, policy=PROD_POLICY, test_cmds=[["true"]], apply_to_repo=True,
    )

    o = report.outcomes[0]
    assert o.remediated is False
    assert o.target_applied is False
    assert o.target_apply_reason.startswith("target apply check failed")
    assert "changed" in open(app).read()


def test_apply_to_repo_rolls_back_when_target_verification_fails(tmp_path):
    repo, sha = _init_repo(tmp_path)
    open(os.path.join(repo, "target-only"), "w").write("present only in caller checkout\n")
    test_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; raise SystemExit(1 if Path('target-only').exists() else 0)",
    ]

    report = run_local_fix(
        INGEST, engine=StubEngine(), patch_generator=gen_parameterize,
        repo_dir=repo, commit=sha, policy=PROD_POLICY, test_cmds=[test_cmd], apply_to_repo=True,
    )

    o = report.outcomes[0]
    assert o.remediated is False
    assert o.target_applied is False
    assert o.target_tests_passed is False
    assert o.target_apply_reason == "target verification failed after apply; patch rolled back"
    assert "?" not in open(os.path.join(repo, "src", "app.py")).read()


def test_ineligible_finding_is_never_fixed(tmp_path):
    repo, sha = _init_repo(tmp_path)
    # default policy autofixes only 'exploitable'; production-reachable is NOT eligible
    report = run_local_fix(
        INGEST, engine=StubEngine(), patch_generator=gen_parameterize,
        repo_dir=repo, commit=sha, policy=Policy(), test_cmds=[["true"]],
    )
    assert report.eligible == 0
    assert report.outcomes[0].fix_applied is False


def test_evasion_patch_blocked_before_apply(tmp_path):
    repo, sha = _init_repo(tmp_path)
    report = run_local_fix(
        INGEST, engine=StubEngine(), patch_generator=gen_evasion,
        repo_dir=repo, commit=sha, policy=PROD_POLICY, test_cmds=[["true"]],
    )
    o = report.outcomes[0]
    assert o.eligible is True and o.integrity_ok is False and o.fix_applied is False
    assert o.remediated is False


def test_test_failure_blocks_remediation(tmp_path):
    repo, sha = _init_repo(tmp_path)
    report = run_local_fix(
        INGEST, engine=StubEngine(), patch_generator=gen_parameterize,
        repo_dir=repo, commit=sha, policy=PROD_POLICY, test_cmds=[["false"]],
    )
    o = report.outcomes[0]
    assert o.fix_applied is True and o.tests_passed is False and o.remediated is False


def test_not_closed_blocks_remediation(tmp_path):
    repo, sha = _init_repo(tmp_path)
    report = run_local_fix(
        INGEST, engine=StubEngine(closure=ClosureStatus.OPEN), patch_generator=gen_parameterize,
        repo_dir=repo, commit=sha, policy=PROD_POLICY, test_cmds=[["true"]],
    )
    o = report.outcomes[0]
    assert o.fix_applied is True and o.closure == "open" and o.remediated is False
