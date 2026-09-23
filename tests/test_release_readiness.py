from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_release_readiness.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_release_readiness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_secret_check_fails_without_telys_license_token():
    tool = _load_script()

    check = tool.check_secret_names({"CODNA_API_KEY"}, set())

    assert check.status == "fail"
    assert "CODNA_TELYS_LICENSE_JWT" in check.message
    assert check.details["configured"] == {"environment": [], "repository": ["CODNA_API_KEY"]}


def test_secret_check_passes_with_repo_only_telys_license_token():
    tool = _load_script()

    check = tool.check_secret_names({"CODNA_TELYS_LICENSE_JWT"}, set())

    assert check.status == "pass"
    assert "Codna OEM Telys license" in check.message
    assert check.details == {"scopes": ["repository"]}


def test_secret_check_passes_with_environment_telys_license_token():
    tool = _load_script()

    check = tool.check_secret_names(set(), {"CODNA_TELYS_LICENSE_JWT"})

    assert check.status == "pass"
    assert check.details == {"scopes": ["environment"]}


def test_secret_check_passes_with_environment_and_repo_telys_license_token():
    tool = _load_script()

    check = tool.check_secret_names({"CODNA_TELYS_LICENSE_JWT"}, {"CODNA_TELYS_LICENSE_JWT"})

    assert check.status == "pass"
    assert check.details == {"scopes": ["environment", "repository"]}


def test_download_secret_check_requires_telys_oem_download_token():
    tool = _load_script()

    missing = tool.check_download_secret_names({"CODNA_TELYS_LICENSE_JWT"}, set())
    present = tool.check_download_secret_names({"CODNA_TELYS_OEM_DOWNLOAD_TOKEN"}, set())

    assert missing.status == "fail"
    assert "CODNA_TELYS_OEM_DOWNLOAD_TOKEN" in missing.message
    assert present.status == "pass"
    assert present.details == {"scopes": ["repository"]}


def test_pypi_token_secret_names_fail_closed():
    tool = _load_script()

    bad = tool.check_no_pypi_token_secret_names(
        {"PYPI_API_TOKEN", "CODNA_TELYS_LICENSE_JWT"},
        {"TWINE_PASSWORD"},
    )
    good = tool.check_no_pypi_token_secret_names({"CODNA_TELYS_LICENSE_JWT", "PYPI_TOKEN"}, set())

    assert bad.status == "fail"
    assert bad.details == {
        "forbidden_secret_names": {
            "environment": ["TWINE_PASSWORD"],
            "repository": ["PYPI_API_TOKEN"],
        }
    }
    assert good.status == "pass"


def test_legacy_telys_token_secret_names_fail_closed():
    tool = _load_script()

    bad = tool.check_no_legacy_telys_secret_names(
        {"CODNA_TELYS_API_KEY", "CODNA_TELYS_LICENSE_JWT"},
        {"CODNA_TELYS_RUNTIME_DOWNLOAD_TOKEN"},
    )
    good = tool.check_no_legacy_telys_secret_names({"CODNA_TELYS_LICENSE_JWT"}, set())

    assert bad.status == "fail"
    assert bad.details == {
        "forbidden_secret_names": {
            "environment": ["CODNA_TELYS_RUNTIME_DOWNLOAD_TOKEN"],
            "repository": ["CODNA_TELYS_API_KEY"],
        }
    }
    assert good.status == "pass"


def test_environment_check_requires_pypi():
    tool = _load_script()

    assert tool.check_environment_names(set()).status == "fail"
    assert tool.check_environment_names({"pypi"}).status == "pass"


def test_environment_protection_warns_when_pypi_has_no_rules():
    tool = _load_script()

    check = tool.check_environment_protection(
        {"environments": [{"name": "pypi", "protection_rules": []}]}
    )

    assert check.status == "warn"
    assert "no protection rules" in check.message
    assert check.details == {"rule_count": 0}


def test_environment_protection_passes_when_pypi_has_rules():
    tool = _load_script()

    check = tool.check_environment_protection(
        {"environments": [{"name": "pypi", "protection_rules": [{"type": "required_reviewers"}]}]}
    )

    assert check.status == "pass"
    assert check.details == {"rule_count": 1}


def test_environment_protection_passes_with_protected_branch_policy():
    tool = _load_script()

    check = tool.check_environment_protection(
        {
            "environments": [
                {
                    "name": "pypi",
                    "protection_rules": [],
                    "deployment_branch_policy": {
                        "protected_branches": True,
                        "custom_branch_policies": False,
                    },
                }
            ]
        }
    )

    assert check.status == "pass"
    assert "branch restrictions" in check.message
    assert check.details == {
        "rule_count": 0,
        "deployment_branch_policy": {
            "protected_branches": True,
            "custom_branch_policies": False,
        },
    }


def test_environment_protection_passes_with_custom_branch_policy():
    tool = _load_script()

    check = tool.check_environment_protection(
        {
            "environments": [
                {
                    "name": "pypi",
                    "protection_rules": [],
                    "deployment_branch_policy": {
                        "protected_branches": False,
                        "custom_branch_policies": True,
                    },
                }
            ]
        }
    )

    assert check.status == "pass"
    assert check.details["deployment_branch_policy"]["custom_branch_policies"] is True


def test_environment_protection_skips_when_pypi_missing():
    tool = _load_script()

    check = tool.check_environment_protection({"environments": [{"name": "staging"}]})

    assert check.status == "skip"
    assert "not found" in check.message


def test_deployment_branch_policy_requires_main():
    tool = _load_script()

    names = tool.parse_deployment_branch_policy_names(
        {"branch_policies": [{"name": "main", "type": "branch"}, {"name": "release/*", "type": "branch"}]}
    )
    check = tool.check_deployment_branch_policy_names(names)

    assert names == {"main", "release/*"}
    assert check.status == "pass"
    assert check.details == {"required_branch": "main", "configured": ["main", "release/*"]}


def test_deployment_branch_policy_fails_without_main():
    tool = _load_script()

    check = tool.check_deployment_branch_policy_names({"release/*"})

    assert check.status == "fail"
    assert "must allow deployments from 'main'" in check.message
    assert check.details == {"required_branch": "main", "configured": ["release/*"]}


def test_missing_deployment_branch_policy_fails_cleanly():
    tool = _load_script()

    check = tool.check_missing_deployment_branch_policy()

    assert check.status == "fail"
    assert "must restrict deployments" in check.message
    assert check.details == {"required_branch": "main", "configured": []}


def test_deployment_branch_policy_payload_requires_branch_policies_list():
    tool = _load_script()

    try:
        tool.parse_deployment_branch_policy_names({"branch_policies": None})
    except tool.ReleaseReadinessError as exc:
        assert "branch_policies[]" in str(exc)
    else:
        raise AssertionError("expected ReleaseReadinessError")


def test_github_app_settings_fail_on_missing_events_and_permissions():
    tool = _load_script()
    payload = {
        "slug": "codna-ai",
        "events": ["check_suite", "issues", "pull_request_review"],
        "permissions": {
            "checks": "write",
            "contents": "write",
            "issues": "write",
            "metadata": "read",
            "pull_requests": "write",
        },
    }

    check = tool.check_github_app_settings(payload, app_slug="codna-ai")

    assert check.status == "fail"
    assert check.details["missing_events"] == [
        "code_scanning_alert",
        "issue_comment",
        "pull_request",
        "pull_request_review_comment",
    ]
    assert check.details["missing_or_insufficient_permissions"] == {
        "security_events": {"required": "read", "actual": None}
    }


def test_github_app_settings_pass_when_events_and_permissions_are_complete():
    tool = _load_script()
    payload = {
        "slug": "codna-ai",
        "events": sorted(tool.REQUIRED_GITHUB_APP_EVENTS),
        "permissions": {
            "checks": "write",
            "contents": "write",
            "issues": "write",
            "metadata": "read",
            "pull_requests": "write",
            "security_events": "read",
        },
    }

    check = tool.check_github_app_settings(payload, app_slug="codna-ai")

    assert check.status == "pass"
    assert check.details["missing_events"] == []
    assert check.details["missing_or_insufficient_permissions"] == {}


def test_github_app_permission_write_satisfies_read():
    tool = _load_script()
    payload = {
        "events": sorted(tool.REQUIRED_GITHUB_APP_EVENTS),
        "permissions": {**tool.REQUIRED_GITHUB_APP_PERMISSIONS, "security_events": "write"},
    }

    check = tool.check_github_app_settings(payload, app_slug="codna-ai")

    assert check.status == "pass"


def test_v1_alignment_requires_tag_to_match_main():
    tool = _load_script()

    mismatch = tool.check_v1_alignment({"refs/heads/main": "a", "refs/tags/v1": "b"})
    pre_publish = tool.check_v1_alignment(
        {"refs/heads/main": "a", "refs/tags/v1": "b"},
        source_version="0.1.30",
        public_version="0.1.29",
    )
    match = tool.check_v1_alignment({"refs/heads/main": "a", "refs/tags/v1": "a"})

    assert mismatch.status == "fail"
    assert pre_publish.status == "skip"
    assert match.status == "pass"


def test_workflow_checks_require_green_main_runs():
    tool = _load_script()
    runs = [
        {"workflowName": "ci", "headSha": "abc", "status": "completed", "conclusion": "success"},
        {"workflowName": "codeql", "headSha": "abc", "status": "completed", "conclusion": "success"},
        {"workflowName": "CodeQL", "headSha": "abc", "status": "completed", "conclusion": "success"},
        {
            "workflowName": "publish cli",
            "headSha": "abc",
            "status": "completed",
            "conclusion": "success",
            "event": "workflow_dispatch",
            "databaseId": 123,
            "url": "https://github.example/runs/123",
        },
    ]
    run_details = {
        123: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "verify PyPI publish auth", "conclusion": "success"},
                {"name": "publish codna to PyPI", "conclusion": "skipped"},
            ]
        }
    }

    assert tool.check_required_workflows(runs, head_sha="abc").status == "pass"
    assert tool.check_safe_publish_workflow(runs, head_sha="abc", run_details_by_id=run_details).status == "pass"
    assert tool.check_required_workflows(runs[:-1], head_sha="abc").status == "pass"

    docs_audit = {
        "workflowName": "docs-audit",
        "headSha": "abc",
        "status": "completed",
        "conclusion": "success",
    }
    assert tool.check_required_workflows([*runs, docs_audit], head_sha="abc").status == "pass"
    missing_publish = tool.check_safe_publish_workflow(
        runs[:-1],
        head_sha="abc",
        repo="thyn-ai/codna",
        run_details_by_id={},
    )

    assert missing_publish.status == "fail"
    assert missing_publish.details["remediation"]["pypi_expected_publisher"] == {
        "project_name": "codna",
        "workflow_filename": "publish-cli.yml",
        "environment_name": "pypi",
        "owner": "thyn-ai",
        "repository_name": "codna",
    }
    assert (
        missing_publish.details["remediation"]["pypi_pending_publishers_url"]
        == "https://pypi.org/manage/account/publishing/"
    )
    assert missing_publish.details["remediation"]["safe_rerun_command"] == (
        "gh workflow run publish-cli.yml --repo thyn-ai/codna --ref main "
        "-f publish=false -f verify_pypi_oidc=true"
    )


def test_safe_publish_requires_publish_auth_job_success_and_upload_skipped():
    tool = _load_script()
    runs = [
        {
            "workflowName": "publish cli",
            "headSha": "abc",
            "status": "completed",
            "conclusion": "success",
            "event": "workflow_dispatch",
            "databaseId": 123,
        }
    ]
    skipped_oidc = {
        123: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "verify PyPI publish auth", "conclusion": "skipped"},
                {"name": "publish codna to PyPI", "conclusion": "skipped"},
            ]
        }
    }
    uploaded = {
        123: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "verify PyPI publish auth", "conclusion": "success"},
                {"name": "publish codna to PyPI", "conclusion": "success"},
            ]
        }
    }

    skipped_check = tool.check_safe_publish_workflow(runs, head_sha="abc", run_details_by_id=skipped_oidc)
    uploaded_check = tool.check_safe_publish_workflow(runs, head_sha="abc", run_details_by_id=uploaded)

    assert skipped_check.status == "fail"
    assert skipped_check.details["required_jobs"]["verify PyPI publish auth"] == "success"
    assert skipped_check.details["remediation"]["pypi_expected_publisher"] == {
        "project_name": "codna",
        "workflow_filename": "publish-cli.yml",
        "environment_name": "pypi",
    }
    assert (
        skipped_check.details["remediation"]["pypi_pending_publishers_url"]
        == "https://pypi.org/manage/account/publishing/"
    )
    assert uploaded_check.status == "fail"
    assert uploaded_check.details["source_version"] is None
    assert uploaded_check.details["public_version"] is None
    assert "without PyPI publish-auth verification" in skipped_check.message
    assert "PyPI does not report the source version" in uploaded_check.message


def test_safe_publish_accepts_real_publish_when_pypi_reports_source_version():
    tool = _load_script()
    runs = [
        {
            "workflowName": "publish cli",
            "headSha": "abc",
            "status": "completed",
            "conclusion": "success",
            "event": "workflow_dispatch",
            "databaseId": 123,
            "url": "https://github.example/runs/123",
        }
    ]
    run_details = {
        123: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "verify PyPI trusted publisher OIDC", "conclusion": "skipped"},
                {"name": "publish codna to PyPI", "conclusion": "success"},
            ]
        }
    }

    check = tool.check_safe_publish_workflow(
        runs,
        head_sha="abc",
        run_details_by_id=run_details,
        pypi_token_secret_scopes=["environment"],
        source_version="0.1.34",
        public_version="0.1.34",
    )

    assert check.status == "pass"
    assert "publish=true" in check.message
    assert check.details["source_version"] == "0.1.34"
    assert check.details["public_version"] == "0.1.34"
    assert check.details["pypi_token_secret_scopes"] == ["environment"]


def test_safe_publish_real_publish_fails_until_pypi_reports_source_version():
    tool = _load_script()
    runs = [
        {
            "workflowName": "publish cli",
            "headSha": "abc",
            "status": "completed",
            "conclusion": "success",
            "event": "workflow_dispatch",
            "databaseId": 123,
        }
    ]
    run_details = {
        123: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "publish codna to PyPI", "conclusion": "success"},
            ]
        }
    }

    check = tool.check_safe_publish_workflow(
        runs,
        head_sha="abc",
        run_details_by_id=run_details,
        source_version="0.1.34",
        public_version="0.1.33",
    )

    assert check.status == "fail"
    assert "PyPI does not report the source version" in check.message
    assert check.details["source_version"] == "0.1.34"
    assert check.details["public_version"] == "0.1.33"


def test_safe_publish_reports_completed_publish_auth_failure_precisely():
    tool = _load_script()
    runs = [
        {
            "workflowName": "publish cli",
            "headSha": "abc",
            "status": "completed",
            "conclusion": "failure",
            "event": "workflow_dispatch",
            "databaseId": 123,
        }
    ]
    run_details = {
        123: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "verify PyPI publish auth", "conclusion": "failure"},
                {"name": "publish codna to PyPI", "conclusion": "skipped"},
            ]
        }
    }

    check = tool.check_safe_publish_workflow(runs, head_sha="abc", run_details_by_id=run_details)

    assert check.status == "fail"
    assert check.message == (
        "latest origin/main publish=false run completed but PyPI publish-auth verification failed"
    )
    assert check.details["inspected"] == [
        {
            "run_id": 123,
            "url": None,
            "jobs": {
                "build and verify codna distribution": "success",
                "verify PyPI publish auth": "failure",
                "publish codna to PyPI": "skipped",
            },
        }
    ]



def test_safe_publish_oidc_failure_warns_when_token_fallback_is_configured():
    tool = _load_script()
    runs = [
        {
            "workflowName": "publish cli",
            "headSha": "abc",
            "status": "completed",
            "conclusion": "failure",
            "event": "workflow_dispatch",
            "databaseId": 123,
        }
    ]
    run_details = {
        123: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "verify PyPI trusted publisher OIDC", "conclusion": "failure"},
                {"name": "publish codna to PyPI", "conclusion": "skipped"},
            ]
        }
    }

    check = tool.check_safe_publish_workflow(
        runs,
        head_sha="abc",
        run_details_by_id=run_details,
        pypi_token_secret_scopes=["repository"],
    )

    assert check.status == "warn"
    assert "PYPI_TOKEN fallback is configured" in check.message
    assert check.details["pypi_token_secret_scopes"] == ["repository"]


def test_safe_publish_uses_latest_dispatch_not_any_older_success():
    tool = _load_script()
    runs = [
        {
            "workflowName": "publish cli",
            "headSha": "abc",
            "status": "completed",
            "conclusion": "success",
            "event": "workflow_dispatch",
            "databaseId": 124,
        },
        {
            "workflowName": "publish cli",
            "headSha": "abc",
            "status": "completed",
            "conclusion": "success",
            "event": "workflow_dispatch",
            "databaseId": 123,
        },
    ]
    run_details = {
        124: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "verify PyPI publish auth", "conclusion": "skipped"},
                {"name": "publish codna to PyPI", "conclusion": "skipped"},
            ]
        },
        123: {
            "jobs": [
                {"name": "build and verify codna distribution", "conclusion": "success"},
                {"name": "verify PyPI publish auth", "conclusion": "success"},
                {"name": "publish codna to PyPI", "conclusion": "skipped"},
            ]
        },
    }

    check = tool.check_safe_publish_workflow(runs, head_sha="abc", run_details_by_id=run_details)

    assert check.status == "fail"
    assert check.details["latest_run"]["run_id"] == 124


def test_release_remediation_omits_rerun_command_for_unsafe_repo_name():
    tool = _load_script()

    remediation = tool.pypi_trusted_publisher_remediation("thyn-ai/codna;echo bad")

    assert remediation == {
        "reason": "PyPI trusted publishing must mint successfully before publish=true is allowed.",
        "pypi_pending_publishers_url": "https://pypi.org/manage/account/publishing/",
        "pypi_expected_publisher": {
            "project_name": "codna",
            "workflow_filename": "publish-cli.yml",
            "environment_name": "pypi",
        },
    }


def test_render_text_report_surfaces_safe_remediation_without_secret_values():
    tool = _load_script()
    report = {
        "ok": False,
        "summary": {"checks": 4, "failed": 1, "warnings": 1},
        "checks": [
            {
                "name": "github_secret",
                "status": "pass",
                "message": "CODNA_TELYS_LICENSE_JWT is configured",
                "details": {"scopes": ["repository"]},
            },
            {
                "name": "safe_publish",
                "status": "fail",
                "message": "latest origin/main publish=false run did not complete a no-upload PyPI publish-auth verification",
                "details": {
                    "latest_run": {
                        "run_id": 28601495113,
                        "url": "https://github.example/runs/28601495113",
                    },
                    "remediation": tool.pypi_trusted_publisher_remediation("thyn-ai/codna"),
                },
            },
            {
                "name": "pypi_project",
                "status": "warn",
                "message": "PyPI project codna is not visible yet",
                "details": None,
            },
            {
                "name": "bundle_download",
                "status": "skip",
                "message": "actual Telys bundle download check was not requested",
                "details": {"token": "secret-value"},
            },
        ],
    }

    output = tool.render_text_report(report)

    assert "Codna release readiness: FAIL" in output
    assert "checks=4 failed=1 warnings=1" in output
    assert "latest_run: 28601495113" in output
    assert "pypi_pending_publishers_url: https://pypi.org/manage/account/publishing/" in output
    assert "project_name: codna" in output
    assert "owner: thyn-ai" in output
    assert "repository_name: codna" in output
    assert "workflow_filename: publish-cli.yml" in output
    assert "environment_name: pypi" in output
    assert (
        "safe_rerun_command: gh workflow run publish-cli.yml --repo thyn-ai/codna --ref main "
        "-f publish=false -f verify_pypi_oidc=true"
    ) in output
    assert "publish=true is blocked" in output
    assert "secret-value" not in output


def test_render_text_report_surfaces_github_app_setting_gaps():
    tool = _load_script()
    report = {
        "ok": False,
        "summary": {"checks": 1, "failed": 1, "warnings": 0},
        "checks": [
            {
                "name": "github_app_settings",
                "status": "fail",
                "message": "GitHub App is missing required event subscriptions or repository permissions",
                "details": {
                    "missing_events": ["issue_comment", "pull_request"],
                    "missing_or_insufficient_permissions": {
                        "security_events": {"required": "read", "actual": None}
                    },
                },
            }
        ],
    }

    output = tool.render_text_report(report)

    assert "missing_events: issue_comment, pull_request" in output
    assert "security_events: required=read actual=missing" in output


def test_parse_args_keeps_json_as_default_and_accepts_text_format():
    tool = _load_script()

    assert tool.parse_args([]).format == "json"
    assert tool.parse_args(["--format", "text"]).format == "text"
    parsed = tool.parse_args(["--check-github-app", "--github-app-slug", "codna-ai-dev"])
    assert parsed.check_github_app is True
    assert parsed.github_app_slug == "codna-ai-dev"


def test_pypi_404_is_warning_not_hard_failure():
    tool = _load_script()

    check = tool.check_pypi_status(404)

    assert check.status == "warn"
    assert "pending trusted publisher" in check.message
    assert "not reserved until the first successful publish" in check.message


def test_local_telys_license_token_check_does_not_print_value():
    tool = _load_script()

    check = tool.check_bundle_download_token({"CODNA_TELYS_OEM_DOWNLOAD_TOKEN": "secret-value"})

    assert check.status == "pass"
    assert "secret-value" not in repr(check)


def test_local_pypi_token_env_fails_without_value_leak():
    tool = _load_script()

    check = tool.check_no_local_pypi_token_env({"TWINE_PASSWORD": "pypi" + "-secret-value"})
    allowed = tool.check_no_local_pypi_token_env({"PYPI_TOKEN": "pypi" + "-secret-value"})

    assert check.status == "fail"
    assert "TWINE_PASSWORD" in repr(check)
    assert "pypi" + "-secret-value" not in repr(check)
    assert allowed.status == "pass"


def test_local_legacy_telys_token_env_fails_without_value_leak():
    tool = _load_script()

    check = tool.check_no_local_legacy_telys_token_env(
        {"CODNA_TELYS_RUNTIME_DOWNLOAD_TOKEN": "legacy-secret-value"}
    )

    assert check.status == "fail"
    assert "CODNA_TELYS_RUNTIME_DOWNLOAD_TOKEN" in repr(check)
    assert "legacy-secret-value" not in repr(check)
