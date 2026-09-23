"""Contracts over the repository's own GitHub Actions configuration.

Every expectation is derived from the files under .github/ (job lists, workflow names), never
from a hardcoded count, so adding a workflow cannot silently escape a rule.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW_FILES = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _jobs(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), path
    return doc["jobs"]


def test_workflows_are_present():
    assert WORKFLOW_FILES, "no workflow files found under .github/workflows"


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_no_workflow_reads_the_retired_fleet_variable(path):
    # `fromJSON(vars.RUNNER_LABELS)` fails at workflow PARSE time (zero jobs, no logs) once the
    # org variable is gone; every job runs on GitHub-hosted runners now.
    assert "fromJSON(vars.RUNNER_LABELS)" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_job_declares_a_timeout(path):
    # GitHub's default is 360 minutes per job, billed on hosted runners. A job that calls a
    # reusable workflow (`uses:`) may not set timeout-minutes itself; the callee's jobs carry it.
    missing = [
        job for job, spec in _jobs(path).items()
        if "timeout-minutes" not in spec and "uses" not in spec
    ]
    assert missing == [], f"{path.name}: jobs without timeout-minutes: {missing}"


def test_security_workflow_pins_the_toolchain_by_sha_with_a_version_comment():
    text = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
    uses = [line.strip() for line in text.splitlines() if line.strip().startswith("uses: thyn-ai/security-toolchain/")]
    assert len(uses) == 1, uses
    assert re.fullmatch(
        r"uses: thyn-ai/security-toolchain/\.github/workflows/security-full\.yml@[0-9a-f]{40} # v\d+\.\d+\.\d+",
        uses[0],
    ), uses[0]


def test_dependabot_leaves_the_toolchain_pin_to_propagate():
    doc = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    actions = [
        u
        for u in doc["updates"]
        if u["package-ecosystem"] == "github-actions" and u.get("directory") == "/"
    ]
    assert actions, "no github-actions block covers .github/workflows"
    ignored = [entry["dependency-name"] for block in actions for entry in block.get("ignore", [])]
    # Dependabot names a reusable workflow by its full path, so only a wildcard can match it.
    assert any(name.startswith("thyn-ai/security-toolchain") and name.endswith("*") for name in ignored), ignored


# --- webhook-ops.yml: the rollout's operator surface --------------------------------------------------

WEBHOOK_OPS = WORKFLOWS / "webhook-ops.yml"
ACTION_LITERAL = re.compile(r"inputs\.action\s*==\s*'([a-z-]+)'")
# A rollout flag is a switch, never a credential: none of these words may appear in an allowed name.
CREDENTIAL_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "DATABASE_URL", "PRIVATE", "CREDENTIAL")


def _webhook_ops() -> dict:
    doc = yaml.safe_load(WEBHOOK_OPS.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _ops_steps() -> list[dict]:
    return _webhook_ops()["jobs"]["run"]["steps"]


def _ops_step(action: str) -> dict:
    hits = [s for s in _ops_steps() if action in ACTION_LITERAL.findall(str(s.get("if", ""))) and "run" in s]
    assert len(hits) == 1, (action, [s.get("name") for s in hits])
    return hits[0]


def test_webhook_ops_action_choices_and_steps_agree():
    doc = _webhook_ops()
    trigger = doc.get("on", doc.get(True))  # PyYAML reads the bare key `on` as the boolean True
    choices = trigger["workflow_dispatch"]["inputs"]["action"]["options"]
    assert len(set(choices)) == len(choices), "duplicate action choice"
    referenced = set(ACTION_LITERAL.findall(WEBHOOK_OPS.read_text(encoding="utf-8")))
    assert set(choices) == referenced, {"choice without a step": set(choices) - referenced,
                                        "step without a choice": referenced - set(choices)}


def test_webhook_ops_flag_allow_list_names_only_switches_the_service_reads():
    names = _webhook_ops()["jobs"]["run"]["env"]["FLAG_ALLOW_LIST"].split()
    assert names and len(set(names)) == len(names)
    assert [n for n in names if any(m in n for m in CREDENTIAL_MARKERS)] == []
    codna = ROOT / "cli" / "codna"
    source = "".join(p.read_text(encoding="utf-8") for p in sorted(codna.glob("*.py")))
    unread = [n for n in names if f'"{n}"' not in source and f"'{n}'" not in source]
    assert unread == [], f"allow-listed flags nothing in cli/codna reads: {unread}"
    # the step validates against THAT list, not a copy of it
    run = _ops_step("set-flags")["run"]
    assert "$FLAG_ALLOW_LIST" in run
    assert not any(n in run.split("case \"$name\" in")[0].split("allowed=")[0] for n in names), \
        "the allow-list must not be duplicated inside the step"


def test_webhook_ops_set_flags_validates_every_pair_before_it_writes():
    run = _ops_step("set-flags")["run"]
    for needle in ("is not an allow-listed rollout flag", "is not NAME=VALUE", "the value must match"):
        assert needle in run
    assert run.index("the value must match") < run.index("flyctl secrets set"), "validation precedes the write"
    assert run.index("flyctl secrets set") < run.index("GITHUB_STEP_SUMMARY")
    assert "redeploys every machine" in run, "the run summary must say Fly redeploys on a secrets change"
    # every value pattern is anchored and closed: nothing can smuggle a second NAME=VALUE through
    for pattern in re.findall(r"pattern='([^']+)'", run):
        assert pattern.startswith("^") and pattern.endswith("$"), pattern


def test_webhook_ops_secret_minting_steps_mask_first_and_never_echo_a_value():
    text = WEBHOOK_OPS.read_text(encoding="utf-8")
    assert "set -x" not in text
    for action in ("rotate-ops-token", "worker-token", "worker-secrets"):
        run = _ops_step(action)["run"]
        assert "::add-mask::" in run, action
        assert run.index("::add-mask::") < run.index("flyctl secrets set"), f"{action}: mask before use"
        secret_var = r"\$\{?(token|value|b64|out|line)\b"
        for line in run.splitlines():
            if "::add-mask::" in line or not re.search(secret_var, line):
                continue
            # an echo whose ARGUMENT names one of these variables would print a secret
            assert not re.search(r"\becho\b[^;|&]*" + secret_var, line), f"{action}: echoes a secret-bearing variable: {line.strip()}"
            if re.search(r"\bprintf\b[^;|&]*" + secret_var, line):
                assert "|" in line, f"{action}: printf of a secret-bearing variable must feed a pipe, not the log: {line.strip()}"


def test_webhook_ops_worker_secrets_refuses_a_partial_set_and_copies_exactly_the_runbook_names():
    run = _ops_step("worker-secrets")["run"]
    runbook = (ROOT / "infra" / "README-webhook-deploy.md").read_text(encoding="utf-8")
    names = re.search(r'names="([^"]+)"', run).group(1).split()
    assert set(names) == {"GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY", "CODNA_ENGINE_URL", "CODNA_WEBHOOK_INTERNAL_SECRET"}
    assert all(f"`{n}`" in runbook for n in names)
    for forbidden in ("CODNA_WEBHOOK_FLY_TOKEN", "FLY_API_TOKEN", "CODNA_GITHUB_WEBHOOK_SECRET", "CODNA_WEBHOOK_OPS_TOKEN"):
        assert forbidden not in names
    assert "refusing to configure" in run and run.index("refusing to configure") < run.index("flyctl secrets set")


def test_webhook_ops_worker_token_sets_the_name_the_scaler_reads_first():
    from codna.webhook_scaler import FLY_TOKEN_ENVS

    run = _ops_step("worker-token")["run"]
    assert FLY_TOKEN_ENVS[0] == "CODNA_WEBHOOK_FLY_TOKEN"
    assert 'flyctl secrets set -a "$APP" "CODNA_WEBHOOK_FLY_TOKEN=$token"' in run
    assert "FLY_API_TOKEN=$token" not in run                    # the flyctl login name is only a fallback the service reads
    runbook = (ROOT / "infra" / "README-webhook-deploy.md").read_text(encoding="utf-8")
    assert "CODNA_WEBHOOK_FLY_TOKEN=$(fly tokens create deploy -a codna-webhook-worker)" in runbook


def test_webhook_ops_mpg_steps_redact_the_connection_string_flyctl_prints():
    for action in ("mpg-create", "mpg-attach"):
        run = _ops_step(action)["run"].replace("\\\n", " ")  # join shell continuation lines
        redact = [line for line in run.splitlines() if re.match(r"\s*flyctl mpg (create|attach)\b", line)]
        assert redact and all("redacted" in line for line in redact), (action, redact)
        assert "--yes" not in run, f"{action}: flyctl mpg create/attach have no --yes flag (cobra rejects unknown flags)"


def test_webhook_ops_actions_that_read_repo_files_check_the_repository_out_first():
    steps = _ops_steps()
    [checkout] = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert checkout["with"]["persist-credentials"] is False
    covered = set(ACTION_LITERAL.findall(str(checkout["if"])))
    needs_repo = {
        action
        for s in steps if "run" in s
        for action in ACTION_LITERAL.findall(str(s.get("if", "")))
        if "infra/" in str(s.get("env", "")) or "infra/" in s["run"]
    }
    assert needs_repo and needs_repo <= covered, {"needs checkout": needs_repo, "checked out": covered}


def test_webhook_ops_worker_pool_takes_its_environment_from_the_worker_toml():
    run = _ops_step("worker-pool-reconcile")["run"]
    assert "machine create" in run and "--skip-launch" not in run, "create = created stopped; run has no --skip-launch"
    assert 'tomllib.load(fh)["env"]' in run and '"${envflags[@]}"' in run
    assert "CODNA_WEBHOOK_ROLE=worker" not in run, "no hand-copied env: the toml is the single source"
    # volumes are attached by id, an unattached one in the region is reused before any is created, and
    # the machine create retries the registry's post-push "failed to get manifest" window
    assert '--volume "$vol:/data"' in run and "codna_webhook_scratch:/data" not in run

    def before(first: str, second: str) -> None:
        assert first in run and second in run, (first, second)
        assert run.index(first) < run.index(second), f"{first!r} must come before {second!r}"

    before("reusing unattached volume", "flyctl volumes create")
    before("for attempt in 1 2 3", "flyctl machine create")
    # volumes handed out earlier in the same run are excluded from reuse; a create whose exit status
    # was lost is recognised by machine NAME before any retry
    assert 'used_vols="$used_vols$vol "' in run and "--arg used" in run
    before("flyctl machine create", 'exists "worker-$i"')
    before('exists "worker-$i"', "retrying in 20 s")


def test_webhook_ops_keeps_the_default_token_read_only():
    doc = _webhook_ops()
    assert doc["permissions"] == {}
    # exactly what actions/checkout needs on a private repository, and nothing that can write
    assert doc["jobs"]["run"]["permissions"] == {"contents": "read"}
    # only the release PAT is ever handed to a step, and only to the one that writes a repo secret
    for step in _ops_steps():
        for value in (step.get("env") or {}).values():
            if "secrets." in str(value):
                assert "RELEASE_PLEASE_TOKEN" in str(value) and "rotate-ops-token" in str(step.get("if")), step["name"]


# --- webhook-slo.yml: an absent gauge is 0, not an alert ------------------------------------------------

def _slo_decide_script() -> str:
    doc = yaml.safe_load((WORKFLOWS / "webhook-slo.yml").read_text(encoding="utf-8"))
    [step] = [s for s in doc["jobs"]["evaluate"]["steps"] if s.get("id") == "decide"]
    return step["run"]


def _run_decide(tmp_path, **env):
    import os
    import subprocess

    out = tmp_path / "out.txt"
    out.write_text("")
    base = dict(VERDICT="not_measured", P95="", SAMPLES="0", BACKEND="sqlite", PG_UP="0", OLDEST="",
                DEAD="", CEILING="", SCALER_ERRORS="", SPOOL="")
    base.update(env)
    proc = subprocess.run(["bash", "-c", _slo_decide_script()], env={**os.environ, **base, "GITHUB_OUTPUT": str(out)},
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return out.read_text()


def test_webhook_slo_absent_gauges_do_not_alert(tmp_path):
    # On the sqlite backend dead_total / scaler_errors_total are not rendered at all: the gauge
    # reader yields "" and that must read as 0 (it alerted as "dead-lettered job(s)" on 2026-09-20).
    assert "alert=false" in _run_decide(tmp_path)


def test_webhook_slo_present_conditions_still_alert(tmp_path):
    body = _run_decide(tmp_path, DEAD="2", SCALER_ERRORS="1")
    assert "alert=true" in body and "2 dead-lettered job(s)" in body and "scaler errors: 1" in body
    body = _run_decide(tmp_path, BACKEND="postgres", PG_UP="0", SPOOL="3")
    assert "alert=true" in body and "pg_up 0" in body and "spooling 3" in body


def test_webhook_slo_oldest_runnable_alerts_only_where_postgres_rows_are_claimed(tmp_path):
    # On shadow every Postgres row is a twin nobody claims (SQLite runs the job): its age climbs
    # forever and is not a backlog. Measured 2026-09-20: 800 s and rising during a healthy shadow.
    assert "alert=false" in _run_decide(tmp_path, BACKEND="shadow", PG_UP="1", OLDEST="900")
    body = _run_decide(tmp_path, BACKEND="postgres", PG_UP="1", OLDEST="900")
    assert "alert=true" in body and "has waited 900s" in body


def test_webhook_slo_gauge_reader_parses_every_metric_shape(tmp_path):
    """The Read step's `gauge` helper, run for real against a /metrics body: a bare gauge, a labelled
    one (max over its series), and an absent one. Its ERE carried a bare `{` -- a repetition operator
    -- so grep failed on every read and pg_up came back "" (read as 0) five evaluations in a row on
    2026-09-20 against a healthy database."""
    import subprocess

    doc = yaml.safe_load((WORKFLOWS / "webhook-slo.yml").read_text(encoding="utf-8"))
    [read] = [s for s in doc["jobs"]["evaluate"]["steps"] if s.get("id") == "read"]
    [gauge_def] = [line.strip() for line in read["run"].splitlines() if line.strip().startswith("gauge()")]
    (tmp_path / "metrics.txt").write_text(
        "# TYPE codna_webhook_pg_up gauge\ncodna_webhook_pg_up 1\n"
        "codna_webhook_queue_depth{kind=\"review\",installation=\"1\"} 3\n"
        "codna_webhook_queue_depth{kind=\"fix\",installation=\"2\"} 7\n"
        "codna_webhook_dead_total 0\n"
        "codna_webhook_oldest_runnable_age_seconds{kind=\"review\"} 622.6\n", encoding="utf-8")
    script = gauge_def + "\nfor g in pg_up queue_depth dead_total oldest_runnable_age_seconds scaler_at_ceiling; do printf '%s=%s\\n' \"$g\" \"$(gauge $g)\"; done"
    proc = subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert proc.returncode == 0 and proc.stderr == "", proc.stderr
    assert proc.stdout.splitlines() == ["pg_up=1", "queue_depth=7", "dead_total=0",
                                        "oldest_runnable_age_seconds=622.6", "scaler_at_ceiling="]


# --- webhook-ops.yml `ops`: the input validation, run for real against a stub flyctl -----------------

def _run_ops_step(tmp_path, **env):
    """Run the `ops` step's script with a stub flyctl that records its argv; returns (rc, argv, stderr)."""
    import os
    import stat
    import subprocess

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    record = tmp_path / "flyctl.argv"
    record.unlink(missing_ok=True)  # one run's argv must never leak into the next assertion
    stub = stub_dir / "flyctl"
    stub.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$FLYCTL_RECORD\"\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    base = dict(APP="codna-webhook", GITHUB_ACTOR="octocat", VERB="metrics", ARGS="")
    base.update(env)
    proc = subprocess.run(["bash", "-c", _ops_step("ops")["run"]],
                          env={**os.environ, **base, "PATH": f"{stub_dir}:{os.environ['PATH']}", "FLYCTL_RECORD": str(record)},
                          capture_output=True, text=True, check=False)
    argv = record.read_text().splitlines() if record.exists() else []
    return proc.returncode, argv, proc.stdout + proc.stderr


def test_webhook_ops_verb_without_args_reaches_the_machine(tmp_path):
    # `ops metrics` and `ops slo` take no flags: an empty args input must pass the validation
    # (it did not on 2026-09-20 -- printf '%s' of "" gives grep no line to match).
    rc, argv, out = _run_ops_step(tmp_path)
    assert rc == 0, out
    assert argv == ["ssh", "console", "-a", "codna-webhook", "-C", "codna webhook ops metrics --actor github-actions:octocat"]


def test_webhook_ops_args_are_flag_value_pairs_from_the_option_list_only(tmp_path):
    rc, argv, _ = _run_ops_step(tmp_path, VERB="jobs", ARGS="--status dead --limit 5")
    assert rc == 0 and argv[-1] == "codna webhook ops jobs --actor github-actions:octocat --status dead --limit 5"
    for bad in ("--status dead; id", "--id 5 --unknown x", "metrics", "--id", "$(id)",
                # a newline anywhere: a line-wise check would pass these on the line that matches
                "\n$(id)", "--id 5\n$(id)", "$(id)\n--id 5", "--id 5\n"):
        rc, argv, out = _run_ops_step(tmp_path, VERB="jobs", ARGS=bad)
        assert rc != 0 and argv == [] and "args must be --<flag> <value> pairs" in out, repr(bad)
    for bad in ("Jobs;", "jobs\n$(id)", "\njobs", "jobs "):
        rc, argv, out = _run_ops_step(tmp_path, VERB=bad)
        assert rc != 0 and argv == [] and "unsafe verb" in out, repr(bad)
    rc, argv, out = _run_ops_step(tmp_path, GITHUB_ACTOR="octo\ncat")
    assert rc != 0 and argv == [] and "unsafe characters in GITHUB_ACTOR" in out


def test_webhook_ops_reconcile_volume_picker_runs_against_real_volume_shapes(tmp_path):
    """The jq program that picks a reusable scratch volume, executed with jq against fly's volume JSON:
    an unattached volume in the region is picked by id; an attached one, one in another region, and
    one handed out earlier in the same run are skipped; nothing matching yields no output. Run
    35503475440 died inside this program (`Cannot index string with string "id"`: `$used |
    contains(...)` rebinds `.` to the string)."""
    import json
    import shutil
    import subprocess

    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")
    run = _ops_step("worker-pool-reconcile")["run"]
    # `[^']*` already spans newlines: no alternation with `\n` (CodeQL py/redos, alert #187)
    m = re.search(r"jq -r --arg r \"\$region\" --arg used \"\$used_vols\" \\\n\s*'([^']*)'\)", run)
    assert m, "the volume picker jq program was not found"
    program = m.group(1)
    volumes = [
        {"id": "vol_attached", "name": "codna_webhook_scratch", "region": "iad", "attached_machine_id": "3287e0"},
        {"id": "vol_other_region", "name": "codna_webhook_scratch", "region": "ord", "attached_machine_id": None},
        {"id": "vol_used_this_run", "name": "codna_webhook_scratch", "region": "iad", "attached_machine_id": None},
        {"id": "vol_other_name", "name": "codna_webhook_data", "region": "iad", "attached_machine_id": None},
        {"id": "vol_free", "name": "codna_webhook_scratch", "region": "iad", "attached_machine_id": None},
    ]

    def pick(region: str, used: str) -> str:
        proc = subprocess.run(["jq", "-r", "--arg", "r", region, "--arg", "used", used, program],
                              input=json.dumps(volumes), capture_output=True, text=True, check=False)
        assert proc.returncode == 0 and proc.stderr == "", proc.stderr
        return proc.stdout.strip()

    assert pick("iad", " vol_used_this_run ") == "vol_free"
    assert pick("iad", " ") == "vol_used_this_run"          # first unattached in the region, by listing order
    assert pick("ord", " ") == "vol_other_region"
    assert pick("iad", " vol_used_this_run vol_free ") == ""  # everything in the region is spoken for
    assert pick("ams", " ") == ""


def _run_worker_secrets_step(tmp_path, remote_lines: str):
    """Run the worker-secrets step against a stub flyctl: `ssh console` prints `remote_lines` (what the
    in-machine script would print), `secrets set` records its argv. Returns (rc, argv, output)."""
    import os
    import stat
    import subprocess

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    record = tmp_path / "secrets.argv"
    record.unlink(missing_ok=True)
    (tmp_path / "remote.txt").write_text(remote_lines)
    stub = stub_dir / "flyctl"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$1\" in\n"
        "  ssh) cat \"$FLY_STUB_REMOTE\" ;;\n"
        "  secrets) shift; printf '%s\\n' \"$@\" > \"$FLY_STUB_RECORD\" ;;\n"
        "esac\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.run(
        ["bash", "-c", _ops_step("worker-secrets")["run"]],
        env={**os.environ, "APP": "codna-webhook", "WORKER_APP": "codna-webhook-worker",
             "PATH": f"{stub_dir}:{os.environ['PATH']}", "FLY_STUB_REMOTE": str(tmp_path / "remote.txt"),
             "FLY_STUB_RECORD": str(record)},
        capture_output=True, text=True, check=False)
    argv = record.read_text().split("\n") if record.exists() else []
    return proc.returncode, argv, proc.stdout + proc.stderr


def test_webhook_ops_worker_secrets_copies_every_value_masked_including_a_multiline_key(tmp_path):
    import base64

    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n-----END RSA PRIVATE KEY-----"
    values = {"GITHUB_APP_ID": "123456", "GITHUB_APP_PRIVATE_KEY": pem,
              "CODNA_ENGINE_URL": "https://engine.example", "CODNA_WEBHOOK_INTERNAL_SECRET": "s3cr3t-internal"}
    remote = "".join(f"SECRET {k} {base64.b64encode(v.encode()).decode()}\n" for k, v in values.items())
    rc, argv, out = _run_worker_secrets_step(tmp_path, remote)
    assert rc == 0, out
    # staged: the pool has no release to redeploy; each machine reads staged secrets at its next start
    assert argv[:4] == ["set", "-a", "codna-webhook-worker", "--stage"]
    assert "\n".join(argv[4:]).strip("\n") == "\n".join(f"{k}={v}" for k, v in values.items())  # multi-line value intact
    masked = [line.removeprefix("::add-mask::") for line in out.splitlines() if line.startswith("::add-mask::")]
    for v in ("123456", "https://engine.example", "s3cr3t-internal", *pem.splitlines()):
        assert v in masked, f"{v!r} was not masked before use"
    assert not any(v in line for v in values.values() for line in out.splitlines() if not line.startswith("::add-mask::"))


def test_webhook_ops_worker_secrets_refuses_a_partial_set_without_writing(tmp_path):
    import base64

    remote = ("SECRET GITHUB_APP_ID " + base64.b64encode(b"123456").decode() + "\n"
              "MISSING GITHUB_APP_PRIVATE_KEY\n"
              "SECRET CODNA_ENGINE_URL " + base64.b64encode(b"https://e").decode() + "\n"
              "SECRET CODNA_WEBHOOK_INTERNAL_SECRET " + base64.b64encode(b"x").decode() + "\n")
    rc, argv, out = _run_worker_secrets_step(tmp_path, remote)
    assert rc != 0 and argv == [] and "GITHUB_APP_PRIVATE_KEY is not set on codna-webhook" in out


def test_webhook_ops_flags_on_the_worker_pool_are_staged_and_on_the_ingress_deployed(tmp_path):
    """The pool's machines come from `fly machine create`, so the app has no release for `fly secrets
    set` to redeploy (run 35504691632). Flags for the worker app are staged; the ingress redeploys."""
    import os
    import stat
    import subprocess

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    record = tmp_path / "flyctl.argv"
    stub = stub_dir / "flyctl"
    stub.write_text("#!/usr/bin/env bash\n[ \"$1\" = secrets ] && printf '%s\\n' \"$@\" >> \"$FLYCTL_RECORD\"\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    allow = _webhook_ops()["jobs"]["run"]["env"]["FLAG_ALLOW_LIST"]

    def run(target: str, mode: str, flags: str) -> list[str]:
        record.unlink(missing_ok=True)
        summary = tmp_path / "summary.md"
        summary.write_text("")
        proc = subprocess.run(["bash", "-c", _ops_step("set-flags")["run"]],
                              env={**os.environ, "APP": "codna-webhook", "WORKER_APP": "codna-webhook-worker",
                                   "FLAG_ALLOW_LIST": allow, "TARGET": target, "MODE": mode, "FLAGS": flags,
                                   "PATH": f"{stub_dir}:{os.environ['PATH']}", "FLYCTL_RECORD": str(record),
                                   "GITHUB_STEP_SUMMARY": str(summary)},
                              capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return record.read_text().splitlines()

    assert run("codna-webhook-worker", "set", "CODNA_WEBHOOK_DRAIN_S=290") == ["secrets", "set", "-a", "codna-webhook-worker", "--stage", "CODNA_WEBHOOK_DRAIN_S=290"]
    assert run("codna-webhook-worker", "unset", "CODNA_WEBHOOK_DRAIN_S") == ["secrets", "unset", "-a", "codna-webhook-worker", "--stage", "CODNA_WEBHOOK_DRAIN_S"]
    assert run("codna-webhook", "set", "CODNA_WEBHOOK_QUEUE_BACKEND=postgres CODNA_WEBHOOK_ROLE=ingress") == \
        ["secrets", "set", "-a", "codna-webhook", "CODNA_WEBHOOK_QUEUE_BACKEND=postgres", "CODNA_WEBHOOK_ROLE=ingress"]


def _modern_bash() -> str:
    """The step uses `mapfile` (bash 4+, the runner's bash 5). macOS ships bash 3.2, so prefer a newer
    bash when one is installed and skip otherwise -- CI is the meaningful run."""
    import shutil
    import subprocess

    for candidate in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash", shutil.which("bash") or "bash"):
        try:
            major = subprocess.run([candidate, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            continue
        if major.isdigit() and int(major) >= 4:
            return candidate
    pytest.skip("no bash >= 4 available for mapfile (the ubuntu runner has bash 5)")


def _run_worker_start_stop_step(tmp_path, mode: str, count: str, machines: list[dict], fail_start: str = ""):
    """Run the worker-start/worker-stop step against a stub flyctl: `machine list --json` returns the
    fixture, every other `machine ...` call is recorded. Returns (rc, recorded argv lines, output)."""
    import json
    import os
    import stat
    import subprocess

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    record = tmp_path / "machine.argv"
    record.unlink(missing_ok=True)
    (tmp_path / "machines.json").write_text(json.dumps(machines))
    stub = stub_dir / "flyctl"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = machine ] && [ \"$2\" = list ]; then case \" $* \" in *\" --json \"*) cat \"$FLY_STUB_MACHINES\" ;; *) echo MACHINE-LIST-TABLE ;; esac; exit 0; fi\n"
        "printf '%s\\n' \"$*\" >> \"$FLY_STUB_RECORD\"\n"
        "[ \"$1 $2 $3\" = \"machine start ${FLY_STUB_FAIL_START:-}\" ] && { echo \"stub: $3 refused to start\" >&2; exit 1; }\n"
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.run([_modern_bash(), "-c", _ops_step("worker-start")["run"]],
                          env={**os.environ, "APP": "codna-webhook", "WORKER_APP": "codna-webhook-worker", "COUNT": count,
                               "MODE": mode, "PATH": f"{stub_dir}:{os.environ['PATH']}",
                               "FLY_STUB_MACHINES": str(tmp_path / "machines.json"), "FLY_STUB_RECORD": str(record),
                               "FLY_STUB_FAIL_START": fail_start},
                          capture_output=True, text=True, check=False)
    return proc.returncode, (record.read_text().splitlines() if record.exists() else []), proc.stdout + proc.stderr


_POOL = [
    {"id": "m2", "name": "worker-2", "state": "stopped", "region": "iad"},
    {"id": "m0", "name": "worker-0", "state": "stopped", "region": "iad"},
    {"id": "m3", "name": "worker-3", "state": "created", "region": "ord"},
    {"id": "m1", "name": "worker-1", "state": "stopped", "region": "ord"},
]


def test_webhook_ops_worker_start_updates_machines_so_staged_secrets_apply(tmp_path):
    # `machine update` pins the machine to the app's current secrets version and starts it; a plain
    # `machine start` reused the old (empty) secrets and both floor machines exited (run 35505335304)
    rc, argv, out = _run_worker_start_stop_step(tmp_path, "start", "2", _POOL)
    assert rc == 0, out
    # update (secrets version) THEN start: Fly leaves a non-started machine stopped after an update
    assert argv == ["machine update m0 -a codna-webhook-worker --yes", "machine start m0 -a codna-webhook-worker",
                    "machine update m1 -a codna-webhook-worker --yes", "machine start m1 -a codna-webhook-worker"]
    assert "MACHINE-LIST-TABLE" in out


def test_webhook_ops_worker_start_tries_every_machine_lists_the_pool_and_still_fails_when_one_refuses(tmp_path):
    # m0's start fails: m1 is still updated and started, the final listing is printed, the step fails
    # naming worker-0 -- no partially started pool left undiagnosed (set -e would have stopped at m0)
    rc, argv, out = _run_worker_start_stop_step(tmp_path, "start", "2", _POOL, fail_start="m0")
    assert rc != 0
    assert argv == ["machine update m0 -a codna-webhook-worker --yes", "machine start m0 -a codna-webhook-worker",
                    "machine update m1 -a codna-webhook-worker --yes", "machine start m1 -a codna-webhook-worker"]
    assert "MACHINE-LIST-TABLE" in out
    assert "::warning::worker-0 (m0, iad) did not start" in out and "::error::machines that did not start: worker-0" in out


def test_webhook_ops_worker_stop_takes_the_highest_names_first_and_keeps_the_floor(tmp_path):
    running = [dict(m, state="started") for m in _POOL]
    rc, argv, out = _run_worker_start_stop_step(tmp_path, "stop", "2", running)
    assert rc == 0, out
    assert argv == ["machine stop m3 -a codna-webhook-worker", "machine stop m2 -a codna-webhook-worker"]
    rc, argv, out = _run_worker_start_stop_step(tmp_path, "stop", "4", running)
    assert rc == 0 and argv == [] and "nothing to stop" in out


def test_webhook_ops_logs_action_reads_either_app():
    run = _ops_step("logs")["run"]
    assert 'flyctl logs -a "$TARGET"' in run and "codna-webhook|codna-webhook-worker" in run
