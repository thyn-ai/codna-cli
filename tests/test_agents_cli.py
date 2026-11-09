"""``codna agents`` and the bridge: the two surfaces a consumer actually calls.

The engine tests cover discovery. These cover the ANSWER a user or a host receives, which is where the
failure modes become visible or invisible:

  * a refused file must appear on stderr (a machine consumer that ignores the JSON must still see it);
  * ``--strict`` must turn a refusal into exit 1, because that is the CI contract;
  * the bridge must carry the ROSTER WITH BODIES (the engine cannot build a tool without them) and the
    refusals beside it.

Every case runs the real subprocess, not a mock: the bridge is the boundary between Python and
TypeScript, and a mock of it would assert the mock.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "agents_repo"
REPO_ROOT = Path(__file__).resolve().parents[1]


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``codna agents`` in a subprocess, with the checkout's ``cli/`` importable."""
    return subprocess.run(
        [sys.executable, "-m", "codna.cli", "agents", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/nonexistent"},
        timeout=60,
    )


def run_bridge(request: dict) -> tuple[int, dict]:
    """Feed one request to ``codna.agents_bridge``; return ``(exit code, parsed answer)``."""
    completed = subprocess.run(
        [sys.executable, "-m", "codna.agents_bridge"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/nonexistent"},
        timeout=60,
    )
    return completed.returncode, json.loads(completed.stdout)


# ---------------------------------------------------------------------------------------------
# codna agents
# ---------------------------------------------------------------------------------------------

def test_cli_lists_the_roster_and_its_profiles():
    result = run_cli(str(FIXTURE))
    assert result.returncode == 0, result.stderr
    assert "SUBAGENTS" in result.stdout
    assert "verifier" in result.stdout
    assert "planner" in result.stdout
    # The PROFILE is shown, not just the boolean: a user asked "is it read-only" and gets the engine's
    # own name for that restriction.
    assert "repository_review" in result.stdout
    assert "inherits session" in result.stdout


def test_cli_json_carries_the_profile_per_agent():
    result = run_cli(str(FIXTURE), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    by_name = {entry["name"]: entry for entry in payload["discovered"]}
    assert by_name["verifier"]["approval_profile"] == "repository_review"
    assert by_name["verifier"]["readonly"] is True
    assert by_name["planner"]["approval_profile"] is None
    assert by_name["planner"]["model"] == "claude-sonnet-5"


def test_cli_reports_a_shadowed_file_on_stderr():
    """A machine consumer reading JSON must STILL see the shadow report on stderr."""
    result = run_cli(str(FIXTURE), "--json")
    assert result.returncode == 0
    assert "agent_name_shadowed" in result.stderr
    assert ".claude/agents/verifier.md" in result.stderr


def test_cli_prints_one_agents_body():
    result = run_cli(str(FIXTURE), "--agent", "verifier")
    assert result.returncode == 0, result.stderr
    assert "You are a verifier." in result.stdout
    assert "profile: repository_review" in result.stdout


def test_cli_refuses_an_unknown_agent_name():
    result = run_cli(str(FIXTURE), "--agent", "nobody")
    assert result.returncode == 1
    payload = json.loads(result.stderr)
    assert payload["error"]["code"] == "agent_not_found"


def test_cli_strict_fails_on_a_refused_file(tmp_path: Path):
    """``--strict`` is the CI contract: a roster that silently lost a file must fail the build."""
    write(tmp_path, ".cursor/agents/broken.md", "---\ndescription: [oops\n---\nbody\n")
    relaxed = run_cli(str(tmp_path))
    assert relaxed.returncode == 0  # reported, not fatal: a broken file must not block the others
    assert "agent_frontmatter_invalid" in relaxed.stderr

    strict = run_cli(str(tmp_path), "--strict")
    assert strict.returncode == 1


def test_cli_strict_passes_for_a_clean_roster():
    assert run_cli(str(FIXTURE), "--strict").returncode == 0


def test_cli_rejects_a_negative_budget():
    result = run_cli(str(FIXTURE), "--budget", "-1")
    assert result.returncode == 1
    assert json.loads(result.stderr)["error"]["code"] == "agents_budget_invalid"


def test_cli_reports_an_empty_repository_honestly(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    result = run_cli(str(tmp_path / "empty"))
    assert result.returncode == 0
    assert "no subagents found" in result.stdout


# ---------------------------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------------------------

def test_bridge_answers_the_roster_with_bodies():
    """The body MUST travel: it is the subagent's system prompt, and the engine builds a tool from it."""
    code, answer = run_bridge({"repo": str(FIXTURE), "home": "/nonexistent"})
    assert code == 0
    assert answer["ok"] is True
    by_name = {entry["name"]: entry for entry in answer["agents"]}
    assert by_name["verifier"]["body"].startswith("You are a verifier.")
    assert by_name["verifier"]["readonly"] is True
    assert by_name["verifier"]["description"].startswith("Verifies a change")
    assert answer["roster_bytes"] > 0


def test_bridge_reports_refusals_and_warnings_separately():
    code, answer = run_bridge({"repo": str(FIXTURE), "home": "/nonexistent"})
    assert code == 0
    # The shadowed .claude copy is a WARNING (a valid file that lost), never a refusal.
    assert [entry["code"] for entry in answer["warnings"]] == ["agent_name_shadowed"]
    assert answer["refusals"] == []


def test_bridge_pulls_one_agent():
    code, answer = run_bridge({"repo": str(FIXTURE), "agent": "verifier", "home": "/nonexistent"})
    assert code == 0
    assert answer["name"] == "verifier"
    assert answer["body"].startswith("You are a verifier.")


def test_bridge_refuses_an_unknown_agent_with_a_code():
    code, answer = run_bridge({"repo": str(FIXTURE), "agent": "nobody", "home": "/nonexistent"})
    assert code == 1
    assert answer["ok"] is False
    assert answer["error"]["code"] == "agent_not_found"


def test_bridge_refuses_an_absent_repo_rather_than_using_the_cwd():
    """The cwd of a spawned bridge is not the user's repository, so guessing would read the wrong tree."""
    code, answer = run_bridge({"home": "/nonexistent"})
    assert code == 1
    assert answer["error"]["code"] == "bridge_bad_request"


def test_bridge_refuses_a_non_list_budget():
    code, answer = run_bridge({"repo": str(FIXTURE), "max_bytes": "big"})
    assert code == 1
    assert answer["error"]["code"] == "bridge_bad_request"


def test_bridge_excludes_an_agent_whose_body_would_exceed_the_roster_budget(tmp_path: Path, monkeypatch):
    """The roster cap NAMES the excluded agent rather than shortening the roster silently."""
    write(tmp_path, ".cursor/agents/huge.md", "---\ndescription: big\n---\n" + "x" * 4096)
    write(tmp_path, ".cursor/agents/small.md", "---\ndescription: small\n---\nsmall body\n")

    completed = subprocess.run(
        [sys.executable, "-c", (
            "import json,sys;"
            "from codna import agents_bridge;"
            f"agents_bridge.MAX_ROSTER_BYTES = 512;"
            f"print(json.dumps(agents_bridge.handle({{'repo': {str(tmp_path)!r}, 'home': '/nonexistent'}})))"
        )],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/nonexistent"},
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    answer = json.loads(completed.stdout)
    assert [entry["name"] for entry in answer["agents"]] == ["small"]
    assert [entry["code"] for entry in answer["refusals"]] == ["agent_body_over_roster_budget"]
