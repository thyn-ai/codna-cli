"""Guard: `ci.yml` must run THIS directory with the `durable` extra installed.

The failure this exists for, measured. `.github/workflows/ci.yml`'s `cli-secure-tests` job runs
`pytest -q` from `cli/` with no `temporalio`/`langgraph` installed — deliberately, it is the offline
contract. Every test here that needs the extra therefore SKIPPED there: the directory collected 803
tests and that job ran none of the ~786 that mattered. A green `ci` check on a durable change was
evidence that the package IMPORTS, not that it WORKS, and 786 tests were proven only on a
developer's machine.

The fix is a separate `durable-tests` job (installing `cli[durable]` against
`cli/constraints-durable.txt`, gating the real-server tests, caching the pinned Temporal binaries).
This file is what stops a future edit from re-hiding that suite — by deleting the job, by dropping
the extra from it, or by "simplifying" it into the whole-tree run, which measurably breaks 7 tests
here (see the comment on the job; the interference is from `tests/test_memory_wrapper.py` and is not
fixed).

Every expectation is derived from the PARSED workflow and from the actual `run:` scripts, never from
comments: a comment that merely claims the suite runs is not coverage, so comment lines are stripped
before any match.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# The extras notation the install must use: `cli[durable]` / `'cli[durable]'` / `"cli[durable]"`.
EXTRA_INSTALL = re.compile(r"cli\[durable\]")
DURABLE_DIR = re.compile(r"tests/durable\b")
# The dedicated job's ID. Pinned deliberately: a job ID is what `needs:` targets and what branch
# protection / required-check settings list, so a rename is a real contract change, not cosmetics —
# and it is the edit that would quietly unhook the suite from the PR gate. A rename MUST be a
# deliberate, reviewed change accompanied by updating this constant.
DURABLE_JOB_ID = "durable-tests"
# The offline contract job (name is fixed in ci.yml) whose behaviour must not change.
OFFLINE_JOB = "cli-secure-tests"


def _jobs() -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{CI_WORKFLOW} did not parse to a mapping"
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{CI_WORKFLOW} has no `jobs:` mapping"
    return jobs


def _strip_comment_lines(text: str) -> str:
    """Drop whole-line `#` comments: a comment is a claim, never evidence of a command."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _run_text(job: dict[str, Any]) -> str:
    """The job's executable shell (every step's `run:`), with comment lines removed."""
    return "\n".join(
        _strip_comment_lines(str(step["run"]))
        for step in job.get("steps", [])
        if isinstance(step, dict) and "run" in step
    )


def _jobs_running_the_durable_directory(jobs: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, job in jobs.items() if DURABLE_DIR.search(_run_text(job))]


def _durable_job(jobs: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """The job that installs the extra AND runs `tests/durable`, or a failure naming what is missing."""
    runners = _jobs_running_the_durable_directory(jobs)
    with_extra = [name for name in runners if EXTRA_INSTALL.search(_run_text(jobs[name]))]
    if len(with_extra) == 1:
        return with_extra[0], jobs[with_extra[0]]
    if not runners:
        raise AssertionError(
            "no job in .github/workflows/ci.yml runs `tests/durable`: the durable suite is not in "
            "CI at all, so its ~786 tests are again proven only on a developer's machine. Add the "
            "`durable-tests` job (it must run the directory on its OWN — see the comment there) and "
            f"do not merge it into `{OFFLINE_JOB}`, which is deliberately offline. "
            f"jobs present: {sorted(jobs)}"
        )
    if not with_extra:
        raise AssertionError(
            f"jobs running `tests/durable` without installing the `durable` extra: {runners}. "
            "Without `pip install -e 'cli[durable]'` (against cli/constraints-durable.txt) every "
            "test that needs temporalio/langgraph SKIPS, which is the exact green-but-empty result "
            f"this guard exists to prevent. jobs present: {sorted(jobs)}"
        )
    raise AssertionError(
        f"more than one job installs `cli[durable]` and runs `tests/durable`: {with_extra}. "
        "One is enough; two would run the same ~786 tests twice and let the two copies drift."
    )


def test_ci_runs_the_durable_suite_with_its_extra():
    """The whole point: one job installs the extra and runs this directory."""
    jobs = _jobs()
    if DURABLE_JOB_ID not in jobs:
        raise AssertionError(
            f"`.github/workflows/ci.yml` has no `{DURABLE_JOB_ID}` job (jobs present: {sorted(jobs)}). "
            "That job is what runs cli/tests/durable with the `durable` extra installed; without it "
            "the suite is proven only on a developer's machine and a green `ci` says nothing about "
            "whether the durable code works. Restore the job — do not fold it into "
            f"`{OFFLINE_JOB}` (offline on purpose) or into a whole-tree run (7 measured failures)."
        )
    name, job = _durable_job(jobs)
    assert name == DURABLE_JOB_ID, (
        f"the job installing `cli[durable]` and running `tests/durable` is now `{name}`, not "
        f"`{DURABLE_JOB_ID}`. A job ID is a required-check contract; if the rename is intended, "
        "update DURABLE_JOB_ID and the branch-protection entry together."
    )
    run_text = _run_text(job)

    # It must use the pinned set, not whatever resolves today: the extra's ranges are a library
    # contract, the constraints file is what we test (see its header for why the pins are exact).
    assert "constraints-durable.txt" in run_text, (
        f"job `{name}` installs `cli[durable]` but not against cli/constraints-durable.txt, so it "
        "tests a DIFFERENT dependency combination than the one we ship and pin."
    )
    assert "-e" in run_text and "cli[durable]" in run_text, (
        f"job `{name}` must install the checkout editable (`-e 'cli[durable]'`): a stale wheel would "
        "test the released code instead of this branch."
    )


def test_the_durable_job_runs_only_the_durable_directory():
    """A whole-tree run is not a simplification, it is a break: measured, 7 tests fail.

    `test_spike_langgraph_temporal.py` dies with `RuntimeError: Failed validating workflow
    SpikeTaskWorkflow` (a `ModuleNotFoundError: No module named 'tests.durable'` raised inside
    Temporal's workflow sandbox) when the durable directory runs alongside the rest of `cli/tests`,
    bisected to `test_memory_wrapper.py`. In isolation all 7 pass. So the job must stay scoped to
    `tests/durable` until that interference is fixed.
    """
    name, job = _durable_job(_jobs())
    pytest_lines = [line for line in _run_text(job).splitlines() if re.search(r"-m\s+pytest\b", line)]
    assert pytest_lines, f"job `{name}` has no `python -m pytest` invocation"
    unscoped = [line for line in pytest_lines if not DURABLE_DIR.search(line)]
    assert not unscoped, (
        f"job `{name}` invokes pytest without scoping it to `tests/durable`: {unscoped}. A whole-tree "
        "run fails 7 durable tests through interference from tests/test_memory_wrapper.py (the sandbox "
        "cannot resolve `tests.durable`); keep this job directory-only."
    )


def test_the_offline_job_does_not_gain_the_durable_extra():
    """`cli-secure-tests` is offline on purpose; installing the extra there is not an improvement.

    It would (a) make an offline contract depend on a network install and (b) drag the durable
    directory into a whole-tree run, which is the combination that fails. The durable job above is
    the complement to it, not a repair of it.
    """
    jobs = _jobs()
    assert OFFLINE_JOB in jobs, f"{OFFLINE_JOB} is gone from ci.yml; it is the offline contract job"
    run_text = _run_text(jobs[OFFLINE_JOB])
    assert not EXTRA_INSTALL.search(run_text), (
        f"`{OFFLINE_JOB}` now installs `cli[durable]`. That job is the OFFLINE base contract and runs "
        "the whole tree; adding the extra makes it network-dependent and triggers the 7 measured "
        "failures in tests/durable/test_spike_langgraph_temporal.py. Leave it offline."
    )
    assert "temporalio" not in run_text, (
        f"`{OFFLINE_JOB}` installs temporalio directly. The durable suite belongs to the "
        "`durable-tests` job, which pins it through cli/constraints-durable.txt."
    )


def test_the_durable_job_runs_the_real_server_tests_and_caches_their_binaries():
    """The real-server tests are gated on `CODNA_TEST_TEMPORAL=1`; this job must set it.

    Setting the gate is only honest if the binaries are fetchable and cached — otherwise the gate
    turns a green job into a flaky one on a slow download. So: the gate is on, and there is a cache
    step keyed on the CONTENT of the pin files (`hashFiles`), which changes when the CLI pin or the
    pinned `temporalio` does.
    """
    jobs = _jobs()
    name, job = _durable_job(jobs)
    env = job.get("env") or {}
    assert str(env.get("CODNA_TEST_TEMPORAL")) == "1", (
        f"job `{name}` must set `CODNA_TEST_TEMPORAL: \"1\"`: without it the 7 real-server tests "
        "(test_dev_server_real.py, test_run_api_real.py, test_run_e2e_real.py) skip, and the job "
        "would prove only the in-process paths."
    )

    cache_steps = [
        step for step in job.get("steps", [])
        if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/cache@")
    ]
    assert cache_steps, (
        f"job `{name}` sets CODNA_TEST_TEMPORAL=1 but caches nothing: both Temporal binaries are "
        "downloaded from temporal.download on first use, so every run pays for 200+ MB (or fails)."
    )
    cache = cache_steps[0]
    key = str((cache.get("with") or {}).get("key", ""))
    paths = str((cache.get("with") or {}).get("path", ""))
    assert "temporal_cli.lock.json" in key and "hashFiles(" in key, (
        f"the cache key must be derived from cli/codna/durable/temporal_cli.lock.json via hashFiles "
        f"so a version bump invalidates it; got {key!r}"
    )
    assert "constraints-durable.txt" in key, (
        f"the cache key must also cover cli/constraints-durable.txt: the SDK's own test-server binary "
        f"is named after the temporalio version, so a bump there must not restore the old one; got {key!r}"
    )
    assert "temporal" in paths, f"the cache does not name either Temporal binary path: {paths!r}"


def test_the_extraction_helpers_are_not_vacuous():
    """The guard itself is tested, so a silent regex/parse regression cannot make it always pass."""
    jobs = _jobs()
    assert OFFLINE_JOB in jobs
    # `cli-secure-tests` really does run pytest over the whole tree, and comment stripping did not
    # eat its steps.
    assert re.search(r"-m\s+pytest\b", _run_text(jobs[OFFLINE_JOB]))
    # A workflow with no durable job must FAIL with a message naming the missing job, not pass.
    try:
        _durable_job({"not-durable": {"steps": [{"run": "python -m pytest -q"}]}})
    except AssertionError as exc:
        assert "tests/durable" in str(exc) and "durable-tests" in str(exc), str(exc)
    else:  # pragma: no cover - the assertion above is the point
        raise AssertionError("a workflow with no durable job was accepted")


def test_the_temporal_preflight_has_a_hard_process_timeout():
    """A hung SDK shutdown must fail the preflight, not consume the whole job budget."""
    _name, job = _durable_job(_jobs())
    preflight = next(
        (
            step
            for step in job.get("steps", [])
            if isinstance(step, dict) and step.get("name") == "Preflight: fetch and verify both Temporal binaries"
        ),
        None,
    )
    assert preflight is not None, "the durable job must preflight both Temporal binaries"
    command = _strip_comment_lines(str(preflight.get("run", "")))
    assert re.search(r"^timeout --signal=TERM --kill-after=10s 120 \"\$PY\" - <<'PY'$", command, re.M), (
        "the Temporal preflight must have an external timeout with a forced-kill backstop; "
        "an unbounded SDK shutdown previously consumed the job's entire 45-minute budget"
    )
