"""Contracts for the PyPI project-size preflight (scripts/check_pypi_project_size.py).

The preflight exists because PyPI's project size cap turned every publish into a bare
`400 Bad Request`: Warehouse puts the explanation in the response body, twine prints the body only
with `--verbose`, and the token publish path did not pass it. These tests pin the arithmetic it
must get right (the same `existing + incoming > limit` comparison Warehouse makes), the exit codes
a workflow branches on, and the fail-closed behaviour when the index cannot be read.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_pypi_project_size.py"

ONE_MIB = 1024 * 1024
ONE_GIB = 1024 * 1024 * 1024


def _load_script():
    spec = importlib.util.spec_from_file_location("check_pypi_project_size", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def size_check():
    return _load_script()


def _artifacts(tmp_path: Path, *specs: tuple[str, int]) -> list[Path]:
    paths = []
    for name, size in specs:
        path = tmp_path / name
        path.write_bytes(b"\0" * size)
        paths.append(path)
    return paths


def _storage(size_check, total: int, *, organization: str | None = None):
    return size_check.ProjectStorage(
        project="codna",
        total_bytes=total,
        organization=organization,
        files_considered=3,
        releases_considered=1,
    )


def test_reproduces_the_real_0_2_114_failure(size_check):
    """The live condition this was written for: 9.9977 GiB used, a 32 MB wheel first."""
    total_on_index = 10_734_889_737  # read from pypi.org/pypi/codna/json
    verdict = size_check.evaluate(
        _storage(size_check, total_on_index),
        (("codna-0.2.114-py3-none-macosx_14_0_arm64.whl", 33_975_000),),
    )
    assert verdict.ok is False
    assert verdict.code == "pypi_project_size_exceeded"
    # 2.41 MiB of headroom against a 32.4 MB wheel: the wheel cannot fit, so PyPI rejects it.
    assert verdict.headroom_bytes == 10 * ONE_GIB - total_on_index
    assert verdict.headroom_bytes < 33_975_000
    assert verdict.shortfall_bytes == 33_975_000 - verdict.headroom_bytes
    assert "HTTP 400" in verdict.message
    assert "yanking does NOT free storage" in verdict.message
    assert "storage-limits" in verdict.message


def test_fits_when_headroom_covers_the_artifacts(size_check):
    verdict = size_check.evaluate(
        _storage(size_check, 5 * ONE_GIB),
        (("codna-0.2.114-py3-none-any.whl", 32 * ONE_MIB),),
    )
    assert verdict.ok is True
    assert verdict.code == "pypi_project_size_ok"
    assert verdict.projected_bytes == 5 * ONE_GIB + 32 * ONE_MIB
    assert verdict.shortfall_bytes == 0


def test_boundary_is_inclusive_like_warehouse(size_check):
    """Warehouse rejects only when `file_size + total_size > limit`; equality is allowed."""
    limit = 10 * ONE_GIB
    exact = size_check.evaluate(
        _storage(size_check, limit - 1000), (("codna-0.2.114.tar.gz", 1000),)
    )
    assert exact.ok is True, "reaching the limit exactly must be allowed"

    one_over = size_check.evaluate(
        _storage(size_check, limit - 1000), (("codna-0.2.114.tar.gz", 1001),)
    )
    assert one_over.ok is False
    assert one_over.shortfall_bytes == 1


def test_all_artifacts_are_summed_not_just_the_first(size_check):
    """The workflow uploads a glob; the first file is only one third of the payload."""
    verdict = size_check.evaluate(
        _storage(size_check, 9 * ONE_GIB),
        (
            ("codna-0.2.114-py3-none-macosx_14_0_arm64.whl", 32 * ONE_MIB),
            ("codna-0.2.114-py3-none-manylinux_2_28_x86_64.whl", 48 * ONE_MIB),
            ("codna-0.2.114.tar.gz", 6 * ONE_MIB),
        ),
    )
    assert verdict.incoming_bytes == 86 * ONE_MIB
    assert verdict.ok is True

    # 1 GiB of headroom: a single 1 GiB artifact still lands exactly ON the limit (allowed, see the
    # boundary test), and the same artifact plus one byte does not.
    at_the_line = size_check.evaluate(
        _storage(size_check, 9 * ONE_GIB), (("codna-0.2.114.tar.gz", ONE_GIB),)
    )
    assert at_the_line.ok is True

    over_the_line = size_check.evaluate(
        _storage(size_check, 9 * ONE_GIB), (("codna-0.2.114.tar.gz", ONE_GIB + 1),)
    )
    assert over_the_line.ok is False
    assert over_the_line.shortfall_bytes == 1


def test_organization_owned_project_gets_the_higher_default(size_check):
    verdict = size_check.evaluate(
        _storage(size_check, 11 * ONE_GIB, organization="thyn-ai"),
        (("codna-0.2.114.tar.gz", 6 * ONE_MIB),),
    )
    assert verdict.ok is True, "an org-owned project may use up to 20 GiB"
    assert verdict.limit_bytes == 20 * ONE_GIB
    assert "organization-owned" in verdict.limit_source


def test_personally_owned_project_gets_the_lower_default(size_check):
    verdict = size_check.evaluate(
        _storage(size_check, 11 * ONE_GIB), (("codna-0.2.114.tar.gz", 6 * ONE_MIB),)
    )
    assert verdict.ok is False
    assert verdict.limit_bytes == 10 * ONE_GIB
    assert "personally owned" in verdict.limit_source


def test_explicit_limit_override_wins(size_check):
    """PyPI grants increases that its JSON API does not report, so an operator must be able to say so."""
    verdict = size_check.evaluate(
        _storage(size_check, 11 * ONE_GIB),
        (("codna-0.2.114.tar.gz", 6 * ONE_MIB),),
        limit_bytes=25 * ONE_GIB,
    )
    assert verdict.ok is True
    assert verdict.limit_bytes == 25 * ONE_GIB
    assert verdict.limit_source == "explicit"


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_limit_override_is_an_error(size_check, bad):
    with pytest.raises(size_check.SizeProbeError) as excinfo:
        size_check.evaluate(
            _storage(size_check, ONE_GIB), (("x.whl", 1),), limit_bytes=bad
        )
    assert excinfo.value.code == "invalid_limit"


def test_missing_artifact_fails_closed(size_check, tmp_path):
    with pytest.raises(size_check.SizeProbeError) as excinfo:
        size_check.incoming_artifact_sizes([tmp_path / "does-not-exist.whl"])
    assert excinfo.value.code == "artifact_missing"


def test_empty_artifact_fails_closed(size_check, tmp_path):
    path = tmp_path / "empty.whl"
    path.write_bytes(b"")
    with pytest.raises(size_check.SizeProbeError) as excinfo:
        size_check.incoming_artifact_sizes([path])
    assert excinfo.value.code == "artifact_empty"


def test_no_artifacts_fails_closed(size_check):
    with pytest.raises(size_check.SizeProbeError) as excinfo:
        size_check.incoming_artifact_sizes([])
    assert excinfo.value.code == "no_artifacts"


def test_fetch_project_storage_sums_every_release_file(size_check, monkeypatch):
    payload = {
        "releases": {
            "0.1.0": [{"size": 10}, {"size": 20}],
            "0.2.0": [{"size": 5}],
            "0.3.0": [],
        },
        "ownership": {"organization": None, "roles": [{"role": "Owner", "user": "x"}]},
    }
    monkeypatch.setattr(size_check, "_get_json", lambda *a, **k: payload)
    storage = size_check.fetch_project_storage("codna")
    assert storage.total_bytes == 35
    assert storage.files_considered == 3
    assert storage.releases_considered == 2
    assert storage.organization is None


def test_fetch_project_storage_reads_organization_ownership(size_check, monkeypatch):
    payload = {
        "releases": {"0.1.0": [{"size": 1}]},
        "ownership": {"organization": "thyn-ai"},
    }
    monkeypatch.setattr(size_check, "_get_json", lambda *a, **k: payload)
    assert size_check.fetch_project_storage("codna").organization == "thyn-ai"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"releases": []},
        {"releases": {"0.1.0": "nope"}},
        {"releases": {"0.1.0": [{"size": "10"}]}},
        {"releases": {"0.1.0": [{"no_size": 1}]}},
        {"releases": {"0.1.0": [{"size": None}]}},
        {"releases": {"0.1.0": [{"size": 10.5}]}},
        {"releases": {"0.1.0": ["not-a-dict"]}},
        # `bool` IS an `int` in Python, so these would otherwise be accepted silently as 1/0 and
        # understate the project's footprint.
        {"releases": {"0.1.0": [{"size": True}]}},
        {"releases": {"0.1.0": [{"size": False}]}},
    ],
)
def test_malformed_index_payload_fails_closed(size_check, monkeypatch, payload):
    monkeypatch.setattr(size_check, "_get_json", lambda *a, **k: payload)
    with pytest.raises(size_check.SizeProbeError) as excinfo:
        size_check.fetch_project_storage("codna")
    assert excinfo.value.code == "pypi_response_malformed"


def test_unreadable_index_fails_closed(size_check, monkeypatch, tmp_path):
    def boom(*_a, **_k):
        raise size_check.SizeProbeError("pypi_unreachable", "index down")

    with pytest.raises(size_check.SizeProbeError):
        size_check.check_project_size(
            "codna",
            _artifacts(tmp_path, ("codna-0.2.114.tar.gz", 10)),
            fetch_storage=boom,
        )


def test_main_exit_codes_and_streams(size_check, monkeypatch, tmp_path, capsys):
    artifacts = _artifacts(tmp_path, ("codna-0.2.114.tar.gz", 100))

    monkeypatch.setattr(
        size_check,
        "check_project_size",
        lambda *a, **k: size_check.evaluate(
            _storage(size_check, 9 * ONE_GIB), (("codna-0.2.114.tar.gz", 100),)
        ),
    )
    assert size_check.main(["--project", "codna", *map(str, artifacts)]) == size_check.EXIT_OK
    out, err = capsys.readouterr()
    assert json.loads(out)["ok"] is True
    assert err == ""

    monkeypatch.setattr(
        size_check,
        "check_project_size",
        lambda *a, **k: size_check.evaluate(
            _storage(size_check, 10 * ONE_GIB), (("codna-0.2.114.tar.gz", 100),)
        ),
    )
    assert size_check.main(["--project", "codna", *map(str, artifacts)]) == (
        size_check.EXIT_OVER_LIMIT
    )
    out, err = capsys.readouterr()
    assert out == ""
    assert json.loads(err)["code"] == "pypi_project_size_exceeded"


def test_main_returns_unknown_not_success_when_the_index_is_unreadable(
    size_check, monkeypatch, tmp_path, capsys
):
    """Exit 2 is distinct from 0 on purpose: an unreadable index must never read as "fits"."""
    artifacts = _artifacts(tmp_path, ("codna-0.2.114.tar.gz", 100))

    def boom(*_a, **_k):
        raise size_check.SizeProbeError("pypi_unreachable", "index down")

    monkeypatch.setattr(size_check, "check_project_size", boom)
    code = size_check.main(["--project", "codna", *map(str, artifacts)])
    assert code == size_check.EXIT_UNKNOWN
    payload = json.loads(capsys.readouterr().err)
    assert payload["ok"] is False
    assert "refusing to guess" in payload["message"]


def test_headroom_is_floored_when_the_project_is_already_over_limit(size_check):
    """A project already past its ceiling must report zero headroom, not a negative one.

    Reachable without a bug in the uploader: an operator can lower --limit-bytes, or the index can
    grow between recording an increase and running this. A consumer reading `headroom_bytes` as
    "space available" must never be handed a negative number.
    """
    verdict = size_check.evaluate(
        _storage(size_check, 12 * ONE_GIB), (("codna-0.2.114.tar.gz", 6 * ONE_MIB),)
    )
    assert verdict.ok is False
    assert verdict.headroom_bytes == 0
    assert verdict.existing_over_limit_bytes == 2 * ONE_GIB
    assert verdict.shortfall_bytes == 2 * ONE_GIB + 6 * ONE_MIB
    report = verdict.as_report()
    assert report["headroom_bytes"] == 0
    assert report["existing_over_limit_bytes"] == 2 * ONE_GIB


def test_report_json_carries_every_deciding_number(size_check):
    verdict = size_check.evaluate(
        _storage(size_check, 9 * ONE_GIB),
        (("codna-0.2.114-py3-none-macosx_14_0_arm64.whl", 32 * ONE_MIB),),
    )
    report = verdict.as_report()
    for key in (
        "existing_bytes",
        "incoming_bytes",
        "projected_bytes",
        "limit_bytes",
        "limit_source",
        "headroom_bytes",
        "shortfall_bytes",
        "organization",
    ):
        assert key in report
    assert report["projected_bytes"] == report["existing_bytes"] + report["incoming_bytes"]
    assert report["incoming_files"] == [
        {"name": "codna-0.2.114-py3-none-macosx_14_0_arm64.whl", "bytes": 32 * ONE_MIB}
    ]
