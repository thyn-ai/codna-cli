from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage_codna_telys_oem_license.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("stage_codna_telys_oem_license", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage_oem_license_writes_single_release_artifact(monkeypatch, tmp_path: Path):
    tool = _load_script()
    monkeypatch.setenv("CODNA_TELYS_LICENSE_JWT", "header.payload.signature")

    output = tool.stage_license(tmp_path, env_name="CODNA_TELYS_LICENSE_JWT")

    assert output == tmp_path / "oem-license.jwt"
    assert output.read_text(encoding="utf-8") == "header.payload.signature\n"


def test_stage_oem_license_requires_secret_without_leaking_value(monkeypatch, tmp_path: Path):
    tool = _load_script()
    monkeypatch.delenv("CODNA_TELYS_LICENSE_JWT", raising=False)

    with pytest.raises(tool.StageLicenseError) as excinfo:
        tool.stage_license(tmp_path, env_name="CODNA_TELYS_LICENSE_JWT")

    assert "CODNA_TELYS_LICENSE_JWT is required" in str(excinfo.value)


def test_stage_oem_license_rejects_multiline_secret(monkeypatch, tmp_path: Path):
    tool = _load_script()
    monkeypatch.setenv("CODNA_TELYS_LICENSE_JWT", "header.payload\nsignature")

    with pytest.raises(tool.StageLicenseError) as excinfo:
        tool.stage_license(tmp_path, env_name="CODNA_TELYS_LICENSE_JWT")

    assert "single-line JWT" in str(excinfo.value)
