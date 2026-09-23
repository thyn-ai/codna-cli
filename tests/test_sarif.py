"""Offline test suite for SARIF ingestion (A–Z test plan: SAR-*, determinism, dedupe).

Every test runs without a live engine — pure parse/logic. Fixtures are built inline as
SARIF dicts so the oracles are self-contained and readable.
"""
from __future__ import annotations

import json

import pytest

from codna import sarif
from codna.findings import FindingKind, Severity
from codna.sarif import (
    BoundedInputError,
    CommitBindingError,
    ProvenanceIncompleteError,
    SarifParseError,
    SarifSchemaError,
    build_security_analysis_request,
    ingest_sarif,
    validate_commit_binding,
)

COMMIT = "a" * 40  # valid 40-hex
REPO = "https://github.com/o/r"


# --------------------------------------------------------------------- builders

def _loc(uri="src/app.py", line=10, base_id=None, logicals=None):
    art = {"uri": uri}
    if base_id:
        art["uriBaseId"] = base_id
    loc = {"physicalLocation": {"artifactLocation": art, "region": {"startLine": line}}}
    if logicals:
        loc["logicalLocations"] = [{"fullyQualifiedName": n} for n in logicals]
    return loc


def _flow(*points):
    locs = [
        {"location": {"physicalLocation": {"artifactLocation": {"uri": u}, "region": {"startLine": ln}}}}
        for (u, ln) in points
    ]
    return {"threadFlows": [{"locations": locs}]}


def _result(rule_id="js/sql-injection", uri="src/app.py", line=10, level="error",
            code_flows=None, props=None, locations=None, related=None, suppressions=None,
            base_id=None):
    r = {
        "ruleId": rule_id,
        "message": {"text": f"{rule_id} finding"},
        "locations": locations if locations is not None else [_loc(uri, line, base_id)],
    }
    if level:
        r["level"] = level
    if code_flows is not None:
        r["codeFlows"] = code_flows
    if props:
        r["properties"] = props
    if related:
        r["relatedLocations"] = related
    if suppressions:
        r["suppressions"] = suppressions
    return r


def _codeql_rule(rid="js/sql-injection", security_severity="9.1",
                 tags=("security", "external/cwe/cwe-089")):
    return {"id": rid, "properties": {"security-severity": security_severity, "tags": list(tags)}}


def _run(driver="CodeQL", version="2.15.0", rules=None, results=None, vcp=True,
         base_ids=None, invocations=None, guid=None, vcp_props=None):
    driver_obj = {"name": driver, "rules": rules or []}
    if version:
        driver_obj["version"] = version
    if guid:
        driver_obj["guid"] = guid
    run = {"tool": {"driver": driver_obj}}
    if results is not None:
        run["results"] = results
    if vcp:
        prov = {"revisionId": vcp if isinstance(vcp, str) else COMMIT, "repositoryUri": REPO}
        if vcp_props:
            prov["properties"] = vcp_props
        run["versionControlProvenance"] = [prov]
    if base_ids:
        run["originalUriBaseIds"] = base_ids
    if invocations:
        run["invocations"] = invocations
    return run


def _doc(*runs, schema=True, version="2.1.0"):
    d = {}
    if schema:
        d["$schema"] = "https://json.schemastore.org/sarif-2.1.0.json"
    if version is not None:
        d["version"] = version
    d["runs"] = list(runs)
    return d


def _complete(results=None, rules=None, **run_kw):
    """A provenance-complete CodeQL doc (driver name+version + 40-hex VCP)."""
    rules = rules if rules is not None else [_codeql_rule()]
    results = results if results is not None else [_result(code_flows=[_flow(("src/in.py", 3), ("src/app.py", 10))])]
    return _doc(_run(rules=rules, results=results, **run_kw))


# ------------------------------------------------------------------- parse/schema

def test_sar01_minimal_valid():
    ing = ingest_sarif(json.dumps(_complete()))
    assert len(ing.findings) == 1
    f = ing.findings[0]
    assert f.rule_id == "js/sql-injection"
    assert ing.schema_version == "2.1.0"


def test_sar02_bad_version_rejected_not_coerced():
    with pytest.raises(SarifSchemaError):
        ingest_sarif(json.dumps(_doc(_run(), version="3.0.0")))


def test_sar03_malformed_and_empty_fail_closed():
    with pytest.raises(SarifParseError):
        ingest_sarif("{not json")
    with pytest.raises(SarifParseError):
        ingest_sarif("   ")


def test_sar04_multi_run_run_index_preserved():
    doc = _doc(
        _run(results=[_result(rule_id="a")]),
        _run(results=[_result(rule_id="b")]),
    )
    ing = ingest_sarif(json.dumps(doc))
    idx = {f.rule_id: f.run_index for f in ing.findings}
    assert idx == {"a": 0, "b": 1}


# --------------------------------------------------------------------- severity

def test_sar18_band_from_rule_props_original_and_rationale():
    ing = ingest_sarif(json.dumps(_complete(rules=[_codeql_rule(security_severity="9.1")])))
    f = ing.findings[0]
    assert f.severity is Severity.CRITICAL
    assert f.original_severity == "9.1"
    assert "security-severity" in f.severity_rationale


def test_sar19_level_alone_is_unknown():
    # rule with no severity property, result carries only `level`
    doc = _complete(rules=[{"id": "x/y", "properties": {}}],
                    results=[_result(rule_id="x/y", level="error", code_flows=None)])
    f = ingest_sarif(json.dumps(doc)).findings[0]
    assert f.severity is Severity.UNKNOWN
    assert "level" in f.severity_rationale.lower() and "insufficient" in f.severity_rationale.lower()


def test_sar20_policy_override_recorded_original_preserved():
    class _Policy:
        severity_overrides = {"js/sql-injection": Severity.HIGH}
        digest = "sha256:policy"

    ing = ingest_sarif(json.dumps(_complete(rules=[_codeql_rule(security_severity="9.5")])),
                       policy=_Policy())
    f = ing.findings[0]
    assert f.severity is Severity.HIGH
    assert f.original_severity == "9.5"  # not silently dropped
    assert "policy override" in f.severity_rationale and "sha256:policy" in f.severity_rationale


# --------------------------------------------------------------------- dedupe

def test_sar21_byte_identical_dups_collapse():
    res = _result(code_flows=[_flow(("src/in.py", 3), ("src/app.py", 10))])
    ing = ingest_sarif(json.dumps(_complete(results=[res, json.loads(json.dumps(res))])))
    assert len(ing.findings) == 1
    assert ing.findings[0].occurrence_count == 2


def test_sar22_distinct_flows_not_merged():
    sink = _loc("src/app.py", 10)
    r1 = _result(locations=[sink], code_flows=[_flow(("src/a.py", 1), ("src/app.py", 10))])
    r2 = _result(locations=[sink], code_flows=[_flow(("src/b.py", 2), ("src/app.py", 10))])
    ing = ingest_sarif(json.dumps(_complete(results=[r1, r2])))
    assert len(ing.findings) == 2
    cids = {f.canonical_id for f in ing.findings}
    assert len(cids) == 2  # distinct identities
    assert all(f.code_flows for f in ing.findings)


# --------------------------------------------------------------------- kind

@pytest.mark.parametrize(
    "rule,result,expected",
    [
        (_codeql_rule(tags=("security",)),
         _result(code_flows=[_flow(("a", 1), ("b", 2))]), FindingKind.TAINT),
        ({"id": "SNYK-JS-LODASH", "properties": {"tags": ["sca"]}},
         _result(rule_id="SNYK-JS-LODASH", props={"cve": ["CVE-2020-8203"]}), FindingKind.SCA),
        ({"id": "secret/aws-key", "properties": {"tags": ["secret"]}},
         _result(rule_id="secret/aws-key"), FindingKind.SECRET),
        ({"id": "tf/open-sg", "properties": {"tags": ["misconfiguration"]}},
         _result(rule_id="tf/open-sg"), FindingKind.CONFIG),
        ({"id": "style/no-tabs", "properties": {}},
         _result(rule_id="style/no-tabs", code_flows=None), FindingKind.UNSUPPORTED),
    ],
)
def test_sar32_finding_kind_classification(rule, result, expected):
    ing = ingest_sarif(json.dumps(_complete(rules=[rule], results=[result])))
    assert ing.findings[0].finding_kind is expected


def test_sar40_multi_location_not_flattened():
    res = _result(
        locations=[_loc("a.py", 1), _loc("b.py", 2)],
        related=[_loc("c.py", 3), _loc("d.py", 4), _loc("e.py", 5)],
    )
    f = ingest_sarif(json.dumps(_complete(results=[res]))).findings[0]
    assert len(f.locations) == 2
    assert len(f.related_locations) == 3


# --------------------------------------------------------------- scanner identity

def test_sar05_scanner_identity_canonicalized():
    ing = ingest_sarif(json.dumps(_complete(driver="CodeQL", version="2.15.0", guid="g-1")))
    assert ing.scanner.name == "codeql"
    assert ing.scanner.version == "2.15.0"
    assert ing.scanner.guid == "g-1"
    assert ing.scanner.rules_count == 1
    assert ing.scanner.rules_digest.startswith("sha256:")


def test_sar06_missing_version_blocks_provenance_and_request():
    ing = ingest_sarif(json.dumps(_doc(_run(version=None, results=[_result()]))))
    assert ing.provenance_valid is False
    assert any("version" in e for e in ing.provenance_errors)
    with pytest.raises(ProvenanceIncompleteError):
        build_security_analysis_request(ing, snapshot_id="s", policy_digest="sha256:p",
                                        model_pack_digest="sha256:m")


def test_sar37_conflicting_drivers():
    doc = _doc(
        _run(driver="CodeQL", version="2.15.0", results=[_result(rule_id="a")]),
        _run(driver="Semgrep", version="1.0.0", results=[_result(rule_id="b")]),
    )
    ing = ingest_sarif(json.dumps(doc))
    assert ing.provenance_valid is False
    joined = " ".join(ing.provenance_errors)
    assert "inconsistent scanner identity" in joined
    assert "codeql" in joined and "semgrep" in joined  # both surfaced


# `tool.driver.name` exactly as each report writer emits it -> the canonical scanner name.
DRIVER_NAMES = {
    "CodeQL": "codeql",
    "Semgrep OSS": "semgrep",
    "Snyk Code": "snyk",
    "Trivy": "trivy",
    "Opengrep OSS": "opengrep",
    "gitleaks": "gitleaks",
    "osv-scanner": "osv-scanner",
}


@pytest.mark.parametrize("raw,canonical", sorted(DRIVER_NAMES.items()))
def test_sar05b_emitted_driver_names_canonicalized(raw, canonical):
    ing = ingest_sarif(json.dumps(_complete(driver=raw)))
    assert ing.scanner.name == canonical
    assert ing.scanner.raw_name == raw
    assert ing.provenance_valid is True


def test_sar05c_alias_map_is_closed_case_insensitive_and_keeps_forks_apart():
    canonical = set(sarif._SCANNER_ALIASES.values())
    # closed: every canonical name is also its own key, so canonicalizing twice is a no-op
    assert all(name in sarif._SCANNER_ALIASES for name in canonical)
    assert set(DRIVER_NAMES.values()) <= canonical
    for raw, want in DRIVER_NAMES.items():
        assert sarif._canon_scanner(f"  {raw.upper()} ") == want
    # Opengrep is a Semgrep fork but a distinct, separately pinned tool
    assert sarif._canon_scanner("Opengrep OSS") != sarif._canon_scanner("Semgrep OSS")
    # unknown drivers pass through lower-cased rather than being guessed at
    assert sarif._canon_scanner("  Some New Scanner ") == "some new scanner"
    # regression floor only (codeql, semgrep, snyk, trivy, opengrep, gitleaks, osv-scanner), not an exact count
    assert len(canonical) >= 7


def test_sar38_mixed_opengrep_semgrep_runs_are_inconsistent():
    doc = _doc(
        _run(driver="Opengrep OSS", version="1.0.0", results=[_result(rule_id="a")]),
        _run(driver="Semgrep OSS", version="1.0.0", results=[_result(rule_id="b")]),
    )
    ing = ingest_sarif(json.dumps(doc))
    assert ing.provenance_valid is False
    joined = " ".join(ing.provenance_errors)
    assert "inconsistent scanner identity" in joined
    assert "opengrep" in joined and "semgrep" in joined


def test_sar39_gitleaks_untagged_rule_is_secret():
    untagged = {"id": "aws-access-token", "properties": {}}
    res = _result(rule_id="aws-access-token", code_flows=None)
    # gitleaks rules carry no tags: the scanner's identity alone makes the finding a SECRET
    gl = ingest_sarif(json.dumps(_complete(driver="gitleaks", rules=[untagged], results=[res])))
    assert gl.findings[0].finding_kind is FindingKind.SECRET
    # the identity rule is gitleaks-only: the same untagged rule under another driver stays UNSUPPORTED
    og = ingest_sarif(json.dumps(_complete(driver="Opengrep OSS", rules=[untagged], results=[res])))
    assert og.findings[0].finding_kind is FindingKind.UNSUPPORTED
    # osv-scanner advisories classify SCA via the GHSA id (pins behaviour that predates the alias)
    ghsa = {"id": "GHSA-abcd-efgh-ijkl", "properties": {}}
    osv = ingest_sarif(json.dumps(_complete(
        driver="osv-scanner", rules=[ghsa], results=[_result(rule_id="GHSA-abcd-efgh-ijkl", code_flows=None)],
    )))
    assert osv.findings[0].finding_kind is FindingKind.SCA


# --------------------------------------------------------------- commit binding

def test_sar09_commit_bound_from_vcp():
    ing = ingest_sarif(json.dumps(_complete(vcp_props={"tree_digest": "sha256:tree"})))
    assert ing.commit == COMMIT
    assert ing.repo_uri == REPO
    assert ing.tree_digest == "sha256:tree"
    assert validate_commit_binding(ing) == COMMIT


def test_sar10_commit_mismatch():
    ing = ingest_sarif(json.dumps(_complete()))
    with pytest.raises(CommitBindingError) as exc:
        validate_commit_binding(ing, expected="b" * 40)
    assert COMMIT in str(exc.value) and "b" * 40 in str(exc.value)


def test_sar11_no_vcp():
    ing = ingest_sarif(json.dumps(_doc(_run(vcp=False, results=[_result()]))))
    with pytest.raises(CommitBindingError):
        validate_commit_binding(ing)


def test_sar12_short_or_branch_ref_rejected():
    ing = ingest_sarif(json.dumps(_doc(_run(vcp="main", results=[_result()]))))
    with pytest.raises(CommitBindingError):
        validate_commit_binding(ing)


# --------------------------------------------------------------- path escape

@pytest.mark.parametrize(
    "uri,base_ids,base_id",
    [
        ("../../../etc/passwd", None, None),
        ("..%2f..%2fetc%2fpasswd", None, None),
        ("..\\..\\windows\\system32", None, None),
        ("passwd", {"SRCROOT": {"uri": "file:///etc/"}}, "SRCROOT"),
    ],
)
def test_sar29_30_31_path_escape_quarantined(uri, base_ids, base_id):
    res = _result(locations=[_loc(uri, 1, base_id=base_id)])
    ing = ingest_sarif(json.dumps(_complete(results=[res], base_ids=base_ids)))
    assert ing.quarantined_artifacts, f"expected quarantine for {uri}"
    assert ing.findings[0].quarantined is True
    # escaping findings are excluded from the engine request
    req = build_security_analysis_request(ing, snapshot_id="s", policy_digest="sha256:p",
                                          model_pack_digest="sha256:m")
    assert req["findings"] == []


def test_benign_in_root_not_quarantined():
    ing = ingest_sarif(json.dumps(_complete(results=[_result(uri="src/app.py")])))
    assert ing.quarantined_artifacts == []
    assert ing.findings[0].quarantined is False


# --------------------------------------------------------------- request / digest

def test_sar33_build_request_exact_shape():
    ing = ingest_sarif(json.dumps(_complete()))
    req = build_security_analysis_request(
        ing, snapshot_id="snap_1", policy_digest="sha256:p", model_pack_digest="sha256:m",
        configuration_digest="sha256:cfg",
    )
    assert set(req) == {"snapshot_id", "sarif_digest", "scanner", "policy_digest",
                        "model_pack_digest", "findings"}
    assert set(req["scanner"]) == {"name", "version", "rules_digest", "configuration_digest"}
    for d in (req["sarif_digest"], req["policy_digest"], req["model_pack_digest"],
              req["scanner"]["rules_digest"]):
        assert isinstance(d, str) and d.startswith("sha256:")
    for f in req["findings"]:
        assert {"canonical_id", "finding_kind", "code_flows"} <= set(f)


def test_sar27_digest_stable_under_reorder_and_whitespace():
    doc = _complete()
    a = ingest_sarif(json.dumps(doc, indent=2)).sarif_digest
    # reorder top-level keys + compact formatting -> same canonical content
    reordered = {k: doc[k] for k in reversed(list(doc))}
    b = ingest_sarif(json.dumps(reordered, separators=(",", ":"))).sarif_digest
    assert a == b


def test_sar28_semantic_mutation_changes_digest():
    base = ingest_sarif(json.dumps(_complete(results=[_result(rule_id="orig")],
                                             rules=[_codeql_rule(rid="orig")]))).sarif_digest
    mutant = ingest_sarif(json.dumps(_complete(results=[_result(rule_id="evil")],
                                               rules=[_codeql_rule(rid="evil")]))).sarif_digest
    assert base != mutant


def test_sar36_toctou_digest_from_ingested_bytes(tmp_path):
    p = tmp_path / "r.sarif"
    p.write_text(json.dumps(_complete(results=[_result(rule_id="orig")],
                                      rules=[_codeql_rule(rid="orig")])))
    ing = ingest_sarif(p)
    d0 = ing.sarif_digest
    # mutate the file on disk AFTER ingest
    p.write_text(json.dumps(_complete(results=[_result(rule_id="evil")],
                                      rules=[_codeql_rule(rid="evil")])))
    req = build_security_analysis_request(ing, snapshot_id="s", policy_digest="sha256:p",
                                          model_pack_digest="sha256:m")
    assert req["sarif_digest"] == d0  # bound to ingest-time bytes, not re-read
    assert ingest_sarif(p).sarif_digest != d0  # re-ingesting the mutated file differs


# --------------------------------------------------------------- determinism

def test_sar25_canonical_id_stable_across_reordering():
    r1 = _result(rule_id="a", uri="a.py", line=1)
    r2 = _result(rule_id="b", uri="b.py", line=2)
    ids_fwd = {f.rule_id: f.canonical_id for f in ingest_sarif(json.dumps(_complete(results=[r1, r2]))).findings}
    ids_rev = {f.rule_id: f.canonical_id for f in ingest_sarif(json.dumps(_complete(results=[r2, r1]))).findings}
    assert ids_fwd == ids_rev


def test_sar35_ingest_twice_byte_equal():
    payload = json.dumps(_complete())
    a = ingest_sarif(payload).to_canonical_json()
    b = ingest_sarif(payload).to_canonical_json()
    assert a == b


# --------------------------------------------------------------- DoS / bounds

def test_sardos_too_many_results(monkeypatch):
    monkeypatch.setattr(sarif, "MAX_RESULTS", 5)
    doc = _complete(results=[_result(rule_id=f"r{i}", line=i) for i in range(6)])
    with pytest.raises(BoundedInputError):
        ingest_sarif(json.dumps(doc))


def test_sardos_too_big(monkeypatch):
    monkeypatch.setattr(sarif, "MAX_BYTES", 100)
    with pytest.raises(BoundedInputError):
        ingest_sarif(json.dumps(_complete()) + " " * 200)


def test_sardos_deep_nesting(monkeypatch):
    monkeypatch.setattr(sarif, "MAX_JSON_DEPTH", 20)
    with pytest.raises(BoundedInputError):
        ingest_sarif("[" * 50)
