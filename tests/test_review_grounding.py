"""Registry grounding of review findings (codna.review_grounding): offline, with recorded registry
answers. The texts are the actual findings codna posted on thyn-ai Dependabot PRs on 2026-09-20 --
every one of them false -- and the recorded answers are what registry.npmjs.org / api.npmjs.org /
pypi.org returned for those packages that day."""
from __future__ import annotations

import json

import pytest

from codna import review_findings as rf
from codna import review_grounding as g
from codna import review_manifests as m

# ---- recorded registry answers (trimmed to the fields the grounder reads) ----------------------
SPEED_INSIGHTS_SRI = "sha512-jwkNcrTeafWxjmWq4AHBaptSqZiJkYU5adLC9QBSqeim0GcqDMgN5Ievh8OG1rJ6W3A4l1oiP7qr9CWxGuzu3w=="
TINYEXEC_SRI = "sha512-QKAl9m8gWWGHV8jZcPeym6j+XULi6tOf1mT83WYJ4Lk2ytW/uwAWkrP0uFsdoYMdueVJ0qs26wZ+23xeB4ibNQ=="
RECORDED = {
    "https://registry.npmjs.org/tinyexec/1.3.0": (200, {
        "name": "tinyexec", "version": "1.3.0", "license": "MIT", "engines": {"node": ">=18"},
        "dist": {"integrity": TINYEXEC_SRI}}),
    "https://registry.npmjs.org/tinyexec/9.9.9": (404, "version not found: 9.9.9"),
    "https://registry.npmjs.org/@vercel%2Fspeed-insights/2.0.0": (200, {
        "name": "@vercel/speed-insights", "version": "2.0.0", "license": "Apache-2.0",
        "peerDependencies": {"vue": "^3", "next": ">= 13", "react": "^18 || ^19 || ^19.0.0-rc", "svelte": ">= 4"},
        "peerDependenciesMeta": {"vue": {"optional": True}, "next": {"optional": True}, "react": {"optional": True},
                                 "svelte": {"optional": True}},
        "dist": {"integrity": SPEED_INSIGHTS_SRI}}),
    "https://registry.npmjs.org/@supabase%2Fsupabase-js/2.116.0": (200, {
        "name": "@supabase/supabase-js", "version": "2.116.0", "license": "MIT", "engines": {"node": ">=22.0.0"},
        "peerDependencies": {"@opentelemetry/api": ">=1.0.0"}, "peerDependenciesMeta": {"@opentelemetry/api": {"optional": True}},
        "dist": {"integrity": "sha512-YyWmKXt2NspV9iO8FPnlswUFJIRnrLd3oTCb+3ZyYRuKZtBH0xCUDgnUqoyA0fGUxpM/UhfwDjYf/dht/9bp7g=="}}),
    "https://registry.npmjs.org/@supabase%2Fssr/0.12.7": (200, {
        "name": "@supabase/ssr", "version": "0.12.7", "license": "MIT",
        "peerDependencies": {"@supabase/supabase-js": "^2.114.0"},
        "dist": {"integrity": "sha512-wiBtEie1KkRJi9RrZWY3R2imRhX1JY7qMyUCH2z9AUk15gQebNEplM+urbCKamdxaTJLXUU6LlpkJsaxhojCEg=="}}),
    "https://registry.npmjs.org/@types%2Fnode/26.0.0": (200, {
        "name": "@types/node", "version": "26.0.0", "license": "MIT", "peerDependencies": {},
        "dist": {"integrity": "sha512-vf2YFi1iY9lHGwNJMs01biZFbKJkrZR1T6/MlzjhJLPdntOHLhTrDSnSVcdtvjihi4VQNlrFRIxLsDBlQpAipA=="}}),
    "https://registry.npmjs.org/@eloqnt%2Fconfig/0.1.0": (200, {
        "name": "@eloqnt/config", "version": "0.1.0", "license": "MIT",
        "dist": {"integrity": "sha512-SLR6ZSHxu6XdZtmOz3xRmC3gsY/sCzqtLMD1gxbW2I3XXPBbupwL5RZbkLU/NcZm74AKZrIUPfz3CtEJfxUKqQ=="}}),
    "https://api.npmjs.org/downloads/point/last-month/@eloqnt/config": (200, {"downloads": 2303119, "package": "@eloqnt/config"}),
    "https://api.npmjs.org/downloads/point/last-month/next-intl": (200, {"downloads": 9184402, "package": "next-intl"}),
    "https://api.npmjs.org/downloads/point/last-month/@acme/brand-new": (200, {"downloads": 3, "package": "@acme/brand-new"}),
    "https://pypi.org/pypi/httpx/0.28.1/json": (200, {
        "info": {"name": "httpx", "version": "0.28.1", "license": "BSD-3-Clause", "license_expression": None, "requires_python": ">=3.8"},
        "urls": [{"packagetype": "bdist_wheel", "digests": {"sha256": "d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad"}},
                 {"packagetype": "sdist", "digests": {"sha256": "75e98c5f16b0f35b567856f597f06ff2270a374470a5c2392242528e3e3e42fc"}}]}),
    "https://pypi.org/pypi/httpx/0.0.999/json": (404, {"message": "Not Found"}),
}


class _Recorder:
    """A ``fetch`` that answers only from the recording and remembers every URL asked."""

    def __init__(self, answers=RECORDED):
        self.answers, self.urls = answers, []

    def __call__(self, url):
        self.urls.append(url)
        return self.answers.get(url, (0, None))


def _registry(**kw):
    return g.Registry(_Recorder(), offline=False, **kw)


def _finding(title, explanation, *, path="package-lock.json", line=3, severity="high", category="correctness"):
    return rf.CodnaReviewFinding(path=path, line=line, severity=severity, category=category, title=title[:80],
                                 explanation=explanation, confidence=0.9, fingerprint=rf.fingerprint(path, category, title))


def _site(tmp_path, *, engines=">=20.19.5", lock_license="Apache-2.0", lock_sri=SPEED_INSIGHTS_SRI, supabase_js="2.116.0"):
    """A Vercel-style site: package.json with an engines RANGE and a v3 package-lock.json."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "site", "engines": {"node": engines},
        "dependencies": {"@supabase/supabase-js": "^2.108.2", "@supabase/ssr": "0.12.7", "@vercel/speed-insights": "^2.0.0",
                         "next-intl": "^4.14.4", "next": "15.5.0"}}), encoding="utf-8")
    lock = {"name": "site", "lockfileVersion": 3, "packages": {
        "": {"name": "site"},
        "node_modules/@vercel/speed-insights": {"version": "2.0.0", "license": lock_license, "integrity": lock_sri},
        "node_modules/@supabase/supabase-js": {"version": supabase_js, "engines": {"node": ">=22.0.0"}},
        "node_modules/@supabase/ssr": {"version": "0.12.7", "peerDependencies": {"@supabase/supabase-js": "^2.114.0"}},
        "node_modules/@eloqnt/config": {"version": "0.1.0"},
        "node_modules/next-intl": {"version": "4.14.4"},
        "node_modules/tinyexec": {"version": "1.3.0", "integrity": TINYEXEC_SRI},
        "node_modules/next/node_modules/tinyexec": {"version": "0.3.2", "integrity": "sha512-nested=="},
    }}
    (tmp_path / "package-lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return tmp_path


def _grounder(repo, changed=("package.json", "package-lock.json"), registry=None, log=None):
    return g.Grounder(str(repo), list(changed), registry=registry or _registry(), log=log or (lambda e, f: None))


# ---- claim detection on the real phrasings ---------------------------------------------------
@pytest.mark.parametrize("text,kinds", [
    ("tinyexec@1.3.0 does not exist on npm; install will fail. That version is not published on npm.", {"version_exists"}),
    ("Malformed SHA-512 integrity hash for @vercel/speed-insights 2.0.0", {"integrity"}),
    ("License field in lock file is wrong for @vercel/speed-insights 2.0.0", {"license"}),
    ("@supabase/supabase-js version range in package.json is below @supabase/ssr peer -- specifies a peer dependency", {"peer"}),
    ("Supabase packages require Node >=22 but project declares Node >=20. The project's own `engines` field", {"engines"}),
    ("Three brand-new (@eloqnt/*) packages with zero download history added as production deps", {"downloads"}),
    ("engines.node '>=20' incompatible with vitest 5's requirement of >=22.12.0", {"engines"}),
    ("Off-by-one in the pagination loop drops the last page", set()),
    ("Unbounded regex on user input (ReDoS) in the search handler", set()),
])
def test_detect_claims_recognises_the_kinds_the_review_agent_actually_wrote(text, kinds):
    assert g.detect_claims(text) == kinds


# ---- the evidence table: every false HIGH is contradicted by the registry and dropped ---------
def test_a_version_the_registry_serves_contradicts_does_not_exist_and_the_finding_is_dropped(tmp_path):
    """sqai#53: HIGH "tinyexec@1.3.0 does not exist on npm" -- registry.npmjs.org/tinyexec/1.3.0 is 200."""
    logged = []
    gr = _grounder(_site(tmp_path), log=lambda e, f: logged.append((e, f)))
    f = _finding("tinyexec@1.3.0 does not exist on npm; install will fail",
                 "The lockfile resolves vitest 5's dependency on tinyexec to version 1.3.0, but that version is not published on npm.")
    kept, counts = gr([f])
    assert kept == [] and counts["registry_contradicted"] == 1
    assert logged == [("review_finding_registry_contradicted", logged[0][1])]
    assert logged[0][1]["title"] == f.title and "tinyexec@1.3.0 is published on npm" in logged[0][1]["checks"][0]
    assert gr.registry.lookups == 1                       # one request for one package


def test_an_integrity_that_matches_dist_integrity_byte_for_byte_is_contradicted(tmp_path):
    """codna-site#54: HIGH "malformed SHA-512 integrity hash" -- the lockfile's SRI IS npm's."""
    gr = _grounder(_site(tmp_path))
    f = _finding("Malformed SHA-512 integrity hash for @vercel/speed-insights 2.0.0",
                 f"The integrity field `{SPEED_INSIGHTS_SRI}` decodes to only 48 bytes. A valid SHA-512 digest is always 64 bytes.",
                 category="security", line=3)
    [v] = gr.verdicts_for(f)
    assert (v.kind, v.package, v.outcome) == ("integrity", "@vercel/speed-insights", g.CONTRADICTED)
    assert gr([f]) == ([], {"registry_contradicted": 1, "registry_confirmed": 0, "registry_unverified": 0})


def test_a_lockfile_license_the_registry_publishes_is_contradicted(tmp_path):
    """codna-site#54: MEDIUM "license field in lock file is wrong" -- Apache-2.0 is what npm publishes for 2.0.0."""
    gr = _grounder(_site(tmp_path))
    f = _finding("License field in lock file is wrong for @vercel/speed-insights 2.0.0",
                 'The lock file still records `"license": "Apache-2.0"` for version 2.0.0; the release changed it to MIT.',
                 severity="medium")
    [v] = gr.verdicts_for(f)
    assert v.outcome == g.CONTRADICTED and "Apache-2.0" in v.note
    assert gr([f])[0] == []


def test_a_peer_the_lockfile_satisfies_is_not_a_violation(tmp_path):
    """cohenta-site#15 / accounts#71: "supabase-js range is below @supabase/ssr's peer ^2.114.0" -- the
    lockfile installs 2.116.0, which satisfies it."""
    gr = _grounder(_site(tmp_path))
    f = _finding("@supabase/supabase-js version range in package.json is below @supabase/ssr peer",
                 'package.json declares `"@supabase/supabase-js": "^2.108.2"`, but `@supabase/ssr@0.12.7` specifies a peer dependency of `"@supabase/supabase-js": "^2.114.0"`.',
                 path="package.json", line=4, severity="medium")
    outcomes = {(v.package, v.outcome) for v in gr.verdicts_for(f)}
    assert outcomes == {("@supabase/ssr", g.CONTRADICTED), ("@supabase/supabase-js", g.CONTRADICTED)}
    assert gr([f])[0] == []


def test_an_optional_peer_is_never_a_violation(tmp_path):
    """@vercel/speed-insights declares vue/next/react/svelte peers, every one optional: a "peer
    dependency not satisfied" finding about it has no violation to point at."""
    gr = _grounder(_site(tmp_path))
    f = _finding("@vercel/speed-insights 2.0.0 peer dependency on vue is not satisfied",
                 "The package lists vue ^3 and svelte >= 4 as peer dependencies but neither is installed.")
    [v] = gr.verdicts_for(f)
    assert v.outcome == g.CONTRADICTED and "optional peers excluded" in v.note


def test_a_required_peer_the_lockfile_violates_is_confirmed_and_keeps_its_severity(tmp_path):
    gr = _grounder(_site(tmp_path, supabase_js="2.110.0"))          # below ^2.114.0
    f = _finding("@supabase/ssr peer dependency on @supabase/supabase-js not satisfied",
                 "@supabase/ssr@0.12.7 needs @supabase/supabase-js ^2.114.0 as a peer dependency.", path="package.json", line=4)
    kept, counts = gr([f])
    assert counts["registry_confirmed"] == 1 and kept[0].severity == "high"
    assert "Registry check: confirmed" in kept[0].explanation and "@supabase/supabase-js@2.110.0 vs ^2.114.0" in kept[0].explanation


def test_zero_downloads_is_contradicted_by_the_download_counts(tmp_path):
    """cohenta-site#15: "@eloqnt/* zero download history" -- ~2.3M downloads last month, each."""
    gr = _grounder(_site(tmp_path))
    f = _finding("Three brand-new (@eloqnt/*) packages with zero download history added as production deps",
                 "next-intl@4.14.4 pulls in @eloqnt/config@0.1.0 and friends, which have no recorded download history.",
                 category="security", severity="medium")
    outcomes = {(v.package, v.outcome) for v in gr.verdicts_for(f)}
    assert outcomes == {("@eloqnt/config", g.CONTRADICTED), ("next-intl", g.CONTRADICTED)}     # @eloqnt/* expanded from the lockfile
    assert gr([f])[0] == []


def test_a_genuinely_obscure_package_confirms_the_downloads_claim(tmp_path):
    repo = _site(tmp_path)
    lock = json.loads((repo / "package-lock.json").read_text())
    lock["packages"]["node_modules/@acme/brand-new"] = {"version": "0.0.1"}
    (repo / "package-lock.json").write_text(json.dumps(lock))
    gr = _grounder(repo)
    kept, counts = gr([_finding("@acme/brand-new has zero downloads", "A brand-new package with no downloads.", category="security")])
    assert counts["registry_confirmed"] == 1 and kept[0].severity == "high" and "3 downloads" in kept[0].explanation


# ---- engines: compared against the runtime the repo declares ----------------------------------
ENGINES_TITLE = "Supabase packages require Node >=22 but project declares Node >=20"
ENGINES_TEXT = ("The lock file resolves @supabase/supabase-js to 2.116.0, whose engine constraint is `node >= 22.0.0`. "
                "The project's own `engines` field in package.json still declares `node >= 20.19.5`.")


def test_engines_range_alone_means_the_newest_supported_node_so_a_vercel_site_on_20_plus_is_fine(tmp_path):
    """cohenta-site#15: engines.node ">=20.19.5" and nothing pinned -> the site runs the newest Node the
    platform offers, which satisfies >=22; the HIGH is contradicted."""
    repo = _site(tmp_path)
    assert m.declared_node_runtimes(str(repo)) == ["26"]
    gr = _grounder(repo)
    [v] = gr.verdicts_for(_finding(ENGINES_TITLE, ENGINES_TEXT, path="package.json", line=2))
    assert v.outcome == g.CONTRADICTED and "Node 26" in v.note


def test_engines_alternation_range_picks_its_newest_major(tmp_path):
    """accounts#71: engines.node "^20.9.0 || ^22.0.0" -> Node 22, which satisfies supabase's >=22."""
    repo = _site(tmp_path, engines="^20.9.0 || ^22.0.0")
    assert m.declared_node_runtimes(str(repo)) == ["22"]
    [v] = _grounder(repo).verdicts_for(_finding(ENGINES_TITLE, ENGINES_TEXT, path="package.json", line=2))
    assert v.outcome == g.CONTRADICTED


def test_a_pinned_runtime_wins_over_the_engines_range(tmp_path):
    repo = _site(tmp_path)
    (repo / ".nvmrc").write_text("v24.8.0\n")
    assert m.declared_node_runtimes(str(repo)) == ["24.8.0"]
    [v] = _grounder(repo).verdicts_for(_finding(ENGINES_TITLE, ENGINES_TEXT, path="package.json", line=2))
    assert v.outcome == g.CONTRADICTED and "Node 24.8.0" in v.note


def test_a_ci_matrix_leg_below_the_requirement_confirms_the_engines_claim(tmp_path):
    repo = _site(tmp_path)
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  test:\n    strategy:\n      matrix:\n        node: [18, 22]\n    steps:\n"
        "      - uses: actions/setup-node@v4\n        with:\n          node-version: ${{ matrix.node }}\n")
    assert m.declared_node_runtimes(str(repo)) == ["18", "22"]
    kept, counts = _grounder(repo)([_finding(ENGINES_TITLE, ENGINES_TEXT, path="package.json", line=2)])
    assert counts["registry_confirmed"] == 1 and kept[0].severity == "high" and "Node 18" in kept[0].explanation


def test_a_dockerfile_node_image_is_a_pinned_runtime(tmp_path):
    repo = _site(tmp_path)
    (repo / "Dockerfile").write_text("FROM --platform=linux/amd64 docker.io/library/node:20-alpine\nCOPY . .\n")
    assert m.declared_node_runtimes(str(repo)) == ["20"]
    [v] = _grounder(repo).verdicts_for(_finding(ENGINES_TITLE, ENGINES_TEXT, path="package.json", line=2))
    assert v.outcome == g.CONFIRMED


def test_a_package_with_no_engines_field_contradicts_any_requires_node_claim(tmp_path):
    """@types/node 22->26 (accounts#71, algenta-site#21, ...): @types/node declares no engines at all."""
    repo = _site(tmp_path)
    lock = json.loads((repo / "package-lock.json").read_text())
    lock["packages"]["node_modules/@types/node"] = {"version": "26.0.0", "dev": True}
    (repo / "package-lock.json").write_text(json.dumps(lock))
    [v] = _grounder(repo).verdicts_for(_finding("@types/node 26 requires Node >=22 (engines violation)",
                                                "@types/node@26.0.0 targets Node 26 APIs and requires Node >= 22 at runtime."))
    assert v.outcome == g.CONTRADICTED and "declares no engines.node" in v.note


def test_engines_with_no_declared_runtime_is_unverified_not_asserted(tmp_path):
    repo = _site(tmp_path)
    (repo / "package.json").write_text(json.dumps({"name": "x", "dependencies": {"@supabase/supabase-js": "^2.116.0"}}))
    assert m.declared_node_runtimes(str(repo)) == []
    kept, counts = _grounder(repo)([_finding(ENGINES_TITLE, ENGINES_TEXT, path="package.json", line=2)])
    assert counts["registry_unverified"] == 1 and kept[0].severity == "low"
    assert kept[0].title.startswith("Unverified: ") and "declares no Node runtime" in kept[0].explanation


# ---- offline / unreachable: unverified LOW, never a fact ---------------------------------------
def test_offline_downgrades_every_registry_claim_to_unverified_low_and_asks_nothing(tmp_path):
    recorder = _Recorder()
    gr = _grounder(_site(tmp_path), registry=g.Registry(recorder, offline=True))
    f = _finding("tinyexec@1.3.0 does not exist on npm; install will fail", "That version is not published on npm.")
    kept, counts = gr([f])
    assert counts == {"registry_contradicted": 0, "registry_confirmed": 0, "registry_unverified": 1}
    assert recorder.urls == []                                        # nothing left the process
    [low] = kept
    assert low.severity == "low" and low.title == "Unverified: tinyexec@1.3.0 does not exist on npm; install will fail"
    assert low.explanation.startswith("Unverified: codna could not check this claim against the package registry")
    assert "Treat it as a question" in low.explanation and low.fingerprint == f.fingerprint   # dedup key intact


def test_offline_policy_env_vars_switch_the_registry_off(monkeypatch):
    monkeypatch.delenv("CI_OFFLINE_CONTRACT_TEST", raising=False)
    monkeypatch.delenv("CODNA_REQUIRE_EGRESS_DENY", raising=False)
    assert g.offline_by_policy() is False
    monkeypatch.setenv("CODNA_REQUIRE_EGRESS_DENY", "1")                 # privacy.egress: fail-closed
    assert g.offline_by_policy() is True and g.Registry(_Recorder()).offline is True
    monkeypatch.delenv("CODNA_REQUIRE_EGRESS_DENY")
    monkeypatch.setenv("CI_OFFLINE_CONTRACT_TEST", "1")
    assert g.Registry(_Recorder()).offline is True


def test_a_fetch_that_raises_or_times_out_is_unverified(tmp_path):
    def boom(url):
        raise OSError("network is down")

    kept, counts = _grounder(_site(tmp_path), registry=g.Registry(boom, offline=False))(
        [_finding("Malformed SHA-512 integrity hash for @vercel/speed-insights 2.0.0", f"`{SPEED_INSIGHTS_SRI}` is truncated.")])
    assert counts["registry_unverified"] == 1 and kept[0].severity == "low" and "unreachable" in kept[0].explanation


def test_the_lookup_budget_bounds_the_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_MAX_LOOKUPS", 1)
    gr = _grounder(_site(tmp_path))
    first = _finding("tinyexec@1.3.0 does not exist on npm", "not published on npm.")
    second = _finding("@eloqnt/config@0.1.0 has zero downloads", "no download history.", category="security")
    kept, counts = gr([first, second])
    assert counts == {"registry_contradicted": 1, "registry_confirmed": 0, "registry_unverified": 1}
    assert [f.severity for f in kept] == ["low"] and gr.registry.lookups == 1


# ---- confirmed claims keep their severity -----------------------------------------------------
def test_a_version_the_registry_lacks_confirms_does_not_exist(tmp_path):
    repo = _site(tmp_path)
    kept, counts = _grounder(repo)([_finding("tinyexec@9.9.9 does not exist on npm", "The lockfile pins tinyexec@9.9.9, which is not published on npm.")])
    assert counts["registry_confirmed"] == 1 and kept[0].severity == "high"
    assert kept[0].explanation.endswith("Registry check: confirmed (npm has no tinyexec@9.9.9).")


def test_a_lockfile_integrity_that_differs_from_the_registry_is_confirmed(tmp_path):
    repo = _site(tmp_path, lock_sri="sha512-AAAAtampered==")
    kept, counts = _grounder(repo)([_finding("Integrity mismatch for @vercel/speed-insights 2.0.0",
                                             "The lockfile's integrity hash does not match the published tarball.", category="security")])
    assert counts["registry_confirmed"] == 1 and kept[0].severity == "high" and "differs from npm's" in kept[0].explanation


# ---- scope: what is (not) grounded -------------------------------------------------------------
def test_a_finding_without_a_registry_claim_passes_through_untouched(tmp_path):
    f = _finding("Race in the retry loop", "Two workers can both take the lease; `next` is read twice.", path="src/app.ts", line=7)
    kept, counts = _grounder(_site(tmp_path))([f])
    assert kept == [f] and counts == {"registry_contradicted": 0, "registry_confirmed": 0, "registry_unverified": 0}


def test_a_source_finding_that_merely_mentions_a_dependency_word_is_not_grounded_away(tmp_path):
    """"hash mismatch" in application code next to the word `next` must not be checked against npm's
    `next` package and dropped: off a manifest, only a package named outright counts."""
    f = _finding("Hash mismatch: the cache key uses the wrong digest",
                 "computeKey() hashes the body but compares against the header hash; next request misses.", path="src/cache.ts", line=12)
    gr = _grounder(_site(tmp_path))
    assert gr.verdicts_for(f) == [] and gr([f])[0] == [f]


def test_a_source_finding_naming_a_package_outright_is_grounded(tmp_path):
    f = _finding("tinyexec@1.3.0 does not exist on npm", "The import in src/run.ts pulls tinyexec@1.3.0, which is not published on npm.",
                 path="src/run.ts", line=1)
    assert _grounder(_site(tmp_path))([f])[0] == []


def test_grounder_for_is_none_when_the_change_touches_no_manifest(tmp_path):
    assert g.grounder_for(str(tmp_path), ["src/a.py", "README.md"]) is None
    assert g.grounder_for(str(tmp_path), ["src/a.py", "package-lock.json"]) is not None
    assert g.grounder_for(str(tmp_path), ["requirements-dev.txt"]) is not None
    assert g.ecosystem_of("packages/cli/pnpm-lock.yaml") == "npm" and g.ecosystem_of("uv.lock") == "pypi"
    assert g.ecosystem_of("src/lock.py") is None


def test_finalize_findings_grounds_before_the_noise_controls_and_reports_the_counts(tmp_path):
    """A refuted HIGH neither reaches the PR nor takes a max_findings slot; the counts join `dropped`.
    Without a grounder the transform is exactly what it was."""
    repo = _site(tmp_path)
    raw = [
        {"path": "package-lock.json", "line": 3, "severity": "high", "category": "correctness", "confidence": 0.95,
         "title": "tinyexec@1.3.0 does not exist on npm; install will fail", "explanation": "not published on npm."},
        {"path": "package.json", "line": 2, "severity": "low", "category": "correctness", "confidence": 0.9,
         "title": "engines field could be tightened", "explanation": "Cosmetic."},
    ]
    files = rf.parse_diff("--- a/package.json\n+++ b/package.json\n@@ -1,3 +1,3 @@\n {\n-  \"a\": 1\n+  \"a\": 2\n }\n")
    cfg = rf.ReviewConfig(max_findings=1)
    inline, summary, dropped = rf.finalize_findings(raw, files, "h" * 40, cfg, grounder=_grounder(repo))
    assert [f.title for f in [*inline, *summary]] == ["engines field could be tightened"]      # the refuted HIGH freed the slot
    assert dropped["registry_contradicted"] == 1 and dropped["over_cap"] == 0
    plain_inline, plain_summary, plain_dropped = rf.finalize_findings(raw, files, "h" * 40, cfg)
    assert [f.title for f in [*plain_inline, *plain_summary]] == ["tinyexec@1.3.0 does not exist on npm; install will fail"]
    assert "registry_contradicted" not in plain_dropped


# ---- PyPI ---------------------------------------------------------------------------------------
def _pyrepo(tmp_path, python=None):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nrequires-python = ">=3.12"\n')
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "httpx"\nversion = "0.28.1"\nsource = { registry = "https://pypi.org/simple" }\n'
        'sdist = { url = "https://files.pythonhosted.org/x.tar.gz", hash = "sha256:75e98c5f16b0f35b567856f597f06ff2270a374470a5c2392242528e3e3e42fc" }\n'
        'wheels = [\n    { url = "https://files.pythonhosted.org/x.whl", hash = "sha256:d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad" },\n]\n')
    if python:
        (tmp_path / ".python-version").write_text(python + "\n")
    return tmp_path


def test_pypi_version_existence_and_hashes_are_grounded(tmp_path):
    gr = _grounder(_pyrepo(tmp_path), changed=("pyproject.toml", "uv.lock"))
    assert gr([_finding("httpx==0.28.1 does not exist on PyPI", "uv.lock pins httpx 0.28.1, which was never published.", path="uv.lock", line=4)])[0] == []
    kept, counts = gr([_finding("httpx==0.0.999 does not exist on PyPI", "uv.lock pins httpx==0.0.999.", path="uv.lock", line=4)])
    assert counts["registry_confirmed"] == 1 and kept[0].severity == "high"
    [v] = gr.verdicts_for(_finding("uv.lock hash for httpx 0.28.1 is corrupted", "The sha256 recorded for httpx==0.28.1 does not match PyPI.", path="uv.lock", line=6))
    assert v.outcome == g.CONTRADICTED


def test_pypi_requires_python_is_compared_against_the_declared_interpreter(tmp_path):
    gr = _grounder(_pyrepo(tmp_path, python="3.12"), changed=("uv.lock",))
    assert m.declared_python_runtimes(str(tmp_path)) == ["3.12"]
    [v] = gr.verdicts_for(_finding("httpx 0.28.1 requires Python >=3.8 which the project does not meet",
                                   "requires_python is >=3.8 for httpx==0.28.1.", path="uv.lock", line=4))
    assert v.outcome == g.CONTRADICTED
    (tmp_path / ".python-version").write_text("3.7.9\n")
    [v] = _grounder(tmp_path, changed=("uv.lock",)).verdicts_for(
        _finding("httpx 0.28.1 requires Python >=3.8 which the project does not meet", "requires_python is >=3.8 for httpx==0.28.1.", path="uv.lock", line=4))
    assert v.outcome == g.CONFIRMED and "3.7.9" in v.note


# ---- the pure helpers ---------------------------------------------------------------------------
@pytest.mark.parametrize("version,rng,expected", [
    ("22", "^22.12.0 || ^24.0.0 || >=26.0.0", True),      # a bare major is any 22.x: the range admits 22.12+
    ("20", "^22.12.0 || ^24.0.0 || >=26.0.0", False),
    ("24.8.0", "^22.12.0 || ^24.0.0 || >=26.0.0", True),
    ("26", ">=20.19.5", True),
    ("22", "^20.9.0 || ^22.0.0", True),
    ("24", "^20.9.0 || ^22.0.0", False),
    ("18", ">=22.0.0", False),
    ("22.11.0", ">=18.17.0 <23", True),
    ("22", "20 - 21", False),
    ("21.5.0", "20 - 21", True),
    ("22", "22.x", True),
    ("2.116.0", "^2.114.0", True),
    ("2.110.0", "^2.114.0", False),
    ("0.12.7", "~0.12.0", True),
    ("0.13.0", "^0.12.0", False),
    ("3.12", ">=3.8,<4", True),
    ("3.12", "~=3.8", True),
    ("3.7", ">=3.8", False),
    ("3.12", ">=3.8, !=3.9.*", True),
    ("3.9", ">=3.8, !=3.9.*", False),                     # the exclusion covers all of 3.9.z
    ("3.9.1", ">=3.8, !=3.9.0", True),                    # ...but a point exclusion only that point
    ("3.8", "!=3.8", False),                              # a bare exclusion is honoured, not a no-op
    ("3.12", "!=3.8", True),
    ("22", "!=22 || >=24", False),
    ("24", "!=22 || >=24", True),
    ("22", "", True),
    ("22", "*", True),
    ("22", "not a range", None),
    ("weird", ">=22", None),
])
def test_satisfies(version, rng, expected):
    assert m.satisfies(version, rng) is expected


def test_packages_in_text_names_explicit_mentions_the_anchor_and_whole_word_candidates_only():
    cands = {"next", "debug", "@supabase/ssr", "@supabase/supabase-js", "@eloqnt/config", "@eloqnt/format-po", "tinyexec", "speed-insights"}
    refs = g.packages_in_text("tinyexec@1.3.0 and `debug` and @eloqnt/* vs version 1.3.0; next steps", cands, "@supabase/ssr", title="t")
    assert {(r.name, r.version) for r in refs} == {("tinyexec", "1.3.0"), ("debug", None), ("@eloqnt/config", None),
                                                    ("@eloqnt/format-po", None), ("@supabase/ssr", None)}
    assert g.packages_in_text("the next release", cands) == []                        # a bare word is not a package
    assert [r.name for r in g.packages_in_text("the next release", cands, title="next 15 breaks")] == ["next"]
    assert [(r.name, r.version) for r in g.packages_in_text("@vercel/speed-insights 2.0.0 hash", set())] == [("@vercel/speed-insights", "2.0.0")]
    assert g.packages_in_text("version 1.3.0 of something", set()) == []


def test_lockfile_readers_cover_npm_pnpm_yarn_and_python(tmp_path):
    pnpm = ("lockfileVersion: '9.0'\n\npackages:\n\n  '@turbo/darwin-64@2.10.2':\n    resolution: {integrity: sha512-wBM3==}\n    cpu: [x64]\n\n"
            "  tinyexec@1.3.0(@types/node@26.0.0):\n    resolution: {integrity: sha512-QKAl==}\n\nsnapshots:\n\n  tinyexec@1.3.0:\n    dependencies: {}\n")
    entries = m._pnpm_lock_entries(pnpm)
    assert entries["@turbo/darwin-64"].version == "2.10.2" and entries["@turbo/darwin-64"].integrity == "sha512-wBM3=="
    assert entries["tinyexec"].version == "1.3.0" and entries["tinyexec"].integrity == "sha512-QKAl=="
    # pnpm v6-8: the same name under several keys. The same version under another peer suffix is the
    # same tarball (and may carry the integrity the first key lacked); a SECOND version of the name
    # never attaches to -- or overwrites -- the first-seen entry.
    dup = ("lockfileVersion: '6.0'\n\npackages:\n\n  /tinyexec@1.3.0(@types/node@26.0.0):\n    resolution: {tarball: https://x/tinyexec-1.3.0.tgz}\n\n"
           "  /tinyexec@1.3.0:\n    resolution: {integrity: sha512-SAME==}\n\n"
           "  /tinyexec@0.3.2:\n    resolution: {integrity: sha512-OLDER==}\n")
    d = m._pnpm_lock_entries(dup)
    assert d["tinyexec"].version == "1.3.0" and d["tinyexec"].integrity == "sha512-SAME=="
    yarn = ('# yarn lockfile v1\n\n\n"@scope/pkg@^1.0.0", "@scope/pkg@~1.2.0":\n  version "1.2.3"\n  resolved "https://x"\n  integrity sha512-YARN==\n\n'
            'left-pad@^1.3.0:\n  version "1.3.0"\n')
    y = m._yarn_lock_entries(yarn)
    assert y["@scope/pkg"].version == "1.2.3" and y["@scope/pkg"].integrity == "sha512-YARN==" and y["left-pad"].version == "1.3.0"
    req = m._requirements_entries("httpx==0.28.1 \\\n    --hash=sha256:" + "a" * 64 + "\nrequests[socks]==2.32.0\n")
    assert req["httpx"].version == "0.28.1" and req["httpx"].hashes == ("a" * 64,) and req["requests"].version == "2.32.0"
    repo = _site(tmp_path)
    mani = m.RepoManifests(str(repo), ["package-lock.json"])
    assert mani.lock_entries("package-lock.json", "npm")["tinyexec"].version == "1.3.0"      # hoisted entry wins over the nested one
    lines = (repo / "package-lock.json").read_text().splitlines()
    sri_line = next(i for i, ln in enumerate(lines, 1) if SPEED_INSIGHTS_SRI in ln)
    assert mani.package_at_line("package-lock.json", sri_line) == "@vercel/speed-insights"
    (repo / "pnpm-lock.yaml").write_text(pnpm)
    assert mani.package_at_line("pnpm-lock.yaml", 6) == "@turbo/darwin-64" and mani.package_at_line("pnpm-lock.yaml", 1) is None
    (repo / "package.json").write_text('{\n  "dependencies": {\n    "@supabase/ssr": "0.12.7"\n  }\n}\n')
    assert mani.package_at_line("package.json", 3) == "@supabase/ssr" and mani.package_at_line("package.json", 2) is None
