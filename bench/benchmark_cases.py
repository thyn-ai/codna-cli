"""Benchmark case manifest helpers.

The executable benchmark needs an explicit case source. Silent random GitHub discovery makes
results non-reproducible; reading prior tracked soak cases keeps the 100-case pool auditable.
"""
from __future__ import annotations

from pathlib import Path
import re

Case = tuple[str, str]

HERE = Path(__file__).resolve().parent
SOAK_CASE_DIR = HERE / "soak"
SOAK_CASE_PATTERN = re.compile(r"^_repo:\s*(https://github\.com/[^\s]+)\s*\u00b7\s*issue:\s*(.+)_$")

MULTILANG_CASES: list[Case] = [
    ("https://github.com/psf/requests", "Session.send does not retry when the underlying connection times out; add a retry-on-timeout path"),
    ("https://github.com/date-fns/date-fns", "format() with the SSS millisecond token drops a leading zero for sub-100ms values"),
    ("https://github.com/gin-gonic/gin", "Context.ShouldBindQuery ignores the `default` struct tag for a missing query parameter"),
    ("https://github.com/BurntSushi/ripgrep", "--count-matches double-counts overlapping matches on a single line"),
    ("https://github.com/google/gson", "the default Date type adapter fails to parse an ISO-8601 timestamp with a trailing Z zone"),
    ("https://github.com/nlohmann/json", "parse() rejects an otherwise-valid document that begins with a UTF-8 BOM"),
    ("https://github.com/JamesNK/Newtonsoft.Json", "serializing a DateTimeOffset with a negative UTC offset emits the wrong offset sign"),
    ("https://github.com/sinatra/sinatra", "a route with an optional named parameter fails to match when the parameter is omitted"),
    ("https://github.com/guzzle/guzzle", "query params with array values are encoded without the [] suffix"),
    ("https://github.com/Alamofire/Alamofire", "URLEncoding does not percent-encode a literal + character in query values"),
]


class BenchmarkCaseError(ValueError):
    """Raised when tracked benchmark case files are malformed."""


def load_tracked_soak_cases(case_dir: Path = SOAK_CASE_DIR) -> list[Case]:
    if not case_dir.is_dir():
        return []
    cases: list[Case] = []
    for path in sorted(case_dir.glob("*.md")):
        parsed = _parse_soak_case_file(path)
        if parsed is not None:
            cases.append(parsed)
    return cases


def benchmark_cases(*, include_tracked_soak: bool = True) -> list[Case]:
    cases = list(MULTILANG_CASES)
    if include_tracked_soak:
        cases.extend(load_tracked_soak_cases())
    return _dedupe_by_repo_url(cases)


def _parse_soak_case_file(path: Path) -> Case | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SOAK_CASE_PATTERN.match(line.strip())
        if match:
            return match.group(1), match.group(2)
    return None


def _dedupe_by_repo_url(cases: list[Case]) -> list[Case]:
    deduped: list[Case] = []
    seen: set[str] = set()
    for repo_url, issue in cases:
        normalized = repo_url.rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((repo_url, issue))
    return deduped
