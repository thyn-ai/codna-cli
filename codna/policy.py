"""Autofix policy — who is eligible for an automated fix PR, and how severity maps.

Encodes gate condition G3 (policy-eligible proof): the default eligible classification is
`exploitable`; `production-reachable` is eligible ONLY when the policy explicitly opts in
(an override the operator must record); `unreachable` and `unknown` are NEVER auto-fixed.
Also carries the scanner-severity overrides `sarif.ingest_sarif` applies, and the set of
new-finding severities that block a PR (G8).

Pure stdlib, no engine/http. Deterministic `digest` so the policy participates in the
immutable-provenance binding (G1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .findings import Classification, Severity, digest_of

# Default new-finding severities that block a PR (G8) and the default eligible set (G3).
DEFAULT_BLOCK_NEW = (Severity.HIGH, Severity.CRITICAL)
DEFAULT_AUTOFIX = ("exploitable",)


def _as_severity(v) -> Severity:
    return v if isinstance(v, Severity) else Severity(str(v).strip().lower())


@dataclass
class Policy:
    autofix_classifications: tuple[str, ...] = DEFAULT_AUTOFIX
    block_new_severities: tuple[Severity, ...] = DEFAULT_BLOCK_NEW
    severity_overrides: dict[str, Severity] = field(default_factory=dict)
    digest: str = ""

    def __post_init__(self):
        # Normalize, then bind a content digest so the policy can be provenance-pinned (G1).
        self.autofix_classifications = tuple(
            c.value if isinstance(c, Classification) else str(c).strip().lower()
            for c in self.autofix_classifications
        )
        self.block_new_severities = tuple(_as_severity(s) for s in self.block_new_severities)
        self.severity_overrides = {k: _as_severity(v) for k, v in self.severity_overrides.items()}
        if not self.digest:
            self.digest = self.compute_digest()

    def compute_digest(self) -> str:
        return digest_of(
            {
                "autofix_classifications": sorted(self.autofix_classifications),
                "block_new_severities": sorted(s.value for s in self.block_new_severities),
                "severity_overrides": {k: v.value for k, v in sorted(self.severity_overrides.items())},
            }
        )

    @classmethod
    def from_dict(cls, d: dict | None) -> "Policy":
        d = d or {}
        return cls(
            autofix_classifications=tuple(d.get("autofix_classifications", DEFAULT_AUTOFIX)),
            block_new_severities=tuple(d.get("block_new_severities", DEFAULT_BLOCK_NEW)),
            severity_overrides={k: _as_severity(v) for k, v in (d.get("severity_overrides") or {}).items()},
        )

    # -- G3: which reachability verdicts may receive an automated fix PR --------------

    def is_autofix_eligible(self, classification: "Classification | str") -> tuple[bool, str]:
        c = classification.value if isinstance(classification, Classification) else str(classification).lower()
        if c in (Classification.UNREACHABLE.value, Classification.UNKNOWN.value):
            return False, f"{c} findings are never auto-fixed"
        if c == Classification.EXPLOITABLE.value:
            ok = "exploitable" in self.autofix_classifications
            return ok, "exploitable (policy default)" if ok else "exploitable not in policy autofix set"
        if c == Classification.PRODUCTION_REACHABLE.value:
            ok = "production-reachable" in self.autofix_classifications
            return (
                ok,
                "production-reachable allowed by explicit policy override"
                if ok
                else "production-reachable requires an explicit policy override",
            )
        return False, f"unrecognized classification {classification!r}"

    # -- G8: does a newly-introduced finding's severity block the PR? -----------------

    def blocks_new_severity(self, severity: "Severity | str") -> bool:
        return _as_severity(severity) in self.block_new_severities
