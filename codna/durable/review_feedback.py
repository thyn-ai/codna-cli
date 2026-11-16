"""What the reviewer SAID when it rejected a change — carried to the next implement attempt.

**The defect this module exists for.** `after_review` routes a rejection back to `implement`, but the
implement prompt was built from the brief and human steering only: there was no parameter for the
reviewer's findings, nothing in `TaskState` carried them, and the retry therefore told the model
"try again" with no new information. Measured (`run-20261116T005743Z`, installed build `6f7a904d`):
verify passed `true` on an unchanged SHA `8cce0fa1` three times, review returned
`changes_requested` three times, and the 3rd and 4th implement prompts in the ledger were
BYTE-IDENTICAL. A `changes_requested` loop can never converge that way — and the run correctly
refused to self-approve and parked instead, so the work became unrecoverable without a human.

**Why this is a record and not a blob of prompt text.** Three properties decide the design:

* **It must be RENDERED, not dumped.** The model gets a path, a line, a severity and an actionable
  statement. Raw findings JSON would be a second, unstated schema for the model to guess at, and the
  severity is what tells it which finding to fix first.
* **It must be BOUNDED, twice.** The retry is already attempt-capped, so a findings list cannot grow
  without limit — but a reviewer may legitimately return hundreds of findings (`max_findings: 0` in
  `codna.yaml` disables the cap). So the RECORD carries at most `MAX_FINDINGS_RECORDED`, remembers
  how many it did not carry, and the RENDERER states that number rather than silently dropping it.
* **It must be DETERMINISTIC and replay-safe.** No clock, no I/O, no randomness, and no dependence
  on dict ordering: the findings are sorted into a TOTAL order (`_sort_key`) before the cap is
  applied, so the same review always yields the same record. The record lives in `TaskState`, which
  is the workflow's memory, so a replayed workflow rebuilds the same prompt from history.

**Leaf module.** stdlib only. `state.py` imports it and `state.py` is imported by Temporal's workflow
sandbox, which refuses anything reaching `urllib.request`. `TaskStateError` is INJECTED as a factory
(`fail=...`) for the same reason `publish.py` and `state_coercion.py` do it: the error type stays
defined in exactly one place and this module never imports back into `state.py`.

**The severity vocabulary is duplicated deliberately.** `review_findings.SEVERITIES` is the source of
truth, but importing it here would pull the review PIPELINE (`subprocess`, `pathlib`, the agent
runner) into the workflow sandbox on every import of `state.py`. `tests/durable/test_review_feedback.py`
parses the tuple instead, so a severities change is a red test rather than a stale copy.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Final, Iterable, Literal, Mapping

from .state_guards import find_secret_like_strings, redact_secret_like

__all__ = [
    "MAX_FINDINGS_RECORDED",
    "MAX_REVIEW_FINDING_EXPLANATION_CHARS",
    "MAX_REVIEW_FINDING_PATH_CHARS",
    "MAX_REVIEW_FINDING_TITLE_CHARS",
    "REVIEW_SEVERITIES",
    "ReviewFeedback",
    "ReviewFinding",
    "review_feedback_from_payload",
    "review_finding_from_report",
    "review_findings_from_report",
]

#: The severities a finding may carry, most severe first. MUST equal `review_findings.SEVERITIES`
#: (pinned by a test) — see the module docstring for why the import is not taken here.
REVIEW_SEVERITIES: Final[tuple[str, ...]] = ("high", "medium", "low")

#: The rank each severity sorts by, so the most severe finding is the one a truncated list keeps.
_SEVERITY_RANK: Final[dict[str, int]] = {"high": 3, "medium": 2, "low": 1}

#: How many findings a record may carry. The review's own `max_findings` defaults to 10, so this only
#: bites when a repository has explicitly removed that cap — and when it does, the count of what was
#: dropped is recorded and then rendered, rather than the list being silently shortened.
MAX_FINDINGS_RECORDED: Final = 12

#: Per-field bounds. Together they cap one rendered finding at roughly 1.3 kB, so the whole block is
#: ~16 kB at the record limit: comfortably inside `TurnRequest`'s 64,000-character prompt limit, and
#: small enough that a hostile or verbose reviewer cannot crowd out the task itself.
MAX_REVIEW_FINDING_PATH_CHARS: Final = 400
MAX_REVIEW_FINDING_TITLE_CHARS: Final = 200
MAX_REVIEW_FINDING_EXPLANATION_CHARS: Final = 600
MAX_REVIEW_FINDING_CATEGORY_CHARS: Final = 40

#: The fingerprint is `review_findings.fingerprint`'s shape: 16 lowercase hex characters. Bounded
#: rather than free text because it reaches workflow history and is only ever used to identify a
#: finding, never to describe one.
_FINGERPRINT_CHARS: Final = 64

#: The two verdicts a feedback record may carry. `approved` is deliberately absent: an approval
#: produces NO record (see `review_findings_from_report`), so a state carrying feedback always means
#: "the last review rejected this".
RejectedVerdict = Literal["rejected", "changes_requested"]


def _clip(text: str, limit: int) -> str:
    """Hard-bound a string, marking the cut. Deterministic: a pure function of (text, limit)."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


@dataclass(frozen=True)
class ReviewFinding:
    """ONE reason the reviewer rejected the change, in the shape the prompt renders.

    Every field is bounded and every field is essential to acting on the finding: the ``path`` and
    ``line`` say WHERE, the ``severity`` says HOW URGENTLY, and ``title`` is the actionable statement.
    ``explanation`` and ``category`` are supporting detail, so they are what the bounds cut first.
    """

    path: str
    line: int
    severity: str
    title: str
    category: str = ""
    explanation: str = ""
    #: `review_findings.fingerprint(path, category, title)` — stable across a force-push, and what
    #: lets a later reader tell "the same finding, moved" from "a new finding". Optional because a
    #: caller that has only the rendered fields can still make a correct record.
    fingerprint: str = ""

    def validate(self, *, fail: Callable[..., Exception]) -> None:
        """Refuse anything the prompt could not render, or that must never enter history."""
        if not isinstance(self.path, str) or not self.path.strip():
            raise fail(
                "a review finding needs a non-empty path",
                code="invalid_review_feedback",
                details={"path": repr(self.path)[:120]},
            )
        # A record that is CONSTRUCTED rather than decoded must not smuggle a credential either:
        # the same value patterns the state guard applies are checked here, at the record, so the
        # refusal names the finding instead of surfacing later as "the task state contains secret-like
        # data" with a JSON path into a workflow payload.
        leaked = find_secret_like_strings(
            {"title": self.title, "explanation": self.explanation, "category": self.category}
        )
        if leaked:
            raise fail(
                "a review finding contains secret-like data and may not enter workflow history",
                code="invalid_review_feedback",
                details={"fields": [entry["path"] for entry in leaked]},
            )
        if len(self.path) > MAX_REVIEW_FINDING_PATH_CHARS:
            raise fail(
                f"review finding path is over {MAX_REVIEW_FINDING_PATH_CHARS} characters",
                code="invalid_review_feedback",
                details={"length": len(self.path)},
            )
        if not isinstance(self.line, int) or isinstance(self.line, bool) or self.line < 1:
            raise fail(
                "a review finding needs a line >= 1",
                code="invalid_review_feedback",
                details={"line": repr(self.line)},
            )
        if self.severity not in REVIEW_SEVERITIES:
            raise fail(
                f"review finding severity must be one of {', '.join(REVIEW_SEVERITIES)}",
                code="invalid_review_feedback",
                details={"severity": repr(self.severity)},
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise fail(
                "a review finding needs a non-empty title",
                code="invalid_review_feedback",
                details={"title": repr(self.title)[:120]},
            )
        if len(self.title) > MAX_REVIEW_FINDING_TITLE_CHARS:
            raise fail(
                f"review finding title is over {MAX_REVIEW_FINDING_TITLE_CHARS} characters",
                code="invalid_review_feedback",
                details={"length": len(self.title)},
            )
        for name, value, limit in (
            ("category", self.category, MAX_REVIEW_FINDING_CATEGORY_CHARS),
            ("explanation", self.explanation, MAX_REVIEW_FINDING_EXPLANATION_CHARS),
            ("fingerprint", self.fingerprint, _FINGERPRINT_CHARS),
        ):
            if not isinstance(value, str):
                raise fail(
                    f"review finding {name} must be a string",
                    code="invalid_review_feedback",
                    details={"field": name, "type": type(value).__name__},
                )
            if len(value) > limit:
                raise fail(
                    f"review finding {name} is over {limit} characters",
                    code="invalid_review_feedback",
                    details={"field": name, "length": len(value)},
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "severity": self.severity,
            "title": self.title,
            "category": self.category,
            "explanation": self.explanation,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_payload(cls, raw: Any, *, fail: Callable[..., Exception]) -> "ReviewFinding":
        """Decode one finding, naming the offending FIELD rather than "invalid feedback"."""
        data = _mapping(raw, "review_finding", fail=fail)
        value = cls(
            path=_text(data, "path", "review_finding.path", fail=fail),
            line=_integer(data, "line", "review_finding.line", fail=fail),
            severity=_text(data, "severity", "review_finding.severity", fail=fail),
            title=_text(data, "title", "review_finding.title", fail=fail),
            category=_optional_text(data, "category", fail=fail),
            explanation=_optional_text(data, "explanation", fail=fail),
            fingerprint=_optional_text(data, "fingerprint", fail=fail),
        )
        value.validate(fail=fail)
        return value


@dataclass(frozen=True)
class ReviewFeedback:
    """The findings a REJECTING review produced, plus what the record could not carry.

    ``omitted`` is a first-class field rather than something a reader recomputes, because the raw
    count is the only evidence that the list it is about to read is incomplete. A reader that had to
    infer it from "fewer than some maximum" would be inventing a fact.
    """

    #: The verdict that produced this record. Never `approved` (see `RejectedVerdict`).
    verdict: RejectedVerdict
    #: The checkpoint SHA the findings are about — the attempt that was rejected. Recorded so a human
    #: (or a later reader) can tell WHICH attempt the reasons belong to when a retry changes the tree.
    sha: str
    findings: tuple[ReviewFinding, ...]
    #: How many findings the reviewer returned that this record does NOT carry, because of
    #: `MAX_FINDINGS_RECORDED`. Rendered as an explicit line, never silently dropped.
    omitted: int = 0

    def validate(self, *, fail: Callable[..., Exception]) -> None:
        if self.verdict not in ("rejected", "changes_requested"):
            raise fail(
                "review feedback may only record a REJECTED review",
                code="invalid_review_feedback",
                details={"verdict": repr(self.verdict)},
            )
        if not isinstance(self.sha, str) or not self.sha.strip():
            raise fail(
                "review feedback needs the checkpoint sha it is about",
                code="invalid_review_feedback",
                details={"sha": repr(self.sha)},
            )
        if not isinstance(self.findings, tuple) or not self.findings:
            raise fail(
                "review feedback needs at least one finding (an approval records no feedback)",
                code="invalid_review_feedback",
                details={"findings": len(self.findings) if isinstance(self.findings, tuple) else None},
            )
        if len(self.findings) > MAX_FINDINGS_RECORDED:
            raise fail(
                f"review feedback carries {len(self.findings)} findings, over the "
                f"{MAX_FINDINGS_RECORDED} a record may hold",
                code="invalid_review_feedback",
                details={"count": len(self.findings), "limit": MAX_FINDINGS_RECORDED},
            )
        if not isinstance(self.omitted, int) or isinstance(self.omitted, bool) or self.omitted < 0:
            raise fail(
                "review feedback omitted count must be a non-negative integer",
                code="invalid_review_feedback",
                details={"omitted": repr(self.omitted)},
            )
        for finding in self.findings:
            finding.validate(fail=fail)

    @property
    def digest(self) -> str:
        """Stable digest of the record: what the ledger stamps, so a log can cite it without the text."""
        body = "|".join(
            f"{item.path}:{item.line}:{item.severity}:{item.title}:{item.fingerprint}"
            for item in self.findings
        )
        raw = f"{self.verdict}:{self.sha}:{self.omitted}:{body}".encode("utf-8", "replace")
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_payload(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "sha": self.sha,
            "omitted": self.omitted,
            "findings": [finding.to_payload() for finding in self.findings],
        }

    @classmethod
    def from_payload(
        cls, raw: Any, *, fail: Callable[..., Exception]
    ) -> "ReviewFeedback | None":
        """Decode a record. ``None`` (absent) is what every state written before this existed means."""
        if raw is None:
            return None
        data = _mapping(raw, "review_feedback", fail=fail)
        findings = data.get("findings")
        if not isinstance(findings, (list, tuple)):
            raise fail(
                "review_feedback.findings must be a list",
                code="invalid_review_feedback",
                details={"type": type(findings).__name__},
            )
        value = cls(
            verdict=_text(data, "verdict", "review_feedback.verdict", fail=fail),  # type: ignore[arg-type]
            sha=_text(data, "sha", "review_feedback.sha", fail=fail),
            findings=tuple(ReviewFinding.from_payload(item, fail=fail) for item in findings),
            omitted=_optional_integer(data, "omitted", fail=fail) or 0,
        )
        value.validate(fail=fail)
        return value

def review_feedback_from_payload(
    raw: Any, *, fail: Callable[..., Exception]
) -> "ReviewFeedback | None":
    """Decode a record, naming the offending FIELD. ``None`` when absent — every pre-change state.

    A module-level function rather than only a classmethod so `state.py` binds ONE import and reads
    exactly like `publish_skip_from_payload(data.get("publish_skip"), fail=…)` beside it. The error
    type is injected for the reason the module docstring gives: this module never imports `state.py`.
    """
    return ReviewFeedback.from_payload(raw, fail=fail)


# -- building a record from what the reviewer returned -------------------------------------


def _field(raw: Any, name: str) -> Any:
    """One attribute of a report finding, whether it is an OBJECT or a MAPPING.

    Both shapes are real: `review_findings.CodnaReviewFinding` is a dataclass, and the durable
    suite's scripted review report carries the same fields as dicts. Reading only attributes would
    silently drop every finding of a dict-shaped report — which is exactly the class of bug this
    module exists to end (a rejection whose reasons never arrive), so it is handled here rather than
    assumed away by whichever fake happens to be in use.
    """
    if isinstance(raw, Mapping):
        return raw.get(name)
    return getattr(raw, name, None)


def review_finding_from_report(raw: Any, *, fail: Callable[..., Exception]) -> ReviewFinding | None:
    """One finding from a review report's own finding, or ``None`` when it cannot be RENDERED.

    Tolerant on purpose. A finding without a path or a line is DROPPED rather than being rendered as
    a broken line: a prompt saying "``:0`` — None" is worse than one that says fewer things, because
    the model can act on "fix this" and cannot act on a malformed row.
    """
    path = str(_field(raw, "path") or "").strip().lstrip("/")
    title = str(_field(raw, "title") or "").strip()
    severity = str(_field(raw, "severity") or "").strip().lower()
    if not path or not title or severity not in REVIEW_SEVERITIES:
        return None
    try:
        line = int(_field(raw, "line") or 0)
    except (TypeError, ValueError):
        return None
    if line < 1:
        return None
    category = _clip(
        redact_secret_like(str(_field(raw, "category") or "").strip()), MAX_REVIEW_FINDING_CATEGORY_CHARS
    )
    explanation = _clip(
        redact_secret_like(" ".join(str(_field(raw, "explanation") or "").split())),
        MAX_REVIEW_FINDING_EXPLANATION_CHARS,
    )
    value = ReviewFinding(
        path=_clip(path, MAX_REVIEW_FINDING_PATH_CHARS),
        line=line,
        severity=severity,
        # The TITLE is redacted too, not only the explanation: a finding that names a hardcoded
        # credential quotes it in its title ("API key sk-… committed in config.py"). Leaving the
        # title raw would fail the state's secret guard — turning a correct finding into a broken
        # run — or, worse, store the token. See `state_guards.redact_secret_like` for why this is
        # redaction rather than refusal.
        title=_clip(redact_secret_like(title), MAX_REVIEW_FINDING_TITLE_CHARS),
        category=category,
        explanation=explanation,
        fingerprint=_clip(str(_field(raw, "fingerprint") or "").strip(), _FINGERPRINT_CHARS),
    )
    value.validate(fail=fail)
    return value


def review_findings_from_report(
    report: Any, *, verdict: str, sha: str, fail: Callable[..., Exception]
) -> ReviewFeedback | None:
    """The feedback record for one review result: ``None`` on an APPROVAL, a bounded record otherwise.

    THIS IS THE ONE PLACE THE ORDER IS DECIDED. `_sort_key` is a TOTAL order over the fields the
    record actually carries (there is no confidence here, and no input-order dependence), so the same
    reviewer output always produces the same record — which is what makes the retry prompt, and a
    replay of it, byte-identical. The cap is applied AFTER the sort, so a truncated list keeps the
    most severe findings rather than an arbitrary prefix of them.
    """
    if verdict == "approved":
        return None
    raw_findings: Iterable[Any] = getattr(report, "findings", None) or ()
    renderable = [
        finding
        for finding in (review_finding_from_report(item, fail=fail) for item in raw_findings)
        if finding is not None
    ]
    if not renderable:
        # A rejection with nothing renderable still HAS to tell the retry something, or the loop is
        # exactly as blind as it was. One synthetic finding names the situation instead of the retry
        # being sent back with no information at all.
        return ReviewFeedback(
            verdict=verdict,  # type: ignore[arg-type]
            sha=sha,
            findings=(
                ReviewFinding(
                    path=".",
                    line=1,
                    severity="medium",
                    title="The reviewer requested changes without a finding it could anchor to a file",
                    explanation=(
                        "No structured finding was returned with a path and a line, so the reason for "
                        "the rejection was not recorded. Re-read the task and the diff and address what "
                        "the review would most likely have raised."
                    ),
                ),
            ),
        )
    ordered = sorted(renderable, key=_sort_key)
    return ReviewFeedback(
        verdict=verdict,  # type: ignore[arg-type]
        sha=sha,
        findings=tuple(ordered[:MAX_FINDINGS_RECORDED]),
        omitted=max(0, len(ordered) - MAX_FINDINGS_RECORDED),
    )


def _sort_key(finding: ReviewFinding) -> tuple[int, str, int, str, str]:
    """A TOTAL order over what the record carries. Deterministic in the finding, never in input order.

    Severity first because the cap keeps the head of this list, so the most urgent finding is the one
    a bounded record is guaranteed to retain. The remaining keys exist only to break ties the same way
    on every run — including ``fingerprint``, which distinguishes two findings at the same location.
    """
    return (
        -_SEVERITY_RANK.get(finding.severity, 0),
        finding.path,
        finding.line,
        finding.title,
        finding.fingerprint,
    )


# -- coercion helpers (the error type is injected, so this module stays a leaf) -------------


def _mapping(raw: Any, label: str, *, fail: Callable[..., Exception]) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise fail(
            f"{label} must be an object",
            code="invalid_review_feedback",
            details={"field": label, "type": type(raw).__name__},
        )
    return raw


def _text(data: Mapping[str, Any], key: str, label: str, *, fail: Callable[..., Exception]) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise fail(
            f"{label} must be a non-empty string",
            code="invalid_review_feedback",
            details={"field": label, "type": type(value).__name__},
        )
    return value


def _optional_text(data: Mapping[str, Any], key: str, *, fail: Callable[..., Exception]) -> str:
    value = data.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise fail(
            f"review_finding.{key} must be a string",
            code="invalid_review_feedback",
            details={"field": f"review_finding.{key}", "type": type(value).__name__},
        )
    return value


def _integer(data: Mapping[str, Any], key: str, label: str, *, fail: Callable[..., Exception]) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise fail(
            f"{label} must be an integer",
            code="invalid_review_feedback",
            details={"field": label, "type": type(value).__name__},
        )
    return value


def _optional_integer(data: Mapping[str, Any], key: str, *, fail: Callable[..., Exception]) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise fail(
            f"review_feedback.{key} must be an integer",
            code="invalid_review_feedback",
            details={"field": f"review_feedback.{key}", "type": type(value).__name__},
        )
    return value
