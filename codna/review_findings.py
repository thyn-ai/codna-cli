"""codna review — turn a diff into validated, deduped, noise-controlled findings.

The review pass runs the read-only Cline agent (``task_kind="review"``) over a diff and normalizes
its output into :class:`CodnaReviewFinding` records: a stable schema plus a resolved GitHub *diff
anchor* (so a finding can be posted as an inline review comment) or, when a finding falls outside the
diff hunks, routed to the summary instead.

Everything except :func:`run_diff_review` is pure/deterministic (diff parsing, anchoring, fingerprint,
noise controls, check conclusion) so it is unit-testable offline with no agent and no network.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SEVERITIES = ("high", "medium", "low")
CATEGORIES = ("correctness", "security", "performance")
_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


# ── Config ──────────────────────────────────────────────────────────────────
@dataclass
class ReviewConfig:
    """The ``review:`` block from codna.yaml (with plan defaults). Non-blocking by default."""

    enabled: bool = True
    min_confidence: float = 0.75
    max_findings: int = 10
    blocking_enabled: bool = False
    blocking_severities: tuple[str, ...] = ("high",)
    # `review.approve` (default on): a review with no medium/high findings and no unresolved codna
    # threads is posted as an APPROVE (review_github.review_event); off = always COMMENT.
    approve_clean: bool = True
    categories: dict[str, bool] = field(
        default_factory=lambda: {c: True for c in CATEGORIES}
    )
    ignore_paths: tuple[str, ...] = ()
    rules_file: str | None = None
    effort: str = "medium"       # low|medium|high — trades review depth (agent iterations) for cost
    incremental: bool = True     # on re-review, look only at commits pushed since the last review

    def category_enabled(self, category: str) -> bool:
        return bool(self.categories.get(category, True))


# Review effort → agent iteration budget (higher effort = more thorough, more cost — mirrors Bugbot's
# effort tiers). Env-overridable ceiling stays authoritative via run_diff_review's max_iterations arg.
_EFFORT_ITERATIONS = {"low": 8, "medium": 12, "high": 20}


def effort_iterations(effort: str) -> int:
    return _EFFORT_ITERATIONS.get((effort or "medium").strip().lower(), _EFFORT_ITERATIONS["medium"])


def load_review_config(repo_dir: str, explicit: str | None = None) -> ReviewConfig:
    """Read the ``review:`` block from the repo's codna.yaml (or an explicit path). Missing file →
    defaults. Malformed YAML fails closed (raises ConfigError via config_file._load)."""
    import os

    from . import config_file

    path = explicit
    if not path:
        for name in ("codna.yaml", ".codna.yaml"):
            cand = os.path.join(repo_dir, name)
            if os.path.isfile(cand):
                path = cand
                break
    if not path or not os.path.isfile(path):
        return ReviewConfig()
    data = config_file._load(path)
    return review_config_from_dict(data.get("review"))


def review_config_from_dict(data: dict | None) -> ReviewConfig:
    """Build a ReviewConfig from a parsed ``review:`` mapping (tolerant of missing/extra keys)."""
    cfg = ReviewConfig()
    if not isinstance(data, dict):
        return cfg
    if "enabled" in data:
        cfg.enabled = bool(data["enabled"])
    if isinstance(data.get("min_confidence"), (int, float)):
        cfg.min_confidence = max(0.0, min(1.0, float(data["min_confidence"])))
    if isinstance(data.get("max_findings"), int) and data["max_findings"] >= 0:
        cfg.max_findings = int(data["max_findings"])
    if isinstance(data.get("approve"), bool):
        cfg.approve_clean = data["approve"]
    blocking = data.get("blocking")
    if isinstance(blocking, dict):
        cfg.blocking_enabled = bool(blocking.get("enabled", False))
        sev = blocking.get("severities")
        if isinstance(sev, list) and sev:
            cfg.blocking_severities = tuple(s for s in sev if s in SEVERITIES)
    cats = data.get("categories")
    if isinstance(cats, dict):
        cfg.categories = {c: bool(cats.get(c, True)) for c in CATEGORIES}
    ignore = data.get("ignore_paths")
    if isinstance(ignore, list):
        cfg.ignore_paths = tuple(str(p) for p in ignore if str(p).strip())
    if isinstance(data.get("rules_file"), str):
        cfg.rules_file = data["rules_file"].strip() or None
    if str(data.get("effort", "")).strip().lower() in _EFFORT_ITERATIONS:
        cfg.effort = str(data["effort"]).strip().lower()
    if "incremental" in data:
        cfg.incremental = bool(data["incremental"])
    return cfg


# ── Finding schema ────────────────────────────────────────────────────────────
@dataclass
class DiffAnchor:
    commit_id: str
    side: str  # "RIGHT" | "LEFT"
    line: int
    start_line: int | None = None
    start_side: str | None = None


@dataclass
class CodnaReviewFinding:
    path: str
    line: int
    severity: str
    category: str
    title: str
    explanation: str
    confidence: float
    fingerprint: str
    end_line: int | None = None
    suggested_patch: str | None = None
    diff_anchor: DiffAnchor | None = None

    @property
    def inline(self) -> bool:
        """A finding is posted inline iff it resolved to a diff anchor; else it goes to the summary."""
        return self.diff_anchor is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.diff_anchor is None:
            d.pop("diff_anchor", None)
        return d


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def fingerprint(path: str, category: str, title: str) -> str:
    """Stable dedup key: identity is (file, category, normalized title) — NOT line number, so a
    finding that merely shifts lines across a force-push dedups to the same fingerprint."""
    raw = f"{path}\x00{category}\x00{_norm_title(title)}".encode("utf-8", "replace")
    return hashlib.sha1(raw).hexdigest()[:16]


def normalize_finding(raw: dict) -> CodnaReviewFinding | None:
    """Validate + coerce one raw agent finding into a CodnaReviewFinding; drop it (return None) if it
    is missing required fields or has an out-of-domain severity/category."""
    if not isinstance(raw, dict):
        return None
    path = str(raw.get("path") or "").strip().lstrip("/")
    title = str(raw.get("title") or "").strip()
    severity = str(raw.get("severity") or "").strip().lower()
    category = str(raw.get("category") or "").strip().lower()
    explanation = str(raw.get("explanation") or "").strip()
    if not path or not title or severity not in SEVERITIES or category not in CATEGORIES:
        return None
    try:
        line = int(raw.get("line"))
    except (TypeError, ValueError):
        return None
    if line < 1:
        return None
    end_line = None
    if raw.get("end_line") is not None:
        try:
            end_line = int(raw["end_line"])
            if end_line < line:
                end_line = None
        except (TypeError, ValueError):
            end_line = None
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    patch = raw.get("suggested_patch")
    patch = str(patch).strip() if isinstance(patch, str) and patch.strip() else None
    return CodnaReviewFinding(
        path=path,
        line=line,
        end_line=end_line,
        severity=severity,
        category=category,
        title=title[:80],
        explanation=explanation,
        confidence=confidence,
        suggested_patch=patch,
        fingerprint=fingerprint(path, category, title),
    )


# ── Diff parsing + anchoring ───────────────────────────────────────────────────
@dataclass
class DiffFile:
    path: str
    # New-side (RIGHT) line numbers that appear in a hunk (added OR context) — the lines GitHub will
    # accept an inline comment on.
    right_lines: set[int] = field(default_factory=set)
    added_lines: set[int] = field(default_factory=set)


_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NEWFILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")


def parse_diff(diff_text: str) -> dict[str, DiffFile]:
    """Parse a unified diff into per-file new-side (RIGHT) line coverage.

    ``right_lines`` = every new-file line number shown in a hunk (added + context) — the anchorable
    set for inline comments. ``added_lines`` = only the ``+`` lines. Deletion-only hunks contribute no
    right-side lines (correctly un-anchorable → routed to the summary).

    Hunk-range driven, like ``diff_changed_lines``: each ``@@ -a,b +c,d @@`` header says how many
    old-side (``b``) and new-side (``d``) body lines follow, and while a hunk still owes lines every
    line is body. Deciding "file header" by prefix instead mis-filed real changes: an added ``++ x``
    renders as ``+++ x`` and used to open a bogus file named ``x``, so every later line of the hunk
    anchored to the wrong file. Only between hunks may ``+++ path`` open a file."""
    files: dict[str, DiffFile] = {}
    current: DiffFile | None = None
    new_line = 0
    old_left = new_left = 0            # body lines the current hunk still owes on each side
    for line in (diff_text or "").splitlines():
        if old_left <= 0 and new_left <= 0:            # between hunks: only headers matter
            m = _NEWFILE_RE.match(line)
            if m:
                path = m.group(1).strip()
                current = None if path == "/dev/null" else files.setdefault(path, DiffFile(path=path))
                continue
            hm = _HUNK_RE.match(line)
            if hm:
                old_left = int(hm.group(1)) if hm.group(1) is not None else 1
                new_line = int(hm.group(2))
                new_left = int(hm.group(3)) if hm.group(3) is not None else 1
            continue
        # inside a hunk: body, whatever it starts with ("+++ x" here is an added "++ x")
        if line.startswith("\\"):
            # "\ No newline at end of file": consumes nothing on either side
            continue
        if line.startswith("+"):
            if current is not None:
                current.right_lines.add(new_line)
                current.added_lines.add(new_line)
            new_line += 1
            new_left -= 1
        elif line.startswith("-"):
            # deletion: consumes an old-side line only; new-side cursor does not advance
            old_left -= 1
        else:
            # context line (present on both sides)
            if current is not None:
                current.right_lines.add(new_line)
            new_line += 1
            old_left -= 1
            new_left -= 1
    return files


def anchor_findings(
    findings: list[CodnaReviewFinding], diff_files: dict[str, DiffFile], head_sha: str | None
) -> None:
    """Resolve each finding to a GitHub diff anchor in place. A finding anchors inline iff its file is
    in the diff, its line is a new-side line shown in a hunk, and we have a head SHA to pin to.
    Findings that don't anchor keep ``diff_anchor is None`` → the caller routes them to the summary."""
    for f in findings:
        df = diff_files.get(f.path)
        if not head_sha or df is None or f.line not in df.right_lines:
            f.diff_anchor = None
            continue
        start = None
        if f.end_line and f.end_line != f.line and f.line in df.right_lines:
            # only a multi-line anchor when the START line is also in the diff
            start = f.line
            end = f.end_line if f.end_line in df.right_lines else f.line
        else:
            end = f.line
        f.diff_anchor = DiffAnchor(
            commit_id=head_sha,
            side="RIGHT",
            line=end,
            start_line=start if start is not None and start != end else None,
            start_side="RIGHT" if start is not None and start != end else None,
        )


# ── Noise controls + dedup ─────────────────────────────────────────────────────
def _path_ignored(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _sort_key(f: CodnaReviewFinding):
    return (-_SEVERITY_RANK.get(f.severity, 0), -f.confidence, f.path, f.line)


def apply_noise_controls(
    findings: list[CodnaReviewFinding], config: ReviewConfig
) -> tuple[list[CodnaReviewFinding], dict[str, int]]:
    """Filter (confidence / category / ignore_paths), dedup by fingerprint, sort by severity+confidence,
    cap at ``max_findings``. Returns ``(kept, dropped)`` where ``dropped`` counts each reason."""
    dropped = {"low_confidence": 0, "category_disabled": 0, "ignored_path": 0, "duplicate": 0, "over_cap": 0}
    kept: list[CodnaReviewFinding] = []
    seen: set[str] = set()
    for f in sorted(findings, key=_sort_key):
        if f.confidence < config.min_confidence:
            dropped["low_confidence"] += 1
            continue
        if not config.category_enabled(f.category):
            dropped["category_disabled"] += 1
            continue
        if _path_ignored(f.path, config.ignore_paths):
            dropped["ignored_path"] += 1
            continue
        if f.fingerprint in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(f.fingerprint)
        kept.append(f)
    if config.max_findings >= 0 and len(kept) > config.max_findings:
        dropped["over_cap"] = len(kept) - config.max_findings
        kept = kept[: config.max_findings]
    return kept, dropped


def check_conclusion(findings: list[CodnaReviewFinding], config: ReviewConfig) -> str:
    """The "codna review" check conclusion. NON-BLOCKING by default: ``failure`` only when blocking is
    explicitly enabled AND a finding hits a blocking severity; ``success`` = clean; else ``neutral``."""
    if not findings:
        return "success"
    if config.blocking_enabled and any(f.severity in config.blocking_severities for f in findings):
        return "failure"
    return "neutral"


# ── Diff computation ────────────────────────────────────────────────────────────
def _git(repo_dir: str, *args: str, timeout: int = 60) -> str:
    out = subprocess.run(
        ["git", "-C", repo_dir, *args], capture_output=True, text=True, timeout=timeout
    )
    return out.stdout


def compute_diff(repo_dir: str, *, diff_range: str | None = None, base: str | None = None,
                 diff_paths: list[str] | None = None) -> tuple[str, list[str]]:
    """Return ``(unified_diff_text, changed_paths)``.

    ``diff_range`` (e.g. ``origin/main...HEAD``) → ``git diff <range>``; else ``base`` (or HEAD) →
    ``git diff <base>`` (working tree vs base). ``diff_paths`` (None = no restriction) limits the diff
    and the changed-path list to those paths -- ``git diff <spec> -- <paths>`` -- which is how an
    incremental range that also swept in base-branch history (an "Update branch" merge commit, see
    ``review._materialize_pr``) is cut back to the pull request's own files. An EMPTY list means
    "restrict to nothing" and returns ``("", [])`` without running git: handing git no pathspec
    would LIFT the restriction, the opposite of what an empty intersection means. Best-effort:
    returns ``("", [])`` if git is unavailable."""
    if diff_paths is not None and not diff_paths:
        return "", []
    spec = diff_range or (base or "HEAD")
    scope = ["--", *diff_paths] if diff_paths else []
    try:
        diff_text = _git(repo_dir, "diff", "--unified=3", spec, *scope)
        names = _git(repo_dir, "diff", "--name-only", spec, *scope)
    except Exception:  # noqa: BLE001 — git absent / not a repo
        return "", []
    changed = [ln.strip() for ln in names.splitlines() if ln.strip()]
    return diff_text, changed


_HUNK_RANGE_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def diff_changed_lines(diff_text: str) -> int:
    """Added + removed lines of a unified diff: the review's own size signal for its turn budget
    (review_budget). Pure.

    Counted from the HUNKS, not by prefix: each ``@@ -a,b +c,d @@`` header says how many old-side
    (``b``) and new-side (``d``) lines follow, and only those lines are body. Skipping every line that
    starts with ``+++ `` or ``--- `` instead -- the obvious "file header" test -- dropped real changes: a
    removed SQL/Lua/Haskell comment ``-- x`` renders as ``--- x`` and an added ``++ x`` as ``+++ x``.
    Between hunks (``diff --git``, ``index``, the ``---``/``+++`` file headers) nothing is counted."""
    count = 0
    old_left = new_left = 0            # body lines the current hunk still owes on each side
    for line in (diff_text or "").splitlines():
        if old_left <= 0 and new_left <= 0:            # between hunks: only a hunk header matters
            m = _HUNK_RANGE_RE.match(line)
            if m:
                old_left = int(m.group(1)) if m.group(1) is not None else 1
                new_left = int(m.group(2)) if m.group(2) is not None else 1
            continue
        if line.startswith("\\"):                     # "\ No newline at end of file": consumes nothing
            continue
        if line.startswith("+"):
            new_left -= 1
            count += 1
        elif line.startswith("-"):
            old_left -= 1
            count += 1
        else:                                           # context: present on both sides
            old_left -= 1
            new_left -= 1
    return count


def resolve_head_sha(repo_dir: str, diff_range: str | None = None) -> str | None:
    """The commit findings anchor to. For ``a...b`` use ``b``; else HEAD. None if not resolvable."""
    ref = "HEAD"
    if diff_range:
        if "..." in diff_range:
            ref = diff_range.split("...", 1)[1].strip() or "HEAD"
        elif ".." in diff_range:
            ref = diff_range.split("..", 1)[1].strip() or "HEAD"
    try:
        sha = _git(repo_dir, "rev-parse", ref).strip()
        return sha or None
    except Exception:  # noqa: BLE001
        return None


# ── Prompt ──────────────────────────────────────────────────────────────────────
_MAX_DIFF_CHARS = 60_000  # keep the review pass fast + within budget; truncate huge diffs


_MAX_FEEDBACK_CHARS = 6_000  # cap the prior-comment context so it can't crowd out the diff


# Restated in the user turn, not only in the sidecar's system prompt (adapter.ts): a review that
# reads like a task to complete got completed -- the model described the change it would make and
# ended the turn, and the run failed with "did not return parseable findings JSON".
REVIEW_OUTPUT_CONTRACT = (
    "You are REVIEWING this diff, not fixing it: do not modify files, do not run commands, and do not "
    "describe changes you would make. A change that looks deliberate (a probe, a failing step, a "
    "removed test) is not a finding unless it is a real defect. "
    'Your FINAL message MUST be a single JSON object and nothing else -- no prose, no code fences: '
    '{"findings": [...]} using the Finding shape from your instructions; '
    'if there are no high-confidence issues, return {"findings": []}.'
)


def build_review_prompt(diff_text: str, changed_files: list[str], guidance: str | None,
                        config: ReviewConfig, prior_feedback: str | None = None) -> str:
    """Build the review task prompt: the diff + in-scope categories + the confidence bar + (optionally)
    prior human/bot review comments so Codna doesn't duplicate points reviewers already raised."""
    cats = [c for c in CATEGORIES if config.category_enabled(c)]
    diff = diff_text or ""
    truncated = ""
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS]
        truncated = "\n\n[diff truncated for length — review what is shown]"
    parts = [
        "Review the following pull-request diff for high-confidence defects.",
        f"In-scope categories: {', '.join(cats) or '(none)'}.",
        f"Only report findings with confidence >= {config.min_confidence:g}. Report at most {config.max_findings} findings, most severe first.",
        f"Changed files: {', '.join(changed_files[:40])}" + (" …" if len(changed_files) > 40 else ""),
    ]
    if guidance:
        parts.append(
            "Project review guidance (authoritative — honor these conventions):\n" + guidance.strip()
        )
    if prior_feedback and prior_feedback.strip():
        parts.append(
            "Comments already on this PR (from human reviewers and prior bot passes). Do NOT repeat a "
            "point that is already raised here; you may build on them:\n" + prior_feedback.strip()[:_MAX_FEEDBACK_CHARS]
        )
    parts.append(REVIEW_OUTPUT_CONTRACT)
    parts.append("\n----- BEGIN DIFF -----\n" + diff + truncated + "\n----- END DIFF -----")
    return "\n\n".join(parts)


# ── Result + orchestration ───────────────────────────────────────────────────────
@dataclass
class ReviewResult:
    repository: str
    base: str
    head_sha: str | None
    changed_files: list[str]
    inline_findings: list[CodnaReviewFinding]
    summary_findings: list[CodnaReviewFinding]
    conclusion: str
    dropped: dict[str, int]
    model: str | None = None
    elapsed_s: float | None = None
    note: str | None = None

    @property
    def findings(self) -> list[CodnaReviewFinding]:
        return [*self.inline_findings, *self.summary_findings]

    def to_dict(self) -> dict:
        return {
            "repository": self.repository,
            "base": self.base,
            "head_sha": self.head_sha,
            "changed_files": self.changed_files,
            "findings": [f.to_dict() for f in self.findings],
            "inline_count": len(self.inline_findings),
            "summary_count": len(self.summary_findings),
            "conclusion": self.conclusion,
            "dropped": self.dropped,
            "model": self.model,
            "elapsed_s": self.elapsed_s,
            **({"note": self.note} if self.note else {}),
        }


def finalize_findings(
    raw_findings: list[dict],
    diff_files: dict[str, DiffFile],
    head_sha: str | None,
    config: ReviewConfig,
    *,
    grounder=None,
) -> tuple[list[CodnaReviewFinding], list[CodnaReviewFinding], dict[str, int]]:
    """Pure pipeline: normalize → [ground registry claims] → noise-control/dedup → anchor → split
    inline vs summary.

    Split out from :func:`run_diff_review` so the whole transform is testable without the agent.
    ``grounder`` (:func:`review_grounding.grounder_for`; None = the transform stays pure) checks the
    findings' claims about packages against the npm / PyPI registries BEFORE the noise controls, so a
    refuted claim neither takes one of the ``max_findings`` slots nor reaches the PR; the counts it
    returns (``registry_contradicted`` / ``registry_confirmed`` / ``registry_unverified``) join
    ``dropped``."""
    normalized = [f for f in (normalize_finding(r) for r in raw_findings) if f is not None]
    grounding: dict[str, int] = {}
    if grounder is not None and normalized:
        normalized, grounding = grounder(normalized)
    kept, dropped = apply_noise_controls(normalized, config)
    dropped.update(grounding)
    anchor_findings(kept, diff_files, head_sha)
    inline = [f for f in kept if f.inline]
    summary = [f for f in kept if not f.inline]
    return inline, summary, dropped


def run_diff_review(
    repo_dir: str,
    *,
    repository: str,
    diff_range: str | None = None,
    base: str | None = None,
    config: ReviewConfig,
    guidance: str | None = None,
    prior_feedback: str | None = None,
    head_sha: str | None = None,
    model: str | None = None,
    provider: str = "anthropic",
    max_iterations: int | None = None,
    timeout_s: int | None = None,
    diff_paths: list[str] | None = None,
) -> ReviewResult:
    """Run the read-only agent over the diff and return a fully finalized :class:`ReviewResult`.

    ``timeout_s`` is a caller's explicit pin on the review turn's budget. Unset (the default), the
    budget is adaptive -- sized from THIS diff's lines/files/tokens and the observed durations of
    earlier review turns (review_budget) -- which is why the diff's size travels to the agent.

    ``diff_paths`` (None = the whole range) confines the diff to those paths (:func:`compute_diff`);
    an incremental review passes the pull request's own file set. A range with nothing left in
    those paths is a clean, agent-free result that says so in its ``note``, not a failure."""
    t0 = time.perf_counter()
    base_label = diff_range or base or "HEAD"
    # Effort → iteration budget (unless the caller pins max_iterations explicitly).
    iterations = max_iterations if max_iterations is not None else effort_iterations(config.effort)
    diff_text, changed = compute_diff(repo_dir, diff_range=diff_range, base=base, diff_paths=diff_paths)
    if not changed:
        return ReviewResult(
            repository=repository, base=base_label, head_sha=head_sha, changed_files=[],
            inline_findings=[], summary_findings=[], conclusion="success", dropped={},
            elapsed_s=round(time.perf_counter() - t0, 1), note=f"no changed files vs {base_label}",
        )
    resolved_head = head_sha or resolve_head_sha(repo_dir, diff_range)
    prompt = build_review_prompt(diff_text, changed, guidance, config, prior_feedback)
    raw, agent = run_review_agent(
        repo_dir, prompt, provider=provider, model=model,
        max_iterations=iterations, timeout_s=timeout_s,
        changed_lines=diff_changed_lines(diff_text), changed_files=len(changed),
    )
    diff_files = parse_diff(diff_text)
    # Registry grounding only when the change touches a dependency manifest or lockfile; every other
    # review keeps the pure transform (no file read, no request).
    from .review_grounding import grounder_for

    inline, summary, dropped = finalize_findings(raw, diff_files, resolved_head, config,
                                                 grounder=grounder_for(repo_dir, changed))
    return ReviewResult(
        repository=repository, base=base_label, head_sha=resolved_head, changed_files=changed,
        inline_findings=inline, summary_findings=summary,
        conclusion=check_conclusion([*inline, *summary], config), dropped=dropped,
        model=(agent or {}).get("model"), elapsed_s=round(time.perf_counter() - t0, 1),
    )


def run_review_agent(
    repo_dir: str,
    prompt: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    max_iterations: int = 12,
    timeout_s: int | None = None,
    changed_lines: int = 0,
    changed_files: int = 0,
) -> tuple[list[dict], dict]:
    """Run the packaged local sidecar in read-only review mode.

    This is the installed-product path. It does not require source-checkout ``node_modules``;
    the bundled sidecar owns the Cline SDK runtime and enforces ``task_kind=review`` below
    the model.

    The turn budget is decided by the runner (packaged_agent_runner._timeout_ms_for_request ->
    review_budget) from ``changed_lines`` / ``changed_files`` and the prompt's token estimate, unless
    ``timeout_s`` pins it. Every turn's observed wall-clock is recorded for the next decision, and a
    turn that outruns its budget is reported as :class:`~codna.cline_agent.ReviewTurnTimeout`.
    """
    from . import cli as _cli
    from . import review_budget
    from .cline_agent import ReviewTurnTimeout, UnparseableFindingsError, extract_findings_json
    from .packaged_agent_runner import SidecarPackagedAgentRunner
    from .packaged_repository_advanced import PackagedAgentRunRequest, PackagedRepositoryAdvancedError
    from .runtime.config import resolve_runtime_config

    repo_root = Path(repo_dir).expanduser().resolve()
    keys = _cli._runtime_keys(include_keychain=True)
    config_runtime = resolve_runtime_config(keys=keys)
    runner = SidecarPackagedAgentRunner(config=config_runtime, keys=keys)

    def _run(text: str, pin_s: int | None):
        """One sidecar turn -> ``(result, elapsed_s, granted_ms)``, so the caller can budget the repair
        turn against what this one used. ``pin_s`` pins the turn through the caller-pin path
        (``review_timeout_s``); None leaves the budget adaptive."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        snapshot_id = "review_" + digest
        prompt_tokens = max(1, len(text) // 4)
        signals: dict = {
            "review_max_iterations": max_iterations,
            review_budget.SIGNAL_CHANGED_LINES: changed_lines,
            review_budget.SIGNAL_CHANGED_FILES: changed_files,
        }
        if pin_s is not None:
            signals["review_timeout_s"] = pin_s     # a caller's pin, or the repair's share of the deadline
        request = PackagedAgentRunRequest(
            repository_id="review_" + hashlib.sha256(str(repo_root).encode("utf-8")).hexdigest()[:16],
            snapshot_id=snapshot_id,
            repo_root=repo_root,
            issue_text=text,
            model=_review_model(provider=provider, model=model),
            signals=signals,
            evidence_bundle={},
            snapshot={
                "snapshot_id": snapshot_id,
                "snapshot_file_count": 0,
                "raw_repo_token_estimate": prompt_tokens,
            },
            task_kind="review",
        )
        started = time.monotonic()
        try:
            result = runner.run(request)
        except PackagedRepositoryAdvancedError as exc:
            elapsed = time.monotonic() - started
            decision = None if pin_s is not None else review_budget.decision_for(snapshot_id)
            budget_ms = (review_budget.budget_from_error(exc.details)
                         or (int(pin_s * 1000) if pin_s else None)
                         or (decision.budget_ms if decision else None))
            if not review_budget.is_turn_timeout(exc.details, cause_code=exc.code, elapsed_s=elapsed, budget_ms=budget_ms):
                raise      # a crash is not a review duration: nothing to learn from, nothing recorded
            # Censored sample: the turn was still running when the budget ended. Recorded so the
            # next budget for this size grows (review_budget.CENSORED_GROWTH) -- "retry" means
            # something.
            review_budget.observe_review_turn(
                duration_s=elapsed, changed_lines=changed_lines, changed_files=changed_files,
                prompt_tokens=prompt_tokens, budget_ms=budget_ms, timed_out=True,
            )
            raise ReviewTurnTimeout(
                _review_timeout_message(changed_lines, changed_files, prompt_tokens, budget_ms, elapsed),
                {
                    "cause_code": exc.code,
                    "status": exc.details.get("status"),
                    "terminal_state": exc.details.get("terminal_state"),
                    "error": exc.details.get("error"),
                    "sidecar_url": exc.details.get("sidecar_url"),
                    "changed_lines": changed_lines,
                    "changed_files": changed_files,
                    "prompt_tokens": prompt_tokens,
                    "budget_ms": budget_ms,
                    "elapsed_s": round(elapsed, 1),
                    "budget_source": decision.source if decision else None,
                    "samples": decision.samples if decision else None,
                },
            ) from exc
        elapsed = time.monotonic() - started
        # A pinned turn never went through review_turn_budget_ms, so nothing was remembered for it;
        # the pin IS its budget. Reading the cache anyway could return an earlier unpinned run's
        # decision for the same prompt text.
        decision = None if pin_s is not None else review_budget.decision_for(snapshot_id)
        granted_ms = int(pin_s * 1000) if pin_s is not None else (decision.budget_ms if decision else None)
        review_budget.observe_review_turn(
            duration_s=elapsed, changed_lines=changed_lines, changed_files=changed_files,
            prompt_tokens=prompt_tokens, budget_ms=granted_ms, timed_out=False,
        )
        _assert_review_left_the_tree_alone(result)
        return result, elapsed, granted_ms

    result, first_elapsed_s, first_budget_ms = _run(prompt, timeout_s)
    agent = _agent_summary(result)
    try:
        return extract_findings_json(result.text), agent
    except UnparseableFindingsError as first:
        # ONE bounded repair: a fresh single-shot run (sessions are one-shot; there is no turn to
        # append to) that restates the contract and shows the model what it said instead. Costs one
        # more model call on the same diff; a second miss fails closed with BOTH replies attached --
        # never an empty findings list, which would read as a clean review.
        #
        # The repair shares the first turn's DEADLINE (review_budget.REVIEW_DEADLINE_MS): pinned to
        # what the first turn left of it, never more than the first turn's own budget, so both turns
        # together stay inside the cap that sits below the webhook's job bound. Two independent
        # adaptive budgets could total 2 x cap = 2400 s > 1800 s, and the job bound's kill reports a
        # generic timeout in place of this classification. Too little left -> no repair: fail closed.
        repair_s = review_budget.repair_turn_budget_s(first_budget_ms=first_budget_ms, elapsed_s=first_elapsed_s)
        if repair_s is None:
            raise ReviewTurnTimeout(
                _repair_skipped_message(changed_lines, changed_files, first_budget_ms, first_elapsed_s),
                {
                    "cause_code": first.code,          # the CAUSE was the prose reply; no time to repair it
                    "changed_lines": changed_lines,
                    "changed_files": changed_files,
                    "budget_ms": first_budget_ms,
                    "elapsed_s": round(first_elapsed_s, 1),
                    "deadline_ms": review_budget.REVIEW_DEADLINE_MS,
                    "repair_skipped": True,
                    "first_attempt": first.details,
                    "model": agent.get("model"),
                    "terminal_state": agent.get("terminal_state"),
                    "agent_run_id": agent.get("agent_run_id"),
                },
            ) from first
        repaired, _repair_elapsed_s, _repair_budget_ms = _run(build_repair_prompt(prompt, result.text), repair_s)
        agent = {**_agent_summary(repaired), "repaired": True, "first_attempt": agent}
        try:
            return extract_findings_json(repaired.text), agent
        except UnparseableFindingsError as second:
            raise UnparseableFindingsError(
                "review agent did not return parseable findings JSON (after one repair attempt)",
                {**second.details, "first_attempt": first.details,
                 "model": agent.get("model"), "terminal_state": agent.get("terminal_state"),
                 "agent_run_id": agent.get("agent_run_id"), "attempts": 2},
            ) from second


_REPAIR_EXCERPT_CHARS = 300


def _review_timeout_message(changed_lines: int, changed_files: int, prompt_tokens: int,
                            budget_ms: int | None, elapsed_s: float) -> str:
    """What the check run says when the review turn outran its budget: the diff's size, the budget
    that was granted, and the two ways forward. It never reads as a clean review."""
    granted = f"{budget_ms / 1000:.0f} s" if budget_ms else "its budget"
    return (
        f"codna review ran out of time on this diff: {changed_lines} changed line(s) across "
        f"{changed_files} file(s), ~{prompt_tokens} prompt tokens. The review turn was granted {granted} "
        f"and had not finished after {elapsed_s:.0f} s. Comment `@codna review` to retry -- the next "
        "budget grows from this run's recorded duration -- or split the pull request into smaller ones."
    )


def _repair_skipped_message(changed_lines: int, changed_files: int, first_budget_ms: int | None,
                            first_elapsed_s: float) -> str:
    """What the check run says when the first turn answered with prose and left too little of the
    review's deadline for the one repair turn. Never reads as a clean review."""
    from . import review_budget

    granted = f"{first_budget_ms / 1000:.0f} s" if first_budget_ms else "its budget"
    return (
        f"codna review ran out of time on this diff: {changed_lines} changed line(s) across "
        f"{changed_files} file(s). The review turn was granted {granted}, answered after {first_elapsed_s:.0f} s "
        "with prose instead of findings JSON, and left too little of the review's "
        f"{review_budget.REVIEW_DEADLINE_MS // 1000} s deadline for the one repair turn. No review was posted. "
        "Comment `@codna review` to retry, or split the pull request into smaller ones."
    )


def build_repair_prompt(original_prompt: str, previous_reply: str) -> str:
    """The re-ask sent when the first reply was not the findings JSON: the contract first, the
    model's own (bounded) reply as the counter-example, then the original task verbatim."""
    excerpt = " ".join((previous_reply or "").split())[:_REPAIR_EXCERPT_CHARS]
    return (
        "Your previous reply to the review task below was NOT the required JSON. "
        f"It began: \u00ab{excerpt}\u00bb. "
        'Reply now with ONLY the JSON object {"findings": [...]} for that diff -- or {"findings": []} '
        "if there are no high-confidence issues. No prose, no code fences, no description of changes. "
        "You are reviewing, not fixing; do not modify files.\n\n"
        + original_prompt
    )


def _agent_summary(result) -> dict:
    return {
        "status": result.status,
        "terminal_state": result.terminal_state,
        "model": result.telemetry.get("model"),
        "usage": result.telemetry,
        "agent_run_id": result.agent_run_id,
        "session_id": result.session_id,
    }


def _assert_review_left_the_tree_alone(result) -> None:
    """Fail closed if a read-only review run touched the workspace. Writes are disabled below the
    model, so this cannot happen -- and if it ever does, the findings must not be posted."""
    from .cline_agent import ClineAgentError

    changed = [f for f in (result.files_changed or []) if str(f).strip()]
    if changed or (result.patch_diff or "").strip():
        raise ClineAgentError(
            "review run modified the workspace; refusing to post its findings",
            {"files_changed": changed[:20], "patch_chars": len((result.patch_diff or "").strip()),
             "agent_run_id": result.agent_run_id},
        )


def _review_model(*, provider: str, model: str | None) -> str:
    if model:
        return model if "/" in model else f"{provider}/{model}"
    env_model = os.environ.get("CODNA_REVIEW_MODEL") or os.environ.get("ALGENTA_REVIEW_MODEL")
    if env_model:
        return env_model
    return "repository.verified_agentic_v1"
