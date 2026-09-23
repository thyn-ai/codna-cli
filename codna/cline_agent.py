"""Drive codna's vendored Cline agent (agent-core) to remediate a repo in place.

This is the agentic-fix executor behind `codna secure --fix`: it shells out to `bun` running the
vendored Cline SDK, which edits files in `working_dir`. The engine seam is optional — with only an
ANTHROPIC_API_KEY the agent runs as a plain coding agent (no HTTP engine/sidecar required).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


class ClineAgentError(Exception):
    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        # Per-instance. `codna review` reports the exception's `.details` verbatim, so anything
        # not put here is invisible to whoever has to act on the failure.
        self.details: dict[str, object] = dict(details or {})


# Enough of the agent's reply to tell the failure modes apart, bounded so a runaway response
# cannot flood a check summary.


class UnparseableFindingsError(ClineAgentError):
    """The review agent's final message was not the findings JSON the contract asks for.

    Its own class so callers can tell an output-contract miss (repairable with one bounded re-ask,
    see review_findings.run_review_agent) from every other agent failure, and so the Check Run
    carries a specific ``cause_code`` (review.py::_wrapped_details reads ``code``).
    """

    code = "review_unparseable_output"


class ReviewTurnTimeout(ClineAgentError):
    """The review turn outran the budget it was granted (review_budget.review_turn_budget).

    Its own class and code so ``codna review`` reports it as ``review_timeout`` -- naming the diff's
    size, the budget granted and how to proceed -- instead of the generic ``review_error`` sentence
    that thyn-ai/algenta-sdk#71 got (``turn exceeded timeout budget of 240000ms`` buried in the
    details). ``details`` carries the numbers: changed_lines, changed_files, prompt_tokens,
    budget_ms, elapsed_s, plus the agent-core fields (cause_code, terminal_state, error).
    """

    code = "review_timeout"


_RAW_EXCERPT_CHARS = 400


def _unparseable_output_details(raw: str) -> dict[str, object]:
    """What the agent actually said, bounded and redacted.

    Discarding `raw` made this failure undiagnosable: prose instead of JSON, JSON truncated by a
    token limit, and an empty response all reported the identical sentence. Head AND tail are
    kept because they show different things -- truncation is only visible at the end, while a
    refusal or a prose preamble is only visible at the start.

    Redacted through the same helper the agent-core log tail uses, because this text is echoed
    into a check summary on a public pull request.
    """
    from codna.agent_core_runtime import _redact_log_text  # noqa: PLC0415 - avoids a cycle at import

    text = _redact_log_text(raw or "")
    return {
        "raw_chars": len(raw or ""),
        "raw_head": text[:_RAW_EXCERPT_CHARS],
        "raw_tail": text[-_RAW_EXCERPT_CHARS:] if len(text) > 2 * _RAW_EXCERPT_CHARS else "",
        "looked_for": [
            "the whole message as JSON",
            "a ```json fence",
            "the first balanced {...} or [...]",
        ],
    }


def _agent_core_dir() -> Path:
    """Locate the vendored Cline SDK root (agent-core/vendor/cline)."""
    env = os.environ.get("CODNA_AGENT_CORE_DIR")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "agent-core" / "vendor" / "cline"
        if (cand / "algenta" / "adapter.ts").exists():
            return cand
    raise ClineAgentError(
        "could not locate agent-core/vendor/cline; set CODNA_AGENT_CORE_DIR to its path."
    )


def _ensure_cline_data_dir(env: dict[str, str]) -> str:
    configured = env.get("CLINE_DATA_DIR", "").strip()
    if configured:
        data_dir = Path(configured).expanduser()
    else:
        data_dir = Path(tempfile.gettempdir()) / "codna-cline-data"
        env["CLINE_DATA_DIR"] = str(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def run_cline_fix(
    working_dir: str,
    prompt: str,
    *,
    task_kind: str = "fix",
    provider: str = "anthropic",
    model: str | None = None,
    max_iterations: int = 20,
    timeout_s: int = 360,
) -> dict:
    """Run the Cline agent against `working_dir` (it edits files in place). Returns the agent's
    result summary ({status, iterations, model, text, usage}). Raises ClineAgentError on failure."""
    core = _agent_core_dir()
    if not (core / "algenta" / "fix-runner.ts").exists():
        raise ClineAgentError(f"fix-runner.ts not found under {core}")
    if not (core / "node_modules").is_dir():
        raise ClineAgentError(f"vendored SDK not built (no node_modules) under {core}")
    if task_kind not in {"fix", "triage", "review"}:
        raise ClineAgentError(f"unsupported task_kind '{task_kind}'")

    env = dict(os.environ)
    _ensure_cline_data_dir(env)
    env["FIX_WORKING_DIR"] = str(working_dir)
    env["FIX_PROMPT"] = prompt
    env["FIX_TASK_KIND"] = task_kind
    env["FIX_PROVIDER"] = provider
    env["FIX_MAX_ITERATIONS"] = str(max_iterations)
    if model:
        env["FIX_MODEL"] = model

    try:
        proc = subprocess.run(
            ["bun", "algenta/fix-runner.ts"],
            cwd=str(core), env=env, capture_output=True, text=True, timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise ClineAgentError("`bun` not found on PATH — required to run the agent.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ClineAgentError(f"the agent timed out after {timeout_s}s") from exc

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise ClineAgentError(
        f"cline agent produced no RESULT (rc={proc.returncode}); stderr tail: {proc.stderr[-300:]}"
    )


def extract_findings_json(text: str) -> list[dict]:
    """Parse the reviewer's final message into a list of raw finding dicts.

    The review prompt asks for a bare ``{"findings": [...]}`` object, but models sometimes wrap it in
    prose or a ```json code fence. Be tolerant: try a strict parse first, then scan for the first
    balanced JSON object/array. Raises ClineAgentError only when nothing parseable is present.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    def _coerce(obj) -> list[dict] | None:
        if isinstance(obj, dict):
            found = obj.get("findings")
            if isinstance(found, list):
                return [f for f in found if isinstance(f, dict)]
            return None
        if isinstance(obj, list):
            return [f for f in obj if isinstance(f, dict)]
        return None

    # 1) Strict: the whole message is the JSON we asked for.
    try:
        coerced = _coerce(json.loads(raw))
        if coerced is not None:
            return coerced
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) Strip a ```json … ``` fence if present.
    if "```" in raw:
        import re

        for block in re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL):
            try:
                coerced = _coerce(json.loads(block.strip()))
                if coerced is not None:
                    return coerced
            except (json.JSONDecodeError, ValueError):
                continue

    # 3) Scan for the first balanced {...} or [...] and try to parse it.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        while start != -1:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == opener:
                    depth += 1
                elif raw[i] == closer:
                    depth -= 1
                    if depth == 0:
                        chunk = raw[start : i + 1]
                        try:
                            coerced = _coerce(json.loads(chunk))
                            if coerced is not None:
                                return coerced
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            start = raw.find(opener, start + 1)

    raise UnparseableFindingsError(
        "review agent did not return parseable findings JSON",
        _unparseable_output_details(raw),
    )


def run_cline_review(
    working_dir: str,
    prompt: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    max_iterations: int = 12,
    timeout_s: int = 240,
) -> tuple[list[dict], dict]:
    """Run the read-only review agent against ``working_dir`` and return ``(raw_findings, result)``.

    ``raw_findings`` is the parsed (but not yet validated/normalized) finding dicts from the agent's
    final message; ``result`` is the raw agent summary ({status, iterations, model, text, usage}).
    Read-only is enforced below the model by the adapter (writes + shell disabled for task_kind=review).
    """
    result = run_cline_fix(
        working_dir,
        prompt,
        task_kind="review",
        provider=provider,
        model=model,
        max_iterations=max_iterations,
        timeout_s=timeout_s,
    )
    return extract_findings_json(result.get("text", "")), result
