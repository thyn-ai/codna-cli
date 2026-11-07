"""Whether a task's work could be PUBLISHED, and why not — a LEAF module.

``open_pr`` is the last doing-step of every durable run, and publishing is not always possible. The
two cases the product actually meets are different in kind, and this module is where the difference
is named rather than inferred:

* **A SKIP** — nothing here can be published. A repository with no remote, or with a remote that is
  not GitHub, has nothing to open a pull request against, and no credential would change that. The
  work is done: the commits exist on the task's branch, verification ran, review approved. Failing
  the task would report a finished run as a broken one, which is the defect this vocabulary exists
  to remove. So the step records a skip with a named reason, and the task finishes
  ``approved_unpublished`` — success, without a pull request.
* **A REFUSAL** — something here *could* be published, but this worker is not configured to do it.
  A repository WITH a GitHub origin is an explicit statement that publishing is wanted, so a missing
  write token is a misconfiguration a user can fix in one command, not a fact about the repository.
  Reporting it as a skip would leave them believing a pull request exists when none does, so it
  stays a named, actionable failure (``activity_pr_token_missing``) plus a ledger note saying what
  to set.

Every value here is a short, closed string. It travels through `TaskState`, workflow history and the
ledger, so it must stay bounded and secret-free — which is also why the reason is a CODE and the
repository's origin URL is never part of it (a remote can embed a credential; the origin is
redacted before it reaches a ledger note and never reaches state at all).

LEAF: stdlib plus ``typing`` only. ``state.py`` imports it, ``state.py`` is imported by the workflow
sandbox, and the sandbox refuses anything that reaches ``urllib.request``.
"""
from __future__ import annotations

from typing import Any, Callable, Final, Literal

__all__ = [
    "GITHUB_TOKEN_ENV",
    "PUBLISH_BLOCKED_NO_CREDENTIAL",
    "PUBLISH_SKIP_NO_REMOTE",
    "PUBLISH_SKIP_REASONS",
    "PUBLISH_SKIP_REASON_VALUES",
    "PUBLISH_SKIP_REMOTE_NOT_GITHUB",
    "PublishSkipReason",
    "is_publish_skip",
    "publish_skip_from_payload",
    "validate_publish_skip",
]

#: The environment variables that may carry the GitHub write token, in precedence order. ONE
#: definition for both readers — `worker.github_token_provider` (environment, then the OS keystore)
#: and the publish step's own refusal — so a token that resolves for one cannot be invisible to the
#: other. `CODNA_` first, so a user's personal token for other tools is not picked up by accident.
GITHUB_TOKEN_ENV: Final[tuple[str, ...]] = ("CODNA_GITHUB_TOKEN", "GITHUB_TOKEN")

#: The repository has no ``origin`` remote at all: a local-only repository, which is the normal
#: shape for the "make me a small script" case the product is used for.
PUBLISH_SKIP_NO_REMOTE: Final = "no_remote"
#: The repository HAS an origin, but it is not a GitHub repository this product can open a pull
#: request against (GitLab, Bitbucket, a self-hosted host, a local path). Recorded separately from
#: ``no_remote`` because the user's next step is different: "add a remote" versus "this remote is
#: not one codna publishes to".
PUBLISH_SKIP_REMOTE_NOT_GITHUB: Final = "remote_not_github"

#: The closed set of reasons a publish step may be SKIPPED. An unknown code is refused wherever it
#: is decoded (`TaskState` validates the field), so a typo cannot become an unqueryable state.
PUBLISH_SKIP_REASONS: Final[tuple[str, ...]] = (
    PUBLISH_SKIP_NO_REMOTE,
    PUBLISH_SKIP_REMOTE_NOT_GITHUB,
)
#: The same set as a name-free sequence of VALUES, for a caller that iterates. One definition: the
#: tuple above is derived from it, never repeated.
PUBLISH_SKIP_REASON_VALUES: Final[tuple[str, ...]] = PUBLISH_SKIP_REASONS

#: The reason a publish was REFUSED rather than skipped. Not part of `PUBLISH_SKIP_REASONS`: a
#: refusal is an error the user must see, never a terminal state.
PUBLISH_BLOCKED_NO_CREDENTIAL: Final = "no_credential"

#: The same closure, as the literal type `TaskState` validates against. Kept beside the tuple so a
#: new reason cannot be added to one and not the other.
PublishSkipReason = Literal["no_remote", "remote_not_github"]


def is_publish_skip(value: Any) -> bool:
    """Whether ``value`` is one of the skip reasons. Total: never raises."""
    return isinstance(value, str) and value in PUBLISH_SKIP_REASONS


def publish_skip_from_payload(
    raw: Any, *, fail: Callable[..., Exception]
) -> PublishSkipReason | None:
    """A ``publish_skip`` from a state payload, refusing an unknown value rather than coercing it.

    ``fail`` builds the state module's own error type. It is INJECTED rather than imported because
    `TaskStateError` lives in `state.py`, which imports this module — the same reason
    `state_coercion.Coercion` takes its error class as a factory, and the reason this stays a leaf.

    The CLOSED SET is named in the message: a typo in a hand-written payload is a bug that would
    otherwise become an unknown reason no reader knows how to interpret.
    """
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in PUBLISH_SKIP_REASONS:
        raise fail(
            f"publish_skip must be null or one of {', '.join(PUBLISH_SKIP_REASONS)} (got {raw!r})",
            code="invalid_publish_skip",
            details={"publish_skip": repr(raw), "allowed": list(PUBLISH_SKIP_REASONS)},
        )
    return raw  # type: ignore[return-value]  # narrowed by the membership check above


def validate_publish_skip(
    publish_skip: Any, *, pr: Any, fail: Callable[..., Exception]
) -> None:
    """Refuse an unknown skip reason, and refuse a skip recorded BESIDE a pull request.

    The second check is what keeps the state's two facts from contradicting each other: a pull
    request EXISTS, so nothing was skipped, and a reader given both could not tell which happened.
    Refused rather than preferred — a downstream reader picking one would be inventing history.
    """
    if publish_skip is None:
        return
    if publish_skip not in PUBLISH_SKIP_REASONS:
        raise fail(
            f"invalid publish_skip {publish_skip!r}",
            code="invalid_publish_skip",
            details={"publish_skip": repr(publish_skip), "allowed": list(PUBLISH_SKIP_REASONS)},
        )
    if pr is not None:
        raise fail(
            "a state may not carry both a pull request and a publish_skip",
            code="invalid_publish_skip",
            details={"publish_skip": publish_skip, "pr": getattr(pr, "number", None)},
        )