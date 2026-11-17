"""The narrow policy decision for completing a local-only task unattended.

This leaf module is deliberately independent of the policy YAML reader. The workflow must not read
repository files during replay: the CLI validates the document and carries its normalized mode and
digest in the `TaskSpec`. This function decides only from that validated grant and durable evidence.
"""
from __future__ import annotations

from typing import Final

from .state import TaskState

__all__ = [
    "AUTO_APPROVAL_REASON",
    "AUTO_APPROVAL_RESUME",
    "auto_approval_eligible",
]

AUTO_APPROVAL_REASON: Final = "policy_auto_approved_local_unpublished"
AUTO_APPROVAL_RESUME: Final[dict[str, bool]] = {"auto_approved": True}


def auto_approval_eligible(state: TaskState, *, enabled: bool) -> bool:
    """Whether completion is local, reviewed, verified, and inside budget.

    Absence of a policy grant, a missing verify result, or any publish result is not an exemption.
    In particular, `remote_not_github` is excluded: Codna did not publish there, but that is not
    proof that the repository has no outbound remote. Only the publish step's `no_remote` evidence
    proves the narrow case this policy permits.
    """
    return bool(
        enabled
        and state.ask is None
        and state.plan_ref is not None
        and state.verify_ref is not None
        and state.verify_ref.passed
        and state.review_verdict == "approved"
        and not state.budget.exhausted
        and state.publish_skip == "no_remote"
        and state.pr is None
    )
