"""Who may call which Run API route — the role model, and the reason it cannot be forgotten.

The Run API is the one surface in codna that a second person can reach (a hosted deployment, a
teammate's IDE pointed at a shared host). "The client hides the button" is not a control: the
transport must decide, per route, against an identity it verified itself. This module is that
decision, and it is deliberately the only place it is written down.

**Three roles, one per kind of act.**

    viewer    read: the run list, one run, its transcripts/events
    operator  change work: start, steer, budget, pause, resume, cancel
    approver  decide: approve (and, by the same act, deny/reject)

They are granted by ROLE_GRANTS, which is an EXPLICIT expansion rather than an ordering compared
with ``<``/``>=``: an operator holds viewer, an approver holds operator and viewer. A hierarchy
expressed as an integer invites "role_value - 1" sprawl; a table is auditable in one screen and a
new role cannot become stronger than intended by being spelled later in an enum.

**Deny by default, structurally.**

``ROUTE_ROLES`` is a closed mapping from a route key to the MINIMUM role that route needs. There is
no wildcard, no prefix rule, and no "if the map is missing this route, allow it". `Route` (the
router's own type, in `run_api_http.py`) refuses to be constructed for a key that is not in this
mapping, so a route added without a role requirement fails when the code is imported — not when it
is exercised, and not in production. `required_role` raises `UnknownRouteError` rather than
returning a permissive default, and `Principal.require` raises `PermissionError`-shaped
`AuthorizationError` rather than returning a boolean a caller could ignore.

**What this module is NOT.** It does not authenticate. It says "this subject, proven by the
identity layer, holds these roles, so it may reach this route". Turning an HTTP request into that
subject is `identity.py`'s job, and it is the only thing that may construct a `Principal`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Mapping

__all__ = [
    "ALL_ROLES",
    "AuthorizationError",
    "KNOWN_ROLES",
    "ROLE_GRANTS",
    "ROUTE_ROLES",
    "Principal",
    "Role",
    "UnknownRouteError",
    "required_role",
    "role_from_claim",
    "route_key",
]


# The closed set. A string, not an enum, because it lands in JSON (the ledger, the SIEM envelope,
# the 403 body) and an Enum's repr in a log is a debugging tax with no safety benefit.
Role = str

VIEWER: Final[Role] = "viewer"
OPERATOR: Final[Role] = "operator"
APPROVER: Final[Role] = "approver"

KNOWN_ROLES: Final[frozenset[Role]] = frozenset({VIEWER, OPERATOR, APPROVER})
ALL_ROLES: Final[frozenset[Role]] = KNOWN_ROLES

# What holding a role actually grants. Explicit, expanded, and the only source of inheritance.
ROLE_GRANTS: Final[Mapping[Role, frozenset[Role]]] = {
    VIEWER: frozenset({VIEWER}),
    OPERATOR: frozenset({VIEWER, OPERATOR}),
    APPROVER: frozenset({VIEWER, OPERATOR, APPROVER}),
}


class UnknownRouteError(LookupError):
    """A route key with no declared requirement.

    Its own type rather than a `KeyError` because "somebody added a route" and "somebody looked up a
    typo" are different bugs, and only the first must be loud at import time.
    """


class AuthorizationError(RuntimeError):
    """An authenticated principal that does not hold the role a route requires.

    Carries the route and the roles involved so a 403 body can say WHAT was missing without the
    caller having to guess, and so a SIEM record can name the refused attempt.
    """

    def __init__(
        self,
        message: str,
        *,
        route: str,
        required: Role,
        held: Iterable[Role] = (),
    ) -> None:
        super().__init__(message)
        self.route = route
        self.required = required
        self.held: tuple[Role, ...] = tuple(sorted(held))


# -- the route table ---------------------------------------------------------------------------
#
# Keys are "<collection>.<verb>" and name a ROUTE, not an HTTP verb: `POST /runs/:id/cancel` is
# `runs.cancel` and its Temporal signal is `stop`. Keeping the key independent of the wire path is
# what lets the router and the role table agree without one implying the other.
ROUTE_ROLES: Final[Mapping[str, Role]] = {
    # reads
    "runs.capabilities": VIEWER,
    "runs.list": VIEWER,
    "runs.get": VIEWER,
    "runs.events": VIEWER,
    # starting and steering work
    "runs.start": OPERATOR,
    "runs.steer": OPERATOR,
    "runs.budget": OPERATOR,
    "runs.pause": OPERATOR,
    "runs.resume": OPERATOR,
    "runs.cancel": OPERATOR,
    # the only human-decision route
    "runs.approve": APPROVER,
}

# How a mutation route's short name becomes its route key. Declared here (not derived by string
# concatenation at the call site) so a renamed route is a single edit and the table above stays the
# authority on what exists.
MUTATION_ROUTE_KEYS: Final[Mapping[str, str]] = {
    "steer": "runs.steer",
    "approve": "runs.approve",
    "budget": "runs.budget",
    "pause": "runs.pause",
    "resume": "runs.resume",
    "cancel": "runs.cancel",
}


def route_key(action: str) -> str:
    """The route key for a mutation's short name, refusing an undeclared one."""
    try:
        return MUTATION_ROUTE_KEYS[action]
    except KeyError as exc:
        raise UnknownRouteError(f"no route is declared for mutation {action!r}") from exc


def required_role(key: str) -> Role:
    """The minimum role ``key`` needs. Raises rather than defaulting to anything.

    A default here is exactly the "if role: allow" bug the plan forbids: the day somebody adds a
    route and forgets the table, a permissive default turns the omission into an open door.
    """
    try:
        return ROUTE_ROLES[key]
    except KeyError as exc:
        raise UnknownRouteError(
            f"route {key!r} has no declared role requirement; add it to ROUTE_ROLES "
            "(there is deliberately no default)"
        ) from exc


# -- the principal -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """A verified caller: who they are, and which roles they hold.

    Constructed ONLY by `identity.py` after a credential has been verified. There is no public
    constructor that takes a role from a request body, and no `admin` escape hatch: `identity` is the
    audit string, `roles` is the granted set, and neither is settable from the wire.

    ``source`` is part of the audit record on purpose: "this approval came from the local owner
    token" and "this approval came from an OIDC subject" are different facts, and an auditor must be
    able to tell them apart after the fact.
    """

    identity: str
    roles: frozenset[Role]
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("a principal needs a non-empty identity")
        unknown = sorted(set(self.roles) - KNOWN_ROLES)
        if unknown:
            # Fail closed at construction: an unrecognised role must never be carried around in the
            # hope that some later check ignores it.
            raise ValueError(f"unknown roles granted: {', '.join(unknown)}")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("a principal needs a source")

    # -- the grant, expanded -------------------------------------------------

    @property
    def granted(self) -> frozenset[Role]:
        """Every role this principal effectively holds (the named roles plus what they imply)."""
        expanded: set[Role] = set()
        for role in self.roles:
            expanded.update(ROLE_GRANTS[role])
        return frozenset(expanded)

    def holds(self, role: Role) -> bool:
        return role in self.granted

    def require(self, key: str) -> None:
        """Raise `AuthorizationError` unless this principal may reach route ``key``.

        Deliberately a raising method rather than ``can(key) -> bool``: a boolean is a value a caller
        can compute and then forget to branch on, and every route here is reached through exactly one
        call site that would have to ignore the answer.
        """
        needed = required_role(key)
        if not self.holds(needed):
            raise AuthorizationError(
                f"this caller does not hold the {needed!r} role that {key} requires.",
                route=key,
                required=needed,
                held=self.roles,
            )

    def as_audit(self) -> dict[str, object]:
        """The principal as a ledger/SIEM row fragment (never a credential)."""
        return {
            "identity": self.identity,
            "roles": sorted(self.roles),
            "source": self.source,
        }


def role_from_claim(value: object) -> frozenset[Role]:
    """The role set from a token claim, keeping only roles this build understands.

    An unknown value is DROPPED, not an error: a provider that adds a role we have not implemented
    must not be able to grant it by naming it. Dropping is also what makes an empty/absent claim a
    deny (the empty set holds nothing) rather than a crash on a malformed token.
    """
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        return frozenset()
    return frozenset(item.strip().lower() for item in candidates if item.strip().lower() in KNOWN_ROLES)
