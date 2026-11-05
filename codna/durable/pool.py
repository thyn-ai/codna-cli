"""POOL task queues (D6): several workers, ONE queue, and the precondition that makes it safe.

D2 gave every ``codna worker`` its own task queue (``codna.agent.<host>``) because a run's state is
per-machine: the checkout, the per-task worktrees and the append-only LEDGER all live on the disk of
the machine that started the run. A hosted pool breaks that assumption — the whole point of a pool is
that any machine may be handed any task — so this module owns the two decisions that keep a pool from
silently corrupting a run:

1. **Which queue a pool worker polls.** ``codna.pool.<name>``, derived from ``CODNA_DURABLE_POOL``.
   Every machine configured with the same pool name resolves the SAME queue, which is what makes
   Temporal distribute that queue's workflow and activity tasks across them — one delivery per task,
   never two. A pool queue is deliberately in a different namespace from the per-host queues, so a
   laptop's ``codna worker`` can never steal a hosted run (and a hosted worker can never be handed a
   developer's run that expects ``/Users/<them>/repo`` to exist).

2. **Whether that queue may have more than one worker.** Temporal guarantees a task is delivered to
   exactly one poller, but it does NOT make two machines share a filesystem. Two pool workers on one
   queue WILL eventually be handed two activities of the same run (two tasks of a run execute
   concurrently by design), and each would append to its own copy of that run's ledger — forking the
   hash chain that ``codna durable verify-ledger`` exists to verify, and losing one side's records
   with the machine. So ``workers > 1`` REQUIRES the operator to declare that the runs root is shared
   storage (``CODNA_DURABLE_POOL_STATE=shared``). The declaration is not a hint: without it this
   module REFUSES at startup (``pool_state_not_shared``), because the alternative failure mode is a
   run whose audit trail is split across machines with nothing saying so.

The refusal is the design, not a limitation being papered over. A single-worker pool needs no
declaration (there is nobody to share with), so a pool still scales HORIZONTALLY the moment the
operator mounts the same volume on every machine — and a Fly volume per machine does not qualify,
which is exactly what ``infra/README-durable-worker.md`` tells an operator.

No ``temporalio`` import: this module is reached from ``config.load_durable_config``, which the base
CLI imports without the ``durable`` extra installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Mapping

__all__ = [
    "ENV_POOL",
    "ENV_POOL_STATE",
    "ENV_POOL_WORKERS",
    "POOL_QUEUE_PREFIX",
    "POOL_STATES",
    "PoolError",
    "PoolSettings",
    "pool_settings_from_env",
    "pool_task_queue",
    "validate_pool_name",
]

#: Set to a pool name (``nightly``) to have this process use the pool queue instead of its host queue.
ENV_POOL: Final = "CODNA_DURABLE_POOL"
#: How many workers poll that one queue. Documentation plus a SAFETY input — see the module docstring.
ENV_POOL_WORKERS: Final = "CODNA_DURABLE_POOL_WORKERS"
#: ``shared`` when every worker's runs root is the same storage; ``local`` when it is per-machine.
ENV_POOL_STATE: Final = "CODNA_DURABLE_POOL_STATE"

POOL_QUEUE_PREFIX: Final = "codna.pool."
POOL_STATES: Final[tuple[str, ...]] = ("shared", "local")

# Deliberately the same character class as run ids and hostnames: a pool name becomes a task-queue
# string, a log field and (in the deploy docs) a Fly app suffix, so nothing outside this class may
# reach any of them. Duplicated from `config._SAFE_TOKEN` rather than imported, because `config`
# imports THIS module — and a cycle here would make `codna --version` fail.
_SAFE_POOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class PoolError(RuntimeError):
    """A pool configuration that must not be used. Always carries a stable ``code``.

    Raised from here and translated into ``DurableConfigError`` by ``load_durable_config``, so the
    CLI's existing structured-error rendering covers it without this module importing the config.
    """

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})


@dataclass(frozen=True)
class PoolSettings:
    """One pool: its queue string, how many workers, and whether their state is shared.

    Frozen and explicit because ``workers``/``state`` are read by the STARTUP refusal: a value that
    changed while a worker was polling would mean the safety decision no longer described the fleet.
    """

    name: str
    task_queue: str
    workers: int
    shared_state: bool

    @property
    def state(self) -> str:
        return "shared" if self.shared_state else "local"

    def to_payload(self) -> dict[str, object]:
        """JSON-safe, secret-free description for a startup banner or a deploy preflight."""
        return {
            "pool": self.name,
            "task_queue": self.task_queue,
            "workers": self.workers,
            "state": self.state,
        }


def validate_pool_name(raw: str) -> str:
    """A pool name as it appears in the queue string, or a refusal naming the rule."""
    value = (raw or "").strip()
    if not _SAFE_POOL.match(value):
        raise PoolError(
            f"{ENV_POOL} must match ^[A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}$ (got {value!r}).",
            code="invalid_pool_name",
            details={"variable": ENV_POOL, "value": raw},
        )
    return value


def pool_task_queue(name: str) -> str:
    """``codna.pool.<name>`` — the queue every worker of that pool polls.

    Derived, never configurable: two operators who agree on a name must get the same queue without
    also agreeing on a prefixed spelling, and the prefix is what keeps a hosted run out of the
    per-host queues. Temporal's own limit is far above 64 characters, so the bounded name is safe.
    """
    return f"{POOL_QUEUE_PREFIX}{validate_pool_name(name)}"


def _int_setting(env: Mapping[str, str]) -> int:
    raw = env.get(ENV_POOL_WORKERS)
    if raw is None:
        return 1
    text = raw.strip()
    # `int("+1")` and `int(" 1 ")` are both accepted by Python; only the plain decimal spelling is
    # accepted here, because this number decides whether a safety assertion is required.
    if not text.isdigit():
        raise PoolError(
            f"{ENV_POOL_WORKERS} must be a positive integer (got {raw!r}).",
            code="invalid_pool_workers",
            details={"variable": ENV_POOL_WORKERS, "value": raw},
        )
    workers = int(text)
    if workers < 1:
        raise PoolError(
            f"{ENV_POOL_WORKERS} must be at least 1 (got {workers}).",
            code="invalid_pool_workers",
            details={"variable": ENV_POOL_WORKERS, "value": raw},
        )
    return workers


def _state_setting(env: Mapping[str, str], *, workers: int) -> bool:
    raw = env.get(ENV_POOL_STATE)
    if raw is None:
        if workers > 1:
            # NO DEFAULT. The two possible defaults are both wrong: "local" would refuse a correctly
            # configured fleet, and "shared" would approve a fleet that forks its ledgers. So the
            # operator states it, once, in the same place they state the fleet size.
            raise PoolError(
                f"{ENV_POOL_WORKERS}={workers} needs {ENV_POOL_STATE} to say whether every worker's "
                "runs root is the same storage: set it to 'shared' (one volume/NFS every worker "
                "mounts) or to 'local' with a single worker.",
                code="pool_state_required",
                details={"variable": ENV_POOL_STATE, "workers": workers, "allowed": list(POOL_STATES)},
            )
        return False
    value = raw.strip().lower()
    if value not in POOL_STATES:
        raise PoolError(
            f"{ENV_POOL_STATE} must be one of {', '.join(POOL_STATES)} (got {raw!r}).",
            code="invalid_pool_state",
            details={"variable": ENV_POOL_STATE, "value": raw, "allowed": list(POOL_STATES)},
        )
    shared = value == "shared"
    if workers > 1 and not shared:
        # THE refusal this module exists for. See the module docstring: one ledger per run is written
        # by whichever machine runs its next activity, so two workers over per-machine disks fork it.
        raise PoolError(
            f"{ENV_POOL_WORKERS}={workers} with {ENV_POOL_STATE}=local would let two workers append "
            "to the same run's ledger on two different disks, forking the hash chain the audit check "
            "verifies. Run one worker, or mount one runs root on every worker and set "
            f"{ENV_POOL_STATE}=shared.",
            code="pool_state_not_shared",
            details={
                "variable": ENV_POOL_STATE,
                "workers": workers,
                "state": value,
                "runs_root_env": "CODNA_DURABLE_RUNS_ROOT",
            },
        )
    return shared


def pool_settings_from_env(env: Mapping[str, str]) -> PoolSettings | None:
    """The pool this environment configures, or ``None`` when it configures a per-host worker.

    ``None`` is the historical (D2) behaviour and stays byte-for-byte: no pool variable means the
    host-derived queue. A pool variable that is PRESENT BUT EMPTY is a refusal, matching every other
    override in this package — a template that rendered an empty value must not silently become
    "no pool" and send a worker to a queue nobody starts runs on.
    """
    if ENV_POOL not in env:
        return None
    name = validate_pool_name(env[ENV_POOL])
    workers = _int_setting(env)
    shared = _state_setting(env, workers=workers)
    return PoolSettings(
        name=name, task_queue=pool_task_queue(name), workers=workers, shared_state=shared
    )
