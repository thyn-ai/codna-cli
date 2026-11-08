"""Which checkout a run works in, read back from its ledger (Run API `GET /runs`).

**Why this is its own module.** The Run API's list route needs one fact per ledger — "which repository
does this run belong to?" — and that fact lives at the FRONT of the file, while every other list fact
(ended / outcome / stalled / last_seq) is read from the TAIL. Two opposite windows, two different
questions. `run_api_support` holds the tail readers and had reached the repository's 1000-line
ceiling; a front-of-file reader is a genuine separate concern rather than a slice taken for size, so
it lives here.

**Why the answer can be `None`, and why that is not a failure.** `repo_dir` was added to a run's
opening notification after the first runs were written, so a ledger that predates it has nothing to
report. The honest answer for such a run is "this ledger does not say" — a caller must present that as
unknown, because the bug this field exists to remove was exactly the opposite: every durable row was
labelled with whichever folder the IDE window happened to have open, so a run from repo A appeared
under repo B after the window was reopened elsewhere.

**Why the read is bounded.** The value is written once, by the `run.started` notification that is a
run's FIRST record. Scanning only the first 64 KiB keeps a list read independent of how large a run's
ledger has grown (a long run writes one line per streamed event), and it is what makes the answer
cheap for the ledgers that have nothing to find.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from .ledger import LedgerError, LedgerRecord

__all__ = ["repo_dir_of"]

# How far FORWARD the scan looks. The opening notification is a run's first record, so a handful of
# lines covers every ledger that recorded one.
_HEAD_SCAN_RECORDS: Final = 8
_HEAD_WINDOW_BYTES: Final = 64 * 1024


def repo_dir_of(path: Path) -> str | None:
    """The checkout the run works in, from the ledger's opening notification. ``None`` when unrecorded.

    The FIRST absolute ``repo_dir`` on a run-level record wins. A ledger can hold several — a resumed
    run re-notifies on ``continue_as_new`` — and they are the same checkout by construction (the run's
    spec is carried across the boundary), so taking the first is deterministic and independent of how
    many generations the run needed.

    Only a ``note`` is a candidate: ``turn.event`` records wrap the engine's own payload, so a
    ``repo_dir`` inside one describes the engine's message rather than the run.
    """
    for record in _head_records(path, limit=_HEAD_SCAN_RECORDS):
        if record.kind != "note":
            continue
        raw = record.payload.get("repo_dir")
        if isinstance(raw, str) and raw.startswith("/"):
            return raw
    return None


def _head_records(path: Path, *, limit: int) -> list[LedgerRecord]:
    """Up to ``limit`` complete records from the START of the file, reading only the first 64 KiB.

    The mirror of `run_api_support._tail_records`, and bounded the same way.

    EVERY chunk is considered, including a final one with no trailing newline — the same rule the tail
    reader follows, and it matters here: a run that has only just started may have exactly one record,
    and dropping the last chunk unconditionally would report "no repository" for the one ledger that
    definitely has one. A record that is genuinely half-written fails to decode as a `LedgerRecord`
    (it is missing `seq`/`hash`/…) and is skipped, so tolerance here does not admit a partial row.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(_HEAD_WINDOW_BYTES)
    except OSError:
        return []
    if not head:
        return []
    found: list[LedgerRecord] = []
    for raw in head.split(b"\n"):
        text = raw.strip()
        if not text:
            continue
        try:
            decoded = json.loads(text.decode("utf-8"))
            if not isinstance(decoded, dict):
                continue
            found.append(LedgerRecord.from_payload(decoded))
        except (UnicodeDecodeError, json.JSONDecodeError, LedgerError):
            # A damaged leading line must not hide the rest of the window: the repository is a
            # convenience for a sidebar, and `ledger_problems` is where an integrity failure is
            # reported. Stopping here would turn one bad line into "this run has no repository".
            continue
        if len(found) >= limit:
            break
    return found
