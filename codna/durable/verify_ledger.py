"""`codna run verify-ledger` — walk the D2 hash chain and REPORT divergence, never crash on it.

The ledger is tamper-*evidence*: each record carries ``prev_hash`` and its own ``hash``, so editing a
past event changes its hash and breaks every link after it. `ledger.verify_chain` already computes
that; this module is what a person runs, and the difference matters:

* **A broken chain is a REPORT, not an exception.** `verify_chain` raises `LedgerError` when a line
  is not JSON or a record is malformed — which is precisely what a tampered file looks like. A
  verifier that tracebacks on tampering is useless to an auditor, so every decoding failure is
  caught and turned into a located finding. The only thing that can still fail the process is "there
  is no ledger at all", and even that is a coded verdict with a non-zero exit, not a stack trace.

* **Every finding names the index and BOTH hashes.** "seq 4 hash mismatch" is not actionable; "seq 4:
  stored abc123…, recomputed def456…" is what lets a reader diff the file against a known-good head.
  The same goes for a link break: the stored `prev_hash` and the hash actually expected are printed.

* **A shortened chain is a finding, not a clean verdict.** The plan's rule is that the chain must not
  be weakenable, and the cheapest way to weaken one is to drop its tail. `verify` therefore compares
  the record count against a caller-supplied expectation (the last `seq` a client or a SIEM row
  remembers) and reports a chain that shrank. "The file verifies" and "nothing was removed" are two
  different questions, and this answers both.

Exit codes follow the CLI's convention: 0 = intact, 2 = it ran and the answer is "not intact".
`1` is reserved for "the command could not do its job" (no ledger), which is a diagnostic failure
rather than a verdict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator, Mapping

from .ledger import GENESIS_HASH, LedgerError, compute_hash

__all__ = [
    "EXIT_DIVERGED",
    "EXIT_INTACT",
    "Finding",
    "LedgerVerification",
    "verify_ledger_file",
]

EXIT_INTACT: Final = 0
EXIT_DIVERGED: Final = 2


@dataclass(frozen=True)
class Finding:
    """One divergence, located by ``seq`` with both hashes so a reader can compare heads.

    ``kind`` is a closed set so a report can be filtered/counted rather than grepped:
    ``line`` (unparseable), ``record`` (malformed), ``seq`` (gap/duplicate), ``link`` (prev_hash
    break), ``hash`` (content changed), ``shrank`` (fewer records than expected).
    """

    kind: str
    seq: int
    detail: str
    stored_hash: str = ""
    computed_hash: str = ""
    expected_prev: str = ""

    def to_payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {"kind": self.kind, "seq": self.seq, "detail": self.detail}
        if self.stored_hash:
            body["stored_hash"] = self.stored_hash
        if self.computed_hash:
            body["computed_hash"] = self.computed_hash
        if self.expected_prev:
            body["expected_prev"] = self.expected_prev
        return body

    def describe(self) -> str:
        """One audit line: index, what broke, and both hashes when they exist."""
        parts = [f"seq {self.seq}: {self.detail}"]
        if self.stored_hash and self.computed_hash:
            parts.append(f"stored {self.stored_hash} computed {self.computed_hash}")
        if self.expected_prev and self.stored_hash:
            parts.append(f"expected_prev {self.expected_prev}")
        return " | ".join(parts)


@dataclass
class LedgerVerification:
    """The verdict: what was read, the findings, and the head a caller can pin a backup against."""

    path: Path
    exists: bool
    records: int
    last_seq: int
    head_hash: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def intact(self) -> bool:
        return self.exists and not self.findings

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "intact": self.intact,
            "records": self.records,
            "last_seq": self.last_seq,
            "head_hash": self.head_hash,
            "findings": [finding.to_payload() for finding in self.findings],
        }


def _iter_raw(path: Path) -> Iterator[tuple[int, Mapping[str, Any] | str]]:
    """Yield ``(line_number, decoded_or_text)`` for every non-blank line.

    Unlike `ledger._iter_records`, a bad line is YIELDED as text rather than raised: the caller wants
    to report it and keep walking, because "the file was truncated at line 9" is more useful than
    "line 9 is invalid" with no idea what follows.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                yield line_number, text
                continue
            yield line_number, decoded if isinstance(decoded, dict) else text


def verify_ledger_file(path: Path, *, expected_records: int | None = None) -> LedgerVerification:
    """Verify a ledger file. Returns a report; raises NOTHING for a broken chain.

    ``expected_records`` is the ``seq`` a caller last saw (a client's resume point, a SIEM row's
    ``last_seq``). When supplied it is a LOWER BOUND: fewer records than that means the tail was
    removed, which a plain chain walk cannot see (a truncated chain is internally consistent).
    """
    target = Path(path)
    if not target.is_file():
        return LedgerVerification(
            path=target,
            exists=False,
            records=0,
            last_seq=0,
            head_hash=GENESIS_HASH,
            findings=[
                Finding(kind="line", seq=0, detail=f"no ledger at {target}")
            ],
        )

    findings: list[Finding] = []
    expected_prev = GENESIS_HASH
    expected_seq = 1
    records = 0
    last_seq = 0
    head_hash = GENESIS_HASH

    for line_number, item in _iter_raw(target):
        if isinstance(item, str):
            findings.append(
                Finding(
                    kind="line",
                    seq=expected_seq,
                    detail=f"line {line_number} is not a JSON object",
                )
            )
            # A line that cannot be decoded breaks the chain for everything after it, because the
            # next record's prev_hash names a hash this line should have supplied.
            expected_prev = ""
            continue
        try:
            seq = int(item["seq"])
            stored_hash = str(item["hash"])
            stored_prev = str(item["prev_hash"])
            body = {key: value for key, value in item.items() if key != "hash"}
            computed = compute_hash(body)
        except (KeyError, TypeError, ValueError) as exc:
            findings.append(
                Finding(
                    kind="record",
                    seq=expected_seq,
                    detail=f"line {line_number} is not a ledger record: {exc}",
                )
            )
            expected_prev = ""
            continue

        if seq != expected_seq:
            findings.append(
                Finding(
                    kind="seq",
                    seq=seq,
                    detail=f"record {records + 1} carries seq {seq}, expected {expected_seq}",
                )
            )
        if expected_prev and stored_prev != expected_prev:
            findings.append(
                Finding(
                    kind="link",
                    seq=seq,
                    detail="prev_hash does not match the previous record's hash",
                    stored_hash=stored_prev,
                    expected_prev=expected_prev,
                )
            )
        if computed != stored_hash:
            findings.append(
                Finding(
                    kind="hash",
                    seq=seq,
                    detail="record content does not match its hash",
                    stored_hash=stored_hash,
                    computed_hash=computed,
                )
            )
        expected_prev = stored_hash
        expected_seq = seq + 1
        records += 1
        last_seq = seq
        head_hash = stored_hash

    if records == 0 and not findings:
        # An empty file is a ledger with no records: intact, but say so rather than implying a run
        # happened. `intact` is true (nothing diverged) and `records` is 0.
        return LedgerVerification(
            path=target, exists=True, records=0, last_seq=0, head_hash=GENESIS_HASH
        )

    if expected_records is not None and last_seq < int(expected_records):
        findings.append(
            Finding(
                kind="shrank",
                seq=last_seq,
                detail=(
                    f"the chain holds {records} records ending at seq {last_seq}, but "
                    f"{int(expected_records)} was expected: records were removed from the tail"
                ),
            )
        )
    return LedgerVerification(
        path=target,
        exists=True,
        records=records,
        last_seq=last_seq,
        head_hash=head_hash,
        findings=findings,
    )


def verify_or_raise(path: Path, *, expected_records: int | None = None) -> LedgerVerification:
    """`verify_ledger_file`, but a malformed LINE is raised as the ledger's own coded error.

    Used by callers that must distinguish "diverged" (a report) from "unreadable" (an error). The CLI
    deliberately does NOT use it: an auditor's tool reports.
    """
    report = verify_ledger_file(path, expected_records=expected_records)
    for finding in report.findings:
        if finding.kind == "line":
            raise LedgerError(
                finding.detail,
                code="ledger_line_invalid",
                details={"path": str(path), "seq": finding.seq},
            )
    return report
