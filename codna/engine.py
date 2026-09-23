"""Engine adapter — binds the `SecurityEngine` protocol to the decision-engine HTTP API.

`EngineAdapter` implements the protocol `secure.run_secure` consumes. It speaks the batch,
snapshot-cached `security-analyses` contract: the whole-program analysis is requested ONCE
per snapshot (build-once / query-many), then each finding's verdict is a cache lookup by
`canonical_id` — never one engine call per finding (RPRV-02).

Two client-side honesty defenses live here so a buggy/hostile engine response can't smuggle
a stronger verdict than it earned:
  * an `unreachable` verdict is downgraded to `unknown` unless the analysis envelope is
    complete and sound ("no path found" in an incomplete envelope is NOT unreachable);
  * the response must be bound to OUR request (snapshot id + sarif digest) or it is rejected
    (RPRV-30 — no response substitution).

The transport is an injected `post(path, body) -> dict` callable, so the adapter is fully
testable offline with a stub; `http_post_factory` wires the real httpx client.
"""
from __future__ import annotations

from typing import Callable

from .findings import Classification, ClosureStatus, IngestResult, NormalizedFinding
from .sarif import build_security_analysis_request
from .secure import ClosureVerdict, ReachVerdict

PostFn = Callable[[str, dict], dict]

_CLASSIFICATIONS = {c.value for c in Classification}
_CLOSURES = {c.value for c in ClosureStatus}


class EngineError(Exception):
    pass


def _to_classification(s: str | None) -> Classification:
    v = (s or "").strip().lower()
    return Classification(v) if v in _CLASSIFICATIONS else Classification.UNKNOWN


def _to_closure(s: str | None) -> ClosureStatus:
    v = (s or "").strip().lower()
    return ClosureStatus(v) if v in _CLOSURES else ClosureStatus.UNKNOWN


class EngineAdapter:
    def __init__(
        self,
        post: PostFn,
        *,
        repository_id: str,
        snapshot_id: str,
        policy_digest: str,
        model_pack_digest: str,
        configuration_digest: str | None = None,
    ):
        self._post = post
        self.repository_id = repository_id
        self.snapshot_id = snapshot_id
        self.policy_digest = policy_digest
        self.model_pack_digest = model_pack_digest
        self.configuration_digest = configuration_digest
        self._verdicts: dict[str, ReachVerdict] | None = None
        self.analysis_id: str | None = None

    # -- batch analysis (build-once / query-many) ------------------------------------

    def analyze_all(self, ingest: IngestResult) -> dict[str, ReachVerdict]:
        if self._verdicts is None:
            req = build_security_analysis_request(
                ingest,
                snapshot_id=self.snapshot_id,
                policy_digest=self.policy_digest,
                model_pack_digest=self.model_pack_digest,
                configuration_digest=self.configuration_digest,
            )
            resp = self._post(f"/v1/repositories/{self.repository_id}/security-analyses", req)
            self._verify_binding(resp, req)
            verdicts: dict[str, ReachVerdict] = {}
            for proof in resp.get("proofs", []) or []:
                cid = proof.get("canonical_finding_id") or proof.get("canonical_id")
                if cid:
                    verdicts[cid] = self._verdict_from_proof(proof)
            self._verdicts = verdicts
            self.analysis_id = resp.get("analysis_id")
        return self._verdicts

    def analyze(self, ingest: IngestResult, finding: NormalizedFinding) -> ReachVerdict:
        verdict = self.analyze_all(ingest).get(finding.canonical_id)
        if verdict is None:
            # No proof for this finding -> UNKNOWN. Absence of a verdict is never silently
            # treated as 'unreachable'.
            return ReachVerdict(Classification.UNKNOWN, envelope_complete=False, proof_type=None)
        return verdict

    def reprove_closure(self, finding: NormalizedFinding, patch) -> ClosureVerdict:
        resp = self._post(
            f"/v1/repositories/{self.repository_id}/security-analyses/closure",
            {
                "snapshot_id": self.snapshot_id,
                "canonical_finding_id": finding.canonical_id,
                "patch_digest": getattr(patch, "patch_digest", None),
            },
        )
        return ClosureVerdict(
            closure_status=_to_closure(resp.get("closure_status")),
            alternate_path_found=bool(resp.get("alternate_path_found", False)),
            new_blocking_findings=list(resp.get("new_blocking_findings") or []),
        )

    # -- defenses --------------------------------------------------------------------

    def _verdict_from_proof(self, proof: dict) -> ReachVerdict:
        cls = _to_classification(proof.get("classification"))
        envelope = proof.get("analysis_envelope") or {}
        complete = bool(envelope.get("is_complete_sound", False))
        if cls is Classification.UNREACHABLE and not complete:
            # 'unreachable' requires a complete sound envelope; otherwise it's 'unknown'.
            cls = Classification.UNKNOWN
        return ReachVerdict(cls, envelope_complete=complete, proof_type=proof.get("proof_type"))

    def _verify_binding(self, resp: dict, req: dict) -> None:
        snap = (resp.get("snapshot") or {})
        echoed_snapshot = snap.get("id") or resp.get("snapshot_id")
        if echoed_snapshot is not None and echoed_snapshot != req["snapshot_id"]:
            raise EngineError(
                f"engine response bound to snapshot {echoed_snapshot!r}, not request {req['snapshot_id']!r}"
            )
        echoed_sarif = resp.get("sarif_digest")
        if echoed_sarif is not None and echoed_sarif != req["sarif_digest"]:
            raise EngineError("engine response sarif_digest does not match the request (possible substitution)")


def http_post_factory(base_url: str, api_key: str | None, timeout: float = 300.0) -> PostFn:
    """Real transport over httpx. Imported lazily so the adapter core stays dependency-free."""
    import httpx

    base = base_url.rstrip("/")

    def post(path: str, body: dict) -> dict:
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        resp = httpx.post(f"{base}{path}", json=body, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise EngineError(f"engine {path} -> {resp.status_code} {resp.text[:200]}")
        return resp.json()

    return post
