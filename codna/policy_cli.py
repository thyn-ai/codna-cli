"""`python3 -m codna.policy_cli verify --file <path> [--required|--optional]`.

The sidecar's ONLY door into the policy validator (`server/policy-document.ts`). Kept as a thin
module so the package has a stable `-m` target even if `policy_document` is renamed.

THIS MODULE IS THE PRODUCT'S SESSION GATE. `policy-document.ts` runs it for every session, and a
non-zero exit or an unparseable answer is `403 policy_verdict_unavailable` — including for a chat that
asks for no policy. So it must be importable with NOTHING but the stdlib and whatever the payload
stages: it has no third-party imports of its own, and its one dependency (`policy_document`'s YAML
reader) degrades to the stdlib subset reader rather than failing. MEASURED on 0.1.48: a payload
without PyYAML made this module answer `policy_reader_unavailable` for every real policy.
"""
from __future__ import annotations

from .policy_document import main

if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by both suites
    raise SystemExit(main())
