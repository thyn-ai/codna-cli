"""`python3 -m codna.policy_document verify --file <path> [--required|--optional]`.

The sidecar's ONLY door into the policy validator (`server/policy-document.ts`). Kept as a thin
module so the package has a stable `-m` target even if `policy_document` is renamed.
"""
from __future__ import annotations

from .policy_document import main

if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by both suites
    raise SystemExit(main())
