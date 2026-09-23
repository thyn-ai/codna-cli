"""The `codna ci` command surface: fleet-wide CI admission.

Split out of `cli.py` for the same reason as `byok_cli`/`mcp_cli`/`webhook_cli` — cli.py is held under
a hard line ceiling by `test_modularity`, and a command's parser plus its handler belong together.

Both the parser and the handler live here so `cli.py` carries one import and one call.
"""
from __future__ import annotations


def cmd_ci(args) -> int:
    """Delegates to `codna.ci_admission.main` so this surface and the workflow-facing
    `python -m codna.ci_admission` path cannot drift apart. Always exits 0."""
    from .ci_admission import main as ci_main

    action = getattr(args, "ci_action", "status") or "status"
    argv = [action]
    if action == "admit":
        argv += ["--job-class", args.job_class]
        for item in getattr(args, "meta", []) or []:
            argv += ["--meta", item]
    elif action == "release" and getattr(args, "lease", None):
        argv += ["--lease", args.lease]
    return ci_main(argv)


def register(sub) -> None:
    """Attach `codna ci {admit,release,status}` to the top-level subparsers."""
    pci = sub.add_parser("ci", help="Keep a self-hosted runner fleet inside a chosen share of one machine.")
    pci_sub = pci.add_subparsers(dest="ci_action")

    admit = pci_sub.add_parser("admit", help="Wait for fleet capacity before a CI job does real work.")
    admit.add_argument("--job-class", required=True,
                       help="stable identity for this kind of job (e.g. 'ci/unit-tests') — the bucket "
                            "its observed durations are learned in")
    admit.add_argument("--meta", action="append", default=[], metavar="K=V",
                       help="annotate the lease (repeatable), e.g. --meta run_id=123")
    admit.set_defaults(func=cmd_ci, ci_action="admit")

    release = pci_sub.add_parser("release", help="Release a fleet slot and record the job's duration.")
    release.add_argument("--lease", help="lease path from `codna ci admit` (default: $CODNA_CI_LEASE)")
    release.set_defaults(func=cmd_ci, ci_action="release")

    status = pci_sub.add_parser("status", help="Show the fleet's capacity budget and what is running.")
    status.set_defaults(func=cmd_ci, ci_action="status")

    pci.set_defaults(func=cmd_ci, ci_action="status")
