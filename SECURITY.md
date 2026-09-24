# Security Policy

This repository contains the public Codna CLI (`codna` on PyPI), including
the `codna mcp` MCP server. We take the security of the CLI seriously and
appreciate responsible disclosure from the community.

## Supported versions

Security fixes land on `main` and in the latest published release of the
package (`codna` on PyPI).

| Channel | Supported |
| --- | --- |
| Latest release / `main` | :white_check_mark: |
| Older tagged releases | Best-effort; please upgrade to the latest |

## Reporting a vulnerability

**Please do not open a public issue, pull request, or discussion for
security problems.** Public disclosure before a fix is available puts other
users at risk.

Report privately through either channel:

1. **GitHub Security Advisories** (preferred) — open a private report from
   this repository's **Security → Report a vulnerability** tab.
2. **Email** — `security@algenta.ai`.

Please include, where possible: a description of the issue and its impact,
the affected command or module, steps to reproduce or a proof of concept,
and the package version you tested.

## What to expect

- Acknowledgement within 3 business days.
- An initial assessment and severity triage within 7 business days.
- Regular updates as we work on a fix, and credit in the published advisory
  (unless you prefer to remain anonymous).
- Coordinated disclosure: we agree on a timeline with you and publish a
  GitHub Security Advisory once a fix is available.

## Scope

**In scope** — this repository's own code, including:

- Credential and key handling (e.g. `codna key` storage in the OS keychain,
  accidental key logging, provider-key transport)
- The sandboxed command execution, worktree, and patch-application paths
  (`sandbox.py`, `worktree.py`, `writer.py`, `patch*`)
- The MCP server's tool surface (input validation, path handling, accidental
  data exposure across tools)
- The webhook/HTTP surfaces and their authentication
- Deserialization, injection, path-traversal, or other logic-safety bugs in
  CLI code
- Insecure defaults in the package

**Out of scope for this repository** (redirect privately to
`security@algenta.ai`, same as above, rather than filing here):

- The closed Algenta engine and its signed runtime (fetched separately)
- The private control-plane license-issuance service
- Any private activation or relay infrastructure

We also want to be upfront about the trust model: the CLI is designed to be
**assumed untrusted** — a report showing that the CLI's own client-side
checks can be bypassed by modifying the CLI is informational, not a
vulnerability, unless it also demonstrates that the closed engine's
independent, server-side enforcement was bypassed.

**Also out of scope:** third-party dependencies (report those upstream; we
still want to hear how they affect Codna), findings in a repository you ran
Codna *against* (report those to that repository's owner), and
social-engineering, physical, or denial-of-service testing against any
hosted environment.

Thank you for helping keep Codna and its users safe.
