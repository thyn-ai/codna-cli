# Governance

This document describes how decisions are made for the Codna CLI
(`thyn-ai/codna-cli`). It is intentionally lightweight and will evolve as
the contributor community grows.

## Roles

- **Contributors** — anyone who opens issues, participates in discussions, or
  submits pull requests.
- **Maintainers** — people with merge and release authority on this
  repository. The project currently has a single maintainer: the `thyn-ai`
  organization owner (see [CODEOWNERS](./CODEOWNERS)), who also holds final
  decision authority on all matters not explicitly delegated.

## Decision-making

Day-to-day decisions are made by **lazy consensus**:

1. Propose the change as a GitHub issue or pull request.
2. Maintainers and contributors discuss in the open.
3. If no maintainer objects within 72 hours (three business days), the
   proposal is considered accepted and may proceed.

Maintainers may fast-track obvious, low-risk changes (typo fixes, CI repairs,
dependency security bumps) without waiting out the window. Any maintainer may
pause lazy consensus by raising an objection, in which case the change waits
until the objection is resolved in discussion. Where consensus cannot be
reached, the organization owner makes the final call.

This repository is an automated mirror of `cli/` in the private
`thyn-ai/codna` repository (see [CONTRIBUTING.md](./CONTRIBUTING.md)).
Mirror mechanics — what the sync carries across, and the upstream port of a
merged community change — are exempt from lazy consensus: they are operated
by the maintainers and cannot be changed by external pull request.

## Becoming a maintainer

External contributors can become maintainers. The path:

1. **Sustained contribution** — a track record of merged pull requests,
   thoughtful reviews, and issue triage over several months.
2. **Nomination** — an existing maintainer nominates the contributor, citing
   specific contributions, in a GitHub discussion visible to all maintainers.
3. **Lazy consensus** — if no maintainer objects within 14 days, the
   nomination carries. The organization owner confirms and grants access.

Maintainers are expected to review pull requests, triage issues, uphold the
[Code of Conduct](./CODE_OF_CONDUCT.md), follow the
[security policy](./SECURITY.md), and keep the repository green on `main`.
Maintainers who become inactive for more than a year may be moved to emeritus
status by the organization owner; emeritus maintainers can regain commit
access on request.

## Releases

- The `codna` package on PyPI follows
  [Semantic Versioning](https://semver.org/); notable changes are recorded in
  [CHANGELOG.md](./CHANGELOG.md).
- Releases are cut from the private source repository by a maintainer
  (release-please) and flow out to this mirror through the automated sync.
  Release authorization currently rests solely with the organization owner.
- Patch releases may be cut whenever a worthwhile fix lands. Minor releases
  are cut when backward-compatible additions have accumulated. Major releases
  require an announced deprecation window for any removal.
- Release decisions — what ships and when — are made by maintainers through
  lazy consensus as described above, with the organization owner holding
  final authority.

## Changing this document

Amendments to this file follow the same lazy-consensus process as any other
change, with one difference: the review window is 14 days, and the
organization owner must approve the merged pull request.
