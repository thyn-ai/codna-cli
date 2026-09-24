# Contributing to the Codna CLI

Thank you for your interest in contributing. This repository holds the
**Codna CLI** — `codna` on PyPI: understand, fix, evolve. It maps a
repository before it spends a token, reviews pull requests, fixes bugs and
proves which scanner findings are reachable, and ships the `codna mcp` MCP
server. This is the part of the product that is open source and meant to be
forked, read, and improved by anyone.

The Algenta engine beneath Codna (the deterministic execution substrate and
its signed runtime) is closed and lives in a separate, private repository.
Nothing in this repository grants access to it, and nothing you contribute
here can change how much execution capacity any license is entitled to —
that's enforced entirely on the engine side. See [SECURITY.md](./SECURITY.md)
for the trust boundary this implies for security reports.

## This repository is an automated mirror

Every file here is mirrored from `cli/` (plus a fixed root allowlist) of the
private `thyn-ai/codna` repository by an automated sync — nothing is exempt,
including these community files. A pull request that lands here is reviewed
and merged normally, and a maintainer then ports the change upstream; it
flows back out here on the next sync run. You do not need to do anything
special — just open the PR here — but please don't be surprised when the
commit that "sticks" arrives via the sync rather than your original commit.

## What you can contribute

| Area | Status | Notes |
|------|--------|-------|
| `codna/` (the CLI package) | ✅ Open | Bug fixes, type fixes, new commands, tests |
| `codna/mcp_server.py` + MCP surface | ✅ Open | Tool fixes, new read-only tools |
| `tests/`, `bench/` | ✅ Open | Test coverage, benchmarks |
| Docs / examples | ✅ Open | Corrections, new guides |
| Staged release artifacts | 🔒 Never | `codna/_agent_core_runtime/`, `_telys_runtime/` are build-time only |

## Getting started

```bash
git clone https://github.com/thyn-ai/codna-cli
cd codna-cli
pip install -e .
pytest tests/ -v
```

Python 3.12–3.13, `git` on your `PATH`. Local commands need no Codna key;
`fix` and `review` use your own model provider key (`codna key set
anthropic`, stored in the OS keychain). You do not need a running Algenta
engine for most of this code.

## Development workflow

### Branch naming
- `feat/short-description` — new feature
- `fix/short-description` — bug fix
- `docs/short-description` — documentation only

### Commit messages
We follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(cli): add --since flag to triage
fix(mcp): correct tool result pagination
docs(readme): document pipx install path
```

### Pull request checklist
- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] New features have tests
- [ ] Documentation updated if needed
- [ ] No hardcoded credentials or secrets
- [ ] I understand a maintainer will port the merged change upstream (see
      "automated mirror" above)

All required checks must pass, including on forked-repository pull
requests — CI runs with no secrets and no elevated permissions, so it's
safe to run automatically on every PR.

## Recognizing contributors

This project follows the [all-contributors](https://allcontributors.org)
specification: everyone who contributes — code, docs, bug reports, reviews,
or any other [contribution type](https://allcontributors.org/docs/en/emoji-key) —
is recognized in the [README](./README.md#contributors). Maintainers add
contributors by commenting `@all-contributors please add @user for code`
(replacing `code` with the relevant contribution type) on an issue or pull
request, and the bot opens a pull request updating the contributors table.

## Licensing

By submitting a pull request you agree that your contribution is licensed
under the project's [Apache-2.0 license](./LICENSE) (inbound=outbound,
[GitHub Terms of Service §D.6](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license)).

## Reporting issues

- **Security vulnerabilities** → see [SECURITY.md](./SECURITY.md) (do NOT
  open a public issue)
- **Bugs** → [GitHub Issues](https://github.com/thyn-ai/codna-cli/issues)
  with the `bug` label
- **Feature requests** → GitHub Issues with the `enhancement` label
- **Questions** → GitHub Issues with the `question` label, or
  https://discord.gg/w8NDsph9an

## Community

- Discord: https://discord.gg/w8NDsph9an
- Web: https://codna.ai
- Email: community@algenta.ai
