# Codna

**Understand. Fix. Evolve.**

Agents read your code. Codna understands it.

Codna maps a repository before it spends a token. It then reviews pull requests, fixes bugs and
proves which scanner findings are reachable. The same `codna` command runs on your machine, in the
GitHub Action and behind the GitHub App.

Runs on your machine or your cloud. Your code never leaves without your key.

## Install

```bash
pip install codna          # or: pipx install codna · uv tool install codna
codna --version
```

Python 3.12–3.13. Wheels for macOS on Apple silicon and Linux x86_64. `git` on your `PATH`.
Nothing else to install: no Node, Bun, Docker or server. The agent runtime ships inside the wheel.

Local commands need no Codna key. `fix` and `review` use your own model provider key, stored in
the OS keychain and never printed:

```bash
codna key set anthropic    # also: openai, gemini, google, groq, mistral, openrouter, xai, cursor
```

## Commands

```bash
codna triage . --issue "checkout total is wrong"     # suspect files. Deterministic. 0 LLM tokens.
codna review . --pr 123 --post                        # findings with a verdict on the pull request
codna fix . --tests --apply --max-iterations 3        # patch, re-run your tests, re-fix until green
codna fix <git url> --ref <sha> --issue "…" --open-pr # push a branch and open a pull request
codna secure . --from-sarif results.sarif             # which scanner findings are reachable. 0 LLM tokens.
```

`codna fix` prints the root cause, the impacted symbols, the blast radius, its confidence and a
regression risk. `--open-pr` needs a git URL and a GitHub write token. `codna review` posts one
inline comment per finding with severity, category and a suggestion block, and an Approve when the
diff is clean at medium and high. `codna secure` reads SARIF 2.1.0 from CodeQL, Semgrep, Snyk,
Trivy or any other scanner.

Other commands: `init`, `status`, `doctor`, `login`, `key`, `impact`, `memory export`, `report`.
Full reference: [docs.codna.ai/reference/cli](https://docs.codna.ai/reference/cli).

## MCP server

```bash
pip install "codna[mcp]"
codna mcp install --client cursor     # or --client claude
```

Tools: `codna_triage`, `codna_fix`, `codna_secure`, `codna_recall`, `codna_report_bug`.

## Code memory

Code memory and recall run on your machine with no login and no extra install. The optional
`codna[memory]` extra adds the on-device semantic reranker; without it, recall ranks lexically.

## What leaves your machine

- Repository mapping, triage, recall and `impact` run offline. No model is involved.
- `fix` and `review` send one issue-specific evidence bundle to the provider you chose, under your
  key. Not the repository.
- Secret redaction is always on and cannot be turned off.
- `privacy.egress: fail-closed` in `codna.yaml` refuses to run tests without network denial and
  skips registry lookups during review.
- Source distributions exclude runtime binaries, keys, `.env` files and logs.

## Links

- Homepage: https://codna.ai
- Documentation: https://docs.codna.ai
- Source: https://github.com/thyn-ai/codna
- Security: https://codna.ai/security
