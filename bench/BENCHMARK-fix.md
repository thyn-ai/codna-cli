# `codna fix` benchmark — real public repos

_Generated 2026-06-26 23:11 UTC · live engine + agent-core sidecar (real Cline, real LLM) · inspect mode (triage → agent plan → patch ref, no apply) · 5/5 repos._

> Each row is one real agent run (real model + USD cost) over the engine's deterministic triage + the agentic Cline decision plan. No mocks. Idle/no-progress timeout only — runs are bounded by agent turns, not a wall clock.

## Results

| Repo | Localized symbol | Blast | Confidence | Regression risk | Context reduction | Model | Cost | Time | Patch | Status |
|---|---|---|--:|--:|---|---|--:|--:|---|---|
| requests | HTTPAdapter, add_headers, build_connection_pool_key_attributes, build_response, get_connection_with_tls_context, proxy_manager_for, request_url, send | low | 55% | 95% | 156,993 → 5,934 tokens  (26× smaller) | claude-sonnet-4-6 | $0.936 | 108s | patch_ac8ad4e26c9b3364c99f7979 | ok |
| click | _NamedTextIOWrapper, make_input_stream, mode | low | 55% | 47% | 290,110 → 5,261 tokens  (55× smaller) | claude-sonnet-4-6 | $4.965 | 2192s | patch_21c7aa3247a6238094cee5d0 | ok |
| flask | get_debug_flag | low | 55% | 21% | 288,845 → 5,769 tokens  (50× smaller) | claude-sonnet-4-6 | $0.282 | 40s | patch_92af1681a7ebaf91d4ab44aa | ok |
| httpx | — | — | — | — | — | — | — | 2409s | — | exit 1 |
| rich | get_character_cell_size | high | 55% | 95% | 976,796 → 5,036 tokens  (194× smaller) | claude-sonnet-4-6 | $0.135 | 38s | patch_5eb8383f210aad8a33b53a1d | ok |

## Transcripts (real output)

### requests — _Session.send does not retry when the underlying connection times out; add a retry-on-timeout path_
```text
codna: fixing /tmp/bench-repos/requests …

✓ codna fixed /tmp/bench-repos/requests in 108s
  root cause   : Repository issue localized from the injected evidence bundle.
  symbol       : HTTPAdapter, add_headers, build_connection_pool_key_attributes, build_response, get_connection_with_tls_context, proxy_manager_for, request_url, send  (blast radius: low)
  confidence   : 55%  ·  regression risk: 95%
  context      : 156,993 → 5,934 tokens  (26× smaller)
  agent        : claude-sonnet-4-6 via codna  ·  cost: $0.936
  patch        : patch_ac8ad4e26c9b3364c99f7979   (--apply for a local branch · --open-pr to open a PR)
```

### click — _a required option in a command group does not raise a clear error when omitted_
```text
codna: fixing /tmp/bench-repos/click …

✓ codna fixed /tmp/bench-repos/click in 2192s
  root cause   : Repository issue localized from the injected evidence bundle.
  symbol       : _NamedTextIOWrapper, make_input_stream, mode  (blast radius: low)
  confidence   : 55%  ·  regression risk: 47%
  context      : 290,110 → 5,261 tokens  (55× smaller)
  agent        : claude-sonnet-4-6 via codna  ·  cost: $4.965
  patch        : patch_21c7aa3247a6238094cee5d0   (--apply for a local branch · --open-pr to open a PR)
```

### flask — _send_file returns the wrong content-type for .webp files_
```text
codna: fixing /tmp/bench-repos/flask …

✓ codna fixed /tmp/bench-repos/flask in 40s
  root cause   : Repository issue localized from the injected evidence bundle.
  symbol       : get_debug_flag  (blast radius: low)
  confidence   : 55%  ·  regression risk: 21%
  context      : 288,845 → 5,769 tokens  (50× smaller)
  agent        : claude-sonnet-4-6 via codna  ·  cost: $0.282
  patch        : patch_92af1681a7ebaf91d4ab44aa   (--apply for a local branch · --open-pr to open a PR)
```

### httpx — _the connection pool is not released when a streaming response is closed early_
```text
codna: fixing /tmp/bench-repos/httpx …
Traceback (most recent call last):
  File "/tmp/bench-venv/bin/codna", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/private/tmp/codna-bench/cli/codna/cli.py", line 705, in main
    args.func(args)
  File "/private/tmp/codna-bench/cli/codna/cli.py", line 214, in cmd_fix
    plan = _dump(c.create_repository_decision_plan(rid, {
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/angel/Developer/decision-engine/packages/python-sdk/decision_engine/client_facade/_mixin_connector_repository.py", line 229, in create_repository_decision_plan
    return _repository_surface_module().create_repository_decision_plan(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/angel/Developer/decision-engine/packages/python-sdk/decision_engine/client_repository_surface.py", line 45, in create_repository_decision_plan
    data = client._request("POST", f"/v1/repositories/{repository_id}/decision-plans", json=request)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/angel/Developer/decision-engine/packages/python-sdk/decision_engine/client_facade/__init__.py", line 366, in _request
    return _transport_module().request_with_retries(self, method, path, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/angel/Developer/decision-engine/packages/python-sdk/decision_engine/client_transport.py", line 40, in request_with_retries
    body = client._handle_response(response)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/angel/Developer/decision-engine/packages/python-sdk/decision_engine/client_facade/__init__.py", line 372, in _handle_response
    return _response_module().handle_response(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/angel/Developer/decision-engine/packages/python-sdk/decision_engine/transport_response.py", line 56, in handle_response
    raise ServerError(
decision_engine.exceptions.ServerError: verified_agentic agent run did not complete (status=failed).
```

### rich — _Table column width is miscalculated when a cell contains wide (CJK) unicode characters_
```text
codna: fixing /tmp/bench-repos/rich …

✓ codna fixed /tmp/bench-repos/rich in 38s
  root cause   : Repository issue localized from the injected evidence bundle.
  symbol       : get_character_cell_size  (blast radius: high)
  confidence   : 55%  ·  regression risk: 95%
  context      : 976,796 → 5,036 tokens  (194× smaller)
  agent        : claude-sonnet-4-6 via codna  ·  cost: $0.135
  patch        : patch_5eb8383f210aad8a33b53a1d   (--apply for a local branch · --open-pr to open a PR)
```

## Method
- `codna fix <repo> --issue "…"` (inspect mode) against a live engine + agent-core sidecar; the agentic patch is generated by the vendored Cline SDK.
- Metrics parsed from real CLI output; cost/model are what the agent actually used.
- Issues are realistic, plausible bugs (not planted) — measures localization + planning quality, not a known-answer pass/fail.
