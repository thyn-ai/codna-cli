"""The CLI test suite as a REAL package — the `__init__.py` is load-bearing, not tidiness.

`cli/tests` used to be a namespace-package candidate, and a namespace portion loses to any regular
`tests` package that appears earlier on `sys.path`. Two independent consumers then disagreed about
what `tests.durable` means:

* the host interpreter, where pytest had already registered `tests.durable` for the modules it
  imported, and
* Temporal's workflow sandbox, which RE-IMPORTS a workflow's defining module through its own
  importer with a fresh `sys.modules` and must therefore resolve `tests` from `sys.path` alone.

In the sandbox that resolution landed on an unrelated regular `tests` package (measured: this
machine's user site-packages carries one), so importing `tests.durable.spike_workflows` raised
`ModuleNotFoundError` and every durable spike test died at worker construction with
`RuntimeError: Failed validating workflow SpikeTaskWorkflow` — the downstream symptom, not a
sandbox flake and not Temporal dev-server flakiness.

A regular package here makes host and sandbox resolve the SAME directory, so no sandbox-specific
configuration is needed. `tests/durable/__init__.py` exists for the same reason.

Consequence worth knowing: with a package root at `cli/tests`, pytest inserts `cli/` on `sys.path`
instead of `cli/tests/`, so intra-suite helpers must be imported as `tests.<name>` (e.g.
`from tests.telys_gate import ...`) rather than as bare top-level names.
"""
