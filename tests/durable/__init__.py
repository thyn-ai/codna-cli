"""The durable suite as a REAL package (see `cli/tests/__init__.py` for the full reason).

Temporal's workflow sandbox re-imports a workflow's defining module (`tests.durable.spike_workflows`)
inside its own importer, with an empty `sys.modules`, so it resolves the package chain from
`sys.path`. Without this file `tests.durable` is only a namespace portion and loses to any regular
`tests` package on the path, which is what produced the "ModuleNotFoundError: No module named
'tests.durable'" / "Failed validating workflow" pair.
"""
