# Subagents fixture

A deliberately-shaped repository whose SUBAGENT layout proves the properties this feature claims. The
same shape as `skills_repo` on purpose, so the two fixtures read side by side.

| Path | Proves |
|---|---|
| `.cursor/agents/verifier.md` with `readonly: true` | a subagent in CURSOR's directory, mapped onto the engine's read-only profile |
| `.claude/agents/verifier.md` | a same-named agent in a LOWER-precedence layer, which `.cursor` shadows and REPORTS |
| `.cursor/agents/planner.md` with no `name:` | identity is the FILENAME, and an explicit `model:` overrides the parent's |

There is no `AGENTS.md` and no skills directory, so this tree tests the subagent engine alone.
`tests/test_agents_engine.py` copies it to a temporary directory for any case that mutates it.
