# Skills fixture

A deliberately-shaped repository whose skill layout proves the properties this feature claims:

| Path | Proves |
|---|---|
| `.cursor/skills/pdf/SKILL.md` with `paths: src/**/*.ts` | a skill in CURSOR's directory, scoped by `paths` |
| `.cursor/skills/pdf/scripts/fill.ts` | a skill ships bundled files, listed relative to its folder |
| `.codna/skills/release/SKILL.md` with `disable-model-invocation: true` | manual-only: catalogued, never auto-attached |
| `.agents/skills/testing/api/SKILL.md` | an identity that is the FOLDER: `testing` is a category, `api` is the skill |

There is no `AGENTS.md` and no rules directory, so this tree tests the skill engine alone.
