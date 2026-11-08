# Demo Repo — rules fixture

A deliberately-shaped monorepo that exercises every part of the rule engine. It is a FIXTURE, not a
real project: the files exist so `codna rules` has something real to select from.

Layout:

```
AGENTS.md                                  project-wide, legacy (always on)
.cursorrules                               legacy (always on)
.codna/rules/
  python.mdc            alwaysApply: true
  typescript.mdc        globs: "**/*.ts,**/*.tsx"
  styles.mdc            globs: "**/*.css"
  migrations.mdc        description only  -> CATALOGUED, pulled by name
  release.mdc           name only         -> MANUAL, @-mention only
.cursor/rules/
  no-default-exports.mdc   globs: "**/*.tsx"
  python.mdc               alwaysApply: true (same name as .codna/rules/python.mdc -> TEAM wins)
web/
  AGENTS.md                                subtree: applies only under web/
  .cursor/rules/react.mdc   globs: "src/**/*.tsx" (relative to web/)
api/
  .codna/rules/pydantic.mdc alwaysApply: true (applies only under api/)
```

Try it:

```
cd cli
PYTHONPATH=. python3 -m codna.cli rules tests/fixtures/rules_repo --files src/app.tsx
PYTHONPATH=. python3 -m codna.cli rules tests/fixtures/rules_repo --files src/app.css
PYTHONPATH=. python3 -m codna.cli rules tests/fixtures/rules_repo --files api/handlers/users.py
PYTHONPATH=. python3 -m codna.cli rules tests/fixtures/rules_repo --files web/src/App.tsx --budget 400
PYTHONPATH=. python3 -m codna.cli rules tests/fixtures/rules_repo --rule migrations
```

`cli/tests/test_rules_fixture_repo.py` pins the properties those commands demonstrate, so this
fixture cannot silently drift into a nice-looking but wrong example.
