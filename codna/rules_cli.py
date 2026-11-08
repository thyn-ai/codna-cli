"""``codna rules`` — see exactly which rules a turn would carry, and why.

This command exists because the defect it replaces was INVISIBLE: the old reader truncated silently,
so "why didn't my rule apply" had no answer anywhere in the product. Every question a user can ask
about the rule engine is answered here from the same ``codna.rules`` call the prompt path uses, so the
command cannot drift from the behaviour it documents:

  * ``codna rules``                       — every rule discovered, with its mode, scope and path
  * ``codna rules --files src/a.tsx``     — what a turn about those files would actually attach
  * ``codna rules --mention @python``     — what an explicit ``@``-mention pulls in
  * ``codna rules --budget 2000``         — what fits, what was dropped, and by how much
  * ``codna rules --rule python``         — the body of one rule (the manual/catalog pull)
  * ``--json``                            — the same answer, structured, for tooling

Exit codes follow the CLI's convention (0 success, 1 refusal) and add one useful distinction: the
command exits 0 even when rules were dropped, because a drop is REPORTED on stderr rather than being
a failure — but ``--strict`` turns a drop into exit 1 for CI use.
"""
from __future__ import annotations

import json
import sys

from .rules import DEFAULT_MAX_BYTES, Discovery, RenderResult, Rule, load, request_rule


def _rule_json(rule: Rule) -> dict:
    return {
        "name": rule.name,
        "mode": rule.mode.value,
        "scope": rule.scope.value,
        "source_path": rule.source_path,
        "base_dir": rule.base_dir,
        "description": rule.description,
        "globs": list(rule.globs),
        "always_apply": rule.always_apply,
        "legacy_default_always": rule.legacy_default_always,
        "bytes": len(rule.body.encode("utf-8")),
    }


def _discovery_json(discovery: Discovery, result: RenderResult, files, mentions) -> dict:
    return {
        "files_in_context": list(files),
        "explicit_mentions": sorted(mentions),
        "budget_bytes": result.budget_bytes,
        "used_bytes": result.used_bytes,
        "discovered": [_rule_json(rule) for rule in discovery.rules],
        "attached": [_rule_json(rule) for rule in result.kept],
        "catalogued": [_rule_json(rule) for rule in result.catalogued],
        "dropped": [
            {
                "name": dropped.name,
                "source_path": dropped.source_path,
                "reason": dropped.reason,
                "bytes_needed": dropped.bytes_needed,
            }
            for dropped in result.dropped
        ],
        "truncated": [
            {
                "name": record.name,
                "source_path": record.source_path,
                "bytes_kept": record.bytes_kept,
                "bytes_total": record.bytes_total,
            }
            for record in result.truncated
        ],
        "warnings": [
            {"code": warning.code, "source_path": warning.source_path, "message": warning.message}
            for warning in discovery.warnings
        ],
        "text": result.text,
    }


def _print_human(discovery: Discovery, result: RenderResult, files, mentions, max_bytes: int,
                 selection=None) -> None:
    print("   in context: " + (", ".join(files) if files else "(not specified)"))
    if mentions:
        print("   mentions:   " + ", ".join(sorted(mentions)))
    print(f"   budget:     {used_label(result.used_bytes, max_bytes)}")
    print()

    attached = {id(rule) for rule in result.kept}
    catalogued = {id(rule) for rule in result.catalogued}
    out_of_scope = {id(rule) for rule in (selection.out_of_scope if selection else ())}
    unmatched = {id(rule) for rule in (selection.unmatched if selection else ())}
    shadowed_paths = {d.source_path for d in result.dropped_for_precedence}
    budget_paths = {d.source_path for d in result.dropped_for_budget}
    if not discovery.rules:
        print("no rules found (looked for .codna/rules/**/*.mdc, .cursor/rules/**/*.mdc, "
              "AGENTS.md at any depth, and the legacy root files)")
        return

    print("RULES")
    for rule in discovery.rules:
        if id(rule) in attached:
            state = "ATTACHED"
        elif rule.source_path in shadowed_paths:
            state = "shadowed"
        elif rule.source_path in budget_paths:
            state = "over budget"
        elif id(rule) in catalogued:
            state = "CATALOGUED"
        elif id(rule) in out_of_scope:
            state = "other subtree"
        elif id(rule) in unmatched:
            state = "no glob match"
        elif rule.mode.value == "manual":
            state = "manual only"
        else:
            state = "not applied"
        where = f", base: {rule.base_dir}" if rule.base_dir else ""
        print(f"  [{state:<14}] {rule.mode.value:<8} {rule.scope.value:<8} {rule.source_path}"
              f"  (name: {rule.name}{where})")
    if result.dropped:
        print()
        print("NOT ATTACHED")
        for dropped in result.dropped:
            print(f"  [{dropped.reason:<10}] {dropped.source_path} (name: {dropped.name}, "
                  f"{dropped.bytes_needed} bytes)")
    if result.truncated:
        print()
        print("TRUNCATED")
        for record in result.truncated:
            print(f"  {record.source_path}: {record.bytes_kept}/{record.bytes_total} bytes kept")
    print()
    print("PROMPT TEXT THIS TURN WOULD CARRY")
    print(result.text if result.text else "(empty)")


def used_label(used: int, budget: int) -> str:
    return f"{used}/{budget} bytes"


def run_rules(args) -> int:
    """Implementation of ``codna rules``. Returns the process exit code."""
    repo = args.repo
    max_bytes = args.budget if args.budget is not None else DEFAULT_MAX_BYTES
    if max_bytes < 0:
        print(json.dumps({"error": {"code": "rules_budget_invalid",
                                    "message": "--budget must be >= 0", "details": {}}}),
              file=sys.stderr)
        return 1

    # Pull ONE rule: the mechanism a catalogued rule needs, and the command a user runs by hand.
    if args.rule:
        rule = request_rule(repo, args.rule, user_rules_dir=args.user_rules_dir)
        if rule is None:
            print(json.dumps({
                "error": {
                    "code": "rule_not_found",
                    "message": f"no rule named {args.rule!r}",
                    "details": {"repo": repo, "requested": args.rule},
                },
            }, indent=2), file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps({**_rule_json(rule), "body": rule.body}, indent=2))
        else:
            print(f"# {rule.source_path}  (name: {rule.name}, mode: {rule.mode.value})")
            print(rule.body)
        return 0

    loaded = load(
        repo,
        files_in_context=args.files or None,
        explicit_mentions=args.mention or None,
        max_bytes=max_bytes,
        user_rules_dir=args.user_rules_dir,
        include_catalog=not args.no_catalog,
    )
    if args.as_json:
        print(json.dumps(
            _discovery_json(loaded.discovery, loaded.result, loaded.selection.files,
                            loaded.selection.mentions),
            indent=2,
        ))
    else:
        _print_human(loaded.discovery, loaded.result, loaded.selection.files,
                     loaded.selection.mentions, max_bytes, loaded.selection)

    # Drops are reported on stderr in BOTH formats, because a machine consumer that ignores the JSON
    # account must still leave a trace a human sees.
    from .rules import emit_diagnostics

    emit_diagnostics(loaded.result, warnings=loaded.discovery.warnings, stream=sys.stderr)
    if args.strict and (loaded.result.dropped_for_budget or loaded.result.truncated):
        return 1
    return 0


def register(sub) -> None:
    """Add ``codna rules`` to the root parser."""
    parser = sub.add_parser(
        "rules",
        help="Show which project rules (.codna/rules, .cursor/rules, AGENTS.md) apply, and why.",
    )
    parser.add_argument("repo", nargs="?", default=".", help="repository path (default: current dir)")
    parser.add_argument("--files", action="append",
                        help="a repository-relative file this turn is about (repeatable); "
                             "glob-scoped rules apply only when one of these matches")
    parser.add_argument("--mention", action="append",
                        help="an explicitly requested rule name, e.g. @python (repeatable)")
    parser.add_argument("--budget", type=int, default=None,
                        help=f"prompt budget in bytes (default: {DEFAULT_MAX_BYTES})")
    parser.add_argument("--rule", help="print one rule's body by name (the manual/catalog pull)")
    parser.add_argument("--user-rules-dir", dest="user_rules_dir", default=None,
                        help="override the user rules directory (default: ~/.codna/rules)")
    parser.add_argument("--no-catalog", dest="no_catalog", action="store_true",
                        help="omit the one-line catalog of description-only rules")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any rule was dropped or truncated (for CI)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="emit the selection as JSON")
    parser.set_defaults(func=run_rules, uses_config=False)


__all__ = ["register", "run_rules"]
