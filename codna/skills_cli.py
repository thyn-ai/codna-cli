"""``codna skills`` — see exactly which skills a turn would carry, and pull one by name.

This is the sibling of ``codna rules`` and it exists for the same reason: the defect it answers is
INVISIBLE. A skill that is never discovered, or is discovered but scoped out by ``paths``, or does not
fit the catalogue budget, produces no error anywhere — the user simply watches a skill do nothing. So
every question a user can ask about the skill engine is answered here from the same ``codna.rules``
call the prompt path uses, and the command cannot drift from the behaviour it documents.

  * ``codna skills``                        — every skill discovered, its scope, path and invocation mode
  * ``codna skills --files src/a.ts``       — what a turn about those files would offer
  * ``codna skills --skill pdf``            — the body of one skill (the ``/skill-name`` pull)
  * ``codna skills --budget 400``           — what fits, what was dropped, and by how much
  * ``--json``                              — the same answer, structured, for tooling

Exit codes follow the CLI's convention (0 success, 1 refusal). A catalogue entry that did not fit is
REPORTED on stderr rather than being a failure — but ``--strict`` turns a drop into exit 1 for CI.
"""
from __future__ import annotations

import json
import sys

from .rules import Skill, SkillDiscovery, discover_skills, load_skill, skill_catalogue

#: The default catalogue ceiling for the STANDALONE command. The prompt path passes its own remainder
#: (see `rules.load_context`), because there the catalogue shares one budget with the rule bodies.
DEFAULT_CATALOGUE_BYTES = 4_096


def _skill_json(skill: Skill) -> dict:
    return {
        "name": skill.name,
        "description": skill.description,
        "scope": skill.scope.value,
        "mode": skill.mode.value,
        "source_path": skill.source_path,
        "paths": list(skill.paths),
        "bundled": list(skill.bundled),
        "disable_model_invocation": skill.disable_model_invocation,
        "bytes": len(skill.body.encode("utf-8")),
    }


def _print_human(
    discovery: SkillDiscovery,
    result,
    files,
    mentions,
    max_bytes: int,
) -> None:
    print("   in context: " + (", ".join(files) if files else "(not specified)"))
    if mentions:
        print("   mentions:   " + ", ".join(sorted(mentions)))
    print(f"   catalogue:  {result.used_bytes}/{max_bytes} bytes")
    print()

    if not discovery.skills:
        print("no skills found (looked for SKILL.md under .codna/skills, .cursor/skills, "
              ".agents/skills, .claude/skills, .codex/skills, and the same directories under ~)")
        return

    offered = {skill.name for skill in result.catalogued}
    dropped = {entry.name for entry in result.dropped}

    print("SKILLS")
    for skill in discovery.skills:
        if skill.name in dropped:
            state = "over budget"
        elif skill.name in offered:
            state = "CATALOGUED"
        else:
            state = "paths mismatch"
        patterns = (", ".join(skill.paths)) if skill.paths else "any"
        print(f"  [{state:<14}] {skill.mode.value:<8} {skill.scope.value:<8} {skill.source_path}"
              f"  (name: {skill.name}, paths: {patterns})")

    if result.dropped:
        print()
        print("NOT CATALOGUED")
        for entry in result.dropped:
            print(f"  [budget      ] {entry.source_path} (name: {entry.name}, "
                  f"{entry.bytes} bytes)")

    print()
    print("CATALOGUE THIS TURN WOULD CARRY (bodies are pulled with codna_load_skill)")
    print(result.text if result.text else "(empty)")


def run_skills(args) -> int:
    """Implementation of ``codna skills``. Returns the process exit code."""
    repo = args.repo
    max_bytes = args.budget if args.budget is not None else DEFAULT_CATALOGUE_BYTES
    if max_bytes < 0:
        print(json.dumps({"error": {"code": "skills_budget_invalid",
                                    "message": "--budget must be >= 0", "details": {}}}),
              file=sys.stderr)
        return 1

    # Pull ONE skill: the mechanism a catalogue entry needs, and the command `/skill-name` expands to.
    if args.skill:
        skill = load_skill(repo, args.skill)
        if skill is None:
            print(json.dumps({
                "error": {
                    "code": "skill_not_found",
                    "message": f"no skill named {args.skill!r}",
                    "details": {"repo": repo, "requested": args.skill},
                },
            }, indent=2), file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps({**_skill_json(skill), "body": skill.body}, indent=2))
        else:
            print(f"# {skill.source_path}  (name: {skill.name}, mode: {skill.mode.value})")
            if skill.bundled:
                print(f"# ships: {', '.join(skill.bundled)}")
            print(skill.body)
        return 0

    discovery = discover_skills(repo)
    result = skill_catalogue(
        discovery,
        files_in_context=args.files or None,
        explicit_mentions=args.mention or None,
        max_bytes=max_bytes,
    )
    if args.as_json:
        print(json.dumps({
            "files_in_context": list(args.files or ()),
            "explicit_mentions": list(args.mention or ()),
            "budget_bytes": result.budget_bytes,
            "used_bytes": result.used_bytes,
            "discovered": [_skill_json(skill) for skill in discovery.skills],
            "catalogued": [_skill_json(skill) for skill in result.catalogued],
            "dropped": [
                {"name": entry.name, "source_path": entry.source_path,
                 "reason": "budget", "bytes_needed": entry.bytes}
                for entry in result.dropped
            ],
            "warnings": [
                {"code": warning.code, "source_path": warning.source_path,
                 "message": warning.message}
                for warning in discovery.warnings
            ],
            "text": result.text,
        }, indent=2))
    else:
        _print_human(discovery, result, tuple(args.files or ()), tuple(args.mention or ()), max_bytes)

    # Drops and read-time warnings are reported on stderr in BOTH formats, because a machine consumer
    # that ignores the JSON account must still leave a trace a human sees.
    for warning in discovery.warnings:
        print(f"codna: warning: [skills] {warning.code}: {warning.source_path}: {warning.message}",
              file=sys.stderr)
    if result.dropped:
        names = ", ".join(f"{entry.name} ({entry.source_path})" for entry in result.dropped)
        print(f"codna: warning: [skills] {len(result.dropped)} skill(s) exceeded the "
              f"{max_bytes}-byte catalogue budget and were NOT catalogued: {names}",
              file=sys.stderr)
    if args.strict and result.dropped:
        return 1
    return 0


def register(sub) -> None:
    """Add ``codna skills`` to the root parser."""
    parser = sub.add_parser(
        "skills",
        help="Show which skills (.codna/skills, .cursor/skills, .agents/skills, …) apply, and why.",
    )
    parser.add_argument("repo", nargs="?", default=".", help="repository path (default: current dir)")
    parser.add_argument("--files", action="append",
                        help="a repository-relative file this turn is about (repeatable); a skill "
                             "with `paths:` is offered only when one of these matches")
    parser.add_argument("--mention", action="append",
                        help="an explicitly requested skill name, e.g. /pdf (repeatable)")
    parser.add_argument("--budget", type=int, default=None,
                        help=f"catalogue budget in bytes (default: {DEFAULT_CATALOGUE_BYTES})")
    parser.add_argument("--skill", help="print one skill's body by name (the /skill-name pull)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any skill was dropped from the catalogue (for CI)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="emit the catalogue as JSON")
    parser.set_defaults(func=run_skills, uses_config=False)


__all__ = ["DEFAULT_CATALOGUE_BYTES", "register", "run_skills"]
