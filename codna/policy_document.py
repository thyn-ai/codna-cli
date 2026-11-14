"""`.codna/policy.yaml` — the repository's policy-as-code document, validated FAIL-CLOSED.

This is the ONE implementation. Two independent consumers read the same file through it and must
reach the same verdict:

* the **durable worker** (`cli/codna/durable/activity_support.py`) imports it in-process;
* the **agent-core sidecar** (`agent-core/vendor/cline/algenta/server/policy-document.ts`) shells
  out to `python3 -m codna.policy verify`, which is `main()` below.

A second parser in TypeScript is how the two sides silently start disagreeing about what "the
policy permits", so there is deliberately no second parser. If the interpreter or this module is
unavailable, the sidecar DENIES (see its `policy_validator_unavailable`).

## Fail-closed, in the specific sense the track asks for

Every one of these is a DENY, never a fall-through to the permissive approval profile:

* the file is missing and a policy is REQUIRED (`policy_missing`);
* the file exists but cannot be read (`policy_unreadable`);
* the YAML does not parse (`policy_unparseable`) — including DUPLICATE KEYS, which PyYAML would
  otherwise silently resolve last-wins (a second `tools:` block is how a policy grows permissions
  without anyone reviewing the change);
* the document is not a mapping (`policy_invalid_shape`);
* an unknown key at any level (`policy_unknown_key`) — a typo'd `tool:` grants nothing, but it also
  silently disables the intent, so it is refused loudly;
* a value of the wrong type (`policy_invalid_type`);
* a required field is absent from a section that is present (`policy_missing_field`);
* the document grants no permission at all (`policy_denies_all`) — an empty file is not "no
  policy", it is a policy that permits nothing, and both readers must say so rather than run with
  the defaults.

The one non-deny outcome that is not "allow" is `mode: "absent"`: NO policy file, and none was
required. That is not a permissive fallback from a present policy — there is no policy to read —
and it is reported as its own mode so a caller can never confuse it with "the policy permitted
this". A run that must be governed by a policy sets `required=True` (the durable path does).

## Schema (version 1)

```yaml
version: 1
tools:
  allow: [read_files, search_codebase]   # required in this section; [] permits no tool
  deny:  [run_commands]                  # optional; deny wins over allow
write_scope:
  paths: ["src/**", "tests/**"]          # required; repo-relative globs; [] permits no write
  deny:  ["src/generated/**"]            # optional
command_allow_list:
  commands: [pytest, git]                # required; bare program names, never paths
budget:
  max_cost_usd: 5.0                      # >= 0, finite
  max_turn_cost_usd: 0.5
  max_tokens: 250000                     # positive integer
required_approvals: [run_commands]       # tool names a human must approve every call to
approval:
  mode: human                            # or auto_for_low_risk: verified local-only completion only
```

`approval.mode` defaults to `human`. `auto_for_low_risk` permits only terminal completion after the
publish step proves there is no `origin` remote, verification passed, review approved and budget
remains. It does NOT permit a tool call, command, edit or publish: those gates are unchanged.
```

Every section is optional at the top level (an absent section permits nothing in that category),
but a section that IS present must carry its required field. Unknown tool/command names are not
schema errors: they match `[a-z][a-z0-9_]*`, resolve to no permission, and therefore fail closed.

Pure stdlib + PyYAML. Deterministic: the digest is over the canonical JSON of the parsed document,
so formatting and key order do not change it.

PyYAML is PREFERRED but not required: `policy_yaml` reads the subset of YAML a policy is written in
(block mappings and sequences, flow sequences of scalars, plain/quoted scalars, numbers, booleans,
nulls, comments), refusing duplicate keys and unknown constructs rather than approximating them. So a
payload that ships without PyYAML still evaluates a policy instead of refusing every session; a
document outside that subset is refused with `policy_reader_subset_exceeded`, which names the
construct and tells the operator to install PyYAML.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import policy_yaml
from .findings import digest_of

__all__ = [
    "DEFAULT_POLICY_RELPATH",
    "MAX_DOCUMENT_DEPTH",
    "PolicyDocumentError",
    "PolicyVerdict",
    "RuntimePermissions",
    "load_policy_document",
    "main",
    "policy_verdict",
    "policy_path_for",
]

#: Where a repository's policy lives, relative to the repository root.
DEFAULT_POLICY_RELPATH = os.path.join(".codna", "policy.yaml")

#: Nesting bound for the parsed document. YAML aliases can build a deep or cyclic graph; the schema
#: is three levels deep, so anything past this is malformed rather than an exotic-but-valid policy.
MAX_DOCUMENT_DEPTH = 16

TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "version", "tools", "write_scope", "command_allow_list", "budget",
        "required_approvals", "approval",
    }
)
APPROVAL_KEYS: frozenset[str] = frozenset({"mode"})
APPROVAL_MODES: frozenset[str] = frozenset({"human", "auto_for_low_risk"})
TOOLS_KEYS: frozenset[str] = frozenset({"allow", "deny"})
WRITE_SCOPE_KEYS: frozenset[str] = frozenset({"paths", "deny"})
COMMAND_KEYS: frozenset[str] = frozenset({"commands"})
BUDGET_KEYS: frozenset[str] = frozenset({"max_cost_usd", "max_turn_cost_usd", "max_tokens"})

#: A tool name as the engine spells it. A typo still matches and still grants nothing.
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
#: A bare program name — the same shape `server/sandbox.ts` accepts for a command allow-list.
_COMMAND_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_SUPPORTED_VERSIONS = frozenset({1})


class PolicyDocumentError(ValueError):
    """A policy document that must DENY. Always carries a stable code and a JSON-path location."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        path: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.details: dict[str, Any] = {"at": path, **dict(details or {})}

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "details": self.details,
            }
        }


@dataclass(frozen=True)
class RuntimePermissions:
    """What a valid policy PERMITS, already normalized so both readers apply identical values.

    Empty tuples/absent caps mean "nothing permitted" in that category, never "unrestricted". The
    consumers are responsible for turning that into refusals; this type never widens anything.
    """

    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    allowed_write_paths: tuple[str, ...] = ()
    denied_write_paths: tuple[str, ...] = ()
    allowed_commands: tuple[str, ...] = ()
    max_cost_usd: float | None = None
    max_turn_cost_usd: float | None = None
    max_tokens: int | None = None
    required_approvals: tuple[str, ...] = ()
    approval_mode: str = "human"

    def effective_tools(self) -> tuple[str, ...]:
        """`allow` minus `deny`: deny always wins, which is the only safe precedence."""
        denied = set(self.denied_tools)
        return tuple(t for t in self.allowed_tools if t not in denied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.effective_tools()),
            "declared_allowed_tools": list(self.allowed_tools),
            "denied_tools": list(self.denied_tools),
            "allowed_write_paths": list(self.allowed_write_paths),
            "denied_write_paths": list(self.denied_write_paths),
            "allowed_commands": list(self.allowed_commands),
            "budget": {
                "max_cost_usd": self.max_cost_usd,
                "max_turn_cost_usd": self.max_turn_cost_usd,
                "max_tokens": self.max_tokens,
            },
            "required_approvals": list(self.required_approvals),
            "approval_mode": self.approval_mode,
        }


@dataclass(frozen=True)
class PolicyVerdict:
    """The single answer both consumers act on. Constructed only by this module.

    ``mode``:
      * ``"policy"`` — a document was read (or was required and missing). Act on ``allowed``: when
        it is False the caller must DENY, and ``permissions`` is None.
      * ``"absent"`` — no document, and none was required. ``allowed`` is False here too, so a
        caller that forgets to check ``mode`` still fails closed; the mode is what tells a caller
        it may fall back to the pre-existing approval profile.
    """

    mode: str
    allowed: bool
    code: str
    reason: str
    source: str | None = None
    policy_digest: str = ""
    version: int | None = None
    permissions: RuntimePermissions | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_policy_absent(self) -> bool:
        return self.mode == "absent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed": self.allowed,
            "code": self.code,
            "reason": self.reason,
            "source": self.source,
            "policy_digest": self.policy_digest,
            "version": self.version,
            "permissions": self.permissions.to_dict() if self.permissions else None,
            "details": dict(self.details),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# ── YAML loading that cannot be made permissive by a duplicate key ─────────────────────────────


def _yaml_load(raw: str, source: str) -> Any:
    """Parse YAML with duplicate keys REFUSED and unknown tags rejected.

    PyYAML's `safe_load` already rejects arbitrary object construction, but it silently keeps the
    last value for a repeated key. For a permission document that is a fail-open: a `tools:` block
    appended below an existing one would silently replace the reviewed list. A duplicate key is
    therefore a parse error.

    PYYAML IS PREFERRED, THE STDLIB SUBSET READER IS THE FALLBACK. `policy_yaml` parses the part of
    YAML a policy uses and refuses everything else with its own code, so a payload that shipped
    without PyYAML answers a real verdict for a real policy instead of refusing every session with
    `policy_reader_unavailable`. The fallback is NOT a permissive path: it refuses duplicate keys by
    the same rule, and it cannot construct an arbitrary object at all (no tags, no aliases), so the
    protections this docstring names hold for both readers.
    """
    yaml = _pyyaml()
    if yaml is None:
        return _yaml_load_without_pyyaml(raw, source)

    class _Loader(yaml.SafeLoader):
        pass

    def _construct_mapping(loader: Any, node: Any, deep: bool = False) -> Any:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise PolicyDocumentError(
                    "policy document has an unhashable mapping key",
                    code="policy_unparseable",
                    details={"source": source},
                ) from exc
            if key in mapping:
                raise PolicyDocumentError(
                    f"duplicate key {key!r} in the policy document",
                    code="policy_duplicate_key",
                    details={"source": source, "key": str(key)},
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _Loader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
    )
    try:
        return yaml.load(raw, Loader=_Loader)  # noqa: S506 - SafeLoader subclass, no constructors
    except PolicyDocumentError:
        raise
    except yaml.YAMLError as exc:
        raise PolicyDocumentError(
            f"policy document is not valid YAML: {exc.__class__.__name__}",
            code="policy_unparseable",
            details={"source": source, "error": str(exc).splitlines()[0][:400]},
        ) from exc


def _pyyaml() -> Any:
    """The `yaml` module, or None when this interpreter cannot PROVIDE it.

    Imported on demand rather than at module scope so importing `codna.policy_document` never depends
    on an optional wheel — which is what made one absent package take down every session. A payload
    that ships it gets the full parser; one that does not gets `policy_yaml`.

    `ImportError` AND NOT ONLY `ModuleNotFoundError`, because "absent" is one way a wheel can be
    unusable and it is not the only one. A `yaml/_yaml` C extension built for a different interpreter
    raises `ImportError` from inside `yaml/__init__.py` (a linker or ABI failure, not a missing file),
    which is NOT a `ModuleNotFoundError`. Catching the narrow class propagated that exception out of
    `_yaml_load` and skipped the fallback this function exists to reach — so a corrupt PyYAML was
    strictly worse than no PyYAML at all. The fallback reader is a real substitute here (it reaches
    the same verdicts on the fixture matrix), and it fails CLOSED, so degrading is the safe direction.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    return yaml


def _yaml_load_without_pyyaml(raw: str, source: str) -> Any:
    """Parse with the stdlib subset reader, mapping each refusal to its own stable code.

    THE THREE OUTCOMES ARE DISTINCT ON PURPOSE. A malformed document is `policy_unparseable` (the
    same verdict PyYAML reaches); a repeated key is `policy_duplicate_key` (the same fail-open
    protection, by name); and VALID YAML the subset does not implement is
    `policy_reader_subset_exceeded` — a refusal that names the construct, because calling valid YAML
    "unparseable" would send an operator to fix a file that is not broken.
    """
    try:
        return policy_yaml.loads(raw)
    except policy_yaml.DuplicateKeyError as exc:
        raise PolicyDocumentError(
            f"duplicate key {exc.key!r} in the policy document",
            code="policy_duplicate_key",
            details={"source": source, "key": str(exc.key), "line": exc.lineno},
        ) from exc
    except policy_yaml.UnsupportedYamlError as exc:
        raise PolicyDocumentError(
            f"the policy document uses {exc.message}, which the built-in reader does not support",
            code="policy_reader_subset_exceeded",
            details={
                "source": source,
                "line": exc.lineno,
                "hint": "install PyYAML (pip install pyyaml) to read this document, or rewrite it "
                "with plain mappings, block sequences and scalars",
            },
        ) from exc
    except policy_yaml.YamlSyntaxError as exc:
        raise PolicyDocumentError(
            f"policy document is not valid YAML: {exc.message}",
            code="policy_unparseable",
            details={"source": source, "error": str(exc)[:400], "line": exc.lineno},
        ) from exc


# ── schema validation ──────────────────────────────────────────────────────────────────────────


def _fail(code: str, message: str, at: str, **details: Any) -> PolicyDocumentError:
    return PolicyDocumentError(message, code=code, path=at, details=details)


def _check_depth(value: Any, at: str, depth: int = 0, seen: frozenset[int] = frozenset()) -> None:
    """Bound nesting, and refuse a CYCLE — a YAML alias can point back at its own ancestor.

    Depth alone does not stop `a: &a {b: *a}`: the walker would descend forever (or until the
    recursion limit raised something that is not a PolicyDocumentError). Tracking the containers on
    the current PATH is what makes it terminate, and a cycle is refused rather than truncated: a
    policy that contains itself is malformed, and guessing which copy to honour is not a decision a
    validator may make.
    """
    if depth > MAX_DOCUMENT_DEPTH:
        raise _fail(
            "policy_invalid_shape",
            f"policy document nests deeper than {MAX_DOCUMENT_DEPTH} levels",
            at,
        )
    if not isinstance(value, (Mapping, list, tuple)):
        return
    identity = id(value)
    if identity in seen:
        raise _fail(
            "policy_invalid_shape",
            f"policy document contains a cycle at {at or 'document'}",
            at,
        )
    nested_seen = seen | {identity}
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_depth(item, f"{at}.{key}" if at else str(key), depth + 1, nested_seen)
    else:
        for index, item in enumerate(value):
            _check_depth(item, f"{at}[{index}]", depth + 1, nested_seen)


def _require_mapping(value: Any, at: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(
            "policy_invalid_type",
            f"{at or 'document'} must be a mapping, got {type(value).__name__}",
            at,
            expected="mapping",
            received=type(value).__name__,
        )
    for key in value:
        if not isinstance(key, str):
            raise _fail(
                "policy_invalid_type",
                f"{at or 'document'} has a non-string key ({type(key).__name__})",
                at,
                expected="string key",
                received=type(key).__name__,
            )
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed: frozenset[str], at: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise _fail(
            "policy_unknown_key",
            f"unknown key(s) {', '.join(unknown)} at {at or 'document'}; "
            f"known keys are {', '.join(sorted(allowed))}",
            at,
            unknown=unknown,
            known=sorted(allowed),
        )


def _string_list(value: Any, at: str, *, required: bool) -> tuple[str, ...]:
    if value is None:
        if required:
            raise _fail("policy_missing_field", f"{at} is required", at)
        return ()
    if not isinstance(value, list):
        raise _fail(
            "policy_invalid_type",
            f"{at} must be a list of strings, got {type(value).__name__}",
            at,
            expected="list",
            received=type(value).__name__,
        )
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise _fail(
                "policy_invalid_type",
                f"{at}[{index}] must be a string, got {type(item).__name__}",
                f"{at}[{index}]",
                expected="string",
                received=type(item).__name__,
            )
        entry = item.strip()
        if not entry:
            raise _fail("policy_invalid_value", f"{at}[{index}] must not be empty", f"{at}[{index}]")
        out.append(entry)
    return tuple(out)


def _tool_names(value: Any, at: str, *, required: bool) -> tuple[str, ...]:
    names = _string_list(value, at, required=required)
    for index, name in enumerate(names):
        if not _TOOL_NAME.match(name):
            raise _fail(
                "policy_invalid_value",
                f"{at}[{index}] ({name!r}) is not a tool name: must match [a-z][a-z0-9_]*",
                f"{at}[{index}]",
                value=name,
            )
    # De-duplicate, preserving the declared order so the digest of the parsed document is stable.
    return tuple(dict.fromkeys(names))


def _write_paths(value: Any, at: str, *, required: bool) -> tuple[str, ...]:
    paths = _string_list(value, at, required=required)
    for index, entry in enumerate(paths):
        where = f"{at}[{index}]"
        if entry.startswith(("/", "~", "\\")) or re.match(r"^[A-Za-z]:", entry):
            raise _fail(
                "policy_invalid_value",
                f"{where} ({entry!r}) must be repo-relative, not absolute",
                where,
                value=entry,
            )
        if ".." in Path(entry).parts:
            raise _fail(
                "policy_invalid_value",
                f"{where} ({entry!r}) must not escape the repository with '..'",
                where,
                value=entry,
            )
        if "\x00" in entry:
            raise _fail("policy_invalid_value", f"{where} contains a NUL byte", where)
    return tuple(dict.fromkeys(paths))


def _command_names(value: Any, at: str, *, required: bool) -> tuple[str, ...]:
    names = _string_list(value, at, required=required)
    for index, name in enumerate(names):
        where = f"{at}[{index}]"
        # Mirror of `server/sandbox.ts::validateCommandAllowList`: a SEPARATOR means the author
        # allowed a path, and allow-listing a path is how a shell escapes the check.
        if "/" in name or "\\" in name or os.sep in name:
            raise _fail(
                "policy_invalid_value",
                f"{where} ({name!r}) must be a bare program name, not a path",
                where,
                value=name,
            )
        if not _COMMAND_NAME.match(name):
            raise _fail(
                "policy_invalid_value",
                f"{where} ({name!r}) is not a program name: must match [A-Za-z0-9][A-Za-z0-9._+-]*",
                where,
                value=name,
            )
    return tuple(dict.fromkeys(names))


def _finite_number(value: Any, at: str, *, integer: bool = False) -> float | int:
    # `bool` is an int subclass; a policy that says `max_cost_usd: true` is a mistake, not a 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(
            "policy_invalid_type",
            f"{at} must be a number, got {type(value).__name__}",
            at,
            expected="integer" if integer else "number",
            received=type(value).__name__,
        )
    number = float(value)
    if not math.isfinite(number):
        raise _fail("policy_invalid_value", f"{at} must be finite", at, value=str(value))
    if number < 0:
        raise _fail("policy_invalid_value", f"{at} must not be negative", at, value=number)
    if integer:
        if int(number) != number:
            raise _fail(
                "policy_invalid_value", f"{at} must be a whole number", at, value=number
            )
        if number <= 0:
            raise _fail("policy_invalid_value", f"{at} must be positive", at, value=number)
        return int(number)
    return number


def _budget(value: Any, at: str) -> tuple[float | None, float | None, int | None]:
    section = _require_mapping(value, at)
    _reject_unknown_keys(section, BUDGET_KEYS, at)
    if not section:
        raise _fail(
            "policy_missing_field",
            f"{at} is present but declares no cap; remove the section or name one",
            at,
            known=sorted(BUDGET_KEYS),
        )
    cost = section.get("max_cost_usd")
    turn = section.get("max_turn_cost_usd")
    tokens = section.get("max_tokens")
    return (
        _finite_number(cost, f"{at}.max_cost_usd") if cost is not None else None,
        _finite_number(turn, f"{at}.max_turn_cost_usd") if turn is not None else None,
        _finite_number(tokens, f"{at}.max_tokens", integer=True) if tokens is not None else None,
    )


def _approval_mode(value: Any) -> str:
    """Validate the terminal-completion mode, defaulting every absent section to human."""
    if value is None:
        return "human"
    section = _require_mapping(value, "approval")
    _reject_unknown_keys(section, APPROVAL_KEYS, "approval")
    if "mode" not in section:
        raise _fail("policy_missing_field", "approval.mode is required", "approval.mode")
    mode = section["mode"]
    if not isinstance(mode, str):
        raise _fail(
            "policy_invalid_type", "approval.mode must be a string", "approval.mode",
            expected="string", received=type(mode).__name__,
        )
    if mode not in APPROVAL_MODES:
        raise _fail(
            "policy_invalid_value",
            f"approval.mode must be one of {', '.join(sorted(APPROVAL_MODES))}",
            "approval.mode", value=mode,
        )
    return mode


def _parse_permissions(document: Mapping[str, Any]) -> tuple[int, RuntimePermissions]:
    """Validate a parsed document and return its normalized permissions.

    Raises `PolicyDocumentError` for every malformed case. An EMPTY document is not a parse error:
    it is returned as empty permissions, and the caller turns that into `policy_denies_all`.
    """
    if "version" in document:
        raw_version = document["version"]
        version = _finite_number(raw_version, "version", integer=True)
        if version not in _SUPPORTED_VERSIONS:
            raise _fail(
                "policy_unsupported_version",
                f"unsupported policy version {version}; this build reads {sorted(_SUPPORTED_VERSIONS)}",
                "version",
                value=version,
                supported=sorted(_SUPPORTED_VERSIONS),
            )
    else:
        version = 1

    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    if "tools" in document:
        tools = _require_mapping(document["tools"], "tools")
        _reject_unknown_keys(tools, TOOLS_KEYS, "tools")
        if "allow" not in tools:
            raise _fail(
                "policy_missing_field",
                "tools.allow is required when the `tools` section is present",
                "tools.allow",
            )
        allowed_tools = _tool_names(tools["allow"], "tools.allow", required=True)
        denied_tools = _tool_names(tools.get("deny"), "tools.deny", required=False)

    allowed_paths: tuple[str, ...] = ()
    denied_paths: tuple[str, ...] = ()
    if "write_scope" in document:
        scope = _require_mapping(document["write_scope"], "write_scope")
        _reject_unknown_keys(scope, WRITE_SCOPE_KEYS, "write_scope")
        if "paths" not in scope:
            raise _fail(
                "policy_missing_field",
                "write_scope.paths is required when the `write_scope` section is present",
                "write_scope.paths",
            )
        allowed_paths = _write_paths(scope["paths"], "write_scope.paths", required=True)
        denied_paths = _write_paths(scope.get("deny"), "write_scope.deny", required=False)

    commands: tuple[str, ...] = ()
    if "command_allow_list" in document:
        section = _require_mapping(document["command_allow_list"], "command_allow_list")
        _reject_unknown_keys(section, COMMAND_KEYS, "command_allow_list")
        if "commands" not in section:
            raise _fail(
                "policy_missing_field",
                "command_allow_list.commands is required when the section is present",
                "command_allow_list.commands",
            )
        commands = _command_names(section["commands"], "command_allow_list.commands", required=True)

    max_cost: float | None = None
    max_turn: float | None = None
    max_tokens: int | None = None
    if "budget" in document:
        max_cost, max_turn, max_tokens = _budget(document["budget"], "budget")

    approvals: tuple[str, ...] = ()
    if "required_approvals" in document:
        approvals = _tool_names(
            document["required_approvals"], "required_approvals", required=True
        )

    return version, RuntimePermissions(
        allowed_tools=allowed_tools,
        denied_tools=denied_tools,
        allowed_write_paths=allowed_paths,
        denied_write_paths=denied_paths,
        allowed_commands=commands,
        max_cost_usd=max_cost,
        max_turn_cost_usd=max_turn,
        max_tokens=max_tokens,
        required_approvals=approvals,
        approval_mode=_approval_mode(document.get("approval")),
    )


def _grants_nothing(permissions: RuntimePermissions) -> bool:
    """True when the document permits nothing AT ALL, in any category.

    A budget cap or a write scope with no tool is not a runnable policy: the agent could not read a
    file or call a model. `policy_verdict` reports that as `policy_denies_all_tools` — a distinct
    code, because the fix is different from "the document is empty" (add a tool, not add a section).
    """
    return not (
        permissions.effective_tools()
        or permissions.allowed_write_paths
        or permissions.allowed_commands
        or permissions.max_cost_usd is not None
        or permissions.max_turn_cost_usd is not None
        or permissions.max_tokens is not None
    )


# ── the public entry points ────────────────────────────────────────────────────────────────────


def policy_path_for(working_dir: str | os.PathLike[str]) -> Path:
    """`<working_dir>/.codna/policy.yaml` — the only place a policy is read from."""
    return Path(working_dir) / DEFAULT_POLICY_RELPATH


def load_policy_document(
    path: str | os.PathLike[str],
) -> tuple[int, RuntimePermissions, str]:
    """Read and validate a policy document. Returns `(version, permissions, digest)`.

    Raises `PolicyDocumentError` — the caller must treat that as a DENY. A missing file raises
    `policy_missing`; use `policy_verdict()` when absence may be permitted.
    """
    target = Path(path)
    try:
        raw_bytes = target.read_bytes()
    except FileNotFoundError as exc:
        raise PolicyDocumentError(
            f"no policy document at {target}",
            code="policy_missing",
            details={"source": str(target)},
        ) from exc
    except OSError as exc:
        raise PolicyDocumentError(
            f"cannot read the policy document at {target}: {exc.__class__.__name__}",
            code="policy_unreadable",
            details={"source": str(target), "error": str(exc)},
        ) from exc

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyDocumentError(
            f"policy document at {target} is not valid UTF-8",
            code="policy_unparseable",
            details={"source": str(target)},
        ) from exc

    parsed = _yaml_load(text, str(target))
    if parsed is None:
        # An empty file. Deliberately NOT an error here: `policy_verdict` reports it as
        # `policy_denies_all`, which is the fail-closed reading the track asks for.
        parsed = {}
    document = _require_mapping(parsed, "")
    _check_depth(document, "")
    _reject_unknown_keys(document, TOP_LEVEL_KEYS, "")
    version, permissions = _parse_permissions(document)
    return version, permissions, digest_of(document)


def policy_verdict(
    path: str | os.PathLike[str] | None,
    *,
    required: bool = True,
) -> PolicyVerdict:
    """The verdict both consumers act on. Never raises for a bad document — it DENIES instead.

    ``required=True`` (the durable/worker path) makes a missing file a denial. ``required=False``
    reports `mode="absent"`, which is the ONLY outcome that lets a caller keep the pre-existing
    approval profile — and even then `allowed` is False, so a caller that ignores `mode` fails
    closed rather than open.
    """
    target = Path(path) if path is not None else None
    if target is None:
        if required:
            return PolicyVerdict(
                mode="policy",
                allowed=False,
                code="policy_missing",
                reason="a policy is required for this run but no path was supplied",
                details={"source": None},
            )
        return PolicyVerdict(
            mode="absent",
            allowed=False,
            code="policy_absent_not_required",
            reason="no policy document was supplied and none was required",
            details={"source": None},
        )

    source = str(target)
    if not target.exists():
        if required:
            return PolicyVerdict(
                mode="policy",
                allowed=False,
                code="policy_missing",
                reason=f"no policy document at {source}",
                source=source,
                details={"source": source},
            )
        return PolicyVerdict(
            mode="absent",
            allowed=False,
            code="policy_absent_not_required",
            reason=f"no policy document at {source}, and none was required",
            source=source,
            details={"source": source},
        )

    try:
        version, permissions, digest = load_policy_document(target)
    except PolicyDocumentError as exc:
        return PolicyVerdict(
            mode="policy",
            allowed=False,
            code=exc.code,
            reason=str(exc),
            source=source,
            details=dict(exc.details),
        )

    if _grants_nothing(permissions):
        return PolicyVerdict(
            mode="policy",
            allowed=False,
            code="policy_denies_all",
            reason=(
                "the policy document grants no permission (no tools, write paths, commands or "
                "budget); an empty policy is a policy that permits nothing"
            ),
            source=source,
            policy_digest=digest,
            version=version,
            details={"source": source},
        )

    if not permissions.effective_tools():
        # A policy that permits a write path or a budget but NO TOOL is not runnable: the agent could
        # not read a file or call a model. Reported separately from `policy_denies_all` because the
        # fix is different — name a tool, do not add a section.
        return PolicyVerdict(
            mode="policy",
            allowed=False,
            code="policy_denies_all_tools",
            reason=(
                "the policy document permits no tool, so no session can run under it; "
                "name at least one tool under tools.allow"
            ),
            source=source,
            policy_digest=digest,
            version=version,
            details={
                "source": source,
                "declared_allowed_tools": list(permissions.allowed_tools),
                "denied_tools": list(permissions.denied_tools),
            },
        )

    return PolicyVerdict(
        mode="policy",
        allowed=True,
        code="policy_ok",
        reason=f"policy v{version} validated",
        source=source,
        policy_digest=digest,
        version=version,
        permissions=permissions,
        details={"source": source},
    )


# ── `python3 -m codna.policy verify` — the sidecar's only door into this module ────────────────

_USAGE = "usage: python -m codna.policy verify --file <path> [--required|--optional]"


def main(argv: list[str] | None = None) -> int:
    """Emit the verdict as ONE JSON object on stdout. Exit 0 whenever a verdict was produced.

    Exit 2 means no verdict exists (bad usage, internal failure); a caller treats that as a DENY
    because it cannot distinguish "denied" from "could not evaluate" — and failing closed is the
    whole point. The verdict itself, including a denial, is a successful evaluation (exit 0) so the
    caller can report the policy's own code instead of a generic spawn failure.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        sys.stderr.write(_USAGE + "\n")
        return 2
    if args[0] != "verify":
        sys.stderr.write(f"unknown command {args[0]!r}\n{_USAGE}\n")
        return 2
    rest = args[1:]
    target: str | None = None
    required = True
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--file" and index + 1 < len(rest):
            target = rest[index + 1]
            index += 2
            continue
        if token in {"--required", "--optional"}:
            required = token == "--required"
            index += 1
            continue
        sys.stderr.write(f"unknown argument {token!r}\n{_USAGE}\n")
        return 2
    try:
        verdict = policy_verdict(target, required=required)
    except Exception as exc:  # noqa: BLE001 - never exit 0 without a verdict
        sys.stdout.write(
            json.dumps(
                {
                    "mode": "policy",
                    "allowed": False,
                    "code": "policy_evaluation_failed",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "details": {},
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        return 2
    sys.stdout.write(verdict.to_json() + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by both suites
    raise SystemExit(main())
