"""Verification manifest — the reproducible, digest-bound plan that replaces raw shell
flags for `--open-pr`.

A manifest (`codna-security.yaml`) pins the scanner image + rule pack, the build/test
commands, the sandbox, and the autofix policy. The whole document is bound by one digest
so baseline and patched phases provably run the *same* configuration — a mutation to any
field (accepted exit codes, sandbox limits, …) changes `digest` and the gate aborts
(MANIFEST-TAMPER). `--open-pr` requires a *resolved* manifest: image pinned by `@sha256`
and network egress denied.

The pure core works on a Python dict (`from_dict`) so it is testable without a YAML dep;
`load_manifest` adds a thin file loader (YAML if available, else JSON).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .findings import digest_of
from .policy import Policy

_SHA256_PIN = re.compile(r"@sha256:[0-9a-f]{64}$")
_NETWORK_OK_FOR_PR = {"deny", "none"}


class ManifestError(Exception):
    pass


def _require_string(value: object, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ManifestError(f"{field} must be a non-empty string")


def _command_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{field} must be a non-empty command array")
    command: list[str] = []
    for index, part in enumerate(value):
        if not isinstance(part, str) or not part:
            raise ManifestError(f"{field}[{index}] must be a non-empty string")
        command.append(part)
    return command


def _command_matrix(value: object, field: str) -> list[list[str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ManifestError(f"{field} must be a list of command arrays")
    return [_command_list(command, f"{field}[{index}]") for index, command in enumerate(value)]


def _exit_codes(value: object, field: str) -> tuple[int, ...]:
    if value is None:
        return (0,)
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{field} must be a non-empty list of integer exit codes")
    codes: list[int] = []
    for index, code in enumerate(value):
        if not isinstance(code, int) or isinstance(code, bool):
            raise ManifestError(f"{field}[{index}] must be an integer exit code")
        codes.append(code)
    return tuple(codes)


@dataclass
class ScannerSpec:
    id: str
    image: str
    command: list[str]
    output: str
    accepted_exit_codes: tuple[int, ...] = (0,)
    rules_digest: str | None = None
    configuration_digest: str | None = None

    @property
    def image_pinned(self) -> bool:
        return bool(_SHA256_PIN.search(self.image or ""))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image": self.image,
            "command": list(self.command),
            "output": self.output,
            "accepted_exit_codes": list(self.accepted_exit_codes),
            "rules_digest": self.rules_digest,
            "configuration_digest": self.configuration_digest,
        }


@dataclass
class Sandbox:
    network: str = "deny"
    timeout_seconds: int = 1800
    cpu_limit: int = 4
    memory_mb: int = 8192

    def to_dict(self) -> dict:
        return {
            "network": self.network,
            "timeout_seconds": self.timeout_seconds,
            "cpu_limit": self.cpu_limit,
            "memory_mb": self.memory_mb,
        }


@dataclass
class VerificationManifest:
    scanner: ScannerSpec
    build: list[list[str]]
    tests: list[list[str]]
    sandbox: Sandbox
    policy: Policy
    digest: str = ""

    def __post_init__(self):
        if not self.digest:
            self.digest = self.compute_digest()

    def compute_digest(self) -> str:
        # Whole-manifest digest — ANY field change (exit codes, sandbox limits, command)
        # flips this, so baseline vs patched config drift is detected (MANIFEST-TAMPER).
        return digest_of(
            {
                "scanner": self.scanner.to_dict(),
                "build": [list(c) for c in self.build],
                "tests": [list(c) for c in self.tests],
                "sandbox": self.sandbox.to_dict(),
                "policy_digest": self.policy.digest,
            }
        )

    @classmethod
    def from_dict(cls, d: dict) -> "VerificationManifest":
        if not isinstance(d, dict):
            raise ManifestError("manifest must be a mapping")
        sc = d.get("scanner") or {}
        if not sc.get("id") or not sc.get("command") or not sc.get("output"):
            raise ManifestError("scanner.{id,command,output} are required")
        scanner = ScannerSpec(
            id=_require_string(sc["id"], "scanner.id"),
            image=sc.get("image", ""),
            command=_command_list(sc["command"], "scanner.command"),
            output=_require_string(sc["output"], "scanner.output"),
            accepted_exit_codes=_exit_codes(sc.get("accepted_exit_codes", [0]), "scanner.accepted_exit_codes"),
            rules_digest=sc.get("rules_digest"),
            configuration_digest=sc.get("configuration_digest"),
        )
        ver = d.get("verification") or {}
        sb = d.get("sandbox") or {}
        sandbox = Sandbox(
            network=str(sb.get("network", "deny")).lower(),
            timeout_seconds=int(sb.get("timeout_seconds", 1800)),
            cpu_limit=int(sb.get("cpu_limit", 4)),
            memory_mb=int(sb.get("memory_mb", 8192)),
        )
        return cls(
            scanner=scanner,
            build=_command_matrix(ver.get("build"), "verification.build"),
            tests=_command_matrix(ver.get("tests"), "verification.tests"),
            sandbox=sandbox,
            policy=Policy.from_dict(d.get("policy")),
        )

    def require_resolved_for_pr(self) -> None:
        """Gate precondition for `--open-pr`: the manifest must be fully pinned and the
        sandbox must deny egress. Raises `ManifestError` otherwise (PROV-02 / G1)."""
        problems = []
        if not self.scanner.image_pinned:
            problems.append(f"scanner.image is not pinned by @sha256 ({self.scanner.image!r})")
        if not self.scanner.rules_digest:
            problems.append("scanner.rules_digest is required (rule pack must be pinned)")
        if self.sandbox.network not in _NETWORK_OK_FOR_PR:
            problems.append(f"sandbox.network must deny egress for --open-pr (got {self.sandbox.network!r})")
        if not self.tests:
            problems.append("verification.tests must be specified for --open-pr")
        if problems:
            raise ManifestError("manifest not resolved for --open-pr: " + "; ".join(problems))


def load_manifest(path: "str | os.PathLike") -> VerificationManifest:
    """Load a manifest file. Uses PyYAML if installed, else falls back to JSON so the
    loader never hard-fails on a missing optional dependency."""
    with open(os.fspath(path), "rb") as fh:
        raw = fh.read()
    try:  # optional dependency
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
    except ModuleNotFoundError:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ManifestError(
                f"cannot parse manifest {os.fspath(path)!r}: PyYAML not installed and content is not JSON ({exc})"
            ) from exc
    return VerificationManifest.from_dict(data)
