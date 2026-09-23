"""`codna status` — concise health (engine · telys · keys), the human-friendly counterpart to the
verbose ``doctor`` JSON. Every probe degrades gracefully: it never spawns the engine and never raises.
"""
from __future__ import annotations


def build_status_lines(keys, *, api_key_present: bool) -> list[str]:
    from . import byok_cli

    lines = ["codna status:"]
    try:
        from .supervisor import read_state

        lines.append(f"  engine        : {'running' if read_state() else 'stopped'}")
    except Exception:  # noqa: BLE001
        lines.append("  engine        : unknown")
    try:
        from . import memory as _mem

        kern = _mem._resolve_telys_kernel(configure_env=False)
        if kern.get("found"):
            lines.append(f"  telys runtime : ready ({kern.get('source', '?')})")
        else:
            lines.append("  telys runtime : not installed — run `codna login`")
        try:
            _tok, src = _mem._configured_license_token()
            lines.append(f"  telys license : {src or 'none'}")
        except Exception:  # noqa: BLE001
            lines.append("  telys license : misconfigured")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  telys runtime : unknown ({type(exc).__name__})")
    lines.append(f"  engine key    : {'configured' if api_key_present else 'missing (set CODNA_API_KEY)'}")
    lines.append(
        "  provider key  : "
        + ("configured" if byok_cli.provider_key_present(keys) else "missing — run `codna key set <provider>`")
    )
    return lines
