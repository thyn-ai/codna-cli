"""Provider endpoint validation shared by the CLI and durable contracts."""
from __future__ import annotations

import ipaddress

__all__ = ["is_loopback_host"]


def is_loopback_host(hostname: str) -> bool:
    """Recognize loopback names and addresses deterministically, without DNS."""
    host = hostname.strip().lower().strip("[]")
    if host == "localhost" or host.endswith(".localhost") or host == "::1":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
