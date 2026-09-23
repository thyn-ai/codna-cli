"""CLI entrypoint for Codna GitHub App webhook."""
from __future__ import annotations

import sys


def _die(message: str) -> None:
    print(f"codna: {message}", file=sys.stderr)
    raise SystemExit(1)


def cmd_webhook(args):
    """Run Codna's GitHub App webhook -- the App channel, owned in Codna -- or operate its queue."""
    action = getattr(args, "action_name", None) or getattr(args, "action", "serve") or "serve"
    if action == "ops":
        from .webhook_ops import cli_ops

        return cli_ops(args)
    if action == "migrate":
        from .webhook_ops import cli_migrate

        return cli_migrate(args)
    if action == "queue":
        from .webhook_ops import cli_queue

        return cli_queue(args)
    if action != "serve":
        _die(f"unknown `codna webhook` action: {action}")
    from .webhook import WebhookError, require_webhook_secret
    from .webhook_service import ROLE_ENV, serve

    import os

    host = getattr(args, "host", "0.0.0.0")
    port = int(getattr(args, "port", 8080))
    role = getattr(args, "role", None)
    if role:
        os.environ[ROLE_ENV] = role  # `--role` wins over the environment for this process
    role = role or os.environ.get(ROLE_ENV, "all")
    try:
        if role != "worker":  # a worker holds no webhook secret: deliveries arrive at the ingress
            require_webhook_secret()
        serve(host=host, port=port)
    except WebhookError as exc:
        _die(str(exc))
    except (RuntimeError, ValueError) as exc:  # a refused backend/role/schema: say why, exit non-zero
        _die(str(exc))
    except KeyboardInterrupt:
        return 0
    return 0
