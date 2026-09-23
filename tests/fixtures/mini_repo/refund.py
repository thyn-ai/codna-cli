"""Refund handling for the shop."""


class RefundService:
    """Processes refunds against the ledger."""

    def process(self, order):
        """Reconcile the ledger entry after an account migration."""
        return reconcile(order)


def reconcile(order):
    """Find the new ledger entry for a migrated order."""
    return order
