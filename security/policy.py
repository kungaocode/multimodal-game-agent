"""Permission and policy engine."""

from typing import Any


class PolicyEngine:
    """Enforce runtime permission and safety policies."""

    def __init__(self, permissions: dict[str, list[str]] | None = None) -> None:
        self.permissions = permissions or {}

    def is_allowed(self, role: str, action: str, target: str | None = None) -> bool:
        allowed = self.permissions.get(role, [])
        if "*" in allowed:
            return True
        return action in allowed

    def require_confirmation(self, action: dict[str, Any]) -> bool:
        """Return True if the action must be confirmed by a human."""
        return action.get("action") in {"donate_troops", "upgrade_building"}
