"""Action validator: ensure every action is safe before execution."""

from typing import Any


class ActionValidator:
    """Validate structured actions against the current state and policy."""

    ALLOWED_ACTIONS: set[str] = {
        "select_building",
        "upgrade_building",
        "attack_target",
        "collect_resources",
        "open_castle",
        "donate_troops",
        "observe",
        "noop",
    }

    def __init__(self, allowed_actions: set[str] | None = None) -> None:
        self.allowed_actions = allowed_actions or self.ALLOWED_ACTIONS

    def validate(self, action: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return validation result with error list if invalid."""
        errors: list[str] = []
        name = action.get("action")
        if not name:
            errors.append("Missing 'action' field")
        elif name not in self.allowed_actions:
            errors.append(f"Action '{name}' is not in allowed set")
        target = action.get("target")
        if state is not None and target is not None:
            # Minimal existence check: state should expose allowed targets
            allowed_targets = state.get("allowed_targets", [])
            if allowed_targets and target not in allowed_targets:
                errors.append(f"Target '{target}' not present in current state")
        return {"valid": not errors, "errors": errors, "action": action}
