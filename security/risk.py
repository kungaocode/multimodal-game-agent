"""Risk scoring engine for agent actions and inputs."""

from enum import Enum


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class RiskScorer:
    """Score risk of actions based on confidence, target and policy."""

    def score(self, action: dict, confidence: float, policy_flags: dict | None = None) -> dict:
        flags = policy_flags or {}
        score = 0.0
        reasons: list[str] = []
        if confidence < 0.6:
            score += 0.4
            reasons.append("Low confidence")
        if flags.get("requires_human_confirmation"):
            score += 0.3
            reasons.append("Requires human confirmation")
        if flags.get("sensitive_target"):
            score += 0.3
            reasons.append("Sensitive target")
        level = RiskLevel.INFO
        if score >= 0.8:
            level = RiskLevel.CRITICAL
        elif score >= 0.6:
            level = RiskLevel.HIGH
        elif score >= 0.3:
            level = RiskLevel.MEDIUM
        elif score > 0:
            level = RiskLevel.LOW
        return {"score": score, "level": level, "reasons": reasons}
