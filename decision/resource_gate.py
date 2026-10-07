"""资源阈值判定（三环仲裁的数据入口）。

原则：模型只负责从画面读数（顶栏 OCR / 视觉模型），是否低于低保线、
是否回升、捐兵需求缺多少，全部由这里的确定性配置和规则决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from state.game_state import ResourceThresholds, Resources


@dataclass(frozen=True)
class ResourceGateDecision:
    """一次资源判定结果。"""

    needs_farm: bool
    low_types: tuple[str, ...] = ()
    missing: dict[str, int] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "needs_farm": self.needs_farm,
            "low_types": list(self.low_types),
            "missing": dict(self.missing),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DonationFeasibility:
    """捐兵需求是否可满足。"""

    enough: bool
    resource_type: str
    available: int
    required: int
    missing: int = 0


_THRESHOLD_FIELDS: dict[str, tuple[str, str]] = {
    "gold": ("gold_low", "gold_resume"),
    "elixir": ("elixir_low", "elixir_resume"),
    "dark_elixir": ("dark_low", "dark_resume"),
}


class ResourceGate:
    """带滞回的阈值判定：首次低于 low 触发，只有回升到 resume 才解除。

    这样避免资源在临界值附近抖动时反复切换打资源/捐兵环。
    """

    def __init__(self, thresholds: ResourceThresholds | None = None) -> None:
        self.thresholds = thresholds or ResourceThresholds()
        self._low_flags: dict[str, bool] = {key: False for key in _THRESHOLD_FIELDS}

    def reset(self) -> None:
        self._low_flags = {key: False for key in _THRESHOLD_FIELDS}

    def evaluate(self, resources: Resources) -> ResourceGateDecision:
        low_types: list[str] = []
        missing: dict[str, int] = {}
        reasons: list[str] = []
        for key, (low_field, resume_field) in _THRESHOLD_FIELDS.items():
            value = getattr(resources, key)
            low = getattr(self.thresholds, low_field)
            resume = getattr(self.thresholds, resume_field)
            active = self._low_flags[key] or value <= low
            recovered = value >= resume
            self._low_flags[key] = active and not recovered
            if self._low_flags[key]:
                low_types.append(key)
                missing[key] = max(0, resume - value)
                reasons.append(f"{key}={value} 未回升到 {resume}")
        return ResourceGateDecision(
            needs_farm=bool(low_types),
            low_types=tuple(low_types),
            missing=missing,
            reason="; ".join(reasons) or "资源充足",
        )

    def donation_feasibility(
        self,
        cost: int,
        resource_type: str,
        resources: Resources,
    ) -> DonationFeasibility:
        required = max(0, int(cost)) + self.thresholds.donation_safety_buffer
        available = getattr(resources, resource_type, 0)
        missing = max(0, required - available)
        return DonationFeasibility(
            enough=available >= required,
            resource_type=resource_type,
            available=available,
            required=required,
            missing=missing,
        )

    def low_requirements(self, decision: ResourceGateDecision) -> dict[str, int]:
        """把低资源类型转成打资源环的回升目标。"""
        return {
            key: max(1, getattr(self.thresholds, _THRESHOLD_FIELDS[key][1]))
            for key in decision.low_types
        }
