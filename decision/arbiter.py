"""三环仲裁器：采集环、打资源环、捐兵环。

采集环与打资源环互斥，采集器优先：村庄一帧先看有没有可收图标，有就只进采集环；
没有采集目标且资源低于低保线/捐兵被阻塞，才进打资源环；两者都不触发时进捐兵环。

仲裁器不调用 API，也不做模型判断。它只消费已有感知数值（资源、采集图标、
捐兵阻塞信号），输出具有确定性的 RingDecision。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from decision.farm_fsm import FarmObjective
from decision.resource_gate import ResourceGate
from state.game_state import ResourceThresholds, Resources


class Ring(str, Enum):
    IDLE = "idle"
    COLLECT = "collect"
    FARM = "farm"
    DONATE = "donate"


@dataclass(frozen=True)
class VillageSnapshot:
    """村庄态快照：由感知层提供，仲裁器只读。"""

    resources: Resources = field(default_factory=Resources)
    collectibles: tuple[tuple[str, tuple[int, int]], ...] = ()
    donation_blocked: dict[str, Any] | None = None


@dataclass(frozen=True)
class RingDecision:
    ring: Ring
    reason: str
    farm_objective: FarmObjective | None = None
    blocked_request: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ring": self.ring.value,
            "reason": self.reason,
            "farm_objective": self.farm_objective.requirements if self.farm_objective else None,
            "blocked_request": dict(self.blocked_request or {}),
        }


class AgentArbiter:
    def __init__(
        self,
        thresholds: ResourceThresholds | None = None,
        gate: ResourceGate | None = None,
    ) -> None:
        self.thresholds = thresholds or ResourceThresholds()
        self.gate = gate or ResourceGate(self.thresholds)

    def decide(self, snapshot: VillageSnapshot) -> RingDecision:
        if snapshot.collectibles:
            return RingDecision(Ring.COLLECT, "存在可收采集器图标")

        if snapshot.donation_blocked:
            resource_type = str(
                snapshot.donation_blocked.get("resource_type", "elixir")
            )
            required = int(
                snapshot.donation_blocked.get("required")
                or snapshot.donation_blocked.get("cost")
                or 0
            )
            available = getattr(snapshot.resources, resource_type, 0)
            if required > 0 and available >= required:
                return RingDecision(Ring.DONATE, "捐兵阻塞需求已补足")
            objective = self._blocked_objective(snapshot.donation_blocked)
            return RingDecision(
                Ring.FARM,
                "捐兵需求阻塞，先补足资源",
                farm_objective=objective,
                blocked_request=dict(snapshot.donation_blocked),
            )

        gate_decision = self.gate.evaluate(snapshot.resources)
        if gate_decision.needs_farm:
            requirements = self.gate.low_requirements(gate_decision)
            return RingDecision(
                Ring.FARM,
                gate_decision.reason,
                farm_objective=FarmObjective(
                    requirements=requirements,
                    target_types=tuple(gate_decision.low_types),
                ),
            )

        return RingDecision(Ring.DONATE, "资源充足，进入捐兵环")

    @staticmethod
    def _blocked_objective(blocked: dict[str, Any]) -> FarmObjective:
        return blocked_farm_objective(blocked)


def blocked_farm_objective(blocked: dict[str, Any] | None) -> FarmObjective:
    """把捐兵阻塞缺口转成打资源环目标（resource_type -> 缺失量）。"""
    if not blocked:
        return FarmObjective()
    resource_type = str(blocked.get("resource_type", "elixir"))
    missing = int(
        blocked.get("missing")
        or blocked.get("required")
        or blocked.get("cost")
        or 1
    )
    return FarmObjective(
        requirements={resource_type: max(1, missing)},
        target_types=(resource_type,),
    )
