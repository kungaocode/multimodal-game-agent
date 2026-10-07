"""三环执行循环：把仲裁结果和本地/云端任务串起来。

真实 API（ADB/OCR/云端 Qwen-VL）由外部注入，这里只定义可测试的端口
（perceive/collect/farm/donate），保证三环互斥和采集优先的调度不依赖具体设备。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from decision.arbiter import AgentArbiter, Ring, VillageSnapshot, blocked_farm_objective
from decision.farm_fsm import FarmObjective


@dataclass(frozen=True)
class RingOutcome:
    ring: Ring
    status: str
    reason: str
    blocked_request: dict[str, Any] | None = None
    objective: FarmObjective | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ring": self.ring.value,
            "status": self.status,
            "reason": self.reason,
            "blocked_request": dict(self.blocked_request or {}),
            "objective": self.objective.requirements if self.objective else None,
        }


@dataclass(frozen=True)
class AgentPorts:
    perceive_village: Callable[[], VillageSnapshot]
    run_collect: Callable[[], RingOutcome]
    run_farm: Callable[[FarmObjective], RingOutcome]
    run_donate: Callable[[], RingOutcome]
    # 组合村庄环：一次调用完成「采集 + 捐兵检测」；返回 BLOCKED/LOW 时由
    # AgentLoop 立即调用 run_farm 补给，下一轮再回到村庄环。缺省使用旧三环。
    run_village: Callable[[], RingOutcome] | None = None


class AgentLoop:
    """在村庄态反复仲裁，直到任务出错、超时或外部显式停止。"""

    def __init__(
        self,
        ports: AgentPorts,
        arbiter: AgentArbiter | None = None,
        max_rounds: int = 50,
    ) -> None:
        self.ports = ports
        self.arbiter = arbiter or AgentArbiter()
        self.max_rounds = max_rounds

    def run(self, max_rounds: int | None = None) -> list[RingOutcome]:
        if self.ports.run_village is not None:
            return self.run_village_first(max_rounds)
        return self.run_legacy(max_rounds)

    def run_village_first(self, max_rounds: int | None = None) -> list[RingOutcome]:
        """村庄环优先：一轮先采集 + 捐兵检测，阻塞/低保才切战斗，打完自动回村。"""
        limit = max_rounds or self.max_rounds
        outcomes: list[RingOutcome] = []
        for _ in range(limit):
            outcome = self.ports.run_village()
            outcomes.append(outcome)
            if outcome.status in ("ERROR", "MAX_STEPS"):
                break
            if outcome.status in ("SUCCESS", "STOP"):
                break
            if outcome.status in ("BLOCKED", "LOW"):
                objective = outcome.objective or blocked_farm_objective(
                    outcome.blocked_request
                )
                farm = self.ports.run_farm(objective)
                outcomes.append(farm)
                if farm.status in ("ERROR", "MAX_STEPS"):
                    break
                continue
            break
        return outcomes

    def run_legacy(self, max_rounds: int | None = None) -> list[RingOutcome]:
        limit = max_rounds or self.max_rounds
        outcomes: list[RingOutcome] = []
        for _ in range(limit):
            snapshot = self.ports.perceive_village()
            decision = self.arbiter.decide(snapshot)
            if decision.ring is Ring.COLLECT:
                outcome = self.ports.run_collect()
            elif decision.ring is Ring.FARM:
                objective = decision.farm_objective or FarmObjective()
                outcome = self.ports.run_farm(objective)
            elif decision.ring is Ring.DONATE:
                outcome = self.ports.run_donate()
            else:
                break
            outcomes.append(outcome)
            if outcome.status in ("ERROR", "MAX_STEPS"):
                break
            if outcome.ring is Ring.DONATE and outcome.status in ("SUCCESS", "STOP"):
                # 本批捐兵需求已处理完，回到外层村庄循环等待下一次调度。
                break
            if outcome.ring in (Ring.COLLECT, Ring.FARM) and outcome.status == "SUCCESS":
                continue
        return outcomes
