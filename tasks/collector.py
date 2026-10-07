"""村庄采集环闭环任务：感知 → FSM → 校验 → 执行 → 验证。

采集器与打资源环在仲裁层互斥，且采集优先：因为采集成本为零，先把免费资源
收掉，剩下的缺口再交给打资源环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.collector_fsm import CollectorAction, CollectorFSM
from executor.collector_simulator import CollectorSimulator
from executor.validator import ActionValidator


@dataclass
class CollectorStepRecord:
    step: int
    fsm_state: str
    action: CollectorAction
    verified: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "fsm_state": self.fsm_state,
            "action": repr(self.action),
            "verified": self.verified,
            "note": self.note,
        }


@dataclass
class CollectorTaskResult:
    status: str
    reason: str
    steps: int
    collected: dict[str, int] = field(default_factory=dict)
    records: list[CollectorStepRecord] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "steps": self.steps,
            "collected": dict(self.collected),
            "records": [r.to_dict() for r in self.records],
        }


class CollectorTask:
    def __init__(
        self,
        fsm: CollectorFSM | None = None,
        validator: ActionValidator | None = None,
    ) -> None:
        self.fsm = fsm if fsm is not None else CollectorFSM()
        self.validator = validator if validator is not None else ActionValidator()

    def run(
        self,
        simulator: CollectorSimulator,
        max_steps: int = 40,
    ) -> CollectorTaskResult:
        self.fsm.reset()
        records: list[CollectorStepRecord] = []
        for step_no in range(1, max_steps + 1):
            signals = simulator.perceive()
            action = self.fsm.step(signals)
            check = self.validator.validate(action)
            if not check.allowed:
                return CollectorTaskResult(
                    "ERROR",
                    f"动作被校验器拦截: {check.reason}",
                    step_no,
                    simulator.collected,
                    records,
                )
            result = simulator.step(action)
            verified = self._verify(action, result)
            records.append(
                CollectorStepRecord(
                    step_no,
                    self.fsm.state.value,
                    action,
                    verified,
                    str(result.info.get("reason", "")),
                )
            )
            if not verified:
                return CollectorTaskResult(
                    "ERROR",
                    f"第 {step_no} 步执行无效: {result.info.get('reason')}",
                    step_no,
                    simulator.collected,
                    records,
                )
            if action.kind == "stop":
                if simulator.done:
                    return CollectorTaskResult(
                        "SUCCESS", "采集完成", step_no, simulator.collected, records
                    )
                return CollectorTaskResult(
                    "STOP", "无采集目标", step_no, simulator.collected, records
                )
            if not simulator.pending:
                return CollectorTaskResult(
                    "SUCCESS", "采集完成", step_no, simulator.collected, records
                )
        return CollectorTaskResult(
            "MAX_STEPS", f"超过最大步数 {max_steps}", max_steps, simulator.collected, records
        )

    @staticmethod
    def _verify(action: CollectorAction, result: Any) -> bool:
        if action.kind == "tap":
            return bool(result.info.get("collected", 0) > 0)
        return True
