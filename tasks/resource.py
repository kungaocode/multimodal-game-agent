"""打资源任务（阶段 3/7）：在模拟环境中闭环执行。

ResourceTask 把已有模块串成完整闭环：

    perceive（模拟检测器+OCR）
        → FarmFSM 决策
        → ActionValidator 校验
        → Simulator 执行
        → verify 前后状态验证
        → 记录日志

结束条件：
    STOP     己方资源已满；
    SUCCESS  所有村庄已搜刮完毕；
    ERROR    某个动作被校验器拦截；
    MAX_STEPS 超过最大步数（防止死循环兜底）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.farm_fsm import FarmAction, FarmFSM
from decision.resource_policy import ResourcePolicy
from executor.simulator import Simulator
from executor.validator import ActionValidator


@dataclass
class StepRecord:
    """单步闭环记录：策略状态、动作、奖励、验证结果。"""

    step: int
    fsm_state: str
    action: FarmAction
    reward: float
    verified: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "fsm_state": self.fsm_state,
            "action": repr(self.action),
            "reward": self.reward,
            "verified": self.verified,
            "note": self.note,
        }


@dataclass
class TaskResult:
    """一次完整打资源任务的总结。"""

    status: str
    reason: str
    steps: int
    total_loot: float
    records: list[StepRecord] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "steps": self.steps,
            "total_loot": self.total_loot,
            "records": [r.to_dict() for r in self.records],
        }


class ResourceTask:
    """打资源闭环任务，默认使用打分策略 + 白名单校验器。"""

    def __init__(
        self,
        fsm: FarmFSM | None = None,
        policy: ResourcePolicy | None = None,
        validator: ActionValidator | None = None,
    ) -> None:
        self.fsm = fsm
        self.policy = policy if policy is not None else ResourcePolicy()
        self.validator = validator if validator is not None else ActionValidator()

    def run(self, simulator: Simulator, max_steps: int = 300) -> TaskResult:
        """在给定模拟环境上跑完一次完整打资源任务。

        默认 FSM 在首次 run 时构建，并把模拟环境当前村庄的防御坐标动态喂给打分策略。
        """
        if self.fsm is None:
            policy = self.policy
            if isinstance(policy, ResourcePolicy):
                policy = ResourcePolicy(
                    config=policy.config,
                    defense_provider=lambda signals, sim=simulator: (
                        [d.coords for d in sim.current_village.defenses]
                        if sim.current_village is not None
                        else []
                    ),
                )
            self.fsm = FarmFSM(policy=policy)
        else:
            self.fsm.state = type(self.fsm.state).VILLAGE
            self.fsm.history = [self.fsm.state.value]

        records: list[StepRecord] = []
        total_loot = 0.0
        for step_no in range(1, max_steps + 1):
            signals = simulator.perceive()
            action = self.fsm.step(signals)

            check = self.validator.validate(action)
            if not check.allowed:
                return TaskResult(
                    "ERROR", f"动作被校验器拦截: {check.reason}", step_no, total_loot, records
                )

            result = simulator.step(action)
            verified = self._verify(action, result)
            total_loot += result.reward
            records.append(
                StepRecord(
                    step_no,
                    self.fsm.state.value,
                    action,
                    result.reward,
                    verified,
                    str(result.info.get("reason", "")),
                )
            )

            if action.kind == "stop":
                return TaskResult("STOP", "己方资源已满，任务停止", step_no, total_loot, records)
            if action.kind == "wait" and simulator.done:
                return TaskResult("SUCCESS", "所有村庄已搜刮完毕", step_no, total_loot, records)

        return TaskResult("MAX_STEPS", f"超过最大步数 {max_steps}", max_steps, total_loot, records)

    @staticmethod
    def _verify(action: FarmAction, result: Any) -> bool:
        """前后状态验证：执行之后必须确认产生了预期效果。"""
        if action.kind == "deploy":
            return bool(result.info.get("looted", 0) > 0)
        if action.kind == "tap":
            return bool(result.info.get("screen_changed", False))
        return True  # wait / stop 只改变决策侧状态，无需验证
