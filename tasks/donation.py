"""捐兵任务（阶段 10）：在确定性模拟环境中闭环执行捐兵流程。

DonationTask 把感知/FSM/校验/执行串成完整闭环：

    perceive（模拟消息列表 + 增援弹窗 + 兵种选择）
        → DonationFSM 决策
        → ActionValidator 校验（确保 tap 始终带白名单内目标与坐标）
        → DonationSimulator 执行
        → verify 前后状态验证
        → 记录日志

结束条件：
    SUCCESS  全部请求已捐完并返回村庄；
    ERROR    某个动作被校验器拦截或执行无效；
    MAX_STEPS 超过最大步数（防死循环兜底）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.donation_fsm import DonationAction, DonationFSM, DonationState
from executor.donation_simulator import DonationSimulator
from executor.validator import ActionValidator


@dataclass
class DonationStepRecord:
    """单步捐兵闭环记录。"""

    step: int
    fsm_state: str
    action: DonationAction
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
class DonationTaskResult:
    """一次完整捐兵任务的总结。"""

    status: str
    reason: str
    steps: int
    donated: int
    records: list[DonationStepRecord] = field(default_factory=list)
    blocked_request: dict[Any, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "steps": self.steps,
            "donated": self.donated,
            "records": [r.to_dict() for r in self.records],
            "blocked_request": self.blocked_request,
        }


class DonationTask:
    """捐兵闭环任务：默认使用 FSM + 完整白名单校验器。"""

    def __init__(
        self,
        fsm: DonationFSM | None = None,
        validator: ActionValidator | None = None,
    ) -> None:
        self.fsm = fsm if fsm is not None else DonationFSM()
        self.validator = validator if validator is not None else ActionValidator()

    def run(
        self,
        simulator: DonationSimulator,
        max_steps: int = 120,
        reset: bool = True,
    ) -> DonationTaskResult:
        if reset:
            self.fsm.reset()
        records: list[DonationStepRecord] = []
        for step_no in range(1, max_steps + 1):
            signals = simulator.perceive()
            action = self.fsm.step(signals)

            check = self.validator.validate(action)
            if not check.allowed:
                return DonationTaskResult(
                    "ERROR",
                    f"动作被校验器拦截: {check.reason}",
                    step_no,
                    simulator.donated,
                    records,
                )

            result = simulator.step(action)
            verified = self._verify(action, result)
            records.append(
                DonationStepRecord(
                    step_no,
                    self.fsm.state.value,
                    action,
                    verified,
                    str(result.info.get("reason", "")),
                )
            )
            if not verified:
                return DonationTaskResult(
                    "ERROR",
                    f"第 {step_no} 步执行无效: {result.info.get('reason')}",
                    step_no,
                    simulator.donated,
                    records,
                )
            if self.fsm.state is DonationState.BLOCKED:
                return DonationTaskResult(
                    "BLOCKED",
                    "捐兵资源不足阻塞，等待采集/打资源补充",
                    step_no,
                    simulator.donated,
                    records,
                    dict(self.fsm.blocked_request or {}),
                )
            if action.kind == "stop":
                return DonationTaskResult(
                    "STOP", "捐兵流程停止", step_no, simulator.donated, records
                )
            if simulator.done and simulator.screen.value == "village":
                return DonationTaskResult(
                    "SUCCESS", "全部请求已捐完", step_no, simulator.donated, records
                )

        return DonationTaskResult(
            "MAX_STEPS",
            f"超过最大步数 {max_steps}",
            max_steps,
            simulator.donated,
            records,
        )

    @staticmethod
    def _verify(action: DonationAction, result: Any) -> bool:
        if action.kind == "tap":
            # 兵种选择不切屏但必须真的选中；其余 tap 必须产生屏幕/流程转移。
            note = str(result.info.get("reason", ""))
            return note == "选中兵种" or bool(result.info.get("screen_changed", False))
        return True
