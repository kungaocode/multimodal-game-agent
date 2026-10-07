"""任务层：把感知、决策、校验、执行、验证串成完整闭环。"""

from .agent_loop import AgentLoop, AgentPorts, RingOutcome
from .collector import CollectorTask, CollectorTaskResult
from .donation import DonationTask
from .resource import ResourceTask, StepRecord, TaskResult

__all__ = [
    "AgentLoop",
    "AgentPorts",
    "RingOutcome",
    "CollectorTask",
    "CollectorTaskResult",
    "DonationTask",
    "ResourceTask",
    "StepRecord",
    "TaskResult",
]
