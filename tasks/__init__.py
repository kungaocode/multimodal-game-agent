"""任务层：把感知、决策、校验、执行、验证串成完整闭环。"""

from .resource import ResourceTask, StepRecord, TaskResult

__all__ = ["ResourceTask", "StepRecord", "TaskResult"]
