"""村庄采集环有限状态机。

本地 YOLO 只负责报告有哪些可收图标（金/圣水/黑油采集器上方的资源气泡），
FSM 负责确定性点击：一次感知一个图标，点击后等待下一帧信号确认已收集。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CollectorState(str, Enum):
    VILLAGE = "村庄-采集检查"
    COLLECTING = "村庄-采集中"
    DONE = "采集完成"


COLLECTOR_TARGET_BY_RESOURCE: dict[str, str] = {
    "gold": "金币采集图标",
    "elixir": "圣水采集图标",
    "dark_elixir": "黑油采集图标",
}

RESOURCE_TYPE_TO_COLLECTOR = {v: k for k, v in COLLECTOR_TARGET_BY_RESOURCE.items()}


@dataclass
class CollectorSignals:
    """一次村庄感知：可收图标列表 (resource_type, icon_coords)。"""

    collectibles: list[tuple[str, tuple[int, int]]] = field(default_factory=list)


@dataclass
class CollectorAction:
    kind: str  # tap / wait / stop
    target: str | None = None
    coords: tuple[int, int] | None = None


class CollectorFSM:
    """采集器 FSM：有目标就点，图标消失即完成本批。"""

    def __init__(self) -> None:
        self.state = CollectorState.VILLAGE
        self.history: list[str] = [self.state.value]

    def step(self, signals: CollectorSignals) -> CollectorAction:
        if not signals.collectibles:
            self.state = CollectorState.DONE
            return CollectorAction("stop", "无采集目标")
        if self.state is CollectorState.DONE:
            self.state = CollectorState.COLLECTING
        else:
            self.state = CollectorState.COLLECTING
        resource_type, coords = signals.collectibles[0]
        return CollectorAction(
            "tap",
            COLLECTOR_TARGET_BY_RESOURCE.get(resource_type, "金币采集图标"),
            coords,
        )

    def reset(self) -> None:
        self.state = CollectorState.VILLAGE
        self.history = [self.state.value]
