"""结构化动作定义（阶段 8）。

Agent 不直接点击屏幕，而是产出结构化 AgentAction，经 ActionValidator 校验后
由执行器（Simulator / 移动端）执行。这是安全层的第一道闸门。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 允许的动作种类
ALLOWED_ACTION_KINDS: tuple[str, ...] = ("tap", "deploy", "wait", "stop")

# 打资源循环允许的目标白名单：6 个资源建筑 + 2 个 UI 按钮
FARM_ALLOWED_TARGETS: tuple[str, ...] = (
    "圣水收集器",
    "金矿",
    "圣水瓶",
    "储金罐",
    "暗黑重油罐",
    "暗黑重油钻井",
    "进攻按钮",
    "返回按钮",
)


@dataclass(frozen=True)
class AgentAction:
    """一条结构化动作。kind: tap(点击) / deploy(下兵) / wait(等待) / stop(停止)。"""

    kind: str
    target: str | None = None
    coords: tuple[int, int] | None = None
    params: dict[str, Any] = field(default_factory=dict, compare=False)
    troop_coords: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "coords": self.coords,
            "params": self.params,
            "troop_coords": self.troop_coords,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentAction":
        coords = data.get("coords")
        if coords is not None:
            coords = (int(coords[0]), int(coords[1]))
        troop_coords = data.get("troop_coords")
        if troop_coords is not None:
            troop_coords = (int(troop_coords[0]), int(troop_coords[1]))
        return cls(
            kind=str(data["kind"]),
            target=data.get("target"),
            coords=coords,
            params=dict(data.get("params") or {}),
            troop_coords=troop_coords,
        )

    @classmethod
    def from_farm_action(cls, action: Any) -> "AgentAction":
        """由 FarmFSM 的 FarmAction 转换（不依赖 decision 包）。"""
        return cls(kind=action.kind, target=action.target, coords=action.coords)
