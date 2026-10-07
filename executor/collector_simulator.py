"""村庄采集环确定性模拟器。

模拟「本地 YOLO 看到可收采集器图标 → 点击图标 → 顶栏资源增加 → 图标消失」的
闭环。不依赖截图或云端模型，感知输出与 decision.collector_fsm.CollectorSignals
兼容，用于验证采集环和资源阈值联动。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from decision.collector_fsm import (
    COLLECTOR_TARGET_BY_RESOURCE,
    CollectorSignals,
)
from executor.simulator import SimulationResult
from state.game_state import Resources


@dataclass
class CollectorSite:
    resource_type: str  # gold / elixir / dark_elixir
    icon_coords: tuple[int, int]
    amount: int = 1000


class CollectorSimulator:
    def __init__(
        self,
        sites: list[CollectorSite] | None = None,
        own_resources: Resources | None = None,
        image_size: tuple[int, int] = (1280, 720),
    ) -> None:
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._initial_resources = Resources(
            gold=int(own_resources.gold if own_resources else 0),
            elixir=int(own_resources.elixir if own_resources else 0),
            dark_elixir=int(own_resources.dark_elixir if own_resources else 0),
        )
        self._sites = [CollectorSite(s.resource_type, tuple(s.icon_coords), int(s.amount)) for s in (sites or [])]
        self.reset()

    def reset(self) -> None:
        self.pending: list[CollectorSite] = [
            CollectorSite(s.resource_type, s.icon_coords, s.amount) for s in self._sites
        ]
        self.own_resources = Resources(
            gold=self._initial_resources.gold,
            elixir=self._initial_resources.elixir,
            dark_elixir=self._initial_resources.dark_elixir,
        )
        self.collected: dict[str, int] = {}
        self.done = not self.pending

    def perceive(self) -> CollectorSignals:
        return CollectorSignals(
            collectibles=[(s.resource_type, s.icon_coords) for s in self.pending]
        )

    def step(self, action: Any) -> SimulationResult:
        kind = getattr(action, "kind", None)
        target = getattr(action, "target", None)
        coords = getattr(action, "coords", None)
        if kind == "tap" and coords is not None:
            site = next(
                (
                    s
                    for s in self.pending
                    if COLLECTOR_TARGET_BY_RESOURCE.get(s.resource_type) == target
                    and math.dist(s.icon_coords, tuple(coords)) <= 12.0
                ),
                None,
            )
            if site is None:
                return SimulationResult(
                    "tap",
                    0.0,
                    self.done,
                    {"reason": "无效采集点击", "screen_changed": False},
                )
            self.pending.remove(site)
            current = getattr(self.own_resources, site.resource_type, 0)
            setattr(self.own_resources, site.resource_type, current + site.amount)
            self.collected[site.resource_type] = (
                self.collected.get(site.resource_type, 0) + site.amount
            )
            self.done = not self.pending
            return SimulationResult(
                "tap",
                float(site.amount),
                self.done,
                {
                    "collected": site.amount,
                    "resource_type": site.resource_type,
                    "reason": "采集成功",
                    "screen_changed": False,
                },
            )
        if kind == "stop":
            self.done = True
            return SimulationResult("stop", 0.0, True, {"reason": "采集结束", "screen_changed": False})
        return SimulationResult(
            kind or "wait",
            0.0,
            self.done,
            {"reason": "等待识别采集图标", "screen_changed": False},
        )

    def summary(self) -> dict[str, Any]:
        return {
            "collected": dict(self.collected),
            "resources": {
                "gold": self.own_resources.gold,
                "elixir": self.own_resources.elixir,
                "dark_elixir": self.own_resources.dark_elixir,
            },
            "done": self.done,
        }
