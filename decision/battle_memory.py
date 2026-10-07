"""战斗内目标记忆：估算每个资源建筑已经被抢掉多少、还剩多少。

感知每一帧都会给出一批资源建筑候选，但候选不代表还有可抢资源：
同一个储金罐在打空后可能在若干帧里仍被检测器/教练继续报出来。
BattleMemory 用「估算容量 - 估算已抢量」给每个目标维护剩余资源，
已经把估算所需兵力打满的目标视为已打空，供下兵规划和撤退判断使用。

容量优先取模拟器/detection 提供的精确值；没有时按建筑类型默认值估算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .loot_estimator import DEFAULT_CAPACITY_BY_TYPE, LootEstimate


@dataclass(frozen=True)
class MemoryConfig:
    """记忆配置：估算容量默认值、打空判定所需连续可见帧数。"""

    capacity_by_type: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_CAPACITY_BY_TYPE)
    )
    loot_per_troop: int = 1000
    min_seen_to_deplete: int = 2     # 至少连续/累计看见 N 帧才把它判成已打空
    min_remaining: int = 0           # 剩余小于等于该值视为已打空
    match_radius: float = 32.0


@dataclass
class SiteMemory:
    """单个资源建筑的战斗内状态。"""

    name: str
    coords: tuple[int, int]
    estimated_capacity: int
    loot_per_troop: int = 1000
    wall_cost: int = 0
    walk_extra: int = 0
    troops_needed_override: int | None = None
    last_estimate: LootEstimate | None = None
    troop_deploys: int = 0
    times_seen: int = 0
    last_seen_step: int = 0

    @property
    def troops_needed(self) -> int:
        if self.troops_needed_override is not None:
            return self.troops_needed_override
        if self.estimated_capacity <= 0:
            return 0
        return (
            int(math.ceil(self.estimated_capacity / max(1, self.loot_per_troop)))
            + self.wall_cost
            + self.walk_extra
        )

    @property
    def looted(self) -> int:
        if self.troop_deploys <= 0:
            return 0
        effective = max(0, self.troop_deploys - self.wall_cost - self.walk_extra)
        return min(self.estimated_capacity, effective * self.loot_per_troop)

    @property
    def remaining(self) -> int:
        return max(0, self.estimated_capacity - self.looted)

    def is_depleted(self, min_seen: int = 2, min_remaining: int = 0) -> bool:
        """已打空判断：看见足够多帧 且（打满估算兵力 或 剩余接近 0）。"""
        if self.times_seen < min_seen:
            return False
        if self.remaining <= min_remaining:
            return True
        return self.troop_deploys >= self.troops_needed

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coords": list(self.coords),
            "estimated_capacity": self.estimated_capacity,
            "looted": self.looted,
            "remaining": self.remaining,
            "troop_deploys": self.troop_deploys,
            "troops_needed": self.troops_needed,
            "wall_cost": self.wall_cost,
            "walk_extra": self.walk_extra,
            "times_seen": self.times_seen,
        }


class BattleMemory:
    """维护当前战斗所有资源目标的估算状态。"""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self.step: int = 0
        self.sites: dict[tuple[str, tuple[int, int]], SiteMemory] = {}

    def reset(self) -> None:
        self.step = 0
        self.sites.clear()

    def find(
        self,
        name: str,
        coords: tuple[int, int] | list[int],
    ) -> SiteMemory | None:
        for key, site in self.sites.items():
            if key[0] == name and math.dist(tuple(site.coords), tuple(coords)) <= self.config.match_radius:
                return site
        return None

    def observe(
        self,
        signals: Any,
        detections: list[dict] | None = None,
        step: int | None = None,
    ) -> None:
        """登记本帧可见的资源建筑；探测结果优先，其次 signal.building_info。"""
        self.step += 1
        current_step = self.step if step is None else int(step)
        building_info = getattr(signals, "building_info", None) or []
        info_by_coords: dict[tuple[int, int], dict] = {}
        for info in building_info:
            if not isinstance(info, dict):
                continue
            coords = info.get("coords")
            if coords:
                info_by_coords[tuple(int(v) for v in coords)] = info
        for name, coords in getattr(signals, "enemy_resources", []):
            coords = (int(coords[0]), int(coords[1]))
            capacity: int | None = None
            info = info_by_coords.get(coords)
            if info:
                cap = info.get("capacity")
                if cap is not None:
                    capacity = int(cap)
            det_capacity = None
            if detections:
                for det in detections:
                    det_coords = det.get("coords")
                    if not det_coords or det.get("type") != name:
                        continue
                    if math.dist(tuple(int(v) for v in det_coords), coords) > self.config.match_radius:
                        continue
                    cap = det.get("capacity")
                    if cap is not None:
                        det_capacity = int(cap)
                    break
            if capacity is None:
                capacity = det_capacity
            if capacity is None:
                capacity = int(self.config.capacity_by_type.get(name, 1000))
            site = self.find(name, coords)
            if site is None:
                self.sites[(name, coords)] = SiteMemory(
                    name,
                    coords,
                    estimated_capacity=capacity,
                    loot_per_troop=self.config.loot_per_troop,
                )
                site = self.sites[(name, coords)]
            site.estimated_capacity = max(site.estimated_capacity, capacity)
            site.times_seen += 1
            site.last_seen_step = current_step

    def mark_deploy(
        self,
        name: str,
        coords: tuple[int, int] | list[int],
        estimate: LootEstimate | None = None,
    ) -> SiteMemory:
        """记录一次真实下兵（或一次估算通过的下兵计划）。"""
        site = self.find(name, coords)
        if site is None:
            capacity = int(self.config.capacity_by_type.get(name, 1000))
            site = SiteMemory(
                name,
                tuple(int(v) for v in coords),
                estimated_capacity=capacity,
                loot_per_troop=self.config.loot_per_troop,
            )
            self.sites[(site.name, site.coords)] = site
        site.troop_deploys += 1
        if estimate is not None:
            site.last_estimate = estimate
            site.wall_cost = estimate.wall_cost
            site.walk_extra = estimate.walk_extra
            site.loot_per_troop = estimate.loot_per_troop
            site.troops_needed_override = estimate.troops_needed
        return site

    def capacity(self, name: str, coords: tuple[int, int]) -> int | None:
        site = self.find(name, coords)
        return site.estimated_capacity if site is not None else None

    def looted(self, name: str, coords: tuple[int, int]) -> int:
        site = self.find(name, coords)
        return site.looted if site is not None else 0

    def remaining(self, name: str, coords: tuple[int, int]) -> int | None:
        site = self.find(name, coords)
        return site.remaining if site is not None else None

    def is_depleted(self, name: str, coords: tuple[int, int]) -> bool:
        site = self.find(name, coords)
        if site is None:
            return False
        return site.is_depleted(
            min_seen=self.config.min_seen_to_deplete,
            min_remaining=self.config.min_remaining,
        )

    def summary(self) -> list[dict[str, Any]]:
        return [site.to_dict() for site in self.sites.values()]
