"""自建模拟环境（阶段 8）：打资源循环的确定性仿真。

Simulator 模拟三个屏幕：村庄待机 / 战斗中 / 战斗结束，对每个动作只做确定性状态转移，
不依赖任何模型或网络：

    tap 进攻按钮   村庄待机 → 战斗中（补满兵力）
    deploy 资源点   战斗中：扣除资源点数量并产出 loot 奖励；目标耗尽或兵力用尽 → 战斗结束
    tap 返回按钮   战斗结束/战斗中 → 村庄待机，并进入下一个村庄
    wait / stop    不改变环境

perceive() 输出与 FarmFSM 兼容的 FarmSignals —— 即「检测器 + OCR」的仿真结果，
用于在无模型环境下闭环验证「感知 → 决策 → 校验 → 执行 → 验证」整条链路。

已满资源跳过规则与 decision.resource_policy 保持一致：己方某类资源已满时，
对应敌方资源建筑视为不可抢目标（不出现在 perception 中，也不计入战斗结束条件）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from state.game_state import ResourceStatus, Resources

from decision.farm_fsm import FarmSignals
from decision.resource_policy import RESOURCE_TYPE_BY_BUILDING, full_resource_names

_OWN_RESOURCE_FIELD: dict[str, str] = {
    "gold": "gold",
    "elixir": "elixir",
    "dark": "dark_elixir",
}


@dataclass
class ResourceSite:
    """敌方一个资源建筑：类别、坐标、剩余/初始可抢资源量。"""

    name: str
    coords: tuple[int, int]
    amount: int = 0
    capacity: int = 0
    # 落点可接近半径：真实下兵只会落在草地/可下兵点，通常不会正好是建筑中心。
    radius: float = 160.0


@dataclass
class DefenseSite:
    """敌方一个防御建筑（只关心坐标，用于安全性打分）。"""

    coords: tuple[int, int]


@dataclass
class SimVillage:
    """一个敌方村庄布局。"""

    resources: list[ResourceSite] = field(default_factory=list)
    defenses: list[DefenseSite] = field(default_factory=list)
    # 直线（墙段起点, 墙段终点）表示的城墙，供落点→建筑中心的穿越估算。
    wall_segments: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)
    name: str = "村庄"


@dataclass(frozen=True)
class SimulationResult:
    """一次 step 的结果：奖励（抢到的资源）、是否结束、附加信息。"""

    action_kind: str
    reward: float
    done: bool
    info: dict[str, Any]


class Simulator:
    SCREEN_VILLAGE = "village"
    SCREEN_BATTLE = "battle"
    SCREEN_BATTLE_OVER = "battle_over"

    def __init__(
        self,
        villages: list[SimVillage],
        image_size: tuple[int, int] = (1280, 720),
        troops_per_battle: int = 30,
        loot_per_deploy: int = 1000,
        optimal_landing_px: float = 120.0,
        max_landing_px: float = 260.0,
        walk_px_per_extra_troop: float = 180.0,
        wall_breaker_cost: int = 3,
        attack_button: tuple[int, int] | None = None,
        return_button: tuple[int, int] | None = None,
        own_resources: ResourceStatus | None = None,
    ) -> None:
        self.villages = list(villages)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.troops_per_battle = troops_per_battle
        self.loot_per_deploy = loot_per_deploy
        self.optimal_landing_px = float(optimal_landing_px)
        self.max_landing_px = float(max_landing_px)
        self.walk_px_per_extra_troop = float(walk_px_per_extra_troop)
        self.wall_breaker_cost = int(wall_breaker_cost)
        w, h = self.image_size
        self.attack_button = attack_button or (w - 160, h - 80)
        self.return_button = return_button or (w - 120, h - 40)
        self.own_resources = own_resources or ResourceStatus()
        self.target_resource_types: tuple[str, ...] | None = None
        self._initial_amounts = Resources(
            gold=self.own_resources.amounts.gold,
            elixir=self.own_resources.amounts.elixir,
            dark_elixir=self.own_resources.amounts.dark_elixir,
        )
        self.reset()

    # ------------------------------------------------------------------ 状态
    def reset(self) -> None:
        """把环境重置到初始村庄与满状态的资源点（可反复跑多个实验）。"""
        self.screen = self.SCREEN_VILLAGE
        self.village_index = 0
        self.troops_left = self.troops_per_battle
        self.done = False
        self.total_loot = 0.0
        self.own_amounts = Resources(
            gold=self._initial_amounts.gold,
            elixir=self._initial_amounts.elixir,
            dark_elixir=self._initial_amounts.dark_elixir,
        )
        for village in self.villages:
            for site in village.resources:
                site.amount = site.capacity

    def set_target_resource_types(self, target_types: tuple[str, ...] | None) -> None:
        """限定本次打资源环只把指定资源类型视为可抢目标。

        由 ResourceTask 从 FarmObjective 注入；不传或传 None 表示不限制。
        """
        self.target_resource_types = tuple(target_types) if target_types else None

    @property
    def current_village(self) -> SimVillage | None:
        if 0 <= self.village_index < len(self.villages):
            return self.villages[self.village_index]
        return None

    def _lootable_sites(self, village: SimVillage | None) -> list[ResourceSite]:
        """当前己方资源状态下「值得抢」的资源点（跳过己方已满的那类）。"""
        if village is None:
            return []
        skip = full_resource_names(self.own_resources)
        sites = [s for s in village.resources if s.amount > 0 and s.name not in skip]
        if self.target_resource_types is not None:
            sites = [
                s
                for s in sites
                if _resource_type_of(s.name) in self.target_resource_types
            ]
        return sites

    def _battle_over(self, village: SimVillage | None) -> bool:
        return self.troops_left <= 0 or not self._lootable_sites(village)

    # ----------------------------------------------------------------- 感知
    def perceive(self) -> FarmSignals:
        """输出一轮感知信号（模拟检测器 + OCR 的结果）。"""
        if self.screen in (self.SCREEN_BATTLE, self.SCREEN_BATTLE_OVER):
            sites = self._lootable_sites(self.current_village)
            return FarmSignals(
                enemy_resources=[(s.name, s.coords) for s in sites],
                resources=self.own_resources,
                return_button=self.return_button if self._battle_over(self.current_village) else None,
            )
        attack = self.attack_button if (self.current_village is not None and not self.done) else None
        return FarmSignals(resources=self.own_resources, attack_button=attack)

    # ----------------------------------------------------------------- 执行
    def step(self, action: Any) -> SimulationResult:
        kind = getattr(action, "kind", None)
        target = getattr(action, "target", None)
        coords = getattr(action, "coords", None)
        if kind == "deploy":
            return self._deploy(target, coords)
        if kind == "tap":
            return self._tap(target, coords)
        if kind == "stop":
            self.done = True
            return SimulationResult("stop", 0.0, True, {"reason": "任务停止", "screen_changed": False})
        return SimulationResult(kind or "wait", 0.0, False, {"reason": "等待", "screen_changed": False})

    def _tap(self, target: str | None, coords: tuple[int, int] | None) -> SimulationResult:
        if (
            target == "进攻按钮"
            and self.screen == self.SCREEN_VILLAGE
            and coords == self.attack_button
        ):
            self.screen = self.SCREEN_BATTLE
            self.troops_left = self.troops_per_battle
            return SimulationResult(
                "tap", 0.0, False, {"reason": "进入战斗", "screen_changed": True}
            )
        if (
            target == "返回按钮"
            and self.screen in (self.SCREEN_BATTLE, self.SCREEN_BATTLE_OVER)
            and coords == self.return_button
        ):
            self.village_index += 1
            self.screen = self.SCREEN_VILLAGE
            if self.current_village is None:
                self.done = True
            return SimulationResult(
                "tap", 0.0, self.done, {"reason": "返回村庄", "screen_changed": True}
            )
        return SimulationResult("tap", 0.0, False, {"reason": "无效点击", "screen_changed": False})

    def _deploy(self, target: str | None, coords: tuple[int, int] | None) -> SimulationResult:
        village = self.current_village
        if self.screen != self.SCREEN_BATTLE or village is None:
            return SimulationResult(
                "deploy", 0.0, False, {"reason": "不在战斗中，无法下兵", "screen_changed": False}
            )
        if self.troops_left <= 0:
            return SimulationResult(
                "deploy", 0.0, False, {"reason": "兵力用尽", "screen_changed": False}
            )
        if coords is None:
            return SimulationResult(
                "deploy", 0.0, False, {"reason": "缺少落点", "screen_changed": False}
            )
        site = min(
            (
                s
                for s in village.resources
                if s.name == target and s.amount > 0
                and math.dist(coords, s.coords) <= max(s.radius, self.max_landing_px)
            ),
            key=lambda s: math.dist(coords, s.coords),
            default=None,
        )
        if site is None:
            return SimulationResult(
                "deploy",
                0.0,
                False,
                {"reason": "目标不存在、已抢空或落点过远", "screen_changed": False},
            )

        # 落点不同 → 步行损耗与收益系数不同；城墙穿越按估算器同一算法计算额外兵力。
        from decision.loot_estimator import wall_crossing_cost

        dist = math.dist(coords, site.coords)
        if dist > self.max_landing_px:
            return SimulationResult(
                "deploy",
                0.0,
                False,
                {"reason": f"落点距建筑 {dist:.0f}px 过远", "screen_changed": False},
            )
        wall_cost = wall_crossing_cost(
            coords,
            site.coords,
            wall_segments=getattr(village, "wall_segments", []),
            wall_breaker_cost=self.wall_breaker_cost,
        )
        walk_extra = 0
        if dist > self.optimal_landing_px and self.walk_px_per_extra_troop > 0:
            walk_extra = int(
                math.ceil((dist - self.optimal_landing_px) / self.walk_px_per_extra_troop)
            )
        troop_cost = 1 + wall_cost + walk_extra
        if self.troops_left < troop_cost:
            return SimulationResult(
                "deploy",
                0.0,
                False,
                {"reason": f"兵力不足支付城墙/步行损耗（需 {troop_cost}）", "screen_changed": False},
            )

        loot_factor = 1.0
        if dist > self.optimal_landing_px and self.max_landing_px > self.optimal_landing_px:
            span = self.max_landing_px - self.optimal_landing_px
            falloff = min(1.0, (dist - self.optimal_landing_px) / span)
            loot_factor = max(0.35, 1.0 - falloff * 0.8)
        loot = min(site.amount, int(self.loot_per_deploy * loot_factor))
        site.amount -= loot
        self.troops_left -= troop_cost
        self.total_loot += loot
        resource_type = RESOURCE_TYPE_BY_BUILDING.get(target)
        own_field = _OWN_RESOURCE_FIELD.get(resource_type) if resource_type else None
        if own_field is not None:
            current = getattr(self.own_amounts, own_field, 0)
            setattr(self.own_amounts, own_field, current + loot)
        if self._battle_over(village):
            self.screen = self.SCREEN_BATTLE_OVER
        return SimulationResult(
            "deploy",
            float(loot),
            False,
            {
                "looted": loot,
                "remaining": site.amount,
                "landing_dist": round(dist, 1),
                "loot_factor": round(loot_factor, 3),
                "wall_cost": wall_cost,
                "walk_extra": walk_extra,
                "troop_cost": troop_cost,
                "reason": "收割资源",
                "screen_changed": False,
            },
        )


def default_farm_world() -> list[SimVillage]:
    """两个固定布局的敌方村庄，容量均为 loot_per_deploy 的整数倍，便于确定性测试。"""
    return [
        SimVillage(
            name="敌方村庄 A",
            resources=[
                ResourceSite("金矿", (300, 300), capacity=3000),
                ResourceSite("圣水收集器", (900, 300), capacity=3000),
                ResourceSite("储金罐", (600, 500), capacity=5000),
                ResourceSite("暗黑重油罐", (120, 600), capacity=2000),
            ],
            defenses=[DefenseSite((500, 400)), DefenseSite((1000, 200))],
        ),
        SimVillage(
            name="敌方村庄 B",
            resources=[
                ResourceSite("圣水瓶", (400, 400), capacity=4000),
                ResourceSite("金矿", (800, 400), capacity=3000),
                ResourceSite("暗黑重油钻井", (600, 200), capacity=2000),
            ],
            defenses=[DefenseSite((300, 500)), DefenseSite((900, 600))],
        ),
    ]


def _resource_type_of(name: str) -> str | None:
    resource_type = RESOURCE_TYPE_BY_BUILDING.get(name)
    if resource_type == "dark":
        return "dark_elixir"
    return resource_type
