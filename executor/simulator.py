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

from dataclasses import dataclass, field
from typing import Any

from state.game_state import ResourceStatus

from decision.farm_fsm import FarmSignals
from decision.resource_policy import full_resource_names


@dataclass
class ResourceSite:
    """敌方一个资源建筑：类别、坐标、剩余/初始可抢资源量。"""

    name: str
    coords: tuple[int, int]
    amount: int = 0
    capacity: int = 0


@dataclass
class DefenseSite:
    """敌方一个防御建筑（只关心坐标，用于安全性打分）。"""

    coords: tuple[int, int]


@dataclass
class SimVillage:
    """一个敌方村庄布局。"""

    resources: list[ResourceSite] = field(default_factory=list)
    defenses: list[DefenseSite] = field(default_factory=list)
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
        attack_button: tuple[int, int] | None = None,
        return_button: tuple[int, int] | None = None,
        own_resources: ResourceStatus | None = None,
    ) -> None:
        self.villages = list(villages)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.troops_per_battle = troops_per_battle
        self.loot_per_deploy = loot_per_deploy
        w, h = self.image_size
        self.attack_button = attack_button or (w - 160, h - 80)
        self.return_button = return_button or (w - 120, h - 40)
        self.own_resources = own_resources or ResourceStatus()
        self.reset()

    # ------------------------------------------------------------------ 状态
    def reset(self) -> None:
        """把环境重置到初始村庄与满状态的资源点（可反复跑多个实验）。"""
        self.screen = self.SCREEN_VILLAGE
        self.village_index = 0
        self.troops_left = self.troops_per_battle
        self.done = False
        self.total_loot = 0.0
        for village in self.villages:
            for site in village.resources:
                site.amount = site.capacity

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
        return [s for s in village.resources if s.amount > 0 and s.name not in skip]

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
        site = next(
            (
                s
                for s in village.resources
                if s.name == target and s.amount > 0 and s.coords == coords
            ),
            None,
        )
        if site is None:
            return SimulationResult(
                "deploy",
                0.0,
                False,
                {"reason": "目标不存在或已抢空", "screen_changed": False},
            )
        loot = min(site.amount, self.loot_per_deploy)
        site.amount -= loot
        self.troops_left -= 1
        self.total_loot += loot
        if self._battle_over(village):
            self.screen = self.SCREEN_BATTLE_OVER
        return SimulationResult(
            "deploy",
            float(loot),
            False,
            {
                "looted": loot,
                "remaining": site.amount,
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
