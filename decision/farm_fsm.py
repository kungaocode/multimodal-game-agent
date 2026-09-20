"""打资源有限状态机（理想状态转移，用户确认版）。

    村庄待机 ── 检查自己资源
      ├─ 全部已满 → 停止
      └─ 有资源未满，且识别到「进攻按钮」 → 点进攻 → 战斗中
    战斗中   ── 识别到「返回按钮」→（战斗结束界面）→ 战斗结束
    战斗结束 ── 点返回 → 村庄待机（循环）

FSM 只负责「看到什么 → 做什么动作」的决策，不直接点击屏幕。
动作由外部执行器（Android 输入）执行，perception 信号由检测器 + OCR 生成。

扩展点：resource_policy 决定打哪些敌方资源建筑。
默认策略跳过自己已满的资源（如圣水满了就不打圣水采集器）。
用户后续要的「圣水满了就跳过圣水采集器、直接退出战斗」就是改这个策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from state.game_state import ResourceStatus

# 敌方资源建筑名 → 对应的己方资源类型（决定满了要不要跳过）
_BUILDING_RESOURCE = {
    "金矿": "gold",
    "储金罐": "gold",
    "圣水收集器": "elixir",
    "圣水瓶": "elixir",
    "暗黑重油钻井": "dark",
    "暗黑重油罐": "dark",
}


class FarmState(str, Enum):
    VILLAGE = "村庄待机"  # 检查自己资源；未满则准备进攻
    BATTLE = "战斗中"  # 下兵收割敌方资源
    BATTLE_OVER = "战斗结束"  # 点返回回村
    STOP = "停止"  # 所有资源已满，终态


@dataclass
class FarmSignals:
    """一次感知得到的信号：由检测器 + OCR 生成，喂给 FSM。"""

    attack_button: tuple[int, int] | None = None  # 识别到「进攻按钮」的中心点
    return_button: tuple[int, int] | None = None  # 识别到「返回按钮」的中心点
    enemy_resources: list[tuple[str, tuple[int, int]]] = field(default_factory=list)  # (建筑名, 中心点)
    resources: ResourceStatus = field(default_factory=ResourceStatus)  # 自己资源是否满


@dataclass
class FarmAction:
    """FSM 给出的动作。kind: tap(点击) / deploy(下兵) / wait(等待) / stop(结束)。"""

    kind: str
    target: str | None = None  # tap/deploy 的对象名，如「进攻按钮」「金矿」
    coords: tuple[int, int] | None = None  # 点击坐标（图像像素）

    def __repr__(self) -> str:  # 简洁日志
        if self.coords:
            return f"FarmAction({self.kind}, {self.target}, {self.coords})"
        return f"FarmAction({self.kind}, {self.target})"


def default_policy(signals: FarmSignals) -> list[tuple[str, tuple[int, int]]]:
    """默认打资源策略：跳过自己已满的那类资源建筑。

    例如圣水满了 → 圣水收集器/圣水瓶不出现在目标里，只打金和黑油。
    这是用户说的「后面再集成」的额外功能，先按默认策略实现。
    """
    full: set[str] = set()
    if signals.resources.gold_full:
        full.update(["金矿", "储金罐"])
    if signals.resources.elixir_full:
        full.update(["圣水收集器", "圣水瓶"])
    if signals.resources.dark_full:
        full.update(["暗黑重油钻井", "暗黑重油罐"])
    return [(name, coords) for name, coords in signals.enemy_resources if name not in full]


class FarmFSM:
    """有限状态机：输入感知信号，输出动作。

    用法（由主循环调用）：
        fsm = FarmFSM()
        while True:
            signals = perceive(screenshot)      # 检测器 + OCR
            action = fsm.step(signals)          # 决策
            if action.kind == "stop": break
            execute(action)                     # 执行器点击/下兵
            sleep(等待下一帧)
    """

    def __init__(self, policy=None) -> None:
        self.state = FarmState.VILLAGE
        self.policy = policy or default_policy

    def step(self, signals: FarmSignals) -> FarmAction:
        if self.state is FarmState.VILLAGE:
            # 满就停
            if signals.resources.all_full:
                self.state = FarmState.STOP
                return FarmAction("stop", "所有资源已满")
            # 有资源未满 → 点进攻
            if signals.attack_button is not None:
                self.state = FarmState.BATTLE
                return FarmAction("tap", "进攻按钮", signals.attack_button)
            return FarmAction("wait", "等待识别到进攻按钮")

        if self.state is FarmState.BATTLE:
            # 战斗结束（返回按钮出现）→ 等战斗结束界面，先不下兵
            if signals.return_button is not None:
                self.state = FarmState.BATTLE_OVER
                return FarmAction("wait", "战斗结束，准备点返回")
            # 否则对敌方资源建筑下兵
            targets = self.policy(signals)
            if targets:
                name, coords = targets[0]
                return FarmAction("deploy", name, coords)
            return FarmAction("wait", "等待识别敌方资源建筑")

        if self.state is FarmState.BATTLE_OVER:
            # 点返回回村
            self.state = FarmState.VILLAGE
            return FarmAction("tap", "返回按钮", signals.return_button or (0, 0))

        # STOP 是终态
        return FarmAction("stop", "已停止")
