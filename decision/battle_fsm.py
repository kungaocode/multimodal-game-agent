"""打资源完整状态机（BattleFSM）：村庄待机 → 搜索对手 → 评估 → 战斗 → 结束 → 返回。

用户叙述的完整打资源流程（含常规战「搜索对手」环节）：

    村庄待机 ── 检查自己资源
      ├─ 全部已满 → 停止
      └─ 识别到「进攻按钮」 → 点进攻 → 搜索中（进入常规战搜索对手）
    搜索中   ── 识别到「搜索对手」界面
      ├─ 识别到「下一个」按钮 → 值不值得由资源评估算法判定
      │     ├─ 不值得 → 点「下一个」继续搜索（保持搜索中）
      │     └─ 值得（资源评估通过）→ 点放兵 → 战斗中
      └─ 识别到「回营/返回」→ 返回村庄待机
    战斗中   ── 对目标资源建筑下兵，打掉罐子/采集器
      ├─ 算法判「有可获取目标」→ 下兵（结束按钮常驻 ≠ 可点，绝不因它提前结束）
      ├─ 识别到结算界面（返回按钮）→ 战斗结束（被动）
      └─ 连续 N 帧无可获取目标 且「结束战斗」可见 → 主动结束
    战斗结束 ── 点「返回」→ 村庄待机（循环）

与 FarmFSM 的区别：新增 SEARCHING（搜索对手/评估）环节；BATTLE 中可用 end_battle_button
主动结束战斗，但必须由资源算法连续判空（EmptyTargetGuard）后才放行。
感知信号仍由检测器 + OCR 生成（信号驱动，FSM 不直接点屏幕）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from state.game_state import ResourceStatus

from decision.farm_fsm import default_policy


class BattleState(str, Enum):
    VILLAGE = "村庄待机"
    SEARCHING = "搜索中"
    BATTLE = "战斗中"
    BATTLE_OVER = "战斗结束"
    STOP = "停止"


@dataclass
class BattleSignals:
    """一次感知得到的信号（检测器 + OCR 生成），喂给 BattleFSM。"""

    attack_button: tuple[int, int] | None = None          # 村庄界面「进攻」按钮
    search_button: tuple[int, int] | None = None          # 联机模式「搜索对手」按钮
    next_button: tuple[int, int] | None = None            # 对手界面「下一个」（放弃当前对手）
    return_button: tuple[int, int] | None = None          # 「回营/返回」按钮
    end_battle_button: tuple[int, int] | None = None      # 战斗中「结束战斗」按钮
    # (建筑名, 中心点) 列表；列表中每一项都是「算法会打分的候选」，
    # 决策顺序：先看算法有无「可获取」候选，再决定是否判空结束。
    enemy_resources: list[tuple[str, tuple[int, int]]] = field(default_factory=list)
    resources: ResourceStatus = field(default_factory=ResourceStatus)  # 自己资源是否满
    # 资源评估算法输出：该对手值不值得打（None = 本轮未评估）
    worth: bool | None = None


@dataclass
class BattleAction:
    """FSM 给出的动作。kind: tap(点击) / deploy(下兵) / wait(等待) / stop(结束)。"""

    kind: str
    target: str | None = None
    coords: tuple[int, int] | None = None

    def __repr__(self) -> str:
        if self.coords:
            return f"BattleAction({self.kind}, {self.target}, {self.coords})"
        return f"BattleAction({self.kind}, {self.target})"


@dataclass
class EmptyTargetGuard:
    """连续判空守卫：连续 N 帧「无可获取资源目标」才允许主动结束战斗。

    由 BattleFSM 与 coach_session 共享同一实例（见 tools/coach_session 的
    _battle_end_guard），避免「判空是否达阈值」的结束判定在两处分叉。
    只要任一处仍给出可获取目标（on_targets），计数即清零。
    """

    threshold: int = 3  # 连续判空多少帧才算「可获取资源全没了」
    streak: int = 0

    def on_targets(self) -> None:
        """发现可获取目标 → 判空计数清零。"""
        self.streak = 0

    def on_empty(self) -> bool:
        """本帧无可获取目标 → 计数 +1；达到阈值返回 True（放行点结束）。"""
        self.streak += 1
        return self.streak >= self.threshold

    def reset(self) -> None:
        """重置计数，通常用于进入一场新战斗。"""
        self.streak = 0


def evaluate_worth(signals: BattleSignals, policy=None) -> bool | None:
    """资源评估：用资源策略计算当前对手值不值得打。

    - 没有敌方资源建筑 → 返回 None（无法评估，继续等待）
    - 有可抢目标（策略 default_policy 会跳过己方已满类型）→ True
    - 全部资源类型己方已满 → False
    """
    if not signals.enemy_resources:
        return None
    policy = policy or default_policy
    targets = policy(signals)
    return bool(targets)


class BattleFSM:
    """完整打资源有限状态机：输入感知信号，输出动作。

    用法与 FarmFSM 一致：
        fsm = BattleFSM()
        while True:
            signals = perceive(screenshot)   # 检测器 + OCR
            action = fsm.step(signals)
            if action.kind == "stop": break
            execute(action)
    """

    def __init__(self, policy=None, empty_frames_threshold: int = 3) -> None:
        self.state = BattleState.VILLAGE
        self.policy = policy or default_policy
        # 主动结束护栏：连续 empty_frames_threshold 帧无可获取目标才 tap 结束战斗
        self.empty_guard = EmptyTargetGuard(threshold=int(max(1, empty_frames_threshold)))
        # 历史状态序列，便于测试与报告
        self.history: list[str] = [self.state.value]

    def step(self, signals: BattleSignals) -> BattleAction:
        if self.state is BattleState.VILLAGE:
            if signals.resources.all_full:
                return self._transit(BattleState.STOP, BattleAction("stop", "所有资源已满"))
            if signals.attack_button is not None:
                action = BattleAction("tap", "进攻按钮", signals.attack_button)
                # 点进攻后进入常规战搜索（联机模式）；若同时看到「搜索对手」按钮也直接进入搜索
                return self._transit(BattleState.SEARCHING, action)
            return BattleAction("wait", "等待识别到进攻按钮")

        if self.state is BattleState.SEARCHING:
            # 回营/返回 → 回村庄
            if signals.return_button is not None and signals.next_button is None:
                return self._transit(BattleState.VILLAGE, BattleAction("tap", "返回按钮", signals.return_button))
            # 搜索动画中「搜索对手」按钮 → 保持搜索（点它继续搜）
            if signals.search_button is not None:
                return BattleAction("tap", "搜索对手按钮", signals.search_button)
            # 先评估该对手值不值得（用户叙述：值得→放兵；不值得→下一个）
            targets = self.policy(signals)
            worth = signals.worth if signals.worth is not None else evaluate_worth(signals, self.policy)
            if worth and targets:
                name, coords = targets[0]
                self.empty_guard.reset()  # 进入新战斗：判空计数复位
                return self._transit(BattleState.BATTLE, BattleAction("deploy", name, coords))
            # 不值得（或资源未检出无法评估）→ 点「下一个」继续搜索
            if signals.next_button is not None:
                return BattleAction("tap", "下一个", signals.next_button)
            if worth is False:
                return BattleAction("wait", "对手不值得，等待「下一个」按钮")
            return BattleAction("wait", "等待搜索结果/评估")

        if self.state is BattleState.BATTLE:
            # ① 资源算法先判「可获取目标」：有就打。红色「结束战斗」按钮整场常驻，
            #    它的出现 ≠ 战斗结束，绝不因它/倒计时就提前结束。
            targets = self.policy(signals)
            if targets:
                name, coords = targets[0]
                self.empty_guard.on_targets()  # 有目标 → 判空计数清零
                return BattleAction("deploy", name, coords)
            # ② 被动结算界面（输了/超时/战斗已结束的画面出现返回按钮）→ 战斗结束，等点返回
            if signals.return_button is not None:
                return self._transit(BattleState.BATTLE_OVER, BattleAction("wait", "战斗结束，准备点返回"))
            # ③ 主动结束：连续 N 帧无可获取目标 且 结束按钮可见 → 才 tap 结束
            if signals.end_battle_button is not None and self.empty_guard.on_empty():
                return self._transit(
                    BattleState.BATTLE_OVER, BattleAction("tap", "结束战斗", signals.end_battle_button)
                )
            return BattleAction("wait", "等待识别敌方资源建筑/判空确认")

        if self.state is BattleState.BATTLE_OVER:
            if signals.return_button is not None:
                return self._transit(BattleState.VILLAGE, BattleAction("tap", "返回按钮", signals.return_button))
            return BattleAction("wait", "等待识别返回按钮")

        return BattleAction("stop", "已停止")

    def _transit(self, new_state: BattleState, action: BattleAction) -> BattleAction:
        self.state = new_state
        if self.history[-1] != new_state.value:
            self.history.append(new_state.value)
        return action
