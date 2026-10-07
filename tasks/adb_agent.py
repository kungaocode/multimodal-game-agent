"""ADB-ready three-ring agent: real device ports for AgentLoop.

This module turns ``executor.adb_executor.AdbExecutor`` + ``CascadeExtractor``
into the ports consumed by ``tasks.agent_loop.AgentLoop``:

    perceive_village() -> VillageSnapshot
    run_village()      -> 采集 + 捐兵检测（一轮村庄环）
    run_farm(objective)-> 打资源补缺口

组合村庄环的优先顺序：免费采集图标 → 资源阈值检查 → 捐兵检测；
`BLOCKED`/`LOW` 由 `AgentLoop` 立即切到打资源环，战斗结束后下一轮自动回村。

Design notes:
- 采集/打资源/捐兵的具体识别仍由感知层负责；本模块只做
  截屏 -> GameState -> 信号 -> FSM -> 白名单校验 -> ADB 执行 的编排。
- 打资源环复用 tools.coach_session 的下兵决策链（RessourcePolicy +
  DeployPlanner + 判空护栏），保证 CLI 与 API 的落点/撤退判断一致。
- 云端模型负责复杂画面识别，三环仲裁（采集优先、阈值滞回、阻塞缺口）
  仍是本地确定性逻辑。
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from decision.arbiter import Ring, VillageSnapshot
from decision.battle_fsm import BattleFSM, BattleSignals, BattleState
from decision.collector_fsm import CollectorFSM, CollectorSignals
from decision.donation_fsm import DonationFSM, DonationSignals, DonationState
from decision.farm_fsm import FarmObjective
from decision.resource_gate import ResourceGate
from decision.resource_policy import RESOURCE_TYPE_BY_BUILDING
from executor.action import ALLOWED_TARGETS, COLLECTOR_TARGETS, AgentAction
from executor.adb_executor import AdbExecutor
from executor.validator import ActionValidator
from state.game_state import GameState, Resources
from tasks.agent_loop import AgentLoop, AgentPorts, RingOutcome
from tools.coach_executor import execute_action as _execute_action
from tools.coach_session import (
    _apply_battle_decision_chain,
    _validated_action,
    make_farm_planner,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ADB_PATH = ROOT / "tools/android-platform-tools/platform-tools/adb"

# 本地 YOLO / 级联感知返回的采集图标类别（可以后续扩展别名）
COLLECT_ICON_TYPES: dict[str, str] = {
    "金币采集图标": "gold",
    "圣水采集图标": "elixir",
    "黑油采集图标": "dark_elixir",
}

_RESOURCE_FIELDS = ("gold", "elixir", "dark_elixir")


def adb_path() -> str:
    """优先使用仓库内置的 platform-tools adb，其次回退到 PATH。"""
    if LOCAL_ADB_PATH.is_file() and os.access(LOCAL_ADB_PATH, os.X_OK):
        return str(LOCAL_ADB_PATH)
    return "adb"


def build_default_cascade(config: Any = None) -> Any:
    """Construct the local-first cascade with cloud Qwen fallback."""
    from perception.cascade import CascadeConfig, CascadeExtractor
    from perception.classifier import BuildingClassifier
    from perception.detector import Detector
    from perception.vision_model import VisionModel

    detector_path = Path(os.getenv("DETECTOR_MODEL", ROOT / "runs/detect/fsm_v4/weights/best.pt"))
    detector = Detector(detector_path) if detector_path.is_file() else None

    classifier_path = Path(os.getenv("CLASSIFIER_MODEL", ROOT / "runs/classifier/best.pt"))
    classifier = BuildingClassifier.load(classifier_path) if classifier_path.is_file() else None

    vision = None
    probe = VisionModel()
    if probe.base_url:
        vision = probe

    return CascadeExtractor(
        detector=detector,
        classifier=classifier,
        vision=vision,
        config=config or CascadeConfig(),
    )


def _copy_resources(resources: Resources) -> Resources:
    return Resources(
        gold=int(resources.gold),
        elixir=int(resources.elixir),
        dark_elixir=int(resources.dark_elixir),
    )


def _position(building: Any) -> tuple[int, int] | None:
    pos = getattr(building, "position", None)
    if not pos:
        return None
    x, y = int(pos[0]), int(pos[1])
    if x <= 0 or y <= 0:
        return None
    return (x, y)


def _first_position(state: GameState, *types: str) -> tuple[int, int] | None:
    for building in state.buildings:
        if building.type in types:
            pos = _position(building)
            if pos is not None:
                return pos
    return None


def _all_positions(state: GameState, *types: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for building in state.buildings:
        if building.type in types:
            pos = _position(building)
            if pos is not None:
                out.append(pos)
    return out


def _collectibles(state: GameState) -> list[tuple[str, tuple[int, int]]]:
    out: list[tuple[str, tuple[int, int]]] = []
    for building in state.buildings:
        resource_type = COLLECT_ICON_TYPES.get(building.type)
        if resource_type is None:
            continue
        pos = _position(building)
        if pos is not None:
            out.append((resource_type, pos))
    return out


def _resources_from_state(state: GameState) -> Resources | None:
    if state.resource_status.detected > 0:
        return _copy_resources(state.resource_status.amounts)
    if any(getattr(state.resources, field, 0) > 0 for field in _RESOURCE_FIELDS):
        return _copy_resources(state.resources)
    return None


def _state_detections(state: GameState) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for building in state.buildings:
        pos = _position(building)
        if pos is None:
            continue
        detections.append(
            {
                "type": building.type,
                "coords": list(pos),
                "bbox": list(building.bbox) if building.bbox else None,
                "confidence": float(building.confidence),
            }
        )
    return detections


def _default_request_hint(state: GameState) -> tuple[int, str] | None:
    """Best-effort donation cost hint from vision/OCR text entries."""
    for item in state.screen_text:
        if not isinstance(item, dict):
            continue
        cost = item.get("request_cost") or item.get("cost")
        if cost is not None:
            return int(cost), str(item.get("resource_type", "elixir"))
    return None


def _objective_matches(name: str, target_types: tuple[str, ...]) -> bool:
    resource_type = RESOURCE_TYPE_BY_BUILDING.get(name)
    if resource_type is None:
        return False
    if resource_type in target_types:
        return True
    return resource_type == "dark" and "dark_elixir" in target_types


class AdbAgent:
    """Bind a real ADB device to the four AgentLoop ports."""

    def __init__(
        self,
        adb: AdbExecutor,
        cascade: Any | None = None,
        *,
        max_steps: int = 60,
        step_delay: float = 0.8,
        troop_budget: int = 30,
        end_empty_frames: int = 3,
        request_hint_provider: Callable[[GameState], tuple[int, str] | None] | None = None,
        gate: ResourceGate | None = None,
    ) -> None:
        self.adb = adb
        self.cascade = cascade
        self.max_steps = int(max_steps)
        self.step_delay = float(step_delay)
        self.troop_budget = int(troop_budget)
        self.end_empty_frames = int(end_empty_frames)
        self.request_hint_provider = request_hint_provider or _default_request_hint
        self.gate = gate or ResourceGate()

        self.resources = Resources()
        self.resource_detected = False
        self.donation_blocked: dict[str, Any] | None = None
        self.donation_fsm = DonationFSM()

    # ------------------------------------------------------------ 感知
    def perceive(self) -> tuple[Image.Image, GameState, Any]:
        """One ADB screencap -> GameState. Returns (screenshot, state, report)."""
        screenshot = self.adb.screencap()
        if self.cascade is None:
            return screenshot, GameState(), None
        state, report = self.cascade.extract(screenshot)
        return screenshot, state, report

    def _perceive_state(self) -> tuple[Image.Image, GameState, Any]:
        screenshot, state, report = self.perceive()
        self._state_cache = state
        resources = _resources_from_state(state)
        if resources is not None:
            self.resources = resources
            self.resource_detected = True
        return screenshot, state, report

    def perceive_village(self) -> VillageSnapshot:
        self._perceive_state()
        return VillageSnapshot(
            resources=_copy_resources(self.resources),
            collectibles=tuple(_collectibles(self._last_state())),
            donation_blocked=dict(self.donation_blocked) if self.donation_blocked else None,
        )

    def run_village(self) -> RingOutcome:
        """组合村庄环：先收免费采集图标，再做捐兵检测；缺资源返回 BLOCKED/LOW。"""
        collect = self.run_collect()
        if collect.status in ("ERROR", "MAX_STEPS"):
            return RingOutcome(Ring.COLLECT, collect.status, collect.reason)

        gate_decision = self.gate.evaluate(self.resources)
        if gate_decision.needs_farm:
            return RingOutcome(
                Ring.FARM,
                "LOW",
                gate_decision.reason,
                objective=FarmObjective(
                    requirements=self.gate.low_requirements(gate_decision),
                    target_types=tuple(gate_decision.low_types),
                ),
            )

        donate = self.run_donate()
        return RingOutcome(
            donate.ring,
            donate.status,
            donate.reason,
            blocked_request=dict(donate.blocked_request or {})
            if donate.blocked_request
            else None,
            objective=donate.objective,
        )

    def _last_state(self) -> GameState:
        # _perceive_state 已经在上面执行过；为保证快照只截一帧，这里用缓存。
        return getattr(self, "_state_cache", GameState())

    # ------------------------------------------------------------ 采集环
    def run_collect(self) -> RingOutcome:
        fsm = CollectorFSM()
        fsm.reset()
        records: list[dict[str, Any]] = []
        for step_no in range(1, self.max_steps + 1):
            screenshot, state, _ = self._perceive_state()
            collectibles = _collectibles(state)
            if not collectibles:
                return RingOutcome(Ring.COLLECT, "SUCCESS", "无采集目标")

            action = fsm.step(CollectorSignals(collectibles))
            if action.kind == "stop":
                break

            check = ActionValidator(
                image_size=screenshot.size,
                allowed_targets=COLLECTOR_TARGETS,
            ).validate(action)
            if not check.allowed:
                return RingOutcome(
                    Ring.COLLECT,
                    "ERROR",
                    f"采集动作被校验器拦截: {check.reason}",
                )

            self.adb.tap(action.coords[0], action.coords[1])
            records.append({"step": step_no, "action": repr(action)})
            if self.step_delay > 0:
                time.sleep(self.step_delay)

        return RingOutcome(
            Ring.COLLECT,
            "MAX_STEPS" if len(records) == self.max_steps else "SUCCESS",
            "采集完成" if records else "无采集目标",
        )

    # ------------------------------------------------------------ 捐兵环
    def run_donate(self) -> RingOutcome:
        fsm = self.donation_fsm
        if fsm.state is not DonationState.BLOCKED:
            fsm.reset()
            self.donation_blocked = None

        for _step_no in range(1, self.max_steps + 1):
            screenshot, state, _ = self._perceive_state()
            hint = self.request_hint_provider(state)
            signals = self._donation_signals(state, hint)
            action = fsm.step(signals)

            if action.kind == "stop":
                self.donation_blocked = None
                fsm.reset()
                return RingOutcome(Ring.DONATE, "STOP", "捐兵流程停止")

            if fsm.state is DonationState.BLOCKED:
                self.donation_blocked = dict(fsm.blocked_request or {})
                close = _first_position(state, "关闭按钮")
                if close is not None:
                    close_check = ActionValidator(
                        image_size=screenshot.size
                    ).validate(AgentAction("tap", "关闭按钮", close))
                    if close_check.allowed:
                        self.adb.tap(close[0], close[1])
                return RingOutcome(
                    Ring.DONATE,
                    "BLOCKED",
                    "捐兵资源不足阻塞",
                    blocked_request=self.donation_blocked,
                )

            if action.kind == "wait":
                if self.step_delay > 0:
                    time.sleep(self.step_delay)
                continue

            check = ActionValidator(image_size=screenshot.size).validate(action)
            if not check.allowed:
                return RingOutcome(
                    Ring.DONATE,
                    "ERROR",
                    f"捐兵动作被校验器拦截: {check.reason}",
                    blocked_request=dict(fsm.blocked_request or {}) if fsm.blocked_request else None,
                )

            self.adb.tap(action.coords[0], action.coords[1])
            if action.target == "捐赠确认按钮" and hint is not None:
                self._spend(hint[0], hint[1])

            if action.target == "关闭按钮" and fsm.state is DonationState.VILLAGE:
                self.donation_blocked = None
                return RingOutcome(Ring.DONATE, "SUCCESS", "全部请求已捐完")

            if self.step_delay > 0:
                time.sleep(self.step_delay)

        return RingOutcome(
            Ring.DONATE,
            "MAX_STEPS",
            f"捐兵超过最大步数 {self.max_steps}",
            blocked_request=dict(fsm.blocked_request or {}) if fsm.blocked_request else None,
        )

    def _donation_signals(
        self,
        state: GameState,
        hint: tuple[int, str] | None,
    ) -> DonationSignals:
        request_entries = _all_positions(state, "请求条目")
        resources = self.resources if self.resource_detected else None
        return DonationSignals(
            message_list_button=_first_position(state, "消息列表按钮"),
            request_entry=request_entries[0] if request_entries else None,
            reinforce_button=_first_position(state, "增援按钮"),
            close_button=_first_position(state, "关闭按钮"),
            confirm_button=_first_position(state, "捐赠确认按钮"),
            troop_bar_coords=_first_position(state, "兵种选择栏"),
            troop_type="气球兵",
            more_requests=len(request_entries) > 1 if request_entries else None,
            request_cost=hint[0] if hint else 0,
            request_resource_type=hint[1] if hint else "elixir",
            resources=resources,
        )

    def _spend(self, cost: int, resource_type: str) -> None:
        if cost <= 0:
            return
        current = getattr(self.resources, resource_type, 0)
        setattr(self.resources, resource_type, max(0, int(current) - int(cost)))

    # ------------------------------------------------------------ 打资源环
    def run_farm(self, objective: FarmObjective | None = None) -> RingOutcome:
        objective = objective or FarmObjective()
        if self._objective_met(objective):
            return RingOutcome(Ring.FARM, "SUCCESS", "目标资源已补足", objective=objective)

        planner = make_farm_planner(self.troop_budget)
        fsm = BattleFSM(
            policy=planner.policy,
            empty_frames_threshold=max(1, self.end_empty_frames),
        )
        guard = fsm.empty_guard
        battle_entered = False

        for step_no in range(1, self.max_steps + 1):
            screenshot, state, _ = self._perceive_state()
            detections = _state_detections(state)
            signals = self._battle_signals(state, objective)
            local_state = fsm.state.value
            action = fsm.step(signals)
            action_dict = {
                "kind": action.kind,
                "target": action.target,
                "coords": list(action.coords) if action.coords is not None else None,
                "troop_coords": (
                    list(_first_position(state, "兵种选择栏"))
                    if _first_position(state, "兵种选择栏")
                    else None
                ),
            }

            action_dict, plan, _signals = _apply_battle_decision_chain(
                signals,
                None,
                action_dict,
                fsm,
                planner,
                guard,
                screenshot=screenshot,
                detections=detections,
                local_state=local_state,
            )
            validated = _validated_action(action_dict, screenshot)

            executed = False
            if validated.get("kind") in ("tap", "deploy") and validated.get("coords"):
                result = _execute_action(
                    validated,
                    self.adb,
                    fsm,
                    None,
                    "farm",
                    image_size=screenshot.size,
                    screenshot=screenshot,
                    detections=detections,
                )
                executed = bool(result.get("executed", False))
                if not executed:
                    return RingOutcome(
                        Ring.FARM,
                        "ERROR",
                        f"ADB 执行失败: {result.get('reason') or result.get('error') or 'unknown'}",
                        objective=objective,
                    )

            if plan is not None and executed:
                planner.commit(plan)
                self._add_planned_loot(plan)

            if fsm.state is BattleState.BATTLE:
                battle_entered = True

            if self._objective_met(objective):
                return RingOutcome(Ring.FARM, "SUCCESS", "目标资源已补足", objective=objective)

            if (
                battle_entered
                and fsm.state is BattleState.VILLAGE
                and action_dict.get("target") == "返回按钮"
            ):
                return RingOutcome(Ring.FARM, "SUCCESS", "战斗结束已回村", objective=objective)

            if action_dict.get("kind") == "stop":
                return RingOutcome(Ring.FARM, "STOP", "已停止", objective=objective)

            if self.step_delay > 0:
                time.sleep(self.step_delay)

        return RingOutcome(
            Ring.FARM,
            "MAX_STEPS",
            f"打资源超过最大步数 {self.max_steps}",
            objective=objective,
        )

    def _battle_signals(
        self,
        state: GameState,
        objective: FarmObjective,
    ) -> BattleSignals:
        enemy_resources: list[tuple[str, tuple[int, int]]] = []
        for name in RESOURCE_TYPE_BY_BUILDING:
            for pos in _all_positions(state, name):
                if not objective.target_types or _objective_matches(name, objective.target_types):
                    enemy_resources.append((name, pos))
        return BattleSignals(
            attack_button=_first_position(state, "进攻按钮"),
            search_button=_first_position(state, "搜索对手按钮"),
            next_button=_first_position(state, "下一个按钮"),
            return_button=_first_position(state, "返回按钮"),
            end_battle_button=_first_position(state, "结束战斗按钮"),
            enemy_resources=enemy_resources,
            resources=state.resource_status,
        )

    def _add_planned_loot(self, plan: Any) -> None:
        resource_type = RESOURCE_TYPE_BY_BUILDING.get(getattr(plan, "target_name", ""))
        if resource_type is None:
            return
        field_name = "dark_elixir" if resource_type == "dark" else resource_type
        expected = getattr(plan, "expected_loot", None)
        if expected is None:
            estimate = getattr(plan, "loot_estimate", None)
            expected = getattr(estimate, "expected_loot", 0)
        current = getattr(self.resources, field_name, 0)
        setattr(self.resources, field_name, current + int(expected or 0))

    def _objective_met(self, objective: FarmObjective) -> bool:
        if not objective.requirements:
            return False
        return all(
            getattr(self.resources, key, 0) >= value
            for key, value in objective.requirements.items()
        )


def build_adb_ports(agent: AdbAgent) -> AgentPorts:
    """Expose one AdbAgent as AgentPorts for AgentLoop."""
    return AgentPorts(
        perceive_village=agent.perceive_village,
        run_collect=agent.run_collect,
        run_farm=agent.run_farm,
        run_donate=agent.run_donate,
        run_village=agent.run_village,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ADB 三环调试入口：村庄环（采集+捐兵检测）> 打资源"
    )
    parser.add_argument("--serial", help="ADB device serial（可选，自动检测）")
    parser.add_argument("--rounds", type=int, default=8, help="外层仲裁最大轮数")
    parser.add_argument("--max-steps", type=int, default=60, help="每个环的最大步数")
    parser.add_argument("--step-delay", type=float, default=0.8)
    parser.add_argument("--troop-budget", type=int, default=30)
    parser.add_argument("--end-empty-frames", type=int, default=3)
    parser.add_argument("--no-cascade", action="store_true", help="不加载本地/云端感知")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    adb = AdbExecutor(serial=args.serial, adb_path=adb_path())
    if not adb.is_connected():
        raise SystemExit(
            "ADB 设备未连接。请先连接手机/模拟器后运行: adb devices"
        )

    cascade = build_default_cascade() if not args.no_cascade else None
    agent = AdbAgent(
        adb,
        cascade=cascade,
        max_steps=args.max_steps,
        step_delay=args.step_delay,
        troop_budget=args.troop_budget,
        end_empty_frames=args.end_empty_frames,
    )
    loop = AgentLoop(build_adb_ports(agent), max_rounds=args.rounds)
    outcomes = loop.run()
    for outcome in outcomes:
        logger.info(
            "ring=%s status=%s reason=%s",
            outcome.ring.value,
            outcome.status,
            outcome.reason,
        )


if __name__ == "__main__":
    main()
