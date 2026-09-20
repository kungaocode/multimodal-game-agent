"""Tests for tools/coach_session.py —— 过早结束护栏与资源回填（dry-run，无需真机）。

聚焦 §4.4：教练拟 tap 结束/返回，但回填后算法仍有可获取目标 → 强制转 deploy；
信号全空 → 连续判空未达阈值前改 wait，达阈值才放行结束。
"""

from __future__ import annotations

from decision.battle_fsm import BattleSignals, EmptyTargetGuard
from decision.coach import CoachVerdict
from decision.deploy_planner import DeployPlanner
from decision.resource_policy import ResourcePolicy
from state.game_state import ResourceStatus
from tools.coach_session import _backfill_resources, _battle_end_guard


def _planner() -> DeployPlanner:
    """无掩码/落点规划的纯打分 planner，便于单测。"""
    return DeployPlanner(policy=ResourcePolicy())


def _end_tap(target: str = "结束战斗") -> dict:
    return {"kind": "tap", "target": target, "coords": [100, 100], "troop_coords": None}


def test_backfill_resources_merges_coach_buildings_dedup():
    sig = BattleSignals(enemy_resources=[("金矿", (100, 100))])
    verdict = CoachVerdict(
        resource_buildings=[
            ("金矿", (100, 100)),      # 与已有重复 → 不重复加
            ("圣水瓶", (200, 300)),
            ("储金罐", (640, 360)),
        ]
    )
    out = _backfill_resources(sig, verdict)
    assert out.enemy_resources == [
        ("金矿", (100, 100)),
        ("圣水瓶", (200, 300)),
        ("储金罐", (640, 360)),
    ]


def test_backfill_resources_none_or_empty_keeps_signals():
    sig = BattleSignals(enemy_resources=[("金矿", (100, 100))])
    assert _backfill_resources(sig, None) is sig
    assert _backfill_resources(sig, CoachVerdict()) is sig


def test_end_guard_forces_deploy_when_targets_remain():
    """教练拟 tap 结束、但算法仍有可获取目标 → 强制转 deploy 并清空判空计数。"""
    guard = EmptyTargetGuard(threshold=3)
    sig = BattleSignals(enemy_resources=[("储金罐", (640, 360))])
    out, plan = _battle_end_guard(sig, _planner(), _end_tap(), guard)
    assert out["kind"] == "deploy"
    assert out["target"] == "储金罐"
    assert out["coords"] == [640, 360]
    assert plan is not None
    assert guard.streak == 0
    # deploy 动作带打分/重复次数元数据，满足验收口径的日志字段
    assert "target_score" in out and "target_repeat_count" in out and "target_reason" in out


def test_end_guard_waits_until_consecutive_empty_threshold():
    """信号全空时前 2 帧改 wait，第 3 帧才放行 tap 结束。"""
    guard = EmptyTargetGuard(threshold=3)
    sig = BattleSignals()  # 无可获取目标
    out1, plan1 = _battle_end_guard(sig, _planner(), _end_tap(), guard)
    assert out1["kind"] == "wait"
    assert guard.streak == 1
    out2, _ = _battle_end_guard(sig, _planner(), _end_tap(), guard)
    assert out2["kind"] == "wait"
    assert guard.streak == 2
    out3, plan3 = _battle_end_guard(sig, _planner(), _end_tap(), guard)
    assert out3 == _end_tap()  # 达阈值 → 原样放行原结束动作
    assert plan3 is None
    assert guard.streak == 3


def test_end_guard_applies_to_return_button_when_in_battle():
    guard = EmptyTargetGuard(threshold=3)
    sig = BattleSignals(enemy_resources=[("金矿", (320, 360))])
    out, _ = _battle_end_guard(sig, _planner(), _end_tap("返回按钮"), guard)
    assert out["kind"] == "deploy"
    assert out["target"] == "金矿"


def test_end_guard_deploy_passes_through_and_clears_streak():
    guard = EmptyTargetGuard(threshold=3)
    _battle_end_guard(BattleSignals(), _planner(), _end_tap(), guard)  # streak=1
    deploy = {"kind": "deploy", "target": "金矿", "coords": [320, 360], "troop_coords": None}
    out, plan = _battle_end_guard(
        BattleSignals(enemy_resources=[("金矿", (320, 360))]), _planner(), deploy, guard
    )
    assert out == deploy
    assert plan is None
    assert guard.streak == 0


def test_end_guard_leaves_non_end_taps_alone():
    guard = EmptyTargetGuard(threshold=3)
    sig = BattleSignals()
    tap_next = {"kind": "tap", "target": "下一个", "coords": [300, 300], "troop_coords": None}
    out, plan = _battle_end_guard(sig, _planner(), tap_next, guard)
    assert out == tap_next
    assert plan is None
    assert guard.streak == 0  # 非结束动作不参与判空计数


def test_end_guard_wait_action_does_not_touch_streak():
    """本帧动作是 wait（非拟结束）→ 不判空也不改动作。"""
    guard = EmptyTargetGuard(threshold=3)
    sig = BattleSignals()
    wait = {"kind": "wait", "target": "等待", "coords": None, "troop_coords": None}
    out, plan = _battle_end_guard(sig, _planner(), wait, guard)
    assert out == wait
    assert plan is None
    assert guard.streak == 0


def test_guard_takes_all_resources_into_account_after_backfill():
    """先回填再判空：己方金/圣水已满 → 本地金矿被打分跳过，算法选教练补报的暗黑罐。"""
    guard = EmptyTargetGuard(threshold=3)
    signals = BattleSignals(
        enemy_resources=[("金矿", (60, 60))],   # 本地唯一检出；己方金已满会被跳过
        resources=ResourceStatus(gold_full=True, elixir_full=True),
    )
    verdict = CoachVerdict(resource_buildings=[("暗黑重油罐", (600, 300))])
    signals = _backfill_resources(signals, verdict)
    assert len(signals.enemy_resources) == 2
    # 高价值「暗黑重油罐」（价值 1.0）是唯一可获取候选 → 强制下兵它而不是结束
    out, _ = _battle_end_guard(signals, _planner(), _end_tap(), guard)
    assert out["kind"] == "deploy"
    assert out["target"] == "暗黑重油罐"
