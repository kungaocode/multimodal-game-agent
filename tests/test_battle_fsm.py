"""Tests for decision/battle_fsm.py — 完整打资源状态机（含搜索对手环节）。"""

from __future__ import annotations

from decision.battle_fsm import (
    BattleFSM,
    BattleSignals,
    BattleState,
    EmptyTargetGuard,
    evaluate_worth,
)
from state.game_state import ResourceStatus


def _sig(**kw):
    defaults = dict(
        attack_button=None,
        search_button=None,
        next_button=None,
        return_button=None,
        end_battle_button=None,
        enemy_resources=None,
        resources=None,
        worth=None,
    )
    defaults.update(kw)
    sig = BattleSignals(
        attack_button=defaults["attack_button"],
        search_button=defaults["search_button"],
        next_button=defaults["next_button"],
        return_button=defaults["return_button"],
        end_battle_button=defaults["end_battle_button"],
        enemy_resources=defaults["enemy_resources"] or [],
        resources=defaults["resources"] or ResourceStatus(),
        worth=defaults["worth"],
    )
    return sig


def test_village_attack_button_goes_searching():
    fsm = BattleFSM()
    action = fsm.step(_sig(attack_button=(100, 200)))
    assert action.kind == "tap"
    assert action.target == "进攻按钮"
    assert fsm.state is BattleState.SEARCHING


def test_village_all_full_stops():
    fsm = BattleFSM()
    resources = ResourceStatus(detected=3, gold_full=True, elixir_full=True, dark_full=True)
    action = fsm.step(_sig(resources=resources))
    assert action.kind == "stop"
    assert fsm.state is BattleState.STOP


def test_searching_next_button_keeps_searching():
    fsm = BattleFSM()
    fsm.state = BattleState.SEARCHING
    action = fsm.step(_sig(next_button=(300, 400)))
    assert action.kind == "tap"
    assert action.target == "下一个"
    assert fsm.state is BattleState.SEARCHING


def test_searching_return_goes_village():
    fsm = BattleFSM()
    fsm.state = BattleState.SEARCHING
    action = fsm.step(_sig(return_button=(10, 10)))
    assert fsm.state is BattleState.VILLAGE


def test_searching_worth_deploys_and_battles():
    fsm = BattleFSM()
    fsm.state = BattleState.SEARCHING
    sig = _sig(
        enemy_resources=[("金矿", (100, 100)), ("圣水收集器", (200, 200))],
        worth=True,
    )
    action = fsm.step(sig)
    assert action.kind == "deploy"
    assert fsm.state is BattleState.BATTLE


def test_searching_not_worth_waits_for_next():
    fsm = BattleFSM()
    fsm.state = BattleState.SEARCHING
    action = fsm.step(_sig(enemy_resources=[("金矿", (100, 100))], worth=False))
    assert action.kind == "wait"
    assert fsm.state is BattleState.SEARCHING


def test_battle_end_button_requires_consecutive_empty_frames():
    """红色结束按钮常驻 ≠ 可点：无资源时首帧 wait，连续 3 帧判空才 tap 结束。"""
    fsm = BattleFSM()
    fsm.state = BattleState.BATTLE
    sig = _sig(end_battle_button=(500, 500))
    action1 = fsm.step(sig)
    assert action1.kind == "wait"
    assert fsm.state is BattleState.BATTLE
    action2 = fsm.step(sig)
    assert action2.kind == "wait"
    assert fsm.state is BattleState.BATTLE
    action3 = fsm.step(sig)
    assert action3.kind == "tap"
    assert action3.target == "结束战斗"
    assert fsm.state is BattleState.BATTLE_OVER


def test_battle_end_button_custom_threshold():
    """自定义判空阈值：empty_frames_threshold=1 → 单帧判空即可结束。"""
    fsm = BattleFSM(empty_frames_threshold=1)
    fsm.state = BattleState.BATTLE
    action = fsm.step(_sig(end_battle_button=(500, 500)))
    assert action.kind == "tap"
    assert fsm.state is BattleState.BATTLE_OVER


def test_battle_return_button_finishes():
    fsm = BattleFSM()
    fsm.state = BattleState.BATTLE
    action = fsm.step(_sig(return_button=(500, 500)))
    assert action.kind == "wait"
    assert fsm.state is BattleState.BATTLE_OVER


def test_battle_deploys_target():
    fsm = BattleFSM()
    fsm.state = BattleState.BATTLE
    sig = _sig(enemy_resources=[("储金罐", (300, 300))])
    action = fsm.step(sig)
    assert action.kind == "deploy"
    assert action.target == "储金罐"
    assert fsm.state is BattleState.BATTLE   # 下兵不改变状态


def test_battle_resources_with_end_button_deploys_not_finishes():
    """结束按钮常驻 ≠ 结束：仍有可获取目标 → 下兵且保持战斗中。"""
    fsm = BattleFSM()
    fsm.state = BattleState.BATTLE
    sig = _sig(enemy_resources=[("储金罐", (300, 300))], end_battle_button=(500, 500))
    action = fsm.step(sig)
    assert action.kind == "deploy"
    assert action.target == "储金罐"
    assert fsm.state is BattleState.BATTLE


def test_battle_deploy_resets_empty_streak():
    """deploy 后判空计数清零：见目标下兵后再判空须从 1 重新计。"""
    fsm = BattleFSM()
    fsm.state = BattleState.BATTLE
    end_sig = _sig(end_battle_button=(500, 500))
    assert fsm.step(end_sig).kind == "wait"  # 判空 1
    assert fsm.step(end_sig).kind == "wait"  # 判空 2
    action = fsm.step(_sig(enemy_resources=[("金矿", (200, 200))], end_battle_button=(500, 500)))
    assert action.kind == "deploy"
    assert fsm.empty_guard.streak == 0
    assert fsm.step(end_sig).kind == "wait"  # 重新判空 1，不立即结束
    assert fsm.state is BattleState.BATTLE


def test_battle_return_button_with_targets_still_deploys():
    """被动结算判定在算法之后：即使画面见返回按钮，仍有可获取目标时优先下兵。"""
    fsm = BattleFSM()
    fsm.state = BattleState.BATTLE
    sig = _sig(enemy_resources=[("金矿", (200, 200))], return_button=(10, 10))
    action = fsm.step(sig)
    assert action.kind == "deploy"
    assert fsm.state is BattleState.BATTLE


def test_empty_target_guard_threshold():
    guard = EmptyTargetGuard(threshold=3)
    assert guard.on_empty() is False  # 1
    assert guard.on_empty() is False  # 2
    assert guard.streak == 2
    assert guard.on_empty() is True   # 3 → 达阈值
    guard.on_targets()
    assert guard.streak == 0
    assert guard.on_empty() is False  # 清零后重新计


def test_empty_target_guard_reset():
    guard = EmptyTargetGuard(threshold=2)
    assert guard.on_empty() is False  # 1
    assert guard.on_empty() is True   # 2 → 达阈值
    guard.reset()
    assert guard.streak == 0
    assert guard.on_empty() is False  # 复位后重新计


def test_battle_over_return_goes_village():
    fsm = BattleFSM()
    fsm.state = BattleState.BATTLE_OVER
    action = fsm.step(_sig(return_button=(600, 600)))
    assert action.kind == "tap"
    assert action.target == "返回按钮"
    assert fsm.state is BattleState.VILLAGE


def test_full_cycle_history():
    """完整循环：村庄 → 搜索 → 战斗 →（连续判空 3 帧）结束 → 村庄。"""
    fsm = BattleFSM()
    fsm.step(_sig(attack_button=(100, 100)))
    assert fsm.state is BattleState.SEARCHING
    fsm.step(_sig(enemy_resources=[("金矿", (200, 200))], worth=True))
    assert fsm.state is BattleState.BATTLE
    end_sig = _sig(end_battle_button=(300, 300))
    fsm.step(end_sig)  # 判空 1
    fsm.step(end_sig)  # 判空 2
    assert fsm.state is BattleState.BATTLE
    fsm.step(end_sig)  # 判空 3 → 战斗结束
    assert fsm.state is BattleState.BATTLE_OVER
    fsm.step(_sig(return_button=(400, 400)))
    assert fsm.state is BattleState.VILLAGE
    assert fsm.history == [
        "村庄待机",
        "搜索中",
        "战斗中",
        "战斗结束",
        "村庄待机",
    ]


def test_evaluate_worth_empty_resources_none():
    assert evaluate_worth(_sig()) is None


def test_evaluate_worth_has_targets():
    sig = _sig(enemy_resources=[("金矿", (100, 100)), ("储金罐", (200, 200))])
    assert evaluate_worth(sig) is True


def test_evaluate_worth_skips_full_resource():
    """己方金色已满 → 金矿/储金罐不算可抢目标 → 评估失败。"""
    resources = ResourceStatus(detected=2, gold_full=True)
    sig = _sig(enemy_resources=[("金矿", (100, 100)), ("暗黑重油罐", (200, 200))], resources=resources)
    assert evaluate_worth(sig) is True   # 黑油还可抢
    sig2 = _sig(enemy_resources=[("金矿", (100, 100))], resources=resources)
    assert evaluate_worth(sig2) is False  # 只剩金色 → 不值得


def test_stop_is_terminal():
    fsm = BattleFSM()
    resources = ResourceStatus(detected=3, gold_full=True, elixir_full=True, dark_full=True)
    action = fsm.step(_sig(attack_button=(1, 1), resources=resources))
    assert action.kind == "stop"
    action2 = fsm.step(_sig(attack_button=(2, 2)))
    assert action2.kind == "stop"
    assert fsm.state is BattleState.STOP


def test_searching_next_button_with_worth_deploys():
    """搜索中发现对手（出现「下一个」按钮）且评估值得 → 先放兵开战，而不是点下一个。"""
    fsm = BattleFSM()
    fsm.state = BattleState.SEARCHING
    action = fsm.step(_sig(next_button=(300, 400), enemy_resources=[("金矿", (100, 100))], worth=True))
    assert action.kind == "deploy"
    assert action.target == "金矿"
    assert fsm.state is BattleState.BATTLE
