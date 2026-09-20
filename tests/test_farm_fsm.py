"""Tests for the farm FSM state machine."""

from decision.farm_fsm import FarmFSM, FarmSignals
from state.game_state import ResourceStatus


def _full() -> ResourceStatus:
    return ResourceStatus(gold_full=True, elixir_full=True, dark_full=True, detected=3)


def test_village_all_resources_full_stops():
    fsm = FarmFSM()
    action = fsm.step(FarmSignals(resources=_full(), attack_button=(100, 100)))
    assert action.kind == "stop"
    assert fsm.state.name == "STOP"


def test_village_not_full_taps_attack():
    fsm = FarmFSM()
    action = fsm.step(FarmSignals(attack_button=(500, 600)))
    assert action.kind == "tap"
    assert action.target == "进攻按钮"
    assert action.coords == (500, 600)
    assert fsm.state.name == "BATTLE"


def test_village_waits_until_attack_button_appears():
    fsm = FarmFSM()
    action = fsm.step(FarmSignals())
    assert action.kind == "wait"
    assert fsm.state.name == "VILLAGE"


def test_battle_deploys_on_enemy_resource():
    fsm = FarmFSM()
    fsm.state = fsm.state.BATTLE
    action = fsm.step(FarmSignals(enemy_resources=[("金矿", (120, 220))]))
    assert action.kind == "deploy"
    assert action.target == "金矿"
    assert action.coords == (120, 220)


def test_battle_return_button_means_battle_over():
    fsm = FarmFSM()
    fsm.state = fsm.state.BATTLE
    action = fsm.step(FarmSignals(enemy_resources=[("金矿", (1, 1))], return_button=(700, 80)))
    assert action.kind == "wait"  # 先不点，等战斗结束界面
    assert fsm.state.name == "BATTLE_OVER"


def test_battle_over_taps_return_and_loops_back():
    fsm = FarmFSM()
    fsm.state = fsm.state.BATTLE_OVER
    action = fsm.step(FarmSignals(return_button=(10, 10)))
    assert action.kind == "tap"
    assert action.target == "返回按钮"
    assert fsm.state.name == "VILLAGE"


def test_policy_skips_full_resource_collectors():
    fsm = FarmFSM()
    fsm.state = fsm.state.BATTLE
    signals = FarmSignals(
        enemy_resources=[("圣水收集器", (1, 1)), ("金矿", (2, 2))],
        resources=ResourceStatus(elixir_full=True, detected=1),
    )
    action = fsm.step(signals)
    assert action.target == "金矿"  # 圣水满了 → 圣水收集器被跳过
