"""Tests for the action schema, validator and simulator (阶段 8)."""

from executor.action import AgentAction
from executor.simulator import (
    DefenseSite,
    ResourceSite,
    SimVillage,
    Simulator,
    default_farm_world,
)
from executor.validator import ActionValidator


def _world():
    return [
        SimVillage(
            name="A",
            resources=[
                ResourceSite("金矿", (300, 300), capacity=3000),
                ResourceSite("圣水瓶", (900, 300), capacity=5000),
            ],
            defenses=[DefenseSite((500, 200))],
        ),
        SimVillage(name="B", resources=[ResourceSite("暗黑重油罐", (600, 400), capacity=2000)]),
    ]


# ---------------------------------------------------------------- 动作 schema
def test_action_roundtrip():
    action = AgentAction("deploy", "金矿", (100, 200), {"units": 3})
    assert AgentAction.from_dict(action.to_dict()) == action


def test_farm_action_conversion():
    class FakeFarmAction:
        kind = "tap"
        target = "进攻按钮"
        coords = (1000, 600)

    converted = AgentAction.from_farm_action(FakeFarmAction())
    assert converted.kind == "tap"
    assert converted.target == "进攻按钮"
    assert converted.coords == (1000, 600)


# ---------------------------------------------------------------- 校验器
def test_validator_allows_valid_actions():
    validator = ActionValidator()
    assert validator.validate(AgentAction("tap", "进攻按钮", (1000, 600))).allowed
    assert validator.validate(AgentAction("deploy", "金矿", (100, 100))).allowed
    assert validator.validate(AgentAction("wait")).allowed
    assert validator.validate(AgentAction("stop")).allowed


def test_validator_blocks_out_of_bounds():
    validator = ActionValidator()
    assert not validator.validate(AgentAction("tap", "进攻按钮", (2000, 2000))).allowed
    assert not validator.validate(AgentAction("deploy", "金矿", (-5, 10))).allowed


def test_validator_ignores_troop_coords_on_tap():
    """教练在非 deploy 动作上会把 troop_coords 填成默认 [0,0]，
    此时 tap 不应因兵种坐标越界被拦（回归：真机 tap 被误降级为 wait）。"""
    validator = ActionValidator(image_size=(2844, 1260))
    assert validator.validate(
        AgentAction("tap", "进攻按钮", (228, 1058), troop_coords=(0, 0))
    ).allowed
    assert validator.validate(
        AgentAction("tap", "进攻按钮", (228, 1058), troop_coords=(2000, 2000))
    ).allowed


def test_validator_still_blocks_bad_troop_coords_on_deploy():
    validator = ActionValidator(image_size=(2844, 1260))
    assert validator.validate(
        AgentAction("deploy", "金矿", (300, 300), troop_coords=(100, 1100))
    ).allowed
    assert not validator.validate(
        AgentAction("deploy", "金矿", (300, 300), troop_coords=(0, 0))
    ).allowed
    assert not validator.validate(
        AgentAction("deploy", "金矿", (300, 300), troop_coords=(2000, 2000))
    ).allowed


def test_validator_blocks_unknown_target_and_kind():
    validator = ActionValidator()
    assert not validator.validate(AgentAction("deploy", "大本营", (100, 100))).allowed
    assert not validator.validate(AgentAction("fly", "进攻按钮", (100, 100))).allowed


def test_validator_requires_coords_for_tap_deploy():
    validator = ActionValidator()
    assert not validator.validate(AgentAction("tap", "进攻按钮")).allowed
    assert not validator.validate(AgentAction("deploy", "金矿")).allowed


def test_validator_risk_levels():
    validator = ActionValidator()
    assert validator.validate(AgentAction("deploy", "金矿", (100, 100))).risk_level == "medium"
    assert validator.validate(AgentAction("wait")).risk_level == "low"


# ---------------------------------------------------------------- 模拟器
def test_village_perception_shows_attack_button():
    simulator = Simulator(_world())
    signals = simulator.perceive()
    assert signals.attack_button == simulator.attack_button
    assert signals.return_button is None
    assert signals.enemy_resources == []


def test_attack_enters_battle_with_enemy_resources():
    simulator = Simulator(_world())
    simulator.step(AgentAction("tap", "进攻按钮", simulator.attack_button))
    assert simulator.screen == Simulator.SCREEN_BATTLE
    signals = simulator.perceive()
    assert signals.return_button is None
    assert {name for name, _ in signals.enemy_resources} == {"金矿", "圣水瓶"}


def test_deploy_loots_and_removes_depleted_site():
    simulator = Simulator(_world(), loot_per_deploy=1000)
    simulator.step(AgentAction("tap", "进攻按钮", simulator.attack_button))
    result = simulator.step(AgentAction("deploy", "金矿", (300, 300)))
    assert result.reward == 1000
    assert result.info["looted"] == 1000
    for _ in range(2):  # 第 2、3 次打空 3000
        simulator.step(AgentAction("deploy", "金矿", (300, 300)))
    signals = simulator.perceive()
    assert "金矿" not in [name for name, _ in signals.enemy_resources]


def test_battle_over_when_all_looted():
    simulator = Simulator(_world(), loot_per_deploy=1000)
    simulator.step(AgentAction("tap", "进攻按钮", simulator.attack_button))
    for target, coords, capacity in [("金矿", (300, 300), 3000), ("圣水瓶", (900, 300), 5000)]:
        for _ in range(capacity // 1000):
            simulator.step(AgentAction("deploy", target, coords))
    assert simulator.screen == Simulator.SCREEN_BATTLE_OVER
    assert simulator.perceive().return_button == simulator.return_button


def test_troop_exhaustion_ends_battle():
    simulator = Simulator(_world(), troops_per_battle=2, loot_per_deploy=1000)
    simulator.step(AgentAction("tap", "进攻按钮", simulator.attack_button))
    simulator.step(AgentAction("deploy", "金矿", (300, 300)))
    simulator.step(AgentAction("deploy", "圣水瓶", (900, 300)))
    assert simulator.screen == Simulator.SCREEN_BATTLE_OVER


def test_return_advances_to_next_village():
    simulator = Simulator(_world())
    simulator.step(AgentAction("tap", "进攻按钮", simulator.attack_button))
    simulator.step(AgentAction("tap", "返回按钮", simulator.return_button))
    assert simulator.screen == Simulator.SCREEN_VILLAGE
    assert simulator.village_index == 1
    simulator.step(AgentAction("tap", "进攻按钮", simulator.attack_button))
    assert {name for name, _ in simulator.perceive().enemy_resources} == {"暗黑重油罐"}


def test_done_after_last_village():
    simulator = Simulator(_world())
    for _ in range(2):
        simulator.step(AgentAction("tap", "进攻按钮", simulator.attack_button))
        simulator.step(AgentAction("tap", "返回按钮", simulator.return_button))
    assert simulator.done
    assert simulator.perceive().attack_button is None


def test_deploy_unknown_target_no_reward():
    simulator = Simulator(_world())
    simulator.step(AgentAction("tap", "进攻按钮", simulator.attack_button))
    result = simulator.step(AgentAction("deploy", "圣水瓶", (9999, 9999)))
    assert result.reward == 0
    assert result.info["reason"]


def test_default_world_is_deterministic():
    world_a = default_farm_world()
    world_b = default_farm_world()
    assert world_a == world_b
    assert sum(site.capacity for v in world_a for site in v.resources) == 22000
