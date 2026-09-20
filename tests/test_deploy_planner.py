"""Tests for final deploy target selection and repeat rotation."""

from decision.battle_fsm import BattleFSM, BattleSignals, BattleState
from decision.deploy_planner import DeployPlanner
from decision.farm_fsm import FarmSignals
from decision.resource_policy import PolicyConfig, ResourcePolicy
from state.game_state import ResourceStatus


def _signals(*resources, status=None):
    return FarmSignals(enemy_resources=list(resources), resources=status or ResourceStatus())


def test_honors_requested_target_when_local_detection_misses_it():
    planner = DeployPlanner(
        policy=ResourcePolicy(PolicyConfig(image_size=(1280, 720), deploy_point=(640, 700))),
        landing_planner=lambda name, coords, _shot, _dets, _req: (coords[0] + 10, coords[1] + 5),
    )
    plan = planner.plan(
        _signals(),
        requested_name="金矿",
        requested_coords=(640, 360),
    )
    assert plan is not None
    assert plan.target_name == "金矿"
    assert plan.target_coords == (640, 360)
    assert plan.landing_point == (650, 365)
    assert plan.reason == "requested_target_honored"


def test_requested_full_resource_is_rejected():
    status = ResourceStatus(gold_full=True, detected=1)
    planner = DeployPlanner()
    plan = planner.plan(
        _signals(("金矿", (100, 100)), status=status),
        requested_name="金矿",
        requested_coords=(100, 100),
    )
    assert plan is None


def test_repeat_limit_rotates_to_next_scored_target():
    config = PolicyConfig(image_size=(600, 400), deploy_point=(100, 380))
    planner = DeployPlanner(policy=ResourcePolicy(config), max_consecutive_per_target=2)
    resources = [
        ("金矿", (120, 340)),
        ("金矿", (520, 340)),
    ]
    first = planner.plan(_signals(*resources))
    second = planner.plan(_signals(*resources))
    third = planner.plan(_signals(*resources))

    assert first is not None and first.target_coords == (120, 340)
    assert second is not None and second.target_coords == (120, 340)
    assert second.reason == "repeat_limit_reached"
    assert third is not None and third.target_coords == (520, 340)
    assert third.reason == "rotated_after_repeat_limit"


def test_landing_planner_and_breakdown_are_exposed():
    planner = DeployPlanner(
        policy=ResourcePolicy(PolicyConfig(image_size=(1280, 720))),
        landing_planner=lambda _name, coords, _shot, _dets, _req: (coords[0], coords[1] - 3),
    )
    plan = planner.plan(
        _signals(("暗黑重油罐", (640, 360)), ("金矿", (640, 360))),
        screenshot=None,
        requested_name="暗黑重油罐",
        requested_coords=(640, 360),
    )
    assert plan is not None
    assert plan.landing_point == (640, 357)
    assert plan.breakdown["value"] > 0.9


def test_battle_search_uses_scored_target_not_raw_first():
    policy = ResourcePolicy(PolicyConfig(deploy_point=(640, 700)))
    fsm = BattleFSM(policy=policy)
    fsm.state = BattleState.SEARCHING
    signals = BattleSignals(
        enemy_resources=[
            ("金矿", (100, 100)),
            ("暗黑重油罐", (620, 680)),
        ]
    )
    action = fsm.step(signals)
    assert action.kind == "deploy"
    assert action.target == "暗黑重油罐"
    assert fsm.state.value == "战斗中"
