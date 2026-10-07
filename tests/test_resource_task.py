"""Tests for the end-to-end resource task closed loop (阶段 3/7)."""

from decision.farm_fsm import FarmFSM, FarmObjective
from executor.simulator import (
    DefenseSite,
    ResourceSite,
    Simulator,
    SimVillage,
    default_farm_world,
)
from executor.validator import ActionValidator
from state.game_state import ResourceStatus, Resources
from tasks.resource import ResourceTask


def test_task_loots_all_villages_to_success():
    task = ResourceTask()
    simulator = Simulator(default_farm_world())
    result = task.run(simulator, max_steps=300)

    assert result.status == "SUCCESS"
    assert result.reason == "所有村庄已搜刮完毕"
    assert result.total_loot > 0
    assert result.steps == len(result.records)
    assert result.records[-1].action.kind == "wait"
    # 每一个 deploy / tap 都必须通过前后状态验证
    assert all(record.verified for record in result.records)
    # 日志可序列化为 JSON 友好 dict
    assert isinstance(result.summary()["records"], list)


def test_task_stops_when_own_resources_full():
    own = ResourceStatus(gold_full=True, elixir_full=True, dark_full=True, detected=3)
    simulator = Simulator(default_farm_world(), own_resources=own)
    result = ResourceTask().run(simulator)

    assert result.status == "STOP"
    assert result.reason == "己方资源已满，任务停止"
    assert result.total_loot == 0
    assert result.records[-1].action.kind == "stop"


def test_task_errors_when_action_blocked_by_validator():
    simulator = Simulator(default_farm_world())  # 按钮在 1120x640 附近，超出小画面
    validator = ActionValidator(image_size=(640, 360))
    result = ResourceTask(validator=validator).run(simulator)

    assert result.status == "ERROR"
    assert result.steps == 1
    assert result.records == []


def test_task_uses_injected_fsm_and_calls_policy():
    calls = {"n": 0}

    class CountingPolicy:
        def __call__(self, signals):
            calls["n"] += 1
            return signals.enemy_resources

    fsm = FarmFSM(policy=CountingPolicy())
    simulator = Simulator(default_farm_world())
    result = ResourceTask(fsm=fsm).run(simulator)

    assert result.status == "SUCCESS"
    assert calls["n"] > 0


def test_task_summary_serializable():
    simulator = Simulator(default_farm_world())
    result = ResourceTask().run(simulator)
    data = result.summary()

    assert set(data) == {"status", "reason", "steps", "total_loot", "records"}
    assert isinstance(data["status"], str)
    assert data["steps"] == len(data["records"])
    first = data["records"][0]
    assert set(first) == {"step", "fsm_state", "action", "reward", "verified", "note"}


def test_simulator_can_be_reused_with_reset():
    task = ResourceTask()
    simulator = Simulator(default_farm_world())

    first = task.run(simulator)
    assert first.status == "SUCCESS"
    assert simulator.done

    simulator.reset()
    second = task.run(simulator)
    assert second.status == "SUCCESS"
    assert second.total_loot == first.total_loot  # 确定性：两次结果一致


def test_objective_only_farms_target_types():
    simulator = Simulator(
        [
            SimVillage(
                name="混合村",
                resources=[
                    ResourceSite("圣水收集器", (400, 400), capacity=3_000),
                    ResourceSite("金矿", (800, 400), capacity=5_000),
                ],
                defenses=[DefenseSite((600, 600))],
            )
        ]
    )

    result = ResourceTask().run(
        simulator,
        max_steps=40,
        objective=FarmObjective(
            requirements={"elixir": 2_000},
            target_types=("elixir",),
        ),
    )

    assert result.status == "SUCCESS"
    assert result.total_loot == 3_000
    assert simulator.own_amounts.elixir == 3_000
    assert simulator.own_amounts.gold == 0


def test_objective_already_met_returns_without_attacking():
    own = ResourceStatus(
        detected=2,
        amounts=Resources(elixir=3_000),
    )
    simulator = Simulator(
        [SimVillage(resources=[ResourceSite("圣水收集器", (400, 400), capacity=3_000)])],
        own_resources=own,
    )

    result = ResourceTask().run(
        simulator,
        objective=FarmObjective(
            requirements={"elixir": 2_000},
            target_types=("elixir",),
        ),
    )

    assert result.status == "SUCCESS"
    assert result.steps == 0
    assert simulator.screen == Simulator.SCREEN_VILLAGE
