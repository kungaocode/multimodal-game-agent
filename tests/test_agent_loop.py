"""三环执行循环测试：仲裁顺序、目标注入与停止条件。"""

from __future__ import annotations

from decision.arbiter import AgentArbiter, Ring, VillageSnapshot
from decision.farm_fsm import FarmObjective
from state.game_state import Resources, ResourceThresholds
from tasks.agent_loop import AgentLoop, AgentPorts, RingOutcome


def test_loop_orders_collect_then_farm_then_donate():
    order: list[str] = []

    def tag(ring: Ring, status: str = "SUCCESS") -> RingOutcome:
        order.append(ring.value)
        return RingOutcome(ring, status, f"{ring.value} done")

    snapshots = [
        VillageSnapshot(collectibles=(("gold", (100, 100)),)),
        VillageSnapshot(
            resources=Resources(
                gold=10_000_000,
                elixir=0,
                dark_elixir=10_000_000,
            )
        ),
        VillageSnapshot(
            resources=Resources(elixir=1_000),
            donation_blocked={
                "resource_type": "elixir",
                "required": 2_000,
                "missing": 1_000,
            },
        ),
        VillageSnapshot(
            resources=Resources(
                gold=10_000_000,
                elixir=10_000_000,
                dark_elixir=10_000_000,
            )
        ),
    ]
    seen_objectives: list[FarmObjective | None] = []

    def perceive():
        return snapshots.pop(0)

    def collect():
        return tag(Ring.COLLECT)

    def farm(objective: FarmObjective) -> RingOutcome:
        seen_objectives.append(objective)
        return tag(Ring.FARM)

    def donate():
        return tag(Ring.DONATE)

    loop = AgentLoop(
        AgentPorts(
            perceive_village=perceive,
            run_collect=collect,
            run_farm=farm,
            run_donate=donate,
        ),
        max_rounds=10,
    )
    outcomes = loop.run()

    assert [o.ring for o in outcomes] == [
        Ring.COLLECT,
        Ring.FARM,
        Ring.FARM,
        Ring.DONATE,
    ]
    assert order == ["collect", "farm", "farm", "donate"]
    assert seen_objectives[0].target_types == ("elixir",)
    assert seen_objectives[1].requirements == {"elixir": 1_000}


def test_loop_stops_on_error():
    snapshots = [
        VillageSnapshot(
            resources=Resources(
                gold=10_000_000,
                elixir=10_000_000,
                dark_elixir=10_000_000,
            )
        ),
        VillageSnapshot(),
    ]

    def perceive():
        return snapshots.pop(0)

    def donate():
        return RingOutcome(Ring.DONATE, "ERROR", "mock failure")

    loop = AgentLoop(
        AgentPorts(
            perceive_village=perceive,
            run_collect=lambda: RingOutcome(Ring.COLLECT, "SUCCESS", "ok"),
            run_farm=lambda objective: RingOutcome(Ring.FARM, "SUCCESS", "ok"),
            run_donate=donate,
        ),
        arbiter=AgentArbiter(),
        max_rounds=10,
    )
    outcomes = loop.run()

    assert [o.ring for o in outcomes] == [Ring.DONATE]
    assert outcomes[0].status == "ERROR"


def test_loop_skips_second_farm_when_block_resource_replenished():
    blocked = {
        "resource_type": "elixir",
        "required": 2_000,
        "missing": 1_000,
    }
    snapshots = [
        VillageSnapshot(
            resources=Resources(gold=10_000_000, elixir=1_000, dark_elixir=10_000_000),
            donation_blocked=dict(blocked),
        ),
        VillageSnapshot(
            resources=Resources(gold=10_000_000, elixir=3_000, dark_elixir=10_000_000),
            donation_blocked=dict(blocked),
        ),
    ]

    def perceive():
        return snapshots.pop(0)

    farm_calls = []
    donate_calls = []

    def farm(objective: FarmObjective) -> RingOutcome:
        farm_calls.append(objective)
        return RingOutcome(Ring.FARM, "SUCCESS", "farm done")

    def donate() -> RingOutcome:
        donate_calls.append(True)
        return RingOutcome(Ring.DONATE, "SUCCESS", "donate done")

    outcomes = AgentLoop(
        AgentPorts(
            perceive_village=perceive,
            run_collect=lambda: RingOutcome(Ring.COLLECT, "SUCCESS", "ok"),
            run_farm=farm,
            run_donate=donate,
        ),
        max_rounds=5,
    ).run()

    assert [o.ring for o in outcomes] == [Ring.FARM, Ring.DONATE]
    assert len(farm_calls) == 1
    assert len(donate_calls) == 1
