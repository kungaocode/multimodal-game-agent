"""三环集成测试：捐兵 BLOCKED → 打资源补缺口 → 捐兵环继续完成。"""

from __future__ import annotations

from decision.arbiter import AgentArbiter, Ring, VillageSnapshot
from decision.donation_fsm import DonationFSM
from executor.donation_simulator import DonationRequest, DonationSimulator
from executor.simulator import DefenseSite, ResourceSite, Simulator, SimVillage
from state.game_state import Resources, ResourceThresholds
from tasks.agent_loop import AgentLoop, AgentPorts, RingOutcome
from tasks.donation import DonationTask
from tasks.resource import ResourceTask


def test_blocked_donation_is_resumed_after_farm_fills_missing_resources():
    donation_fsm = DonationFSM()
    donation_sim = DonationSimulator(
        [
            DonationRequest(
                (500, 300),
                troop_type="气球兵",
                cost=20_000,
                resource_type="elixir",
            )
        ],
        own_resources=Resources(elixir=10_000),
    )
    donation_task = DonationTask(fsm=donation_fsm)

    resource_sim = Simulator(
        [
            SimVillage(
                name="目标资源村",
                resources=[
                    ResourceSite("圣水收集器", (400, 400), capacity=10_000),
                ],
                defenses=[DefenseSite((600, 600))],
            )
        ],
        loot_per_deploy=10_000,
    )
    resource_task = ResourceTask()

    total_elixir = 10_000
    blocked_request = None

    def perceive() -> VillageSnapshot:
        return VillageSnapshot(
            resources=Resources(
                gold=10_000_000,
                elixir=total_elixir,
                dark_elixir=10_000_000,
            ),
            donation_blocked=blocked_request,
        )

    def run_farm(objective) -> RingOutcome:
        nonlocal total_elixir
        result = resource_task.run(resource_sim, max_steps=20, objective=objective)
        if result.status == "SUCCESS":
            total_elixir += resource_sim.own_amounts.elixir
            donation_sim.own_resources.elixir = total_elixir
        return RingOutcome(Ring.FARM, result.status, result.reason, objective=objective)

    def run_donate() -> RingOutcome:
        nonlocal total_elixir, blocked_request
        result = donation_task.run(
            donation_sim,
            max_steps=10,
            reset=blocked_request is None,
        )
        if result.status == "SUCCESS":
            total_elixir = donation_sim.own_resources.elixir
        if result.status == "BLOCKED":
            blocked_request = dict(result.blocked_request or {})
        return RingOutcome(
            Ring.DONATE,
            result.status,
            result.reason,
            blocked_request=blocked_request,
        )

    outcomes = AgentLoop(
        AgentPorts(
            perceive_village=perceive,
            run_collect=lambda: RingOutcome(Ring.COLLECT, "SUCCESS", "无采集目标"),
            run_farm=run_farm,
            run_donate=run_donate,
        ),
        arbiter=AgentArbiter(
            ResourceThresholds(
                gold_low=0,
                gold_resume=1,
                elixir_low=0,
                elixir_resume=1,
                dark_low=0,
                dark_resume=1,
            )
        ),
        max_rounds=10,
    ).run()

    assert [o.ring for o in outcomes] == [Ring.DONATE, Ring.FARM, Ring.DONATE]
    assert [o.status for o in outcomes] == ["BLOCKED", "SUCCESS", "SUCCESS"]
    assert donation_sim.summary()["donated"] == 1
    assert donation_sim.own_resources.elixir == 0
