"""三环仲裁器测试：互斥、采集优先、阻塞缺口触发打资源。"""

from __future__ import annotations

from decision.arbiter import AgentArbiter, Ring, VillageSnapshot
from decision.resource_gate import ResourceGate
from state.game_state import Resources, ResourceThresholds


def _arbiter(gate: ResourceGate | None = None) -> AgentArbiter:
    return AgentArbiter(
        ResourceThresholds(
            gold_low=1_000,
            gold_resume=3_000,
            elixir_low=1_000,
            elixir_resume=3_000,
            dark_low=50,
            dark_resume=200,
        ),
        gate=gate,
    )


def test_collect_has_priority_over_farm_and_donation():
    decision = _arbiter().decide(
        VillageSnapshot(
            collectibles=(("gold", (100, 100)),),
            donation_blocked={
                "resource_type": "elixir",
                "required": 2_000,
                "missing": 1_500,
            },
        )
    )
    assert decision.ring is Ring.COLLECT
    assert decision.farm_objective is None
    assert decision.blocked_request is None


def test_donation_block_uses_missing_amount_as_farm_objective():
    decision = _arbiter().decide(
        VillageSnapshot(
            resources=Resources(gold=5_000, elixir=1_000),
            donation_blocked={
                "resource_type": "elixir",
                "required": 2_500,
                "available": 1_000,
                "missing": 1_500,
            },
        )
    )

    assert decision.ring is Ring.FARM
    assert decision.farm_objective is not None
    assert decision.farm_objective.requirements == {"elixir": 1_500}
    assert decision.farm_objective.target_types == ("elixir",)
    assert decision.blocked_request == {
        "resource_type": "elixir",
        "required": 2_500,
        "available": 1_000,
        "missing": 1_500,
    }


def test_low_resource_triggers_farm_to_resume_level():
    decision = _arbiter().decide(
        VillageSnapshot(
            resources=Resources(
                gold=10_000,
                elixir=100,
                dark_elixir=10_000,
            )
        )
    )

    assert decision.ring is Ring.FARM
    assert decision.farm_objective is not None
    assert decision.farm_objective.requirements == {"elixir": 3_000}
    assert decision.farm_objective.target_types == ("elixir",)


def test_high_resources_and_no_block_enter_donation_ring():
    gate = ResourceGate(ResourceThresholds())
    gate.evaluate(Resources(gold=10_000_000, elixir=10_000_000, dark_elixir=100_000))
    decision = _arbiter(gate=gate).decide(
        VillageSnapshot(resources=Resources(gold=10_000_000, elixir=10_000_000, dark_elixir=100_000))
    )

    assert decision.ring is Ring.DONATE
    assert decision.reason == "资源充足，进入捐兵环"


def test_supplied_blocked_request_bypasses_farm():
    blocked = {
        "resource_type": "elixir",
        "required": 2_000,
        "missing": 1_000,
    }
    decision = _arbiter().decide(
        VillageSnapshot(
            resources=Resources(gold=10_000, elixir=2_000, dark_elixir=10_000),
            donation_blocked=blocked,
        )
    )

    assert decision.ring is Ring.DONATE
    assert decision.reason == "捐兵阻塞需求已补足"
    assert decision.farm_objective is None
