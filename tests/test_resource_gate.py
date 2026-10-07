"""资源阈值滞回与捐兵缺口判定测试。"""

from __future__ import annotations

from decision.resource_gate import ResourceGate
from state.game_state import Resources, ResourceThresholds


def test_gate_triggers_below_low_and_keeps_farm_until_resume():
    gate = ResourceGate(
        ResourceThresholds(
            gold_low=1_000,
            gold_resume=3_000,
            elixir_low=2_000,
            elixir_resume=6_000,
            dark_low=100,
            dark_resume=300,
        )
    )

    first = gate.evaluate(Resources(gold=0, elixir=500, dark_elixir=50))
    assert first.needs_farm is True
    assert first.low_types == ("gold", "elixir", "dark_elixir")
    assert first.missing["elixir"] == 5_500

    # 已回归到 low 之上、resume 之下时仍保持打资源，避免临界抖动。
    between = gate.evaluate(Resources(gold=2_000, elixir=5_999, dark_elixir=299))
    assert between.needs_farm is True
    assert between.low_types == ("gold", "elixir", "dark_elixir")

    recovered = gate.evaluate(Resources(gold=3_000, elixir=6_000, dark_elixir=300))
    assert recovered.needs_farm is False
    assert recovered.low_types == ()

    # 解除后再次下探到 low 才重新触发。
    after_safe = gate.evaluate(Resources(gold=1_500, elixir=4_000, dark_elixir=200))
    assert after_safe.needs_farm is False

    retriggered = gate.evaluate(Resources(gold=900, elixir=1_000, dark_elixir=50))
    assert retriggered.needs_farm is True


def test_donation_feasibility_counts_safety_buffer():
    gate = ResourceGate(
        ResourceThresholds(donation_safety_buffer=100)
    )

    enough = gate.donation_feasibility(1_000, "elixir", Resources(elixir=1_100))
    assert enough.enough is True
    assert enough.required == 1_100
    assert enough.missing == 0

    short = gate.donation_feasibility(1_000, "elixir", Resources(elixir=900))
    assert short.enough is False
    assert short.required == 1_100
    assert short.missing == 200


def test_low_requirements_are_resume_values():
    gate = ResourceGate(
        ResourceThresholds(
            gold_low=500,
            gold_resume=1_500,
            elixir_low=400,
            elixir_resume=1_200,
        )
    )
    decision = gate.evaluate(Resources(gold=100, elixir=200, dark_elixir=10_000))
    assert gate.low_requirements(decision) == {"gold": 1_500, "elixir": 1_200}
