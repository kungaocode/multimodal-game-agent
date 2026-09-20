"""Tests for the farm resource-target priority scoring (阶段 4)."""

import pytest

from decision.farm_fsm import FarmSignals
from decision.resource_policy import (
    BASE_RESOURCE_VALUE,
    DEFAULT_WEIGHTS,
    RESOURCE_NAMES,
    PolicyConfig,
    ResourcePolicy,
    full_resource_names,
    score_target,
)
from state.game_state import ResourceStatus


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS) == pytest.approx(1.0)


def test_all_resource_names_have_base_value():
    assert RESOURCE_NAMES
    assert all(name in BASE_RESOURCE_VALUE for name in RESOURCE_NAMES)


def test_scores_are_normalized():
    config = PolicyConfig()
    for name in RESOURCE_NAMES:
        for coords in [(0, 0), (640, 360), (1200, 700), (10, 700)]:
            target = score_target(name, coords, config)
            assert 0.0 <= target.score <= 1.0


def test_breakdown_contains_components():
    target = score_target("金矿", (640, 360), PolicyConfig())
    assert set(target.breakdown) == {"value", "accessibility", "safety", "distance"}


def test_full_resource_names_mapping():
    status = ResourceStatus(gold_full=True, elixir_full=True, dark_full=False, detected=3)
    assert full_resource_names(status) == {"金矿", "储金罐", "圣水收集器", "圣水瓶"}


def test_policy_skips_full_resource_types():
    policy = ResourcePolicy()
    status = ResourceStatus(gold_full=True, detected=2)
    out = policy(
        FarmSignals(
            enemy_resources=[("金矿", (1, 1)), ("储金罐", (2, 2)), ("圣水收集器", (3, 3))],
            resources=status,
        )
    )
    names = [name for name, _ in out]
    assert "金矿" not in names
    assert "储金罐" not in names
    assert "圣水收集器" in names


def test_higher_value_target_ranks_first():
    # 同一坐标下，价值更高的暗黑重油罐（1.0）应排在金矿（0.55）前面
    out = ResourcePolicy()(
        FarmSignals(enemy_resources=[("金矿", (640, 360)), ("暗黑重油罐", (640, 360))])
    )
    assert out[0][0] == "暗黑重油罐"


def test_closer_target_preferred_when_value_equal():
    config = PolicyConfig(deploy_point=(640, 700))
    out = ResourcePolicy(config)(
        FarmSignals(enemy_resources=[("金矿", (100, 100)), ("金矿", (620, 680))])
    )
    assert out[0][1] == (620, 680)


def test_safer_target_preferred():
    defenses = [(640, 360)]
    policy = ResourcePolicy(defense_provider=lambda signals: defenses)
    out = policy(
        FarmSignals(enemy_resources=[("金矿", (630, 350)), ("金矿", (900, 200))])
    )
    assert out[0][1] == (900, 200)


def test_top_k_limits_results():
    config = PolicyConfig(top_k=2)
    out = ResourcePolicy(config)(
        FarmSignals(enemy_resources=[("金矿", (1, 1)), ("储金罐", (2, 2)), ("圣水瓶", (3, 3))])
    )
    assert len(out) == 2


def test_empty_targets_yield_empty_result():
    assert ResourcePolicy()(FarmSignals()) == []


def test_policy_callable_matches_fsm_interface():
    out = ResourcePolicy()(FarmSignals(enemy_resources=[("金矿", (1, 1))]))
    assert isinstance(out, list)
    assert all(isinstance(name, str) and isinstance(coords, tuple) for name, coords in out)


def test_scored_exposes_breakdown_for_logging():
    scored = ResourcePolicy().scored(
        FarmSignals(enemy_resources=[("金矿", (640, 360))])
    )
    assert len(scored) == 1
    target = scored[0]
    w_v, w_a, w_s, w_d = DEFAULT_WEIGHTS
    assert target.score == pytest.approx(
        w_v * target.breakdown["value"]
        + w_a * target.breakdown["accessibility"]
        + w_s * target.breakdown["safety"]
        + w_d * target.breakdown["distance"]
    )
